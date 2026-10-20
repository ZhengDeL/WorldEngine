"""CPU-only path-contract fixtures; do not import Torch, MMCV or the model."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'train.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
HELPER = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
              and node.name == '_resolve_dataset_filter_paths')
NAMESPACE = {'Path': Path}
exec(compile(ast.Module(body=[HELPER], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
RESOLVE = NAMESPACE[HELPER.name]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class DatasetFilterPaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        self.alg = self.repo / 'projects/AlgEngine'
        self.external = self.root / 'external'
        self.alg.mkdir(parents=True)
        self.external.mkdir()
        self.relative = {
            'train': 'configs/navsim_splits/navtrain_split/navtrain.yaml',
            'val': 'configs/navsim_splits/navtest_split/navtest.yaml',
            'test': 'configs/navsim_splits/navtest_split/navtest.yaml',
        }
        for name in set(self.relative.values()):
            self.make_file(self.alg / name)
        self.cfg = SimpleNamespace(data={phase: {'nav_filter_path': name, 'unrelated': [1, 2]}
                                        for phase, name in self.relative.items()})

    def make_file(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('tokens: [official-fixture-token]\n', encoding='utf-8')
        return path.resolve()

    def test_repository_and_external_cwd_use_alg_base(self):
        for cwd in (self.repo, self.external):
            cfg = copy.deepcopy(self.cfg)
            with working_directory(cwd):
                RESOLVE(cfg, self.alg)
            for phase, relative in self.relative.items():
                self.assertEqual(cfg.data[phase]['nav_filter_path'], str((self.alg / relative).resolve()))
                self.assertEqual(cfg.data[phase]['unrelated'], [1, 2])

    def test_absolute_cli_filter_is_not_rebased(self):
        chosen = self.make_file(self.external / 'selected_navtrain.yaml')
        self.cfg.data['train']['nav_filter_path'] = str(chosen)
        with working_directory(self.repo):
            RESOLVE(self.cfg, self.alg)
        self.assertEqual(self.cfg.data['train']['nav_filter_path'], str(chosen))

    def test_mini_absolute_filters_and_bytes_are_preserved(self):
        before = {}
        for phase in self.relative:
            path = self.make_file(self.external / ('mini_%s_navtest.yaml' % phase))
            self.cfg.data[phase]['nav_filter_path'] = str(path)
            before[phase] = (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
        RESOLVE(self.cfg, self.alg)
        for phase, (path, sha) in before.items():
            self.assertEqual(self.cfg.data[phase]['nav_filter_path'], path)
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), sha)

    def test_rare_string_becomes_one_path_in_each_phase(self):
        name = 'configs/navsim_splits/rare_case.yaml'
        expected = str(self.make_file(self.alg / name))
        for phase in self.relative:
            self.cfg.data[phase]['finetune_yaml'] = name
        with working_directory(self.external):
            RESOLVE(self.cfg, self.alg)
        for phase in self.relative:
            self.assertEqual(self.cfg.data[phase]['finetune_yaml'], [expected])

    def test_rare_list_keeps_order_duplicates_and_absolute_entries(self):
        second = self.make_file(self.alg / 'configs/second.yaml')
        first = self.make_file(self.alg / 'configs/first.yaml')
        absolute = self.make_file(self.external / 'cli_rare.yaml')
        self.cfg.data['train']['finetune_yaml'] = ['configs/second.yaml', str(absolute), 'configs/first.yaml', 'configs/second.yaml']
        RESOLVE(self.cfg, self.alg)
        self.assertEqual(self.cfg.data['train']['finetune_yaml'],
                         [str(second), str(absolute), str(first), str(second)])

    def test_none_empty_and_missing_rare_filters_add_no_entries(self):
        self.cfg.data['train']['finetune_yaml'] = None
        self.cfg.data['val']['finetune_yaml'] = []
        RESOLVE(self.cfg, self.alg)
        self.assertIsNone(self.cfg.data['train']['finetune_yaml'])
        self.assertEqual(self.cfg.data['val']['finetune_yaml'], [])
        self.assertNotIn('finetune_yaml', self.cfg.data['test'])
        self.assertEqual(list(self.cfg.data['train'].get('finetune_yaml') or []), [])

    def test_missing_nav_filter_is_an_error(self):
        self.cfg.data['train']['nav_filter_path'] = 'configs/missing_navtrain.yaml'
        with self.assertRaises(FileNotFoundError):
            RESOLVE(self.cfg, self.alg)

    def test_missing_rare_filter_is_an_error(self):
        self.cfg.data['train']['finetune_yaml'] = ['configs/missing_rare.yaml']
        with self.assertRaises(FileNotFoundError):
            RESOLVE(self.cfg, self.alg)

    def test_directories_empty_paths_and_wrong_types_are_errors(self):
        for value, error in [(str(self.alg), FileNotFoundError), ('', ValueError), (7, TypeError)]:
            cfg = copy.deepcopy(self.cfg)
            cfg.data['train']['nav_filter_path'] = value
            with self.assertRaises(error):
                RESOLVE(cfg, self.alg)
        cfg = copy.deepcopy(self.cfg)
        cfg.data['train']['finetune_yaml'] = {'path': 'configs/file.yaml'}
        with self.assertRaises(TypeError):
            RESOLVE(cfg, self.alg)

    def test_navtrain_and_navtest_basenames_are_unchanged(self):
        RESOLVE(self.cfg, self.alg)
        for phase, relative in self.relative.items():
            self.assertEqual(Path(self.cfg.data[phase]['nav_filter_path']).name, Path(relative).name)

    def test_normalization_is_idempotent(self):
        RESOLVE(self.cfg, self.alg)
        expected = copy.deepcopy(self.cfg.data)
        with working_directory(self.external):
            RESOLVE(self.cfg, self.alg)
        self.assertEqual(self.cfg.data, expected)

    def test_helper_runs_after_overrides_and_before_runtime_digest(self):
        main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == HELPER.name]
        self.assertEqual(len(calls), 1)
        contract = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'contract' for target in node.targets))
        dump = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'dump')
        filter_overrides = [node for node in ast.walk(main) if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Attribute) and target.attr in ('nav_filter_path', 'finetune_yaml')
                                    for target in node.targets)]
        self.assertGreater(calls[0].lineno, max(node.lineno for node in filter_overrides))
        self.assertLess(calls[0].lineno, contract.lineno)
        self.assertLess(calls[0].lineno, dump.lineno)


if __name__ == '__main__':
    unittest.main()
