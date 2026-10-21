"""Consume the actual training-entry argv after its real cwd transition; CPU only."""
import ast
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest


class TrainingOutputPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(__file__).resolve().parents[2] / 'train.py'
        tree = ast.parse(cls.source.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        cls.normalizer = [n for n in main.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                  and t.value.id == 'args' and t.attr == 'output' for t in n.targets)]
        if len(cls.normalizer) != 1:
            raise AssertionError('One production output normalization is required')
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'path' for t in n.targets))
        cls.tail = main.body[start:]

    def consume(self, relative, normalize=True):
        previous_cwd, previous_argv = Path.cwd(), sys.argv
        with tempfile.TemporaryDirectory(prefix='training-output-path-') as directory:
            root = Path(directory)
            alg = root / 'projects/AlgEngine'
            (alg / 'scripts').mkdir(parents=True)
            # This downstream fixture really opens sys.argv[1] after production
            # os.chdir(ALG). Model/data/device imports are intentionally absent.
            (alg / 'scripts/train.py').write_text(
                'from pathlib import Path\nimport json,runpy,sys\n'
                'loaded=runpy.run_path(sys.argv[1])\n'
                'assert loaded["fixture_value"]==47\n'
                'Path(sys.argv[sys.argv.index("--work-dir")+1],"consumed.json").write_text('
                'json.dumps(dict(config=str(Path(sys.argv[1]).resolve()),cwd=str(Path.cwd()))))\n',
                encoding='utf-8')
            output = Path('work/result') if relative else root / 'work/result'
            (root / output).mkdir(parents=True)

            class Config:
                def dump(self, path):
                    Path(path).write_text('fixture_value=47\n', encoding='utf-8')

            args = SimpleNamespace(output=output, profile='fixture', seed=20261012,
                                   deterministic=False, steps=2, runtime=True,
                                   validate=False, precision='fp32')
            nodes = (self.normalizer if normalize else []) + self.tail
            namespace = dict(args=args, cfg=Config(), ALG=alg, os=os, sys=sys, runpy=runpy)
            try:
                os.chdir(root)
                exec(compile(ast.Module(body=nodes, type_ignores=[]), str(self.source), 'exec'), namespace)
                consumed = json.loads((root / 'work/result/consumed.json').read_text())
                self.assertEqual(Path(consumed['cwd']), alg)
                self.assertEqual(Path(consumed['config']), root / 'work/result/fixture_rank0.py')
                self.assertTrue(Path(sys.argv[1]).is_absolute())
                self.assertIn('--no-validate', sys.argv)
            finally:
                os.chdir(previous_cwd)
                sys.argv = previous_argv

    def test_relative_output_config_is_consumed_after_chdir(self):
        self.consume(relative=True)

    def test_absolute_output_config_is_consumed_after_chdir(self):
        self.consume(relative=False)

    def test_original_relative_tail_without_normalization_fails_to_consume(self):
        with self.assertRaises(FileNotFoundError):
            self.consume(relative=True, normalize=False)


if __name__ == '__main__':
    unittest.main()
