"""Import-system fixtures for the separately pinned planning checkout."""
import importlib
import importlib.machinery
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest

from simengine_ascend.closed_loop_runtime import assert_module_origins


NAME='mmdet3d_plugin.datasets.data_utils'


class ModuleOriginsTest(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)/'selected'
        self.plugin=self.root/'projects/AlgEngine/mmdet3d_plugin'
        self.namespace=self.plugin/'datasets/data_utils'
        self.namespace.mkdir(parents=True)
        (self.root/'ascend').mkdir()
        for folder in (self.plugin,self.plugin/'datasets'):
            (folder/'__init__.py').write_text('')
        (self.namespace/'vector_map.py').write_text('VALUE=1\n')
        self.saved={key:value for key,value in list(sys.modules.items())
                    if key.split('.')[0] in ('ascend','mmdet3d_plugin')}
        for key in self.saved:del sys.modules[key]
        self.old_path=list(sys.path)
        sys.path.insert(0,str(self.plugin.parent))
        importlib.invalidate_caches()
        self.addCleanup(self.restore_imports)

    def restore_imports(self):
        for key in list(sys.modules):
            if key.split('.')[0] in ('ascend','mmdet3d_plugin'):del sys.modules[key]
        sys.modules.update(self.saved)
        sys.path[:]=self.old_path
        importlib.invalidate_caches()

    def load(self):
        return importlib.import_module(NAME)

    def assert_rejected(self):
        with self.assertRaises(RuntimeError):assert_module_origins(self.root,require_plugin=True)

    def test_real_pep420_namespace_and_child_file_are_audited(self):
        namespace=self.load()
        self.assertIsNone(namespace.__file__)
        self.assertIsNone(namespace.__spec__.origin)
        child=importlib.import_module(NAME+'.vector_map')
        self.assertEqual(child.VALUE,1)
        origins=assert_module_origins(self.root,require_plugin=True)
        self.assertEqual(origins[NAME],dict(kind='namespace_package',search_locations=[str(self.namespace)]))
        self.assertEqual(origins[NAME+'.vector_map'],str(self.namespace/'vector_map.py'))

    def test_namespace_requires_matching_spec_name(self):
        namespace=self.load();namespace.__spec__.name=NAME+'.foreign'
        self.assert_rejected()

    def test_namespace_without_spec_is_not_exempted(self):
        namespace=self.load();namespace.__spec__=None
        self.assert_rejected()

    def test_namespace_rejects_file_origin_or_has_location(self):
        namespace=self.load();namespace.__spec__.origin='foreign.py'
        self.assert_rejected()
        namespace.__spec__.origin=None;namespace.__spec__.has_location=True
        self.assert_rejected()

    def test_namespace_rejects_foreign_path_even_when_spec_agrees(self):
        namespace=self.load()
        foreign=self.root.parent/'foreign';foreign.mkdir()
        namespace.__path__=[str(self.namespace),str(foreign)]
        namespace.__spec__.submodule_search_locations=list(namespace.__path__)
        self.assert_rejected()

    def test_namespace_rejects_spec_only_path_injection(self):
        namespace=self.load()
        foreign=self.root.parent/'foreign';foreign.mkdir()
        namespace.__spec__.submodule_search_locations=[str(foreign)]
        self.assert_rejected()

    def test_namespace_rejects_wrong_directory_inside_selected_tree(self):
        namespace=self.load()
        other=self.plugin/'other';other.mkdir()
        namespace.__path__=[str(other)]
        namespace.__spec__.submodule_search_locations=[str(other)]
        self.assert_rejected()

    def test_regular_package_rejects_external_search_path(self):
        self.load()
        package=sys.modules['mmdet3d_plugin']
        foreign=self.root.parent/'foreign';foreign.mkdir()
        package.__path__=[str(self.plugin),str(foreign)]
        package.__spec__.submodule_search_locations=list(package.__path__)
        self.assert_rejected()

    def test_namespace_rejects_empty_and_duplicate_paths(self):
        namespace=self.load()
        for paths in ([],[str(self.namespace),str(self.namespace)]):
            namespace.__path__=paths
            namespace.__spec__.submodule_search_locations=list(paths)
            self.assert_rejected()

    def test_plain_no_file_module_cannot_claim_namespace_status(self):
        self.load()
        fake=ModuleType(NAME)
        fake.__spec__=importlib.machinery.ModuleSpec(NAME,loader=None)
        sys.modules[NAME]=fake
        self.assert_rejected()

    def test_namespace_symlink_cannot_escape_selected_tree(self):
        namespace=self.load()
        foreign=self.root.parent/'foreign';foreign.mkdir()
        link=self.plugin/'linked'
        try:link.symlink_to(foreign,target_is_directory=True)
        except (OSError,NotImplementedError) as error:self.skipTest('Symlink fixture unavailable: '+str(error))
        namespace.__path__=[str(link)]
        namespace.__spec__.submodule_search_locations=[str(link)]
        self.assert_rejected()

    def test_regular_child_file_outside_selected_tree_is_rejected(self):
        self.load()
        foreign=self.root.parent/'foreign.py';foreign.write_text('')
        child=importlib.import_module(NAME+'.vector_map');child.__file__=str(foreign)
        self.assert_rejected()


if __name__=='__main__':unittest.main(verbosity=2)
