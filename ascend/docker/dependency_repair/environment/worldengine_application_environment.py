"""Real bevformer3.10 application environment; does not import Torch or edit SDK files."""
import os
from pathlib import Path
import sys

EXPECTED_PREFIX='/opt/conda/envs/bevformer'
BLOCKED=Path('/usr/local/Ascend')

def sdk_python_path(value):
    if not value:return False
    path=Path(value)
    raw=path.as_posix()
    resolved=path.resolve().as_posix()
    return any(x=='/usr/local/Ascend' or x.startswith('/usr/local/Ascend/') for x in (raw,resolved))

STATUS={'active':False,'python':sys.version,'prefix':sys.prefix,'before_sys_path':list(sys.path),
 'before_PYTHONPATH':os.environ.get('PYTHONPATH'),'removed':[]}
if sys.version_info[:2]==(3,10) and Path(sys.prefix).resolve()==Path(EXPECTED_PREFIX).resolve():
    STATUS['removed']=[x for x in sys.path if sdk_python_path(x)]
    sys.path[:]=[x for x in sys.path if not sdk_python_path(x)]
    actual=[x for x in os.environ.get('PYTHONPATH','').split(os.pathsep) if x and not sdk_python_path(x)]
    os.environ['PYTHONPATH']=os.pathsep.join(actual)
    STATUS.update(active=True,after_sys_path=list(sys.path),after_PYTHONPATH=os.environ['PYTHONPATH'],
      compiler_python=os.environ.get('HI_PYTHON'))

