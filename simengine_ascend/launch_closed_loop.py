"""Launch a reviewed simulation/planner argv pair with an explicit UTC cutoff.

Commands run without a shell. Only this launcher's direct child process groups
are signalled. Run Ray/container workloads in dedicated containers whose owner
also enforces the cutoff; this helper never issues a global Ray/NPU stop/reset.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from simengine_ascend.file_protocol import atomic_write,deadline_epoch
from simengine_ascend.owned_process import cleanup_owned,peek_exit


def publish(path,value):
    atomic_write(path,lambda stream:stream.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode()))


def validate_command(value):
    if not isinstance(value,dict) or not isinstance(value.get('argv'),list) or not value['argv']:
        raise ValueError('Each phase requires a nonempty argv list')
    if any(not isinstance(item,str) or not item or '\0' in item for item in value['argv']):
        raise ValueError('Command argv entries must be nonempty strings')
    if not isinstance(value.get('cwd'),str):raise ValueError('Each phase requires an explicit cwd')
    if any(not isinstance(key,str) or not isinstance(item,str) for key,item in value.get('env',{}).items()):
        raise ValueError('Explicit environment keys and values must be strings')


def validate_runner_reports(output,expected_scenarios,expected_scene_ids=None):
    """The original runner may exit zero while recording failed scenarios."""
    if not isinstance(expected_scenarios,int) or isinstance(expected_scenarios,bool) or expected_scenarios<1:
        raise ValueError('Expected scenario count must be positive')
    reports=sorted(Path(output).glob('runner_report_*.json'))
    if not reports:raise ValueError('Simulation produced no original runner report')
    rows=[];identities=set();sources={}
    for path in reports:
        if path.is_symlink():raise ValueError('Runner report must not be a symlink')
        raw=path.read_bytes();items=json.loads(raw)
        if not isinstance(items,list):raise ValueError('Original runner report must contain a list')
        sources[str(path)]=hashlib.sha256(raw).hexdigest()
        for item in items:
            if not isinstance(item,dict):raise ValueError('Invalid original runner report row')
            key=tuple(item.get(name) for name in ('scenario_name','log_name','planner_name'))
            # BaseEnv.run() deliberately publishes planner_name=None.
            if any(not isinstance(value,str) or not value for value in key[:2]) or \
                    (key[2] is not None and not isinstance(key[2],str)) or key in identities:
                raise ValueError('Duplicate or incomplete scenario identity in runner reports')
            identities.add(key)
            if item.get('succeeded') is not True or item.get('error_message') not in (None,''):
                raise ValueError('Simulation did not complete a scenario successfully: '+str(key))
            duration=item.get('duration_s')
            if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
                raise ValueError('Invalid completed simulation duration')
            rows.append(item)
    if len(rows)!=expected_scenarios:
        raise ValueError('Expected %d scenarios; runner reported %d'%(expected_scenarios,len(rows)))
    actual_ids=[item['scenario_name'] for item in rows]
    if len(set(actual_ids))!=len(actual_ids):raise ValueError('Scenario IDs are not unique across original runner reports')
    if expected_scene_ids is not None and set(actual_ids)!=set(expected_scene_ids):
        raise ValueError('Runner scene IDs differ from the explicitly selected input scenes')
    return dict(expected_scenarios=expected_scenarios,completed_unique_scenarios=len(rows),
                expected_scene_ids=sorted(expected_scene_ids) if expected_scene_ids is not None else None,
                original_runner_report_sha256=sources,scenarios=rows)


def launch(plan_path,run_root,deadline_utc,expected_scenarios=None,expected_scene_ids=None):
    deadline=deadline_epoch(deadline_utc)
    if deadline is None or deadline<=time.time():raise ValueError('A future timezone-aware --deadline-utc is required')
    if expected_scenarios is not None and (isinstance(expected_scenarios,bool) or not isinstance(expected_scenarios,int) or expected_scenarios<1):
        raise ValueError('Expected scenario count must be a positive integer')
    if expected_scene_ids is not None and (not isinstance(expected_scene_ids,list) or \
            any(not isinstance(value,str) or not value for value in expected_scene_ids) or \
            len(set(expected_scene_ids))!=len(expected_scene_ids) or len(expected_scene_ids)!=expected_scenarios):
        raise ValueError('Expected scene IDs must be a unique string list matching the expected scenario count')
    raw=Path(plan_path).read_bytes();plan=json.loads(raw)
    if set(plan)!={'simulation','planner'}:raise ValueError('Plan must contain exactly simulation and planner')
    for phase in plan.values():validate_command(phase)
    raw_root=Path(run_root)
    if '..' in raw_root.parts:raise ValueError('Run root must not contain parent traversal components')
    run_root=raw_root.resolve()
    if run_root.exists() or run_root.is_symlink():raise ValueError('Use a fresh run root')
    run_root.mkdir(parents=True,exist_ok=False)
    paths={name:run_root/name for name in ('frames','plan_traj','merged_ann_files','WE_output')}
    # The simulator wrapper requires a fresh output path and creates WE_output.
    # Original Hydra also creates output_dir, so direct original commands work.
    for name,path in paths.items():
        if name!='WE_output':path.mkdir()
    stop=run_root/'STOP';status=run_root/'launcher_report.json'
    replacements=dict(run_root=str(run_root),repo_root=str(Path(__file__).resolve().parents[1]),
                      deadline_utc=deadline_utc,stop_file=str(stop),**{name:str(path) for name,path in paths.items()})
    def expand(value):
        for name,replacement in replacements.items():value=value.replace('{'+name+'}',replacement)
        return value
    report=dict(result='running',started_utc=datetime.now(timezone.utc).isoformat(),
                deadline_utc=deadline_utc,plan_sha256=hashlib.sha256(raw).hexdigest(),
                children={},run_root=str(run_root),real_closed_loop_accepted=False)
    children={};interrupted=[]
    previous={number:signal.getsignal(number) for number in (signal.SIGTERM,signal.SIGINT)}
    for number in previous:signal.signal(number,lambda *_:interrupted.append('launcher_signal'))
    result=1
    try:
        publish(run_root/'launch_plan.json',plan);publish(status,report)
        for name in ('planner','simulation'):
            if time.time()>=deadline or interrupted:raise TimeoutError('Stopped before both phases could start')
            phase=plan[name];command=[expand(item) for item in phase['argv']]
            cwd=Path(expand(phase['cwd'])).resolve(strict=True)
            environment=dict(os.environ)
            environment.update({key:expand(value) for key,value in phase.get('env',{}).items()})
            environment.update(WORLDENGINE_CLOSED_LOOP_DEADLINE_UTC=deadline_utc,
                               WORLDENGINE_CLOSED_LOOP_STOP_FILE=str(stop))
            log=(run_root/(name+'.log')).open('xb',buffering=0)
            try:
                child=subprocess.Popen(command,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:
                log.close();raise
            children[name]=(child,log)
            report['children'][name]=dict(pid=child.pid,command=command,cwd=str(cwd),log=str(run_root/(name+'.log')))
            publish(status,report)
        while True:
            if interrupted or stop.exists():raise InterruptedError('Launcher stop requested')
            if time.time()>=deadline:raise TimeoutError('Closed-loop launcher deadline reached')
            codes={name:peek_exit(child) for name,(child,_) in children.items()}
            if any(code is not None and code!=0 for code in codes.values()):
                raise RuntimeError('Owned closed-loop phase exited nonzero: '+str(codes))
            if all(code==0 for code in codes.values()):
                if expected_scenarios is not None:
                    report['simulation_report_validation']=validate_runner_reports(paths['WE_output'],expected_scenarios,expected_scene_ids)
                else:
                    report['simulation_report_validation']='Not requested by caller; process protocol fixture only'
                report['result']='both_owned_commands_completed';result=0;break
            time.sleep(.1)
    except BaseException as error:
        report.update(result='stopped_or_error',error_type=type(error).__name__,error=str(error))
        result=2 if isinstance(error,(TimeoutError,InterruptedError,KeyboardInterrupt)) else 1
    finally:
        cleanup_errors=[]
        try:atomic_write(stop,lambda stream:stream.write(b'owned launcher finished\n'))
        except OSError as error:cleanup_errors.append('STOP publication: '+str(error))
        for name,(child,log) in children.items():
            try:
                # The simulator wrapper supervises its own separate session
                # with a three-second grace; let it finish that cleanup first.
                cleanup_owned(child,grace_seconds=8)
            except (OSError,RuntimeError,subprocess.TimeoutExpired) as error:
                cleanup_errors.append('WAIT/KILL pid '+str(child.pid)+': '+str(error))
            finally:
                log.close();report['children'][name]['actual_exit']=child.returncode
        for number,handler in previous.items():signal.signal(number,handler)
        if cleanup_errors:
            result=result or 1
            report.update(result='cleanup_error',cleanup_errors=cleanup_errors)
        report.update(actual_exit=result,completed_utc=datetime.now(timezone.utc).isoformat(),
                      cleanup_scope='Linux retained direct-child process groups before leader reap; detached sessions remain with the container owner')
        publish(status,report)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--deadline-utc',required=True)
    parser.add_argument('--expected-scenarios',type=int,required=True,
                        help='Exact number of unique successful scenarios required in original runner reports')
    parser.add_argument('--expected-scene-ids',type=Path,required=True,
                        help='JSON list containing the exact selected scenario IDs')
    args=parser.parse_args()
    if args.expected_scenarios<1:parser.error('--expected-scenarios must be positive')
    return launch(args.plan,args.run_root,args.deadline_utc,args.expected_scenarios,
                  json.loads(args.expected_scene_ids.read_bytes()))


if __name__=='__main__':raise SystemExit(main())
