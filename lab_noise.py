"""Run a Noise training script under the lab recorder, resuming from the output's checkpoint if there is one.

    python -m torch.distributed.run --standalone --nproc_per_node=1 lab_noise.py train/val/run.py --output logs/x ...

The recorder (lab_run.py) writes the run's own page on rank 0: settings, environment, commit, and at the end the
script's metrics.jsonl and result.json. LAB_RUN_WORK, LAB_RUN_DIR, LAB_RUN_BRANCH, LAB_RUN_NAME, LAB_RUN_OBJECTIVE
and NOISE_COMMIT (the copy on the server has no .git) come from the environment.
"""
import json
import logging
import os
import runpy
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _output_of(argv: List[str]) -> Path:
    return Path(argv[argv.index('--output') + 1]) if '--output' in argv else Path('logs/val')


def _segments(output: Path) -> List[Path]:
    """Earlier segments of a resumed run, in order, then the current output."""
    return sorted(output.parent.glob(output.name + '.seg*'), key=lambda p: int(p.name.rsplit('seg', 1)[1])) + [output]


def _record(run, output: Path) -> None:
    for folder in _segments(output):
        metrics = folder / 'metrics.jsonl'
        if not metrics.exists():
            continue
        for line in metrics.read_text().splitlines():
            row = json.loads(line)
            for key in ('train_loss', 'val_loss', 'train_seconds'):
                if key in row:
                    run.metric(key, row[key], step=row['step'])
        run.attach(metrics, f'{folder.name}.metrics.jsonl')
    result = output / 'result.json'
    if result.exists():
        data = json.loads(result.read_text())
        for key in ('val_loss', 'val_loss_plain', 'train_seconds', 'peak_cuda_gb'):
            if key in data:
                run.metric(key, data[key], final=True)
        run.attach(result)
    for name in ('config.json', 'segment.json'):
        if (output / name).exists():
            run.attach(output / name)


def main() -> None:
    script, argv = sys.argv[1], sys.argv[2:]
    output = _output_of(argv)
    if (output / 'checkpoint.pt').exists() and '--resume' not in argv and not (output / 'result.json').exists():
        # The record script wants an empty output even on resume: move this segment aside and resume from it.
        segment = output.parent / f'{output.name}.seg{len(_segments(output))}'
        output.rename(segment)
        argv += ['--resume', str(segment / 'checkpoint.pt')]
        logger.warning('resuming from %s', segment / 'checkpoint.pt')
    sys.argv = [script] + argv
    if int(os.environ.get('RANK', 0)) or not os.environ.get('LAB_RUN_WORK'):
        runpy.run_path(script, run_name='__main__')
        return
    from lab_run import LabRun

    with LabRun(work=os.environ['LAB_RUN_WORK'], name=os.environ['LAB_RUN_NAME'],
                settings={'script': script, 'argv': ' '.join(argv), 'commit': os.environ.get('NOISE_COMMIT', '')},
                objective=os.environ.get('LAB_RUN_OBJECTIVE', ''),
                command=' '.join([script] + argv)) as run:
        try:
            runpy.run_path(script, run_name='__main__')
        finally:
            _record(run, output)


if __name__ == '__main__':
    main()
