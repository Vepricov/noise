"""Credit-quality bench for Noise's Cut: how much of the full-cycle estimate survives fewer forward moves.

    python zo_bench.py --checkpoint logs/<run>/checkpoint.pt --sequences 1 --out bench.json

One fixed batch, one model state. The reference is the record's own estimate (full chain cycles, its key and future
draws). Each variant changes one budget and is compared with the reference matrix by matrix: cosine of the raw
estimate, cosine of its polar factor (what ANVIL steps along), and the wall time of one estimate. The reference is
measured twice with different seeds, which gives the noise floor of its own partial draws. Autograd of the true loss
is reported once as context only; nothing here trains.
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


def load_record(path: Path):
    spec = importlib.util.spec_from_file_location('record', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partial_chain(record, rows: int):
    """Cut whose chain moves each root along `rows` random rows of the signed Hadamard cycle instead of all W.

    (1/rows) sum over a uniform row subset of <c, h_j> h_j is unbiased for c, so the credit is rescaled by W/rows."""

    class PartialCut(record.Cut):
        def basis(self):
            full = super().basis()
            return full[torch.randperm(self.width, device=self.device, generator=self.generator)]

        def chain_credits(self, *args):
            whole = self.share
            self.share = lambda total: (0, rows) if total == self.width else whole(total)
            try:
                credits = super().chain_credits(*args)
            finally:
                del self.share
            for key in credits:
                if not isinstance(key, tuple):
                    credits[key] = credits[key] * (self.width / rows)
            return credits

    return PartialCut


def shaped_chain(record, transform):
    """Cut whose chain credits pass through `transform(site, credit)`: simulates measuring only some directions.

    Measuring the full cycle and then keeping c U U^T is exactly what moving along the orthonormal columns of U
    would measure, so the cosine is the real one; the time is not (it is the full chain's)."""

    class ShapedCut(record.Cut):
        def chain_credits(self, *args):
            credits = super().chain_credits(*args)
            self.recorded = {k: v for k, v in credits.items() if not isinstance(k, tuple)}
            return {k: (v if isinstance(k, tuple) else transform(k, v, self.generator)) for k, v in credits.items()}

    return ShapedCut


def top_subspace(credits: Dict, k: int) -> Dict:
    """Per site, the top-k eigenvectors of the credit covariance C^T C (W x k, orthonormal columns)."""
    return {site: torch.linalg.eigh(c.T @ c)[1][:, -k:] for site, c in credits.items()}


def project(bases: Dict, extra: int):
    """c -> c U U^T, plus an unbiased random estimate of the rest from `extra` orthonormal complement directions."""

    def transform(site, c, generator):
        u = bases[site]
        kept = (c @ u) @ u.T
        if not extra:
            return kept
        w = c.shape[1]
        g = torch.randn(w, extra, device=c.device, generator=generator)
        q = torch.linalg.qr(g - u @ (u.T @ g))[0]  # `extra` random orthonormal directions orthogonal to U
        return kept + (c @ q) @ q.T * ((w - u.shape[1]) / extra)

    return transform


def estimate(record, cls, model, settings, x, y, seed: int):
    estimator = cls(model, settings, seed, 0, 1, 1, None)
    torch.cuda.synchronize()
    start = time.perf_counter()
    gradients = estimator.estimate(x, y)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    estimator.close()
    return {k: v.float().clone() for k, v in gradients.items()}, seconds


def autograd_reference(model, x, y) -> Dict[str, torch.Tensor]:
    params = dict(model.named_parameters())
    for p in params.values():
        p.requires_grad_(True)
    with torch.autocast('cuda', dtype=torch.bfloat16):  # as utils/eval.py scores the model
        loss = model(x, y, loss_reduction='sum')
    grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
    for p in params.values():
        p.requires_grad_(False)
    return {k: (g if g is not None else torch.zeros_like(p)).float() for (k, p), g in zip(params.items(), grads)}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()).clamp_min(1e-30))


def compare(record, ref: Dict[str, torch.Tensor], other: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Mean cosines over block matrices (raw and polar) and over everything else (raw)."""
    raw, polar, rest = [], [], []
    for name, g in ref.items():
        if g.norm() == 0:
            continue
        if '.h.' in name and g.dim() == 2:
            raw.append(cosine(g, other[name]))
            polar.append(cosine(record.polar(g), record.polar(other[name])))
        else:
            rest.append(cosine(g, other[name]))
    mean = lambda v: sum(v) / max(1, len(v))
    return dict(block_raw=mean(raw), block_polar=mean(polar), block_raw_min=min(raw), other_raw=mean(rest))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--record', type=Path, default=ROOT / 'train/val/run.py')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--sequences', type=int, default=1)
    parser.add_argument('--row', type=int, default=2000, help='first training row of the fixed batch')
    parser.add_argument('--chain-rows', type=int, nargs='*', default=[256, 128, 64])
    parser.add_argument('--key-draws', type=int, nargs='*', default=[256, 128])
    parser.add_argument('--future-draws', type=int, nargs='*', default=[64, 32])
    parser.add_argument('--subspace', type=int, nargs='*', default=[64, 128],
                        help='chain directions from the credit covariance of a neighbouring batch')
    parser.add_argument('--complement', type=int, default=64, help='random complement directions added to each')
    parser.add_argument('--out', type=Path, default=Path('bench.json'))
    args = parser.parse_args()

    record = load_record(args.record)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    settings = record.Settings()
    model = record.make_model(device, record.SEED, settings.model, settings.projection_std, settings.head_std)
    step: Optional[int] = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        step = state['step']
    rows = record.load_rows(args.data_dir / 'train.pt', record.TRAIN_ROWS)
    batch = rows[args.row:args.row + args.sequences].to(device)
    x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    neighbour = rows[args.row + args.sequences:args.row + 2 * args.sequences].to(device)

    variants: List[tuple] = [('reference', record.Cut, settings, 1), ('reference_reseeded', record.Cut, settings, 2)]
    variants += [(f'chain_rows_{r}', partial_chain(record, r), settings, 1) for r in args.chain_rows]
    variants += [(f'key_draws_{k}', record.Cut, record.Settings(key_path_draws=k), 1) for k in args.key_draws]
    variants += [(f'future_draws_{f}', record.Cut, record.Settings(future_draws=f), 1) for f in args.future_draws]

    if args.subspace:  # the credit covariance of the neighbouring batch, measured with the full chain
        recorder = shaped_chain(record, lambda site, c, generator: c)(model, settings, 3, 0, 1, 1, None)
        recorder.estimate(neighbour[:, :-1].contiguous(), neighbour[:, 1:].contiguous())
        recorder.close()
        for k in args.subspace:
            bases = top_subspace(recorder.recorded, k)
            variants.append((f'subspace_{k}', shaped_chain(record, project(bases, 0)), settings, 1))
            variants.append((f'subspace_{k}+rand_{args.complement}',
                             shaped_chain(record, project(bases, args.complement)), settings, 1))
            variants.append((f'rand_{k + args.complement}_simulated',
                             shaped_chain(record, project({s: b[:, :0] for s, b in bases.items()},
                                                          k + args.complement)), settings, 1))
        del recorder
    ref = None
    rows_out = []
    for name, cls, recipe, seed in variants:
        estimate(record, cls, model, recipe, x, y, seed + 100)  # compiles this variant's shapes; untimed
        grads, seconds = estimate(record, cls, model, recipe, x, y, seed)
        ref = grads if ref is None else ref
        row = dict(variant=name, seconds=round(seconds, 2), **compare(record, ref, grads))
        logger.info(json.dumps(row))
        rows_out.append(row)
    truth = autograd_reference(model, x, y)
    context = dict(reference_vs_autograd=compare(record, truth, ref))
    logger.info(json.dumps(context))
    result = dict(step=step, sequences=args.sequences, row=args.row, rows=rows_out, **context)
    args.out.write_text(json.dumps(result, indent=2) + '\n')

    if os.environ.get('LAB_RUN_WORK'):
        from lab_run import LabRun

        with LabRun(work=os.environ['LAB_RUN_WORK'], name=os.environ['LAB_RUN_NAME'],
                    settings=dict(vars(args), commit=os.environ.get('NOISE_COMMIT', '')),
                    objective=os.environ.get('LAB_RUN_OBJECTIVE', ''), command=' '.join(sys.argv)) as run:
            for row in rows_out:
                for key in ('seconds', 'block_raw', 'block_polar', 'other_raw'):
                    run.metric(f"{row['variant']}.{key}", row[key], final=True)
            run.attach(args.out)


if __name__ == '__main__':
    main()
