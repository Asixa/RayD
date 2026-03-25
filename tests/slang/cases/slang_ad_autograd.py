"""Test: Slang-compiled traceAD returns IntersectionAD with dt_do/dt_dd.
Backward uses traceAD in a single call (no redundant AD passes).
"""
import json, torch
import rayd as rd, rayd.slang as rs
import drjit.cuda as cuda
from pathlib import Path

SHADER = str(Path(__file__).resolve().parent.parent / "shaders" / "rayd_query_ad.slang")
M = rs.load_module(SHADER, link_rayd=True)

mesh = rd.Mesh(
    cuda.Array3f([0, 1, 0], [0, 0, 1], [0, 0, 0]),
    cuda.Array3i([0], [1], [2]),
)
scene = rd.Scene(); scene.add_mesh(mesh); scene.configure()
H = scene.slang_handle


class SlangTrace(torch.autograd.Function):
    """Forward and backward both use traceAD (single AD pass)."""
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        hit = M.traceAD(H, 0.25, 0.25, oz.item(), 0.0, 0.0, 1.0)
        return torch.tensor(hit.t, device=oz.device)

    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        hit = M.traceAD(H, 0.25, 0.25, oz.item(), 0.0, 0.0, 1.0)
        return torch.tensor(hit.dt_do.z * g.item(), device=oz.device)


oz = torch.tensor(-1.0, device="cuda", requires_grad=True)
t = SlangTrace.apply(oz)
t.backward()

print(json.dumps({
    "t": t.item(),
    "grad_oz": oz.grad.item(),
    "t_correct": abs(t.item() - 1.0) < 1e-5,
    "grad_correct": abs(oz.grad.item() - (-1.0)) < 1e-5,
}))
