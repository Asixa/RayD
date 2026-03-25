"""Forward / backward through Slang→rayd: verify FD gradient for ray origin z.

All rayd calls happen inside Slang (traceRayT). The Python side wraps
the scalar Slang function in torch.autograd.Function with central-FD backward.
"""
import json, torch
import rayd as rd, rayd.slang as rs
import drjit.cuda as cuda
from pathlib import Path

SHADER = str(Path(__file__).resolve().parent.parent / "shaders" / "rayd_query.slang")
M = rs.load_module(SHADER, link_rayd=True)

# Build scene: triangle at z=3
mesh = rd.Mesh(
    cuda.Array3f([0, 1, 0], [0, 0, 1], [0, 0, 0]),
    cuda.Array3i([0], [1], [2]),
)
scene = rd.Scene(); scene.add_mesh(mesh); scene.configure()
H = scene.slang_handle

class SlangTraceT(torch.autograd.Function):
    """f(oz) = traceRayT(scene, 0.25, 0.25, oz, 0, 0, 1)"""
    EPS = 1e-3
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        return torch.tensor(M.traceRayT(H, 0.25, 0.25, oz.item(), 0, 0, 1),
                            device=oz.device)
    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        e = SlangTraceT.EPS
        tp = M.traceRayT(H, 0.25, 0.25, oz.item() + e, 0, 0, 1)
        tm = M.traceRayT(H, 0.25, 0.25, oz.item() - e, 0, 0, 1)
        return g * (tp - tm) / (2 * e)

oz = torch.tensor(-1.0, device="cuda", requires_grad=True)
t = SlangTraceT.apply(oz)
t.backward()

# Analytical: ray at (0.25, 0.25, oz) going +z hits z=0 plane at t = -oz
# dt/doz = -1
print(json.dumps({
    "t": t.item(),
    "grad_oz": oz.grad.item(),
    "grad_close_to_minus_one": abs(oz.grad.item() - (-1.0)) < 0.01,
}))
