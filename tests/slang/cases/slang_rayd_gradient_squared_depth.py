"""Gradient of squaredDepth (t^2) w.r.t. ray origin z, through Slang→rayd.

squaredDepth = t^2,  t = -oz  →  d(t^2)/d(oz) = 2t * dt/doz = 2*(-oz)*(-1) = 2*oz
At oz=-2: grad = 2*(-2) = -4   (note: negative because moving origin further
away increases t, but squared growth means d(t^2)/doz = -2t = -2*2 = -4)

More carefully: t = 0 - oz = -oz = 2,  d(t^2)/d(oz) = 2*t * (-1) = -4
"""
import json, torch
import rayd as rd, rayd.slang as rs
import drjit.cuda as cuda
from pathlib import Path

SHADER = str(Path(__file__).resolve().parent.parent / "shaders" / "rayd_query.slang")
M = rs.load_module(SHADER, link_rayd=True)

mesh = rd.Mesh(
    cuda.Array3f([0, 1, 0], [0, 0, 1], [0, 0, 0]),
    cuda.Array3i([0], [1], [2]),
)
scene = rd.Scene(); scene.add_mesh(mesh); scene.configure()
H = scene.slang_handle

class SlangSqDepth(torch.autograd.Function):
    EPS = 1e-3
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        return torch.tensor(M.squaredDepth(H, 0.25, 0.25, oz.item(), 0, 0, 1),
                            device=oz.device)
    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        e = SlangSqDepth.EPS
        fp = M.squaredDepth(H, 0.25, 0.25, oz.item() + e, 0, 0, 1)
        fm = M.squaredDepth(H, 0.25, 0.25, oz.item() - e, 0, 0, 1)
        return g * (fp - fm) / (2 * e)

oz = torch.tensor(-2.0, device="cuda", requires_grad=True)
loss = SlangSqDepth.apply(oz)
loss.backward()

print(json.dumps({
    "sq_depth": loss.item(),
    "grad_oz": oz.grad.item(),
    "expected_grad": -4.0,
    "match": abs(oz.grad.item() - (-4.0)) < 0.05,
}))
