"""Gradient of inverseDepth (1/t) w.r.t. ray origin z, through Slang→rayd.

inverseDepth = 1/t,  t = -oz  (for z-directed ray hitting z=0 plane)
d(1/t)/d(oz) = d(1/(-oz))/d(oz) = 1/oz^2
At oz=-1: grad = 1.0
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

class SlangInvDepth(torch.autograd.Function):
    EPS = 1e-3
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        return torch.tensor(M.inverseDepth(H, 0.25, 0.25, oz.item(), 0, 0, 1),
                            device=oz.device)
    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        e = SlangInvDepth.EPS
        fp = M.inverseDepth(H, 0.25, 0.25, oz.item() + e, 0, 0, 1)
        fm = M.inverseDepth(H, 0.25, 0.25, oz.item() - e, 0, 0, 1)
        return g * (fp - fm) / (2 * e)

oz = torch.tensor(-1.0, device="cuda", requires_grad=True)
loss = SlangInvDepth.apply(oz)
loss.backward()

print(json.dumps({
    "inv_depth": loss.item(),
    "grad_oz": oz.grad.item(),
    "grad_close_to_one": abs(oz.grad.item() - 1.0) < 0.02,
}))
