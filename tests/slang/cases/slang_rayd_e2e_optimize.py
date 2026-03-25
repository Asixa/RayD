"""End-to-end optimization through Slang→rayd pipeline.

Optimize ray origin z so that t (returned by Slang's traceRayT calling
raydSceneIntersect internally) matches a target depth.

Triangle at z=0, ray going +z.  t = -oz.  Target t = 5  →  oz should → -5.
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
TARGET = 5.0

class SlangTraceT(torch.autograd.Function):
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
opt = torch.optim.Adam([oz], lr=0.3)

for _ in range(80):
    opt.zero_grad()
    t = SlangTraceT.apply(oz)
    loss = (t - TARGET) ** 2
    loss.backward()
    opt.step()

final_t = M.traceRayT(H, 0.25, 0.25, oz.item(), 0, 0, 1)
print(json.dumps({
    "oz": oz.item(),
    "final_t": final_t,
    "converged": abs(final_t - TARGET) < 0.2,
}))
