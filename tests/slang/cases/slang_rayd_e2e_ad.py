"""End-to-end optimization using Dr.Jit AD gradients from Slang→rayd.

Optimize ray origin z so that t matches a target depth.
All rayd calls go through Slang. Backward uses AD (not FD).
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
TARGET = 5.0


class SlangTraceAD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        return torch.tensor(M.traceT(H, 0.25, 0.25, oz.item(), 0, 0, 1),
                            device=oz.device)

    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        # Dr.Jit AD gradient — NOT finite differences
        dtdoz = M.dtdoz(H, 0.25, 0.25, oz.item(), 0, 0, 1)
        return g * dtdoz


oz = torch.tensor(-1.0, device="cuda", requires_grad=True)
opt = torch.optim.Adam([oz], lr=0.3)

for _ in range(80):
    opt.zero_grad()
    t = SlangTraceAD.apply(oz)
    loss = (t - TARGET) ** 2
    loss.backward()
    opt.step()

final_t = M.traceT(H, 0.25, 0.25, oz.item(), 0, 0, 1)
print(json.dumps({
    "oz": oz.item(),
    "final_t": final_t,
    "converged": abs(final_t - TARGET) < 0.2,
}))
