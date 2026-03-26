"""Test: Adam optimization using Slang-compiled traceAD.

All gradient computation happens in Slang/C++ via traceAD (single AD pass).
Python only bridges into torch.optim.
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
scene = rd.Scene(); scene.add_mesh(mesh); scene.build()
H = scene.slang_handle
TARGET = 5.0


class SlangTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        hit = M.traceAD(H, 0.25, 0.25, oz.item(), 0, 0, 1)
        return torch.tensor(hit.t, device=oz.device)

    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        hit = M.traceAD(H, 0.25, 0.25, oz.item(), 0, 0, 1)
        return torch.tensor(hit.dt_do.z * g.item(), device=oz.device)


oz = torch.tensor(-1.0, device="cuda", requires_grad=True)
opt = torch.optim.Adam([oz], lr=0.3)

for _ in range(80):
    opt.zero_grad()
    t = SlangTrace.apply(oz)
    loss = (t - TARGET) ** 2
    loss.backward()
    opt.step()

final = M.traceAD(H, 0.25, 0.25, oz.item(), 0, 0, 1)
print(json.dumps({
    "oz": oz.item(),
    "final_t": final.t,
    "converged": abs(final.t - TARGET) < 0.2,
}))
