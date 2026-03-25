"""Test: two independent Slang→rayd AD traces, combined loss, gradient composition.

All gradient math in Slang/C++.  Python only orchestrates torch.autograd.
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
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        return torch.tensor(M.traceTFwd(H, 0.25, 0.25, oz.item(), 0, 0, 1), device=oz.device)

    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        grad_o = M.traceTBwdOrigin(H, 0.25, 0.25, oz.item(), 0, 0, 1, g.item())
        return torch.tensor(grad_o.z, device=oz.device)


oz1 = torch.tensor(-1.0, device="cuda", requires_grad=True)
oz2 = torch.tensor(-3.0, device="cuda", requires_grad=True)

t1 = SlangTrace.apply(oz1)  # t=1
t2 = SlangTrace.apply(oz2)  # t=3
loss = (t1 - 2.0) ** 2 + (t2 - 2.0) ** 2
loss.backward()

# dloss/doz1 = 2*(1-2)*(-1) = 2,  dloss/doz2 = 2*(3-2)*(-1) = -2
print(json.dumps({
    "t1": t1.item(),
    "t2": t2.item(),
    "grad_oz1": oz1.grad.item(),
    "grad_oz2": oz2.grad.item(),
    "oz1_correct": abs(oz1.grad.item() - 2.0) < 0.01,
    "oz2_correct": abs(oz2.grad.item() - (-2.0)) < 0.01,
}))
