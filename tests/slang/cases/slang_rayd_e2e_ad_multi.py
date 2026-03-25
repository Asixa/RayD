"""Two independent Slang→rayd AD queries, combined loss, verify gradients.

Ray 1: oz1=-1, t1=1.  Ray 2: oz2=-3, t2=3.  Target t=2 for both.
loss = (t1-2)^2 + (t2-2)^2 = 1 + 1 = 2

dloss/doz1 = 2*(t1-2) * dt1/doz1 = 2*(-1)*(-1) = 2
dloss/doz2 = 2*(t2-2) * dt2/doz2 = 2*(1)*(-1)  = -2
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


class SlangTraceAD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        return torch.tensor(M.traceT(H, 0.25, 0.25, oz.item(), 0, 0, 1),
                            device=oz.device)

    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        dtdoz = M.dtdoz(H, 0.25, 0.25, oz.item(), 0, 0, 1)
        return g * dtdoz


oz1 = torch.tensor(-1.0, device="cuda", requires_grad=True)
oz2 = torch.tensor(-3.0, device="cuda", requires_grad=True)

t1 = SlangTraceAD.apply(oz1)
t2 = SlangTraceAD.apply(oz2)
loss = (t1 - 2.0) ** 2 + (t2 - 2.0) ** 2
loss.backward()

print(json.dumps({
    "t1": t1.item(),
    "t2": t2.item(),
    "grad_oz1": oz1.grad.item(),
    "grad_oz2": oz2.grad.item(),
    "oz1_correct": abs(oz1.grad.item() - 2.0) < 0.01,
    "oz2_correct": abs(oz2.grad.item() - (-2.0)) < 0.01,
}))
