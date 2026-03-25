"""End-to-end: multiple Slang→rayd queries with combined loss.

Two rays at different z-origins, both queried via Slang's traceRayT.
Loss = sum of squared depth errors.  Verify both gradients are non-zero
and consistent with FD.
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

oz1 = torch.tensor(-1.0, device="cuda", requires_grad=True)
oz2 = torch.tensor(-3.0, device="cuda", requires_grad=True)

t1 = SlangTraceT.apply(oz1)
t2 = SlangTraceT.apply(oz2)
loss = (t1 - 2.0) ** 2 + (t2 - 2.0) ** 2
loss.backward()

# t1 = 1, t2 = 3.  loss = (1-2)^2 + (3-2)^2 = 2
# dloss/dt1 = 2*(1-2) = -2,  dloss/dt2 = 2*(3-2) = 2
# dt/doz = -1 for both.
# dloss/doz1 = -2 * (-1) = 2,  dloss/doz2 = 2 * (-1) = -2
print(json.dumps({
    "t1": t1.item(),
    "t2": t2.item(),
    "grad_oz1": oz1.grad.item(),
    "grad_oz2": oz2.grad.item(),
    "oz1_close": abs(oz1.grad.item() - 2.0) < 0.05,
    "oz2_close": abs(oz2.grad.item() - (-2.0)) < 0.05,
}))
