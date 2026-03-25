"""Test: Slang-compiled forward (traceTFwd) + backward (traceTBwdOrigin)
wrapped in torch.autograd.Function. The backward logic is in Slang/C++.

Python only bridges fwd/bwd into torch autograd — no gradient math in Python.
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
    """Forward and backward both call Slang-compiled host functions."""
    @staticmethod
    def forward(ctx, oz):
        ctx.save_for_backward(oz)
        t = M.traceTFwd(H, 0.25, 0.25, oz.item(), 0.0, 0.0, 1.0)
        return torch.tensor(t, device=oz.device)

    @staticmethod
    def backward(ctx, g):
        oz, = ctx.saved_tensors
        # Backward is computed in Slang/C++ via Dr.Jit AD
        grad_o = M.traceTBwdOrigin(H, 0.25, 0.25, oz.item(), 0.0, 0.0, 1.0, g.item())
        return torch.tensor(grad_o.z, device=oz.device)


oz = torch.tensor(-1.0, device="cuda", requires_grad=True)
t = SlangTrace.apply(oz)
t.backward()

print(json.dumps({
    "t": t.item(),
    "grad_oz": oz.grad.item(),
    "t_correct": abs(t.item() - 1.0) < 1e-5,
    "grad_correct": abs(oz.grad.item() - (-1.0)) < 1e-5,
}))
