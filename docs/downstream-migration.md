# Downstream hard-cut inventory

The 2026-07-09 workspace scan found active consumers in:

- `E:/Code/witwin-platform/channel`, `channel_native`, and `radar`;
- `E:/Code/rfdt-pose`;
- `E:/Code/Research/RFDT/Mini-Differentiable-RF-Digital-Twin`.

Generated benchmark installs, `bin/` comparison trees, vendored `ext/raydn` and
`ext/raydtorch` snapshots, and `RayDi_stage2_baseline_snapshot` are historical
or generated copies and are not authoritative downstream sources.

The coordinated cut for active consumers is, pinning version `0.6.0`:

| Old | New |
| --- | --- |
| `import rayd as rd` | `import rayd.drjit as rd` |
| `import raydn as rt` | `import rayd.torch as rt` |
| `torch.ops.raydn` | `torch.ops.rayd_torch` |
| `torch.classes.raydn` | `torch.classes.rayd_torch` |
| dependency `rayd` (single backend) | `rayd-drjit` and/or `rayd-torch` |
| dependency `rayd-native` | `rayd-torch` |

- projects needing both backends: dependency `rayd`;
- Dr.Jit code: `import rayd.drjit as rd` and dependency `rayd-drjit`;
- Torch code: `import rayd.torch as rt`, dependency `rayd-torch`, dispatcher
  `torch.ops.rayd_torch`, and custom classes `torch.classes.rayd_torch`;
- no default `rayd` API and no compatibility package. `rayd` is a PEP 420
  namespace, so `import rayd` alone gives no API.

Dr.Jit consumers must also apply the 2026-05-21 class and method renames, which
are breaking and independent of the namespace cut. In particular the AD/non-AD
convention flipped (`Ray` is now non-AD, `RayAD` is the autodiff variant), and
`trace_segment_visibility` / `trace_reflection_epc` became `visible` /
`trace_refl_epc`. See [`backends/drjit/API_RENAME.md`](../backends/drjit/API_RENAME.md)
for the complete old-to-new table.

Land these downstream changes only after the pre-release wheels are available
in the selected release channel. This prevents downstream main branches from
depending on package names that cannot yet be installed. The release workflow
publishes the two backend distributions independently, then the exact-version
`rayd` meta-distribution, and the coexistence gate must pass before the
coordinated project tag.
