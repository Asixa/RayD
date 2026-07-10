# Downstream hard-cut inventory

The 2026-07-09 workspace scan found active consumers in:

- `E:/Code/witwin-platform/channel`, `channel_native`, and `radar`;
- `E:/Code/rfdt-pose`;
- `E:/Code/Research/RFDT/Mini-Differentiable-RF-Digital-Twin`.

Generated benchmark installs, `bin/` comparison trees, vendored `ext/raydn` and
`ext/raydtorch` snapshots, and `RayDi_stage2_baseline_snapshot` are historical
or generated copies and are not authoritative downstream sources.

The coordinated cut for active consumers is:

- projects needing both backends: dependency `rayd`;
- Dr.Jit code: `import rayd.drjit as rd` and dependency `rayd-drjit`;
- Torch code: `import rayd.torch as rt`, dependency `rayd-torch`, dispatcher
  `torch.ops.rayd_torch`, and custom classes `torch.classes.rayd_torch`;
- no default `rayd` API and no compatibility package.

Land these downstream changes only after both pre-release wheels are available
in the selected release channel. This prevents downstream main branches from
depending on package names that cannot yet be installed. The release workflow
publishes the two distributions independently and the coexistence gate must
pass before the coordinated project tag.
