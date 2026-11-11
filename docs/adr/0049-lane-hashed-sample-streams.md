# ADR-0049: Hash the lane before combining it with the stream and seed

- Status: Accepted; pending review by Xingyu Chen
- Date: 2026-11-11
- Decision ID: `lane-hashed-sample-streams`
- Scope: the counter-based uniforms and index hashes of the diffraction accumulation samplers
  (`src/diffraction/accumulation.h`, `src/diffraction/accumulation_ad.cuh`,
  `src/diffraction/diffraction_jit.cpp`) and of the stochastic reflection sampler
  (`src/reflection/reflection_algorithms.cuh`), plus the regenerated
  `generated/drjit/ptx/{diffraction_accumulation,reflection_accumulation,reflection_epc,reflection_trace}_ptx.h`.

## Context

Every sampler drew `hash_u32(lane ^ (stream * 0x9e3779b9) ^ seed)` (the reflection sampler uses
the ray index and depth). For a launch of `N` lanes and two seeds that differ only below
`log2(N)` bits, `{lane ^ c ^ seed}` is the same set of hash inputs for both seeds: the second
seed only permutes the first seed's draws across lanes. Seeds are usually small integers, so the
estimates a caller averages over seeds `1..K` were strongly correlated. Channel measured it on its
MC Basic diffraction map: 200 adjacent seeds at 4096 samples gave an eps_r gradient mean of
`-1.79e-7 +- 0.04e-7`, 200 seeds spread over the high bits `-2.27e-7`, and 262144 samples
`-2.20e-7`; the adjacent-seed mean looked like a small-sample bias of 7 standard errors.

## Decision

Every such hash hashes the lane (or ray index) first:
`hash_u32(hash_u32(lane) ^ (stream * 0x9e3779b9) ^ seed)`, and likewise for the second- and
third-state index hashes. Primal (`accumulation.h`), AD (`accumulation_ad.cuh`) and Dr.Jit
(`diffraction_jit.cpp`) change in lockstep, so the backends and the primal/derivative pairs still
draw identical values. Two seeds now share a draw only where `hash(l1) ^ hash(l2)` equals their
XOR difference, about `N^2 / 2^32` lane pairs, instead of all `N` lanes. One extra integer hash
per draw; no launch, memory or reduction-order change.

## Consequences

- Sample sequences of diffraction accumulation (Torch and Dr.Jit), its AD companions and the
  stochastic reflection accumulation change for every seed. Results are statistically equivalent;
  tests that compare a sampled result with a stored value move.
- The committed PTX of the four modules that include the edited headers is regenerated.
- Downstream callers no longer need to premix their seeds.
