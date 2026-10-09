# Initial migration map

This first extraction is deliberately a rewrite of a small contract slice;
the historical source is not modified.

| Source repository | Source commit | Original source | Disposition | Target | Scope / limit |
|---|---|---|---|---|---|
| `alessbarb/symbiont-lab` | `80d372a186513035f85a769db886e505cabba52d` | `symbiont/src/symbiont/core/orchestration/runtime.py` identity handling | `REWRITE` | `biopeia_organism.identity` | Current runtime assigns identity internally; this extraction requires identity input rather than drawing from host entropy. |
| `alessbarb/symbiont-lab` | `80d372a186513035f85a769db886e505cabba52d` | `symbiont/src/symbiont/host/checkpoint.py` and `core/orchestration/runtime.py` checkpoint composition | `REWRITE` | `biopeia_organism.checkpoint` | Only an opaque owner checkpoint boundary and verified content references are included. Current mixed Organism/body state is not copied. |
| `alessbarb/symbiont-lab` | `80d372a186513035f85a769db886e505cabba52d` | `symbiont/src/symbiont/core/embodiment/` | `SPLIT` / `DEFER` | `embodiment` candidate | Somatic state is excluded from this Organism package and will be mapped separately. |
| `alessbarb/symbiont-lab` | `80d372a186513035f85a769db886e505cabba52d` | cognition, memory, agency, learning and private model implementation | `DEFER` | `organism` | Source dependencies and checkpoint closure need finer mapping before extraction. |

The commit above is the actual source checkout used for this extraction. The
architecture/audit baseline remains separately pinned at
`cef998a55aae32ed8b2f7373079c11667b01e915`.

## Deliberate non-goals

- No compatibility with the historical `symbiont` import/package name.
- No runtime implementation, cognition algorithm, or persistence wire format.
- No World manifest, cross-authority commit protocol, or restore policy.
- No claim that current runtime/checkpoint behavior complies with TARGET.
