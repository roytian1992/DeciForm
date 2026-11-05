# Final D2 position audit and architecture ablations

## Scope

The completed independent-candidate D2 run is frozen as the final main model:
`runs/formal_independent_20261105_v1`. This follow-up keeps its data cache,
fixed sampled training views, ordinal score objective, all-target LoRA setup,
one epoch, 200,000 training rows, global batch 64, and full-native dev/test
evaluation protocol.

The position audit uses the D2 epoch-1 checkpoint and the complete native
in-domain test inventory. It compares original order, reverse order, one-step
left/right rotations, and deterministic shuffles with seeds 11, 29, and 47.
Only Choice and Noul are audited, matching the established order-sensitivity
protocol; Score is not included in this audit.

The three fresh-base ablations are isolated on two GPUs each:

| Arm | Disabled component | GPUs | Run root |
| --- | --- | --- | --- |
| A1 | task routing | 2,3 | `runs/formal_independent_ablations_20261105_v1/A1_no_task_routing` |
| A2 | quadratic QF | 4,5 | `runs/formal_independent_ablations_20261105_v1/A2_no_quadratic` |
| A3 | task geometry | 6,7 | `runs/formal_independent_ablations_20261105_v1/A3_no_head_geometry` |

GPU 0,1 run the position audit at
`runs/formal_independent_position_20261105_v1`.

## Fairness and initialization

Each ablation starts from the same local Qwen3.5-9B base and fresh zero-B
LoRA initialization as the final D2 run. No static-full, D2, or other trained
checkpoint is inherited. The parent D2 constructor is initialized with all
components enabled and then exactly one component is zeroed/frozen before the
optimizer is built. Preflight checks compare base, LoRA, pointer, and head
initialization hashes against the final D2 fresh initialization.

## Monitoring and completion criteria

Every arm has a preflight proof, progress/status file, failure alert, resumable
checkpoint, and automatic native dev/test evaluation. A successful arm must
report 3,125 optimizer steps, 200,000 sample visits, and both `dev` and
`test_in_domain` evaluated with complete native candidates. The position audit
must report all seven orders over the full 10,000 Choice/Noul test rows.

## Position audit result

The position audit completed successfully on 2026-11-05. Across all seven
orders and all 10,000 native Choice/Noul rows, the final D2 produced zero
decision flips and zero maximum probability delta. Choice accuracy remained
0.7312 and Noul accuracy remained 0.8878 for every order. The detailed report
is `runs/formal_independent_position_20261105_v1/position_sensitivity.json`.
