# OpenHarmony C++ Acceptance

This directory contains the PipeANN-native C++ acceptance runner. It uses PipeANN
tools and C++ APIs directly: index build, official groundtruth computation,
filtered search, dynamic delete/merge/insert, and single-query resource checks.

Latency is measured with `SEARCH_THREADS=1` by default. Build, groundtruth, and
background update threads are configured separately.

## Dynamic Tests

Dynamic update acceptance is split into two independent tests.
Both dynamic tests start from zero live vectors. The runner keeps the first
`ZERO_BOOTSTRAP_NPOINTS` vectors in a ZeroStart in-memory phase, where filtered
queries use prefilter plus exact KNN semantics. At the bootstrap threshold it
writes PipeANN-compatible vectors and attributes, builds the initial disk index,
then inserts the remaining base vectors through PipeANN's native dynamic insert
path. After that point, delete, merge/save, insert, and search all use PipeANN
native APIs; the runner does not do lazy rebuild or maintain a second index.

1. Foreground interference test
   - Starts from zero, bootstraps at `ZERO_BOOTSTRAP_NPOINTS`, inserts to the
     target live count, then runs one cycle: delete 60% live vectors, merge,
     then insert new vectors.
   - Uses `FOREGROUND_UPDATE_THREADS=4` by default.
   - Runs foreground filtered search during delete/merge/insert.
   - Pass condition: for each foreground phase (`after_mark_delete`, `merge`,
     `insert`, `after_insert`), the mean of that phase's probe
     `avg_latency_ms` values stays below 10 ms. Individual foreground probe rows
     above 10 ms are emitted as warnings, and the summary reports their count
     and ratio.

2. Batch quality test
   - Starts from zero independently from the foreground test, bootstraps at
     `ZERO_BOOTSTRAP_NPOINTS`, and inserts to the target live count before the
     5-cycle chain starts.
   - Runs five cycles by default.
   - Uses `BATCH_UPDATE_THREADS=32` by default.
   - Does not run foreground search during update.
   - After each cycle returns to 1M live vectors, publishes the post-insert
     state with `save()` so the checkpoint is a consistent disk snapshot.
   - Then runs the full filtered search matrix and checks recall and average
     latency.

The two tests produce separate artifacts:

- `dynamic_foreground_chain.jsonl`
- `dynamic_foreground_latency.jsonl`
- `dynamic_foreground_progress.jsonl`
- `zero_start_exact.jsonl`
- `dynamic_batch_chain.jsonl`
- `dynamic_batch_checkpoint_search.jsonl`
- `dynamic_batch_progress.jsonl`

`acceptance_summary.json` checks that both dynamic tests actually ran.
The single-query runner is a resource/RSS check. Its latency is reported as a
cold single-query diagnostic, but it is not used as a pass/fail latency gate;
steady-state search latency is judged by the 1000-query static and dynamic
checkpoint searches.

Tag reuse follows PipeANN's native delete semantics: a tag deleted by
`lazy_delete/remove` cannot be reinserted until `save()/merge_deletes()` has
completed. The dynamic tests therefore use `delete -> save/merge -> insert same
tag range` for every cycle.

This suite intentionally does not perform PQ retraining. In a zero-start run the
initial PQ codebook is trained at the bootstrap threshold, so later PQ drift is
reported as an experimental risk rather than hidden by rebuilding the graph,
disk node layout, or attribute index.
