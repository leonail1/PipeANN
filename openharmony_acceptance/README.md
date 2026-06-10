# OpenHarmony C++ Acceptance

This directory contains the PipeANN-native C++ acceptance runner. It uses PipeANN
tools and C++ APIs directly: index build, official groundtruth computation,
filtered search, dynamic delete/merge/insert, and single-query resource checks.

Latency is measured with `SEARCH_THREADS=1` by default. Build, groundtruth, and
background update threads are configured separately.

## Dynamic Tests

Dynamic update acceptance is split into two independent tests.

1. Foreground interference test
   - Runs one cycle: delete 60% live vectors, merge, then insert new vectors.
   - Uses `FOREGROUND_UPDATE_THREADS=4` by default.
   - Runs foreground filtered search during delete/merge/insert.
   - Pass condition: foreground search average latency stays below 10 ms.

2. Batch quality test
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
- `dynamic_batch_chain.jsonl`
- `dynamic_batch_checkpoint_search.jsonl`
- `dynamic_batch_progress.jsonl`

`acceptance_summary.json` checks that both dynamic tests actually ran.
