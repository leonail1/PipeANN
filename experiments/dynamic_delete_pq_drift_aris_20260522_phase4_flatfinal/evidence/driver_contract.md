# Dynamic Delete/PQ Drift Driver Contract

The experiment runner assumes these driver modes and fields. If the C++ driver cannot emit them, the corresponding claim must be marked UNSUPPORTED rather than inferred.

Duplicate arg policy: Driver must reject duplicate scalar CLI args; the runner avoids emitting duplicates.

ID/tag semantics:
- Phase1/2 delete file: One uint32 tag/id per line, interpreted as current live external tag when tags are enabled.
- Phase3 delete scope: `current-live-tags` means sample uniformly without replacement from all tags live at the start of that cycle.

Live corpus contract:
- live_data_bin: PipeANN save_bin float matrix. Row i is the exact vector for live_tag_file row i.
- live_base_label_file: spmat with nrow equal to live_data_bin npoints. Row i contains labels for live_tag_file row i.
- live_tag_file: PipeANN save_bin uint32/TagT vector with nrow equal to live_data_bin npoints.
- GT scope: All post-cycle GT must be computed against these live files, not original base files.

## Required Modes

- `measure-delete-only`: Load an existing disk index, lazy-delete a supplied tag/id set, do not call final_merge.
  Required args: `--source-prefix`, `--jsonl-output`, `--delete-id-file`, `--delete-count`
- `measure-delete-then-merge`: Lazy-delete a supplied tag/id set and then materialize through final_merge into --dest-prefix.
  Required args: `--source-prefix`, `--dest-prefix`, `--jsonl-output`, `--delete-id-file`, `--delete-count`
- `delete-batch`: Phase3 composition step: lazy-delete supplied tags and final_merge to an intermediate prefix.
  Required args: `--source-prefix`, `--dest-prefix`, `--jsonl-output`, `--delete-id-file`, `--delete-count`
- `insert-only`: Phase3 composition step: insert replacement vectors using the supplied tag file and final_merge to the next-cycle prefix.
  Required args: `--source-prefix`, `--dest-prefix`, `--jsonl-output`, `--data-bin`, `--insert-tag-file`, `--insert-count`
- `cycle-delete-insert`: Runner-composed cycle: delete-batch+merge to an intermediate index, then insert-only+merge equal-count replacement vectors into the deleted tag set and emit live-corpus files for GT.
  Implementation status: implemented_as_runner_composition_of_delete-batch_and_insert-only
  Required args: `--source-prefix`, `--dest-prefix`, `--jsonl-output`, `--delete-id-file`, `--insert-tag-file`, `--data-bin`, `--insert-count`
- `pq-drift`: Compare direct-build PQ with zero-data incremental PQ using seed-trained pivots and no full-corpus retrain, plus optional retrain cost proxies from build logs.
  Implementation status: implemented_smoke_for_direct_vs_zero_insert_seed_pq_no_retrain
  Required args: `--jsonl-output`, `--data-bin`, `--base-label-file`, `--query-bin`
- `zero-insert-only`: Driver mode used by Phase4: insert from an empty flat index, materialize once threshold is crossed, optionally using seed-trained PQ pivots.
  Required args: `--source-prefix`, `--jsonl-output`, `--data-bin`, `--insert-count`, `--flat-threshold`, `--pq-bytes`, `--flat-pq-pivots`
- `measure-dynamic-search`: Run search for a fixed route/L/query/GT/selector and append full route, recall, latency, and RSS stats.
  Required args: `--source-prefix`, `--jsonl-output`, `--query-bin`, `--truthset-bin`, `--query-label-file`, `--selector-type`, `--route`, `--search-l`

## Global JSON Fields

- `mode`
- `phase`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `threads`
- `source_prefix`
- `dest_prefix`
- `delete_count`
- `deleted_tag_hash`
- `delete_scope`
- `insert_count`
- `insert_segment`
- `live_point_count`
- `live_data_bin`
- `live_base_label_file`
- `live_tag_file`
- `live_gt_scope`
- `route`
- `actual_route`
- `search_l`
- `recall@10`
- `avg_latency_us`
- `p95_latency_us`
- `candidate_count`
- `prefilter_count`
- `graph_count`
- `fallback_count`
- `tau_m`
- `threshold_version`
- `delete_wall_s`
- `merge_wall_s`
- `insert_wall_s`
- `wall_s`
- `max_rss_kb`
- `pq_bytes`
- `pq_codebook_hash`
- `pq_code_hash`
- `pq_retrained`
- `pq_train_core_count`
- `pq_train_wall_s`
- `pq_recode_wall_s`
- `pq_training_points`
- `requested_points`
- `code_point_count`
- `code_chunks`
- `point_count_consistent`
- `label_storage_mode`
- `disk_format_version`
- `main_index_label_size`
- `raw_command`

## Mode-Specific Required JSON Fields

### `measure-delete-only`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `cpu_affinity_allowed_cpus`
- `source_prefix`
- `delete_count`
- `deleted_tag_hash`
- `delete_scope`
- `live_point_count`
- `delete_wall_s`
- `raw_command`
### `measure-delete-then-merge`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `cpu_affinity_allowed_cpus`
- `source_prefix`
- `dest_prefix`
- `delete_count`
- `deleted_tag_hash`
- `delete_scope`
- `live_point_count`
- `delete_wall_s`
- `merge_wall_s`
- `main_index_label_size`
- `label_storage_mode`
- `raw_command`
### `cycle-delete-insert`
- `delete_step`
- `insert_step`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `cpu_affinity_allowed_cpus`
- `source_prefix`
- `dest_prefix`
- `delete_count`
- `deleted_tag_hash`
- `delete_scope`
- `insert_count`
- `insert_segment`
- `live_point_count`
- `live_data_bin`
- `live_base_label_file`
- `live_tag_file`
- `live_gt_scope`
- `raw_command`
### `delete-batch`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `cpu_affinity_allowed_cpus`
- `source_prefix`
- `dest_prefix`
- `delete_count`
- `deleted_tag_hash`
- `delete_scope`
- `delete_elapsed_s`
- `merge_elapsed_s`
- `raw_command`
### `insert-only`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `cpu_affinity_allowed_cpus`
- `source_prefix`
- `dest_prefix`
- `insert_count`
- `insert_scope`
- `inserted_tag_hash`
- `insert_elapsed_s`
- `merge_elapsed_s`
- `live_point_count`
- `raw_command`
### `pq-drift`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `requested_points`
- `insert_count`
- `live_point_count`
- `code_point_count`
- `code_chunks`
- `point_count_consistent`
- `pq_bytes`
- `pq_codebook_hash`
- `pq_code_hash`
- `pq_retrained`
- `pq_train_core_count`
- `pq_train_wall_s`
- `pq_recode_wall_s`
- `pq_training_points`
- `pq_training_corpus_points`
- `seed_points`
- `flat_threshold`
- `variant`
- `final_index_prefix`
- `raw_command`
### `zero-insert-only`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `source_prefix`
- `final_index_prefix`
- `insert_count`
- `insert_wall_s`
- `merge_wall_s`
- `live_point_count`
- `pq_bytes`
- `flat_threshold`
- `flat_pq_pivots`
- `main_index_label_size`
- `label_sidecar_loadable`
- `raw_command`
### `measure-dynamic-search`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `source_prefix`
- `route`
- `actual_route`
- `search_l`
- `recall@10`
- `avg_latency_us`
- `p95_latency_us`
- `candidate_count`
- `prefilter_count`
- `graph_count`
- `raw_command`

## Field Schema and Units

- `cpu_cap`: int cores requested by runner
- `cpu_cap_enforced`: bool true only if taskset/numactl or equivalent affinity was applied
- `cpu_affinity_allowed_cpus`: string or array from sched_getaffinity/proc status in child process
- `delete_wall_s`: float seconds spent inside lazy-delete loop only
- `merge_wall_s`: float seconds spent inside final_merge/merge_deletes only
- `insert_wall_s`: float seconds spent inserting vectors only
- `avg_latency_us`: float microseconds per query
- `p95_latency_us`: float microseconds per query
- `candidate_count`: float or int mean candidates for this selector/query batch
- `main_index_label_size`: int bytes of label payload embedded in main node record; must be 0 for sidecar-only claim
