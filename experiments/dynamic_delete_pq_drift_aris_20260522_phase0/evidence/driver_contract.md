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
- `cycle-delete-insert`: Sample 60% of the current live tags, lazy-delete them, insert equal-count vectors from a SIFT100M segment, and emit live-corpus files for GT.
  Implementation status: contract_only_not_implemented_in_current_cpp_slice
  Required args: `--source-prefix`, `--dest-prefix`, `--jsonl-output`, `--delete-fraction`, `--delete-seed`, `--delete-scope`, `--insert-segment`, `--insert-count-policy`, `--sift100m-bin`, `--emit-live-corpus`
- `pq-drift`: Compare direct-build PQ with zero-data incremental PQ, optional retrain, and emit PQ/recode artifacts.
  Implementation status: contract_only_not_implemented_in_current_cpp_slice
  Required args: `--jsonl-output`, `--sift100m-bin`, `--core-sweep`, `--out-prefix`
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
### `pq-drift`
- `mode`
- `status`
- `cpu_cap`
- `cpu_cap_enforced`
- `cpu_affinity_allowed_cpus`
- `pq_bytes`
- `pq_codebook_hash`
- `pq_code_hash`
- `pq_retrained`
- `pq_train_core_count`
- `pq_train_wall_s`
- `pq_recode_wall_s`
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
