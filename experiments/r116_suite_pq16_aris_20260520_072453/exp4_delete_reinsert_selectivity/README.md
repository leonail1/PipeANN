# PQ16 delete/reinsert selectivity

Runs r116/PQ16 1M initial, 750k after deleting the last 250k tags, and 1M after reinserting them. The PPT figure is `fixed_graph_recall_high_selectivity.png`.

Completion repair: the first run stopped after 29 calibrated rows and had stale fixed-graph table rows; the final u100 reinsert row was filled and fixed-graph rows were rerun cleanly for all 3 states x 5 high-selectivity buckets.
