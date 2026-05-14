# YFCC10M Mixed Posting vs Bitmap

## Space

- bitmap sidecar: 233.280 GiB
- mixed sidecar: 0.335 GiB
- savings: 232.945 GiB (99.86%)
- mixed encoding split: 29 bitmap labels, 200334 posting labels, 23 empty labels

## Performance

| bucket | selectivity | candidates | bitmap avg us | mixed avg us | avg delta | bitmap qps | mixed qps | qps delta | bitmap peak KB | mixed peak KB | peak delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t1e-05 | 0.000010 | 99 | 13133.8 | 13251.7 | +0.90% | 3717.4 | 3904.5 | +5.03% | 47764 | 56708 | +18.73% |
| t3e-05 | 0.000043 | 428 | 12157.8 | 13027.4 | +7.15% | 3998.4 | 3971.2 | -0.68% | 50252 | 70616 | +40.52% |
| t1e-04 | 0.000096 | 960 | 12271.6 | 13129.2 | +6.99% | 3964.2 | 3939.2 | -0.63% | 53360 | 95828 | +79.59% |
| t3e-04 | 0.000289 | 2885 | 12420.8 | 13123.7 | +5.66% | 3914.0 | 3939.0 | +0.64% | 66028 | 162268 | +145.76% |
| t1e-03 | 0.001000 | 9996 | 12554.7 | 13254.5 | +5.57% | 3844.1 | 3893.9 | +1.30% | 100456 | 237104 | +136.03% |
| t3e-03 | 0.002999 | 29991 | 12370.7 | 12952.0 | +4.70% | 3799.4 | 3969.9 | +4.49% | 182384 | 309636 | +69.77% |
| t1e-02 | 0.009794 | 97939 | 16586.2 | 16881.9 | +1.78% | 2733.2 | 3012.1 | +10.21% | 307192 | 309792 | +0.85% |

## Notes

- All seven buckets are real YFCC10M single-label workloads and all are encoded as posting in the mixed sidecar.
- Comparison fixes route=prefilter with intersect selector, threads=52, beamwidth=4, k=10, L=100.
- The search driver reports hybrid_enabled=false because no _hybrid.meta was provisioned, but prefilter_count=100000 for every bucket in both runs, so the forced prefilter path did execute on the densebit sidecar.
- Recall is null in these runs because no filtered exact ground truth was supplied; this experiment isolates latency/QPS/memory differences only.
