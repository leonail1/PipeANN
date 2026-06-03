# PPT-ready PQ-only Summary

- PQ-only rebuild works as a narrow maintenance primitive: it only rewrites PQ pivots/codes; disk graph, labels, layout, tag-id map, tombstone cleanup, and live compact are unchanged.
- Smoke passed: recall 98.05, avg 8.44 ms, p95 8.77 ms.
- Chain passed 13 selected rows, then failed at cycle3/intersect/u75: recall 98.25, avg 10.96 ms, p95 12.33 ms.
- Diagnosis: graph route no longer reaches 98 recall by L500; prefilter reaches recall but scans 750k candidates and about 400 4KB pages. This is non-PQ degradation.
- PQ-only cost: 16-core max wall 13.20s, train 4.22s, recode 4.13s. 1-core background wall 42.69s.
- Foreground under 1-core PQ-only background stayed within gates: max avg 9.90 ms, max p95 9.94 ms, recall 98.08.
- Engineering strategy: use PQ-only for frequent PQ drift repair; trigger graph/layout/compact maintenance separately under low-peak or shadow-prefix publish.
