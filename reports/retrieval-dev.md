# FinanceBench retrieval, dev split

93 questions. Page-level: a hit means a gold evidence page (the page the FinanceBench annotator used) is among the top-k retrieved pages.

| config | scope | hit@1 | hit@5 | hit@10 | recall@10 | MRR@20 | filing found | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bm25, no filing filter | corpus | 7.5% | 18.3% | 22.6% | 22.0% | 0.125 | n/a | 254 |
| bm25 | resolved | 9.7% | 24.7% | 39.8% | 36.9% | 0.193 | 97% | 120 |
| dense | resolved | 8.6% | 17.2% | 26.9% | 25.3% | 0.140 | 97% | 20 |
| hybrid | resolved | 6.5% | 29.0% | 43.0% | 39.6% | 0.174 | 97% | 136 |
| hybrid + section router | resolved | 28.0% | 53.8% | 61.3% | 57.9% | 0.392 | 97% | 149 |
| hybrid + section router, filing given | oracle | 34.4% | 68.8% | 75.3% | 71.9% | 0.493 | given | 141 |
