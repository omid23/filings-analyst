# FinanceBench retrieval, test split

57 questions. Page-level: a hit means a gold evidence page (the page the FinanceBench annotator used) is among the top-k retrieved pages.

| config | scope | hit@1 | hit@5 | hit@10 | recall@10 | MRR@20 | filing found | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bm25, no filing filter | corpus | 7.0% | 22.8% | 26.3% | 25.4% | 0.134 | n/a | 282 |
| bm25 | resolved | 19.3% | 38.6% | 50.9% | 43.9% | 0.279 | 96% | 136 |
| dense | resolved | 10.5% | 35.1% | 43.9% | 41.2% | 0.209 | 96% | 24 |
| hybrid | resolved | 17.5% | 40.4% | 49.1% | 43.0% | 0.265 | 96% | 154 |
| hybrid + section router | resolved | 31.6% | 57.9% | 64.9% | 60.5% | 0.428 | 96% | 163 |
| hybrid + section router, filing given | oracle | 40.4% | 70.2% | 80.7% | 75.4% | 0.545 | given | 154 |
