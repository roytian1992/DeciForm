# Fixed evaluation: DeciForm and released decision models

All three models evaluated the same 68,156 records. Accuracy is expressed as a percentage.

| Source | N | DeciForm-9B | Kev-9B | Open-Jev-9B |
|---|---:|---:|---:|---:|
| banking77 | 153 | 72.55 | 87.58 | 63.40 |
| dbpedia14 | 66,012 | 99.23 | 99.03 | 94.56 |
| emotion | 382 | 66.23 | 49.21 | 47.12 |
| f2_priority | 45 | 97.78 | 84.44 | 84.44 |
| mmlu | 255 | 73.73 | 79.61 | 69.02 |
| mmlu_pro | 640 | 50.31 | 52.66 | 47.66 |
| paws | 128 | 94.53 | 81.25 | 83.59 |
| qnli | 128 | 92.97 | 88.28 | 90.62 |
| sciq | 64 | 96.88 | 96.88 | 96.88 |
| sst5 | 120 | 51.67 | 57.50 | 37.50 |
| trec | 104 | 93.27 | 91.35 | 83.65 |
| tweet_offensive | 125 | 77.60 | 79.20 | 64.80 |
| **Source-equal mean** | 68,156 | 80.56 | 78.92 | 71.94 |
| **Record-weighted accuracy** | 68,156 | 98.27 | 98.03 | 93.48 |

DBpedia14 contributes 66,012 of 68,156 records. Source-equal accuracy gives each of the 12 sources equal weight; pooled accuracy weights records equally.

Fixed local engineering comparison of released models with different training histories. Exact external training/test overlap is not yet audited. These are Static results, not RSI results.

The per-source counts for the other 11 sources are small. This table is descriptive, not a claim of statistical significance or equal training budgets. The full benchmark text and per-record predictions are not redistributed in this repository.
