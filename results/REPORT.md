# Results

Agreement with the teacher's labels on 342 held-out emails (the newest 20% of the mail).

| Model | Accuracy | Macro-F1 | Employer match | Valid JSON (unconstrained) | Latency p50 / p95 (s) | Decode tok/s | Prefill tok/s | API-equivalent $ per 1k emails ($2 in, $10 out per Mtok) |
|---|---|---|---|---|---|---|---|---|
| base-q4_k_m | 98.0% | 0.901 | 98.2% (n=54) | 6.1% | 4.151 / 7.281 | 62.5 | 1926.0 | 3.851 |
| tuned-q4_k_m | 96.8% | 0.744 | 90.7% (n=54) | 99.7% | 4.296 / 7.791 | 61.9 | 1084.8 | 3.795 |
| tuned-q8_0 | 97.7% | 0.758 | 94.4% (n=54) | 99.7% | 5.505 / 10.361 | 41.9 | 1203.4 | 3.615 |

## base-q4_k_m

Confusion matrix (rows: teacher, columns: model)

| teacher \ model | applied | rejected | interview | incomplete | other | invalid |
|---|---|---|---|---|---|---|
| applied | 27 | 0 | 0 | 0 | 1 | 0 |
| rejected | 0 | 21 | 0 | 0 | 0 | 0 |
| interview | 0 | 0 | 2 | 0 | 0 | 0 |
| incomplete | 0 | 0 | 0 | 2 | 0 | 1 |
| other | 4 | 0 | 1 | 0 | 283 | 0 |

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| applied | 28 | 0.871 | 0.964 | 0.915 |
| rejected | 21 | 1.000 | 1.000 | 1.000 |
| interview | 2 | 0.667 | 1.000 | 0.800 |
| incomplete | 3 | 1.000 | 0.667 | 0.800 |
| other | 288 | 0.997 | 0.983 | 0.990 |

## tuned-q4_k_m

Confusion matrix (rows: teacher, columns: model)

| teacher \ model | applied | rejected | interview | incomplete | other | invalid |
|---|---|---|---|---|---|---|
| applied | 27 | 0 | 0 | 0 | 1 | 0 |
| rejected | 0 | 19 | 0 | 0 | 0 | 2 |
| interview | 0 | 0 | 0 | 0 | 0 | 2 |
| incomplete | 0 | 0 | 0 | 2 | 0 | 1 |
| other | 0 | 0 | 0 | 0 | 283 | 5 |

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| applied | 28 | 1.000 | 0.964 | 0.982 |
| rejected | 21 | 1.000 | 0.905 | 0.950 |
| interview | 2 | 0.000 | 0.000 | 0.000 |
| incomplete | 3 | 1.000 | 0.667 | 0.800 |
| other | 288 | 0.997 | 0.983 | 0.990 |

## tuned-q8_0

Confusion matrix (rows: teacher, columns: model)

| teacher \ model | applied | rejected | interview | incomplete | other | invalid |
|---|---|---|---|---|---|---|
| applied | 28 | 0 | 0 | 0 | 0 | 0 |
| rejected | 0 | 21 | 0 | 0 | 0 | 0 |
| interview | 0 | 0 | 0 | 0 | 0 | 2 |
| incomplete | 0 | 0 | 0 | 2 | 0 | 1 |
| other | 0 | 0 | 0 | 0 | 283 | 5 |

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| applied | 28 | 1.000 | 1.000 | 1.000 |
| rejected | 21 | 1.000 | 1.000 | 1.000 |
| interview | 2 | 0.000 | 0.000 | 0.000 |
| incomplete | 3 | 1.000 | 0.667 | 0.800 |
| other | 288 | 1.000 | 0.983 | 0.991 |
