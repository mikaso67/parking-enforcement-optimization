# Exploratory analysis

Figures come from `notebooks/01_eda.ipynb` (seed 42). Train has 6,076,546 rows and test 2,028,750.

## 1. Target and `total_count`

| total_count | rows | mean ratio | share at 0 | share at 1 |
|-------------|------|-----------|-----------|-----------|
| 1 | 1,532,442 | 0.66 | 0.34 | 0.66 |
| 2 to 5 | 1,829,486 | 0.55 | 0.18 | 0.29 |
| 6 to 44 | 2,606,842 | 0.38 | 0.04 | 0.02 |
| 45 and above | 107,776 | 0.30 | 0.00 | 0.01 |

- Only 1.8% of rows have 45 checks or more. Small counts give ratios stuck at 0 or 1.
- `total_count` alone has Spearman -0.295 with the ratio: more checks, lower ratio.

## 2. Time

- Hours present: 6 to 19. The ratio peaks at 19 (0.77) and 6 (0.63).
- Train covers months 1 to 12 and the mean ratio falls from 0.56 to 0.46. Test covers months 1 to 5 and month 5 is partial (35,132 rows).

## 3. Weather

Spearman with the ratio on rows with 45 checks or more: precipitation -0.005, snow -0.001, temperature 0.047, wind -0.013. Weather carries almost no signal.

## 4. Space

- Coordinates are min-max scaled with outliers at 0, so almost all points sit in a tiny range.
- 2,487,129 distinct points in train and 53% of test points exist in train.
- 32% of train rows sit on points seen at least 10 times (the most repeated point has 399 rows).

| cell size | train cells | median rows per cell | cells with 30 rows or more | test rows in a known cell |
|-----------|------------|---------------------|---------------------------|--------------------------|
| 1e-3 | 14 | 157 | 7 | 99.9998% |
| 3e-4 | 48 | 3,359 | 33 | 99.9998% |
| 1e-4 | 244 | 1,010 | 164 | 99.999% |
| 3e-5 | 1,733 | 1,279 | 1,223 | 99.9975% |
| 1e-5 | 10,605 | 125 | 6,768 | 99.986% |

Grid cells solve the unseen points problem: almost every test row lands in a cell seen in train. The cell size trades resolution against rows per cell (D6).

## 5. How the benchmark could be measured

Random forest, 10 trees, seed 42, trained on rows with 45 checks or more from months 1 to 9, scored on a sample of 600,000 rows from months 10 to 12:

| scored on | Spearman |
|-----------|---------|
| all rows | 0.188 |
| 45 and above | 0.738 |
| 6 to 44 | 0.317 |
| 2 to 5 | 0.144 |
| 1 | 0.106 |

This almost reproduces the 0.197 benchmark. My reading is that the benchmark is trained on the clean rows and scored on all test rows. It is an inference: the organisers do not state it. With a random split the same forest reaches 0.78 on the 45+ rows because repeated points leak across the split, which is why validation follows time.

## 6. Conclusions

1. The ranking is mostly lost on noisy small counts. D4 (how to treat them) is the main lever.
2. Validate on later months (D5). A random split inflates scores through repeated points.
3. Build zones as grid cells (D6). Cell size is still open.
4. Weather adds little. Hour and `total_count` carry the signal.
