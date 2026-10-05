# Frozen aggregate tables

Generated from the completed aggregate result and frozen configuration. No new model evaluation or inference is performed.

## 1. Cohort

| Group | Patients | Breasts | Positive breasts |
| --- | --- | --- | --- |
| Development | 2,383 | 4,766 | 99 |
| Holdout | 9,530 | 19,060 | 393 |

## 2. Classification

| Group | Breasts | Positive | TP | FN | TN | FP | Sensitivity | Specificity | ROC-AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | 19060 | 393 | 354 | 39 | 8276 | 10391 | 90.08% | 44.33% | 0.77245 | 0.12907 |
| site_1 | 9308 | 203 | 173 | 30 | 4732 | 4373 | 85.22% | 51.97% | 0.78662 | 0.16452 |
| site_2 | 9752 | 190 | 181 | 9 | 3544 | 6018 | 95.26% | 37.06% | 0.78327 | 0.13857 |
| site_1_density_AB | 4906 | 104 | 83 | 21 | 2830 | 1972 | 79.81% | 58.93% | 0.78315 | 0.14828 |
| site_1_density_CD | 4386 | 99 | 90 | 9 | 1896 | 2391 | 90.91% | 44.23% | 0.79134 | 0.18828 |
| site_1_density_missing | 16 | 0 | 0 | 0 | 6 | 10 | NA | 37.50% | NA | NA |

## 3. Review at the four prespecified budgets

| Method | Budget | Selected breasts | Misses flagged / all misses | Capture | Yield |
| --- | --- | --- | --- | --- | --- |
| Cancer score alone | 1% | 83 | 0/39 | 0.00% | 0.00% |
| View disagreement | 1% | 83 | 0/39 | 0.00% | 0.00% |
| Between-member variance | 1% | 83 | 0/39 | 0.00% | 0.00% |
| Cancer score + disagreement | 1% | 83 | 0/39 | 0.00% | 0.00% |
| Cancer score + variance | 1% | 83 | 0/39 | 0.00% | 0.00% |
| Combined score | 1% | 83 | 0/39 | 0.00% | 0.00% |
| Cancer score alone | 5% | 415 | 3/39 | 7.69% | 0.72% |
| View disagreement | 5% | 415 | 1/39 | 2.56% | 0.24% |
| Between-member variance | 5% | 415 | 1/39 | 2.56% | 0.24% |
| Cancer score + disagreement | 5% | 415 | 2/39 | 5.13% | 0.48% |
| Cancer score + variance | 5% | 415 | 2/39 | 5.13% | 0.48% |
| Combined score | 5% | 415 | 2/39 | 5.13% | 0.48% |
| Cancer score alone | 10% | 831 | 8/39 | 20.51% | 0.96% |
| View disagreement | 10% | 831 | 3/39 | 7.69% | 0.36% |
| Between-member variance | 10% | 831 | 4/39 | 10.26% | 0.48% |
| Cancer score + disagreement | 10% | 831 | 5/39 | 12.82% | 0.60% |
| Cancer score + variance | 10% | 831 | 7/39 | 17.95% | 0.84% |
| Combined score | 10% | 831 | 5/39 | 12.82% | 0.60% |
| Cancer score alone | 20% | 1663 | 18/39 | 46.15% | 1.08% |
| View disagreement | 20% | 1663 | 6/39 | 15.38% | 0.36% |
| Between-member variance | 20% | 1663 | 13/39 | 33.33% | 0.78% |
| Cancer score + disagreement | 20% | 1663 | 16/39 | 41.03% | 0.96% |
| Cancer score + variance | 20% | 1663 | 14/39 | 35.90% | 0.84% |
| Combined score | 20% | 1663 | 13/39 | 33.33% | 0.78% |

## 4. Primary difference and uncertainty

| Bootstrap scheme | Difference (percentage points) | 95% interval | Valid draws |
| --- | --- | --- | --- |
| calibration_inclusive | -2.56 | [-14.29, +8.00] | 2000 |
| conditional | -2.56 | [-13.79, +8.11] | 2000 |

Difference = combined minus cancer score alone. Draws resample patients within site, retain both breasts and use the same holdout draws for paired comparisons. An interval crossing zero does not establish equivalence.

## 5. Subgroups at 5% global review capacity

| Group | Method | AI-negative breasts | Misses | Selected in group | Misses flagged | Capture |
| --- | --- | --- | --- | --- | --- | --- |
| site_1 | Cancer score alone | 4762 | 30 | 252 | 2 | 6.67% |
| site_2 | Cancer score alone | 3553 | 9 | 163 | 1 | 11.11% |
| site_1_density_AB | Cancer score alone | 2851 | 21 | 117 | 0 | 0.00% |
| site_1_density_CD | Cancer score alone | 1905 | 9 | 135 | 2 | 22.22% |
| site_1_density_missing | Cancer score alone | 6 | 0 | 0 | 0 | NA |
| site_1 | View disagreement | 4762 | 30 | 197 | 1 | 3.33% |
| site_2 | View disagreement | 3553 | 9 | 218 | 0 | 0.00% |
| site_1_density_AB | View disagreement | 2851 | 21 | 113 | 1 | 4.76% |
| site_1_density_CD | View disagreement | 1905 | 9 | 84 | 0 | 0.00% |
| site_1_density_missing | View disagreement | 6 | 0 | 0 | 0 | NA |
| site_1 | Between-member variance | 4762 | 30 | 150 | 1 | 3.33% |
| site_2 | Between-member variance | 3553 | 9 | 265 | 0 | 0.00% |
| site_1_density_AB | Between-member variance | 2851 | 21 | 86 | 1 | 4.76% |
| site_1_density_CD | Between-member variance | 1905 | 9 | 63 | 0 | 0.00% |
| site_1_density_missing | Between-member variance | 6 | 0 | 1 | 0 | NA |
| site_1 | Cancer score + disagreement | 4762 | 30 | 232 | 2 | 6.67% |
| site_2 | Cancer score + disagreement | 3553 | 9 | 183 | 0 | 0.00% |
| site_1_density_AB | Cancer score + disagreement | 2851 | 21 | 118 | 2 | 9.52% |
| site_1_density_CD | Cancer score + disagreement | 1905 | 9 | 114 | 0 | 0.00% |
| site_1_density_missing | Cancer score + disagreement | 6 | 0 | 0 | 0 | NA |
| site_1 | Cancer score + variance | 4762 | 30 | 202 | 2 | 6.67% |
| site_2 | Cancer score + variance | 3553 | 9 | 213 | 0 | 0.00% |
| site_1_density_AB | Cancer score + variance | 2851 | 21 | 99 | 1 | 4.76% |
| site_1_density_CD | Cancer score + variance | 1905 | 9 | 103 | 1 | 11.11% |
| site_1_density_missing | Cancer score + variance | 6 | 0 | 0 | 0 | NA |
| site_1 | Combined score | 4762 | 30 | 184 | 2 | 6.67% |
| site_2 | Combined score | 3553 | 9 | 231 | 0 | 0.00% |
| site_1_density_AB | Combined score | 2851 | 21 | 92 | 2 | 9.52% |
| site_1_density_CD | Combined score | 1905 | 9 | 92 | 0 | 0.00% |
| site_1_density_missing | Combined score | 6 | 0 | 0 | 0 | NA |

Selection is global; no subgroup receives its own 5% budget. Density groups are available only at site 1; site 2 has no recorded density. These sparse subgroup comparisons are descriptive.

## 6. Probability quality

| Group | Score | Brier | Log loss |
| --- | --- | --- | --- |
| pooled | raw | 0.030244 | 0.126135 |
| pooled | calibrated | 0.019251 | 0.089818 |
| pooled | constant_development_prevalence | 0.020194 | 0.100439 |
| site_1 | raw | 0.021093 | 0.097295 |
| site_1 | calibrated | 0.020200 | 0.093756 |
| site_1 | constant_development_prevalence | 0.021335 | 0.105025 |
| site_2 | raw | 0.038979 | 0.153662 |
| site_2 | calibrated | 0.018345 | 0.086060 |
| site_2 | constant_development_prevalence | 0.019105 | 0.096063 |
| site_1_density_AB | raw | 0.020704 | 0.094672 |
| site_1_density_AB | calibrated | 0.019734 | 0.093071 |
| site_1_density_AB | constant_development_prevalence | 0.020749 | 0.102672 |
| site_1_density_CD | raw | 0.021562 | 0.100315 |
| site_1_density_CD | calibrated | 0.020793 | 0.094800 |
| site_1_density_CD | constant_development_prevalence | 0.022066 | 0.107964 |
| site_1_density_missing | raw | 0.011788 | 0.073950 |
| site_1_density_missing | calibrated | 0.000724 | 0.017666 |
| site_1_density_missing | constant_development_prevalence | 0.000431 | 0.020991 |

Calibration was fitted only on development patients. The constant comparator uses development prevalence. Lower values are better; study-sample performance does not establish population calibration.
