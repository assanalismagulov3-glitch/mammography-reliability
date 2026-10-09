# Understanding our mammography study

Prepared 2026-10-01 from the frozen protocol and completed results. Read sections 1–3 first; the remaining sections add technical detail. This is a teaching guide, not a new analysis or evidence that the student has already mastered the work.

## 1. The question in plain English

A classifier assigns an image-based cancer score. A cutoff turns that score into a positive or negative prediction. Some cancer-positive breasts fall below the cutoff. These are **false negatives**: the model predicts negative, but the dataset label is positive.

We asked: **among breasts the model predicts negative, can disagreement between image views and between models help prioritize missed cancers for possible review, beyond the cancer score alone?**

There are two separate decisions:

| Decision | What it does |
|---|---|
| Base classification | Uses one frozen cutoff to label every breast AI-positive or AI-negative. |
| Review ranking | Orders the AI-negative breasts and flags a fixed number for possible review. It does not change the original classifier's predictions. |

The review ranking has no access to holdout cancer labels. We compare its selections with those labels afterward to measure performance. A selected positive label is a flagged miss; it is not evidence that a doctor diagnosed or recovered that cancer.

## 2. What went into the experiment

The main unit is **one breast**, identified by patient and side. One patient contributes two breasts; each breast can have several images.

**CC** means craniocaudal, a roughly top-down projection. **MLO** means mediolateral oblique, an angled projection. They provide different views of the same breast. We retained repeated standard-view images rather than choosing a favorable image.

| Patient group | Patients | Breasts | Cancer-positive breasts | Purpose |
|---|---:|---:|---:|---|
| Development | 2,383 | 4,766 | 99 | Development decisions, final calibration, cutoff and score reference distributions |
| Holdout | 9,530 | 19,060 | 393 | Final evaluation of the frozen rules |

Patients were separated, so both breasts from one patient stayed in the same group. Otherwise information associated with one person could cross the development/evaluation boundary. Holdout metadata and labels had already been used for auditing and stratified assignment; the final rules were frozen before examining holdout model outcomes.

We used five externally trained GMIC models. A model's **weights** are numerical parameters learned during training. In this final experiment those weights stayed fixed: we did not train them on RSNA images. Earlier unsuccessful training trials belong to the documented development history.

For each member, we averaged repeated images within CC and within MLO, then gave the two views equal weight. We then averaged the five member scores. This prevents extra images of one view from automatically giving that view more weight.

The image-processing pipeline was adapted and technically inspected. That inspection does not establish preservation of every subtle lesion.

## 3. Follow the actual numbers

At the frozen cutoff, the holdout produced this confusion matrix. “Positive” and “negative” refer to dataset labels in the columns and model predictions in the rows.

| Prediction | Cancer-positive label | Cancer-negative label | Total |
|---|---:|---:|---:|
| AI-positive | 354 true positives | 10,391 false positives | 10,745 |
| AI-negative | 39 false negatives | 8,276 true negatives | 8,315 |
| Total | 393 | 18,667 | 19,060 |

The classifier's **sensitivity** is the fraction of labeled positive breasts it predicts positive: 354 / 393 = 90.08%. Its **specificity** is the fraction of labeled negative breasts it predicts negative: 8,276 / 18,667 = 44.33%. The relatively low specificity matters when interpreting this operating point. These are model classifications, not observed clinical recall decisions.

Our primary review budget was 5% of **AI-negative breasts**. Therefore each method selected floor(0.05 × 8,315) = **415 breasts**. The floor operation rounds down, so the realized fraction is 4.99%.

| Review method | Breasts selected | Misses flagged | Capture of all 39 misses | Positive-label yield among selections |
|---|---:|---:|---:|---:|
| Cancer score alone | 415 | 3 | 3 / 39 = 7.69% | 3 / 415 = 0.72% |
| Combined score | 415 | 2 | 2 / 39 = 5.13% | 2 / 415 = 0.48% |

**Capture and yield answer different questions.** Capture asks how many of the classifier's misses we selected. Yield asks how many selected breasts had a positive label. Neither number is the classifier's sensitivity.

Both methods had equal capacity, but they did not necessarily select the same breasts. The combined score selected one fewer missed positive breast. The primary capture difference is (2 − 3) / 39 × 100 = **−2.56 percentage points**. The denominator is 39 because the question concerns the classifier's missed positive breasts.

## 4. What the three signals mean

| Signal | Calculation in this study | Interpretation limit |
|---|---|---|
| Cancer score | The calibrated pooled score; larger scores rank first among AI-negative breasts. | A baseline priority score; a low value does not guarantee absence of cancer. |
| View disagreement | Absolute difference between the ensemble's mean CC and mean MLO scores. | Disagreement can reflect useful complementary views, acquisition differences or model error; it is not automatically evidence of cancer. |
| Between-member variation | Population variance of the five pooled breast scores. | A descriptive feature. The members differ through external model selection, so this is not a pure seed-only uncertainty estimate or a validated probability of error. |

We put the three signals on a comparable scale using development-reference **percentiles**: where each value falls relative to the values from AI-negative development breasts. The combined score is the mean of those three percentiles, with equal weights.

For an invented arithmetic example, percentiles 0.60, 0.90 and 0.30 give (0.60 + 0.90 + 0.30) / 3 = 0.60. This combined value does not mean a 60% cancer probability. Actual reference percentiles handle ties using their midpoint; selection ties use a frozen label-independent ordering.

Equal weighting is a declared heuristic to evaluate. It is not a learned optimal weighting or a new neural architecture. We also retained individual-signal and two-signal comparisons.

## 5. Calibration, cutoff and uncertainty

**Calibration** fits a mapping from model scores toward observed outcome frequencies using development data. We used a monotone logistic mapping. Monotone means it preserves score order, so it cannot improve ranking just by changing the numeric scale. Fitting this mapping does not guarantee calibration in a new population.

The raw cutoff, 0.02242065602913494, was chosen on development patients to attain at least 90% empirical sensitivity. It detected 90 of 99 development-positive breasts. That was a fitting target, not a guarantee for the holdout or future patients. We retained the cutoff when evaluating the holdout.

The main statistical question concerns the difference between the two review methods. A single dataset gives one estimate, and the 39-miss denominator makes that estimate sensitive to individual misses. One additional flagged miss changes capture by about 2.56 percentage points.

We estimated sampling uncertainty with a **paired patient bootstrap**:

1. Sample patients with replacement within each site. Keep both breasts together and retain the original number of patients per site.
2. Apply both review methods to the same resampled patients and recalculate the review capacity and capture difference.
3. Repeat 2,000 times. Use the 2.5th and 97.5th percentiles of the differences as the reported 95% interval.

The primary interval keeps the fitted rules fixed: **−13.79 to +8.11 percentage points**. A secondary set of 2,000 draws also resamples development patients and refits calibration, cutoff and references: **−14.29 to +8.00 points**. These are two uncertainty calculations, not 4,000 new independent experiments.

The intervals contain negative and positive differences. Our result therefore does not establish an incremental benefit. It also does not prove equivalence or show that these signals are useless in every setting. A confidence interval is not a direct probability that the true effect lies between its endpoints. These intervals do not capture every source of bias or adaptive development uncertainty.

