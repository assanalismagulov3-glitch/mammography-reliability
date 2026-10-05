# Deferred extension: uncertainty flags and review policies

Recorded on 2026-09-09 at the student's request. This is a planning note, not an implemented method, a protocol freeze, or a research finding.

The student would like to combine a technical uncertainty/failure-flagging prototype with a clinically motivated policy for additional human review. They explicitly asked to assess the idea, remember it, and revisit it later without distracting from the current work.

## Assessment

Keep one focused study: technical failure detection as the primary contribution, with a possible retrospective simulation of how review candidates would be selected. The existing comparison of false-negative capture at fixed review budgets already provides part of this second component. A new uncertainty metric is not required unless the literature review and experiments justify one; established metrics must remain comparison methods.

The model does not know which predictions are wrong. Its flag indicates a candidate for review and can itself produce false alarms or miss errors. Current ranking scores are not calibrated probabilities that a prediction is incorrect. A negative cancer prediction is not proof that a breast is cancer-free.

An additional policy could compare uncertainty-based selection with a simple density-based or age-based rule using information available at screening. Define the decision unit explicitly: the current endpoint is breast-level; referring a patient and reviewing both breasts changes the workload calculation. Compare policies at the same workload and report missed-cancer flags, false alarms, subgroup counts, and uncertainty intervals. Flagging a labeled cancer in a simulation does not show that a radiologist would find it or that patient outcomes would improve.

## Important correction to the suggested example

HR-positive means the tumor has estrogen receptors and/or progesterone receptors. For the cancer being screened for, receptor status is normally established from tissue obtained by biopsy or surgery after diagnosis. [National Cancer Institute](https://www.cancer.gov/types/breast/diagnosis/breast-cancer-biomarker-tests)

Our audited RSNA metadata contain no ER, PR, or HR-status labels. Using subsequently obtained tumor receptor status to decide who should have been reviewed at initial screening would introduce unavailable future information. A documented history of a previous tumor is a different variable and would require an appropriate dataset and explicit availability dates. We have no evidence establishing dense tissue plus HR positivity as a review rule for this project. Density is usable only at site 1 in the current release.

## Timing and scope

1. Continue the current image preparation and isolated development pilots.
2. During the substantive literature review and pilot assessment, decide whether a small review-policy comparison adds a defensible contribution. Check variable availability, sample sizes, and breast-versus-patient workload definitions.
3. If included in the same study, specify rules, thresholds, budgets, and endpoints using development data before freezing the final protocol and opening held-out results. A rule invented after inspecting final results is exploratory and requires fresh validation.
4. After the model and evaluation are stable, optionally build a minimal demonstration showing the cancer score, candidate review flag, and evidence behind that flag. A user interface is optional; reproducible experiments are the essential deliverable.
5. A clinical double-reading protocol requires collaboration with clinicians and a separate evaluation of the actual human workflow and its outcomes. Lack of a medical degree does not prevent rigorous computational research, but a working prototype does not establish clinical validity. [DECIDE-AI reporting guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC9116198/)

Decision: retain as a conditional extension; continue the current technical work. This note makes no clinical-validity or novelty claim.
