# Contribution record

The student selected the breast-cancer AI reliability direction through mentoring and operated the initial Kaggle metadata audit. The aggregate counts in `observed_metadata.json` came from outputs the student shared in the conversation.

On 2026-09-09, the student authorized the assistant to implement and execute the practical work, with AI assistance documented. The assistant authored this source package, the working experiment protocol, the notebook, and software tests. Local tests used artificial fixtures. Real-data training and evaluation had not been executed when this version was prepared.

Update this record as work proceeds: note who designed each experiment, wrote or modified its code, ran it, interpreted results, and reviewed the conclusions. Only claim work that has actually been performed and understood. Venue-specific disclosure requirements should be checked when a submission venue is selected.
# Execution update — 2026-09-09

After the student signed in, the assistant operated private Kaggle notebooks, reran the real metadata audit, decoded and visually checked 20 technical examples, and ran a GPU resource benchmark using artificial tensors. It implemented and tested private cache partitioning to fit notebook storage limits and submitted two full preprocessing jobs. These actions do not constitute trained-model research results, clinical image annotation, or a novelty review. The assistant also identified a pilot-versus-cross-validation leakage risk that must be resolved before outcome-guided model development.

In version 0.2, before any cancer-model performance was inspected, the assistant resolved that risk by implementing a reserved development cohort and a prospective configuration gate. It added software checks for patient isolation, unchanged partitions/code at evaluation, and higher-resolution cache identity. It prepared a separate 1536 × 1152 pilot cache workflow and recorded a targeted prior-art check. The student authorized this practical work; the assistant made and documented these implementation and methodological decisions. Real model learning and final results remain to be evaluated.

## Execution update — 2026-09-10

The assistant verified successful completion of all three private preprocessing jobs from their versioned Kaggle logs. It authored and locally validated the two-resolution development launcher, attached the three private outputs and official dataset, checked private visibility, and submitted the GPU pilot job. The practical work remains AI-authored and AI-operated under the student's authorization. No completed real-model or held-out research result was available when this update was recorded.

## Execution update — 2026-09-23

The assistant reconnected to the private account and verified both pilot trials completed. It extracted their learning curves and recognized that the selected models had weak ranking performance. It authored and checked a separate diagnostic launcher, using only reserved development patients, and recorded the adaptive optimization decisions in `DEVELOPMENT_DECISIONS.md`. The assistant also retained the targeted primary-literature assessment and its access limitations. These experiments, code, and interpretations are AI-assisted work; the student has not yet been mentored through every implementation detail. No final test performance or clinical benefit has been established.

The assistant submitted the diagnostic notebook privately as version 1 (352083246), with the verified cache inputs and GPU T4 x2. Submission is distinct from successful diagnostic completion.

Later on 2026-09-23, the assistant verified diagnostic completion, extracted its full aggregate learning curves, and interpreted the successful tiny-set memorization and weak full-cohort tuning results. It designed and implemented a class-balanced update trial, with the original expected loss scale preserved and explicit gradient/AMP instrumentation. It added and passed three local sampler tests and checked notebook syntax and unchanged embedded source. The hypothesis, code, interpretation, and execution remain AI-assisted work; the student has not yet reviewed every detail. No final test results were opened.

The assistant submitted the balanced-update experiment privately as notebook 08, version 1 (352110149), on GPU T4 x2, after verifying private visibility and attaching the official data and three verified prior outputs.

## Execution update — 2026-09-24

The assistant verified notebook 08 completed, extracted all nine epochs and the selected fit/tune/site summaries, and identified the substantial fitting–tuning gap. It selected a mammography-transfer feasibility check as the next development step, inspected the authors' GMIC methods, source and license, and authored the standalone adapter and preflight. Three synthetic invariant tests and notebook/source checks passed locally. The assistant configured and submitted notebook 09 privately. GPU reference reproduction and real-image visual review are distinct later checks; local tests do not establish medical validity. The upstream classifier and weights are the NYU authors' work. The student's research question and empirical comparisons remain provisional, and the student has not yet reviewed every implementation detail.

Notebook 09 version 1 failed at the strict checkpoint gate after verified downloads. The assistant inspected the original `strict=False` loader and absence of `shared_rep_filter` from the network, added a named unused-tensor exception and a regression test, and retained the failed attempt. It did not silently disable strict loading or alter the network to obtain a result.

The assistant investigated the version-2 reference mismatch by running the unmodified original current CLI in version 3. All processed arrays matched the adapter exactly. It inspected repository history and found a later replacement of sample images/preprocessing while the reference table and checkpoint blobs stayed unchanged. It then declared a version-4 check of historical fixtures plus current-CLI parity before allowing fit-only image inspection. This revised gate, its rationale, and all unsuccessful attempts are documented; the reference tolerance was not increased and no RSNA outcome was used to choose the change.

The assistant verified successful version-4 compatibility checks, then identified a clinically material geometric exclusion in the technical image display: a fixed window omitted most visible breast tissue in one large image. It rejected that adaptation without consulting RSNA cohort accuracy, implemented full-field resizing and padding, and passed three geometric software tests. It also corrected a thumbnail-only Pillow error, retained both versioned audit attempts, and began visual review of the corrected 20-image audit. This is AI-performed engineering inspection, not a radiologist's diagnostic assessment.

The assistant prepared a reserved-pilot fixed-GMIC evaluation with explicit probability pooling, complete image/patient identity checks, and separate fit/tune/site descriptive metrics. Three synthetic tests check repeat-view weighting, prediction coverage, malignant-channel use, invalid probabilities, and rejection of nondevelopment roles. GPU execution of this pilot remains contingent on the technical visual review.

The assistant completed technical visual review of all five corrected contact sheets (20 pairs), recorded the limitations and exact reviewed-plan identity, and prepared notebook 11 for all five fixed GMIC models on the reserved pilot. It validated notebook syntax/schema and confirmed all ten original package modules remain unchanged. The technical visual review is AI-assisted and not a diagnostic image review. No final research result or clinically actionable protocol is claimed.

The assistant submitted the fixed-model development pilot privately as notebook 11 version 1 (352363590) with GPU T4 x2, after verifying private visibility and attaching the two reviewed outputs plus official data. Submission and completed cohort inference are separate; current observed status is saved with the validation records.

Notebook 11 version 1 stopped before cohort inference because a pandas Series method shadowed the view column in the reproduction probe. The assistant reproduced the accessor problem locally, changed the probe row to a named tuple, checked the exact scalar view/identifier expression, regenerated the notebook, and submitted a corrected version. The main experiment settings were unchanged. All failed attempts are retained for transparent provenance.

The assistant verified notebook 11 version 2 (352364754) passed its startup gates and exactly reproduced all five sample-11 audit predictions. Full pilot inference was still running at that observation; no completed cohort AP/AUC or final test result was claimed.

## Execution update — 2026-09-28

The assistant verified completion of notebook 11 version 2, recovered exact scored development tables and provenance, and independently reproduced breast aggregation and all original ranking summaries. It authored the development-only operating-point plan after inspecting AP/AUC but before calculating thresholds/review endpoints. It fit calibration and the 90% sensitivity threshold on the fit group only, evaluated unchanged rules on tune patients, retained all six methods and four budgets, and calculated a conditional paired patient bootstrap. Two new artificial isolation/global-budget checks passed.

The assistant interpreted the 18/26 detected tune cancers, 8 misses, and sparse 1-versus-0 primary flagging result, including the unfavorable 20% review comparison. It authored a private CPU Kaggle reproduction and a display audit of all three previously flagged fitting images. The first CPU attempt reproduced all 6 confusion groups and 24 review comparisons, then failed at image decoding because plugin discovery preceded decoder installation. The setup was repaired to install required decoders and run image review in a fresh process; no prediction or analysis rule changed. Failed attempts remain documented.

These code, analysis and interpretation steps were performed by the assistant under the student's authorization. The student has not yet been mentored through every implementation detail. No human diagnostic review, final test result, novelty claim or public release has occurred.

Notebook 12 version 2 (353577896) subsequently completed in 46.7 seconds. All six confusion groups and 24 review comparisons reproduced. The assistant inspected all three original/processed image pairs and found no obvious gross orientation, inversion or boundary-exclusion error at contact-sheet scale. This is technical inspection, not diagnostic review; it does not establish fine-detail preservation. All cases and scores remain unchanged.


## Development threshold stability — 2026-09-28

The assistant declared an adaptive resampling plan after the original operating-point findings, implemented weighted patient-cluster resampling, and executed all 2,000 paired fit/tune bootstrap draws and 100 alternative development partitions. Seven artificial checks verified duplicate-row equivalence, patient clustering, evaluation-label isolation, original-evaluator agreement and split separation. Existing input hashes, pooling/ranking and original outcome counts reproduced. Both breasts stayed together; site patient counts were preserved in bootstrap draws. No new image inference, classifier fitting, final patient access or GPU run occurred.

The assistant authored the report, inspected the figure and retained every aggregate draw. It recorded the instability and unfavorable combined-score comparisons without changing the original cutoff, original roles or final pipeline. These are assistant-performed analyses under the student's authorization; the student has not yet explained or defended the full implementation independently. The next substantive prior-art/protocol review remains outstanding.


## Related-work comparison and final-design specification — 2026-09-28

The assistant searched both available web engines, followed primary references, inspected accessible publisher/arXiv/conference methods, and read/rendered the CEI-Net and M&M main PDFs. It distinguished examined full text from indexed abstracts and unresolved access failures. It identified direct overlap with uncertainty referral and cross-view disagreement; no novelty claim was made.

The assistant authored the evidence matrix, access/query log, fixed-GMIC final protocol and machine-readable design. It proposed refitting calibration on all reserved development patients and evaluating once on the original holdout, retaining the original primary score/budget and all negative development findings. This is an adaptive design decision before final model outcomes, not a public preregistration.

No final calibration fit, compatible final evaluator, execution freeze or holdout inference was completed in this step. No paper, clinical protocol or public upload was produced. The student has authorized this practical assistance and still needs mentoring to independently explain the code and defend the scientific decisions.


## Evaluator, calibration and launch preparation — 2026-09-28

The assistant wrote the fixed-model evaluator, preparation and gate modules, inference/analysis programs, notebook generator and 18 software checks. It verified historical pooling/scores and all 24 review comparisons; repaired an empty-input validation error; recovered complete original cohort identities through the visible Kaggle output interface; fitted all-development calibration; and created a source/artifact freeze. The raw cutoff is 0.02242065602913494. Ninety of 99 calibration cancers are detected by construction, not independent validation.

The assistant prepared six private notebooks and created the separate Kaggle draft 13. Its file-chooser upload was interrupted before import completed. No Kaggle CPU verification, final image inference, final performance calculation, paper or public release occurred. The code, private checkpoint and exact restart state were preserved. These practical steps were assistant-performed under authorization; the student's later understanding and defense of the methods remain necessary.


## Private final execution — 2026-09-29 to 2026-10-01

The assistant verified the successful private CPU check, imported and launched all five frozen inference jobs, and recorded their exact versions, completion hashes and startup reproduction probes. All five completed without outcome evaluation. On October 1, following the user's response to the explicit private-upload authorization request, the assistant imported the prepared final-analysis notebook and submitted private notebook 19 version 1 (354389561) on CPU. It attached the official dataset and six completed outputs, retained the pinned environment and left every frozen scientific file and calibration artifact unchanged. The September 30 automatic-approval rejection is retained as history. At this launch checkpoint the global integrity and statistical analysis is running; its completion and scientific interpretation are recorded separately. These are assistant-operated research steps; student understanding, diagnostic review, clinical benefit and novelty are not established by successful execution.


## Final result extraction and interpretation — 2026-10-01

The assistant verified successful completion of the original frozen global analysis (notebook 19 v1, 354389561), recovered the full aggregate JSON and all 4,000 bootstrap rows through the visible browser interface, checked row coverage and exact agreement with the original log, and independently reproduced both percentile intervals from the saved draws. Kaggle's truncated preview was not treated as a complete file; a 200,000-character browser return limit in the first CSV read was detected and resolved with complete DOM-text chunks before verification. The bulk download did not yield a local archive; full private outputs remain in the original Kaggle versions.

The assistant authored the final report and documented the inconclusive primary comparison (2 versus 3 of 39 missed cancer-positive breasts flagged at 415 selections per method; −2.56 percentage-point difference, primary 95% interval −13.79 to +8.11), all prespecified secondary tables and study limitations. No frozen rule or model was changed after outcomes. These analyses, execution checks and interpretations are assistant-performed under the student's authorization. The student still needs to understand and defend the work; no human diagnostic review, clinical benefit, established novelty, paper submission or public release occurred.


## Post-outcome literature and contribution review — 2026-10-01

The assistant continued the targeted search using both available web engines, inspected Verboom supplemental PDF pages 1–4, examined selected sections of Gudhe's published-paper transcription, and rechecked the CEI-Net official record. It retained inaccessible leads as unverified and did not treat indexed abstracts or reviewer opinions as fully examined methods. Exact retained queries and access limits are recorded separately.

The assistant authored the October 1 literature addendum and proposed cautious manuscript wording based on the completed, inconclusive comparison. This interpretation was written after final outcomes; it is not a preregistration or a change to the frozen analysis. The earlier frozen literature decision, scientific code, configuration and calibration artifacts remain unchanged. No new experiment, student demonstration of understanding, manuscript submission or public upload occurred. Four preparation blocks remain.


## Student methods/results review — 2026-10-01

The assistant checked the current final results and frozen protocol/implementation, prepared `STUDY_EXPLANATION.md`, and began explaining the two-stage classification/review design in English. The guide distinguishes capture from yield and sensitivity, explains calibration and patient-cluster resampling, and preserves the inconclusive interpretation. Its percentile arithmetic example is explicitly invented and contains no patient data.

Preparing explanations is not evidence that the student independently understands the work. The review remains in progress, with an initial request for the student to explain the 39-miss and 415-selection denominators. Four preparation blocks remain. No scientific code, frozen rule, original result or public-release state changed.


## Practical release preparation — 2026-10-01

The user explicitly prioritized finishing the practical work and GitHub upload and deferred detailed explanations until before paper writing. The assistant removed teach-back questions as a release prerequisite, preserved the optional teaching guide and did not claim independent student mastery.

The assistant generated three aggregate-only figures and six table sections, independently reproduced both original bootstrap intervals from the saved draw rows, and ran the 18 original fixed-GMIC tests successfully on the recorded CPU release environment. It made the final GMIC study the README entry point, documented exact commands and private execution prerequisites, and prepared an explicit source-and-aggregate release. An initial reporting-script run used the wrong bootstrap column names and stopped before exporting; it was corrected to the original CSV schema without altering saved results. The flow diagram was rendered and its text spacing corrected.

The frozen 32 scientific files and original five calibration/cohort artifacts remain unchanged. No GPU inference, threshold revision or new statistical endpoint was introduced. Original restricted records, images, private embedded notebooks, weights and raw logs are excluded from the release. No clean-room reproduction of the entire private pipeline or new project license grant is claimed. GitHub installation/connection is pending; no remote upload is claimed at this checkpoint.

Release audit completed: all 32 frozen source hashes and five private rule/cohort hashes match. The staged source-and-aggregate package passed its input/hash verification and regenerated the figures/tables. Eleven historical launcher notebooks have no cell outputs, attachments or embedded patient-table literals. The original private generated final notebooks are excluded. One practical step remains: GitHub upload and remote verification, pending installation/connection.
