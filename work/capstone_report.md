# Ranking pages for next-month search growth

**Author:** Vincent Oei

**Lane:** Search growth prediction and human review prioritization

**Decision date:** 31 March 2026

**Repository:** [flyrank-starter](https://github.com/vincentoei/flyrank-starter)
**Paper site:** [flyrank-research](https://flyrank-research.vercel.app/)

## Abstract

Can search signals available by 31 March 2026 help a content team decide which pages to inspect for April impression growth? I constructed a page-level cohort from FlyRank's v20260703 internship warehouse using January–March performance and a future April label. A transparent momentum-and-volume rule and three classifiers were compared on the same five client-grouped validation folds. Gradient Boosting achieved mean Precision@50 of 79.6% versus 42.8% for the rule, against a 24.7% growth prevalence, although one held-out fold favored the rule. These rankings support human review of content opportunities; they do not show that editing a page causes growth.

## 1. Problem framing

A content team has limited review capacity. The unit is a page at the close of 31 March 2026; the output is a ranking score for which pages to inspect first for April impression growth. The human editor checks the page, search results, and business context before deciding to expand, refresh, improve a snippet, monitor, or defer work. A false positive consumes scarce review time; a false negative may delay attention to a useful page. Ranking helps allocate that capacity, but it does not prescribe edits automatically.

## 2. Data and public safety

The source is the gated [FlyRank ML Internship warehouse, release v20260703](https://huggingface.co/datasets/FlyRank/internship-warehouse). The analysis joins `fact_content_daily_performance` with `dim_content`. One row represents one eligible page. Features aggregate 1 January–31 March 2026 (90 days); the reference total uses 1–31 March (31 days); the outcome total uses 1–30 April (30 days). The earlier variable name `last_30d` is retained in the original notebook but refers to all 31 March days. These are calendar-month totals with unequal exposure, not equal-length 30-day windows.

The cohort has 67,478 pages in 45 pseudonymized client groups. Eligibility requires at least 300 January–March impressions, a page created by 31 March, and a non-missing word count. The positive label requires April impressions at least 1.20 times March impressions **and** at least 100 March impressions. The observed positive prevalence is 24.7% under the original label rule. The source query assigned a negative label when a monthly aggregation was absent; 3,022 pages have no April total. That assumption can mix unobserved outcomes with real non-growth.

Pseudonymous page and client IDs are used for joins and client grouping only, never model inputs. April impressions and any future-derived fields are excluded from features. Starter-slice label-derived `trend_direction` and `trend_pct`, post-decision update fields, product decision flags, page URLs, titles, client names, and raw queries are excluded. The report and site expose aggregate metrics only; the row-level feature cache and queue remain gitignored. GA4 organic sessions are counted only where its availability flag is true in the source construction; missing analytics coverage remains a limitation.

## 3. Transparent baseline

The baseline scores a page `recent_share × log1p(impressions_90d)` if it has at least 500 impressions over January–March, and zero otherwise. Here `recent_share` is March impressions divided by January–March impressions, clipped to [0, 1]. The rule uses only information available at the decision date. It gives editors a simple high-volume, recent-momentum ordering to compare with the learned rankers. Its five-fold mean Precision@50 is 42.8%.

## 4. Model and label method

The Week 5 experiment compares logistic regression, random forest, and Gradient Boosting. The feature vector has 24 columns: pre-decision impressions, clicks, click-through rate, mean search position, organic sessions, days with impressions, March impressions, search volume, competition, cost per click, word count, page age, recent impression share, log transforms, a momentum-by-volume interaction, and one-hot content-type and intent categories. Missing numerical values are filled with zero in the Week 5 modeling frame; this can leave availability patterns in the features and deserves future testing.

The Gradient Boosting model uses 200 estimators, maximum depth 5, learning rate 0.05, seed 42, and balanced sample weights. These scores are used for ordering; probability calibration was not evaluated, so they are not reported as literal chances of growth. The source code for exact feature construction and training is [Week 5's notebook](https://github.com/vincentoei/flyrank-starter/blob/main/work/notebooks/w05_model.ipynb).

## 5. Validation and results

The primary evaluation is five-fold `StratifiedGroupKFold`, grouped by pseudonymized client ID with seed 42. A client never appears in both training and held-out data for the same fold. Precision@50 is the share of positive labels among the top 50 pages in **each held-out fold**, then averaged over five folds. The rule and models are ranked on the same held-out pages. This is a transfer-to-unseen-clients test within one historical snapshot, not a later-month test.

| Ranker | Mean Precision@50 | Mean ROC AUC | Mean average precision |
|:--|--:|--:|--:|
| Gradient Boosting | 79.6% | 0.730 | 0.483 |
| Random forest | 78.0% | 0.724 | 0.486 |
| Logistic regression | 57.6% | 0.677 | 0.430 |
| Momentum-and-volume rule | 42.8% | 0.599 | 0.351 |

The 24.7% base growth rate is the prevalence, not a held-out random-ranking experiment. Per-fold Gradient Boosting Precision@50 is 90%, 76%, 92%, 48%, and 92%; the rule scores 22%, 36%, 36%, 62%, and 58%. The rule beats Gradient Boosting in fold 4. The mean difference of 36.8 percentage points corresponds to roughly 18 more labeled growers per 50 reviews on these folds, not a guaranteed gain on future work. The detailed receipt is [Week 5 metrics](https://github.com/vincentoei/flyrank-starter/blob/main/work/outputs/w05_model_metrics.json).

A random row split diagnostic scored 92% Precision@50 and ROC AUC 0.821 for Gradient Boosting, versus 79.6% and 0.730 in grouped validation. Those are different validation designs and sample partitions, so the gap is evidence that row-random evaluation is optimistic for this use case, not a measured amount of client memorization. The [Week 6 audit](https://github.com/vincentoei/flyrank-starter/blob/main/work/notebooks/w06_validation_audit.ipynb) also tests a deliberately future-derived field as a leakage probe; it is excluded from the final feature set.

In a **post-hoc sensitivity check**, pages without both March and April totals were removed. The resulting 64,456-page, 43-client subset has a 25.9% positive prevalence; on new five client-grouped folds, Gradient Boosting mean Precision@50 is 89.2% and the rule is 42.8%. This shows the original missing-outcome convention matters. It does not supply a sealed future-time evaluation or repair the unequal calendar months. See the [script](https://github.com/vincentoei/flyrank-starter/blob/main/work/scripts/w08_observed_outcome_sensitivity.py) and [aggregate receipt](https://github.com/vincentoei/flyrank-starter/blob/main/work/outputs/w08_observed_outcome_sensitivity.json).

## 6. Interpretation and limits

In the Week 5 tree impurity ranking, page age and recent impression share receive the largest shares. These are model-use diagnostics, not signed effects, causal explanations, or independent evidence of what an edit would change. The Week 4 signal audit found **mixed** evidence for simple recent-impression concentration and active-day signals; this is one reason to compare against a rule instead of treating momentum as a settled truth.

The study uses one historical feature/outcome pair and chooses the strongest ranker using the same cross-validation results it reports. It has no prospective or later-month holdout, no uncertainty interval for future deployments, no calibrated score assessment, sparse GA4 coverage, and a label that treats absent monthly aggregates as negative. March and April totals have different numbers of days. Site demand, search changes, client mix, and eligibility may shift. The model estimates a historical ranking association; neither the score nor the queue establishes that editing a selected page causes impressions to rise.

## 7. Ranked recommendations

Use the ranked output as a weekly human-review queue. Inspect the top 50 first, check the reason code and current search result, and record whether an action was taken. Candidate actions from the Week 7 playbook are: **expand and protect** a promising younger page where depth is useful; **refresh** an older page with renewed momentum where substantive updates are possible; **optimize snippet** when visibility exists but click capture looks weak; **monitor** pages without an urgent intervention; and **deprioritize** low-volume or fading candidates for this cycle. A label is a review prompt, not a treatment recommendation proven to work.

The Week 7 full-cohort fit assigns 5,232 pages to expand/protect (7.8%), 1,516 to refresh (2.2%), 7,328 to snippet review (10.9%), 49,437 to monitor (73.3%), and 3,965 to deprioritize (5.9%). It fits and scores the same cohort, so these counts describe an illustrative policy output, not independent validation. Review capacity, content quality, and editorial judgment still decide the actual workload. Monitor monthly feature availability, client mix, queue composition, and realized later-window Precision@50 before making this an operational workflow.

## 8. Reproducibility

The committed notebooks, figures, scripts, and JSON receipts are in [`work/`](https://github.com/vincentoei/flyrank-starter/tree/main/work). The warehouse is gated; a fresh runner needs approved dataset access and a `HF_TOKEN` environment secret. Never paste a token in a notebook or commit the feature vector. In an environment with the repository requirements installed:

1. Run `work/notebooks/w05_model.ipynb` top to bottom from its notebook directory. It creates a gitignored `work/outputs/model_feature_vector.csv` cache and the public aggregate `w05_model_metrics.json` receipt.
2. Run `work/notebooks/w06_validation_audit.ipynb` and `work/notebooks/w07_action_playbook.ipynb` from the same notebook directory for the validation diagnostic and action counts.
3. Run `python work/scripts/w08_observed_outcome_sensitivity.py` from the repo root to regenerate the post-hoc aggregate receipt.
4. Run `work/notebooks/capstone.ipynb` top to bottom from `work/notebooks/` to regenerate the paper's charts and summary tables from the committed receipts.

Random seed is 42 in grouped splits and model fits. The original training run used scikit-learn models recorded in Week 5; the capstone synthesis needs `pandas`, `numpy`, `matplotlib`, and `nbclient`/Jupyter. The public site is a Next.js publication built separately in `projects/flyrank-research` and hosted on Vercel, with no runtime API.

## 9. Acknowledgments and data credit

[Built on the FlyRank ML Internship dataset](https://flyrank.ai). The gated warehouse release and the internship's research workflow supplied the source and framing; all interpretations and limitations here are the author's.
