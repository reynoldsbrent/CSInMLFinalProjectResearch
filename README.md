# How Data Splitting Affects RAG Hallucination Detection

This repository contains the research code and documentation for comparing
response-level and source-task-level data splitting strategies for binary
hallucination detection on RAGTruth.

## Repository contents

- `notebooks/ragtruth_split_experiments.ipynb` - reproducible TF-IDF +
  Logistic Regression experiments.
- `research/audit_dataset.py` - dataset schema, quality, annotation, and split
  audit.
- `research/audit_results/` - compact audit outputs that do not contain raw
  response text.
- `How Data Splitting Affects RAG Hallucination Detection.pdf` - project
  proposal.

The raw RAGTruth dataset is intentionally excluded from this repository.
`RAGTruth/` is an independent upstream clone and is ignored by the parent
repository.

## Dataset setup

The notebook expects these files:

```text
RAGTruth/dataset/response.jsonl
RAGTruth/dataset/source_info.jsonl
```

In Google Colab, the recommended location is:

```text
/content/drive/MyDrive/RAGTruth/dataset/
```

Run the notebook's Google Drive mounting cell first, then run the path
configuration and data-loading cells. The notebook uses the Drive dataset when
it exists and otherwise falls back to the local project layout.

## Running in VS Code with Colab

1. Open `notebooks/ragtruth_split_experiments.ipynb` in VS Code.
2. Select `Kernel -> Colab -> Auto Connect`.
3. Mount Google Drive when prompted.
4. Run the cells from top to bottom.
5. Commit notebook source changes to GitHub from the local project directory.

The Colab runtime is temporary. The notebook writes tables and figures to
`/content/drive/MyDrive/RAGTruth-experiment-results/` when that Drive location
is available. This preserves results across runtime resets. If Drive is not
mounted, outputs fall back to the local `results/` directory.

## Experimental design

Only records with `quality == "good"` are modeled. The binary label is one when
the response has at least one annotated hallucination span and zero otherwise.
The notebook evaluates:

- response-only TF-IDF input;
- context-plus-response TF-IDF input;
- response-level random splitting;
- source-task-level grouped splitting;
- five fixed random seeds;
- F1, precision, recall, PR-AUC, and accuracy;
- overall and task-type results for QA, Summary, and Data2txt;
- final evaluation on the untouched official test split.

The official test set is not used for model selection or tuning.

The notebook also includes a pretrained encoder extension using
`distilroberta-base`. This section requires a GPU runtime and uses
context-plus-response input with both split strategies and three fixed seeds.
It uses two epochs and a 256-token maximum length to keep the experiment
feasible while preserving a meaningful encoder split comparison. Validation
results are checkpointed after every completed run, and the final encoder is
trained once on all official training data before official-test evaluation.

The notebook selects the primary input condition using mean internal-validation
PR-AUC. It then reports the selected condition separately on the untouched
official test set. Response-only and context-plus-response test results are
also retained as a secondary comparison.

The source-task-level splitter guarantees that no source task appears in both
training and validation. It does not directly stratify the response-level
hallucination label, so the notebook reports validation positive rates for
every seed and computes paired split-strategy differences with 95% confidence
intervals. This limitation should be disclosed when interpreting the results.

## Reproducibility

The notebook records the split seeds, validation size, feature settings,
classifier settings, label definition, and package versions in
`experiment_configuration.json`. CSV tables and figures are written to the
configured results directory. Raw JSONL data, model artifacts, predictions,
and notebook checkpoints are excluded by `.gitignore`.

After rerunning the notebook in Colab, verify that the following files exist in
the persistent Google Drive results directory:

- `compact_validation_results.csv`
- `paired_strategy_differences_with_ci.csv`
- `input_condition_selection.csv`
- `selected_official_test_results.csv`
- `official_test_results_all_input_conditions.csv`
- `encoder_validation_results_by_seed.csv`
- `encoder_validation_summary.csv`
- `encoder_input_condition_selection.csv`
- `encoder_selected_official_test_results.csv`
- `encoder_validation_checkpoint.csv`

Commit the notebook and README from the local project directory:

```powershell
cd "C:\Users\brent\OneDrive\Desktop\CSInMLFinalProjectResearch"
git add notebooks\ragtruth_split_experiments.ipynb README.md
git commit -m "Complete reproducible split analysis"
git push
```
