# Team Model-Selection Protocol

This protocol gives all six members the same evaluation observations while allowing different training samples and modeling approaches.

## Dataset Status

| Dataset | Status | Evaluation data |
|---|---|---|
| HIGGS | Ready | `validation.csv`; `official_test.csv` is reserved |
| KDD Cup 1999 | Ready | `validation.csv`; `internal_test.csv` is internal only |
| RLCP | Ready for exploration | Source-block split; entity independence is not guaranteed |
| Epsilon | Pending | Do not use yet |

RLCP source-block assignment is blocks 1-8 for training, block 9 for validation, and block 10 for `internal_test`. Measured entity overlap must remain part of the limitation discussion.

## HIGGS Assignments

| Member | Training sample |
|---|---|
| edgar | `train_1000000.csv` |
| luis | `train_200000.csv` |
| fercho | `train_3000000.csv` |
| isaac | `train_1000000.csv` |
| caleb | `train_1000000.csv` |
| michelle | `train_200000.csv` |

Everyone uses the same `validation.csv` and `validation.manifest.csv`. Do not use `official_test.csv` yet.

## Member Workflow

1. Clone the repository.
2. Switch to `feat/experiment-foundation`.
3. Create a personal branch named `exp/<member>`.
4. Download the assigned training sample and the common validation files from Drive.
5. Do not create another train, validation, or test split.
6. Train and compare at least two models.
7. Fit every preprocessing step using training data only.
8. Evaluate every model on the common validation set.
9. Record the experiment metrics.
10. Generate `validation_predictions.csv`.
11. Push the personal branch to GitHub.
12. Upload `validation_predictions.csv` to Drive.

Create the prediction file with exactly this header:

```csv
observation_id,prediction
```

Copy `observation_id` from `validation.manifest.csv` without reordering or regenerating it.

## Metrics Record

Record one row per experiment with these fields:

```text
member,dataset,sample,model,accuracy,recall,f1,training_time_seconds,notes
```

Keep preprocessing decisions, model hyperparameters, and any class-label assumptions in `notes` or an adjacent experiment log. Do not use test labels for model selection or tuning.

## Methodological Boundaries

- HIGGS is the recommended first dataset.
- KDD `internal_test` is not the official KDD test.
- RLCP source blocks provide usable partitions but do not guarantee entity independence.
- Epsilon remains pending because full bzip2 streaming/copying is currently too slow.
- The Hadoop/Spark infrastructure is preserved but is outside this competition workflow.
