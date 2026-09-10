# Big Data Model Selection

Reproducible preparation and shared evaluation data for the team's classification experiments on HIGGS, KDD Cup 1999, RLCP, and Epsilon.

## Quick Start

```bash
git clone <repository-url>
cd bigdata-cluster
git switch feat/experiment-foundation
git switch -c exp/<member>
```

Download the assigned training sample and the common validation files from Drive. Do not create new splits and do not use `official_test.csv` yet. Follow the complete workflow in [`docs/experiment-protocol.md`](docs/experiment-protocol.md).

Required prediction format:

```csv
observation_id,prediction
```

The current priority is model selection. Hadoop and Spark cluster work remains preserved separately.
