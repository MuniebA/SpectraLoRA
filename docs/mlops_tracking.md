# MLOps Database Tracking

SpectraLoRA includes a built-in, zero-friction MLOps database tracker. By default, every time you run a training loop, the library automatically creates a local SQLite database (`spectralora_experiments.db`) to log your hyperparameters, epoch durations, and loss metrics.

## Using PostgreSQL for Enterprise Tracking
If you are running massive training loops across a cluster or want to centralize your team's experiments, you can seamlessly override the local SQLite fallback and connect SpectraLoRA directly to a PostgreSQL database. 

You do not need to write any SQL or manually create tables. SpectraLoRA handles namespace protection (prefixing all tables with `spectralora_`) and automatically generates the required schema.

### How to Connect
Set the `SPECTRALORA_DB_URL` environment variable in your terminal before running your training script:

**Linux/macOS:**
```bash
export SPECTRALORA_DB_URL="postgresql://username:password@server_address:5432/database_name"
python experiments/train.py

```

**Windows (PowerShell):**

```powershell
$env:SPECTRALORA_DB_URL="postgresql://username:password@server_address:5432/database_name"
python experiments/train.py

```

### Tracked Metrics

The database tracks both standard ML performance and GeoAI-specific data:

* **Hyperparameters:** LoRA Rank, Alpha, Gate Temperature, Number of Experts.
* **Epoch Data:** Training Loss, Validation Loss, Learning Rate.
* **GeoAI Metrics:** Mean Intersection over Union (mIoU), Pixel Accuracy, and Physics Violation Scores.
