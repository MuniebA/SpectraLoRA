# API Reference

## Configuration

### `spectra_lora.config.SpectraConfig`
The central dataclass controlling the library's hyperparameters.

**Attributes:**
* `BAND_MAP` *(dict)*: Maps channel indices to their spectral names. Default assumes Prithvi-100M HLS ordering: `{'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3, 'SWIR': 4, 'SWIR2': 5}`.
* `LORA_R` *(int)*: The rank of the adapter matrices. Default: `4`.
* `LORA_ALPHA` *(int)*: The scaling factor. Default: `8`.
* `NUM_ADAPTERS` *(int)*: The number of expert adapters per layer. Default: `3`.
* `GATE_HIDDEN_DIM` *(int)*: The hidden dimension of the MLP router. Default: `32`.
* `GATE_TEMPERATURE` *(float)*: Controls the sharpness of the routing decision. Default: `1.0`.

---

## Model Surgery

### `load_prithvi_model(repo_id, filename)`
Downloads and initializes the bare Prithvi-100M Vision Transformer architecture.

**Returns:** * `torch.nn.Module`: The frozen foundation model encoder.

### `inject_spectra_lora(model, config)`
Recursively traverses a PyTorch model, finds the QKV (Query-Key-Value) linear layers within the Attention blocks, and replaces them with `SpectraLoRALayer`.

**Returns:**
* `torch.nn.Module`: The modified model, ready for training.

---

## The Neural Architecture

### `spectra_lora.layers.SpectraLoRALayer`
The custom PyTorch module that wraps a frozen Linear layer and adds the physics-aware sidecar. 

**Forward Pass Requirements:**
Because this layer requires the global physics context $z$, standard PyTorch forward passes (which only pass $x$) will fail. You must implement a context manager or monkey-patch the forward pass in your training loop to supply $z$.

### `spectra_lora.gating_network.SpectralGate`
A Multi-Layer Perceptron (MLP) that maps the 5-dimensional physics vector to softmax routing probabilities for the adapter bank.

---

## Utilities

### `count_parameters(model)`
Prints a statistical breakdown of the model, showing the exact efficiency gains (e.g., "1.2% trainable parameters") achieved by the LoRA injection.

---

## MLOps Database Tracking (`spectra_lora.db`)

### `log_experiment_start(run_name, device, configs)`
Initializes a new database entry, auto-generating a unique UUID and logging hardware/hyperparameters.
**Returns:** * `str`: The unique `run_id` required for subsequent logging.

### `log_epoch_metrics(run_id, epoch, metrics)`
Logs execution data for a specific epoch.
**Parameters:**
* `metrics` *(dict)*: Expects keys like `train_loss`, `val_loss`, `learning_rate`, `miou`, `pixel_accuracy`, and `physics_violations`.

### `log_experiment_end(run_id, weights_path, status)`
Finalizes the database entry, recording the time of completion and the path to the saved `.pth` weights.