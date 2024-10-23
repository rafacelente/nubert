# nubert

NuBERT is an approach to modeling time-series tabular data through the adaptation of transformer-based language models.

## Overview
NuBERT demonstrates that language model architectures can effectively learn representations of tabular time-series data. The model is pre-trained on the Oklahoma State Fiscal Year 2013 Purchase Card dataset and can be fine-tuned for various downstream tasks such as transaction amount classification.

Key features:

- Encoding scheme for tabular data that preserves structural and temporal information
- Quantization approach for numerical values
- Temporal binning for datetime features
- Pre-training using masked language modeling
- Fine-tuning capabilities for downstream tasks

## Requirements

- Python 3.9 or higher.

## Installation

To install Nubert, clone the repository and install either with pip or poetry.

1. 
    ```bash
    git clone git@github.com:rafacelente/nubert.git
    cd nubert
    pip install .
    ```
2. 
    ```bash
    git clone git@github.com:rafacelente/nubert.git
    cd nubert
    poetry install
    poetry shell
    ```

## Usage

### Scripted

The `scripts` directory contains scripts for easily training, fine-tuning and evaluating the model. To run the training and fine-tuning scripts, create an YAML config file with both the trainer and model configurations.

```yaml
model_name: "distilbert/distilbert-base-uncased"
file_name: "nubank_raw.cleaned"
dataset_path: "/notebooks/nubank/"
max_length: 512
num_transactions: 5
stride: 1
num_bins: 20
randomize_column_order: False
from_cleaned_data: True
trainer:
  learning_rate: 5e-5
  num_train_epochs: 1.0
  bf16: True
  per_device_train_batch_size: 64
  per_device_eval_batch_size: 64
  output_dir: "/notebooks/nubank/"
  overwrite_output_dir: True
  save_total_limit: 1
  evaluation_strategy: "steps"
  eval_steps: 5000
  report_to: "wandb"
  remove_unused_columns: True
  load_best_model_at_end: True
  logging_steps: 5
```

Then, call the training script by doing

```bash
python scripts/train_nubert.py --config_path /path/to/config.yaml
```

### Modular

You can use nubert as a library and import its modules.

```python
from nubert.config import NubertPreTrainConfig
from nubert.datasets import NuDataset
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

config = NubertPreTrainConfig.from_yaml(args.config_path)
dataset = NuDataset.from_config(
    config=config,
)

model = AutoModelForMaskedLM.from_pretrained(config.model_name)
tokenizer = dataset.tokenizer.base_tokenizer

tokenizer.save_pretrained(config.trainer.output_dir)
model = resize_model_embeddings(model, tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    
training_args = TrainingArguments(
    **config.trainer.model_dump()
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


```