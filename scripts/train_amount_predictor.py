import os
import argparse
import logging

from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset
import wandb
from sklearn.model_selection import train_test_split
from nubert.datasets import AmountDataset
from nubert.config import AmountConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NuBERT on the Amount prediction task")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--wandb_project", type=str, default="nubert-amount-predictor", help="Wandb project name")
    parser.add_argument("--log_model", action="store_true", help="Log model to wandb")
    return parser.parse_args()

def split_dataset(dataset: AmountDataset, val_size: float = 0.1, seed: int = 42):
    train, val = train_test_split(dataset, test_size=val_size, random_state=seed)
    
    return train, val

def create_hf_dataset(data):
    input_ids = [example["input_ids"] for example in data]
    labels = [example["label"] for example in data]
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})

def train_model(
    dataset: AmountDataset,
    config: AmountConfig,
    num_labels: int,
    ):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
    )
    tokenizer = dataset.tokenizer.base_tokenizer

    train_data, val_data = split_dataset(dataset)

    train_dataset = create_hf_dataset(train_data)
    val_dataset = create_hf_dataset(val_data)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
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

    trainer.save_model()
    tokenizer.save_pretrained(config.trainer.output_dir)
    wandb.finish()


def main():
    args = parse_args()

    config_path = args.config_path
    wandb_project = args.wandb_project
    log_model = args.log_model

    assert os.path.exists(config_path), f"Configuration file {config_path} does not exist"
    assert config_path.endswith(".yaml"), "Configuration file must be in YAML format"

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOG_MODEL"] = "end" if log_model else "no"

    config = AmountConfig.from_yaml(config_path)
    
    full_dataset = AmountDataset.from_config(config)
    train_model(dataset=full_dataset, config=config, num_labels=config.num_bins)

if __name__ == "__main__":
    main()