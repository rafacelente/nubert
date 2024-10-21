import os
import gc
import argparse
import logging

from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset
import wandb
import torch
from sklearn.model_selection import train_test_split
from nubert.datasets import NuDataset
from nubert.config import NubertPreTrainConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train NuBERT")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    
    return parser.parse_args()

def split_dataset(dataset, val_size: float = 0.1, seed: int = 42):
    train, val = train_test_split(dataset, test_size=val_size, random_state=seed)
    
    return train, val

def create_hf_dataset(data):
    return Dataset.from_dict({"input_ids": data})

def resize_model_embeddings(model, tokenizer):
    """Resize the model's embeddings to match the tokenizer's vocabulary size."""
    model.resize_token_embeddings(len(tokenizer))
    return model

def train_model(
    dataset: NuDataset,
    config: NubertPreTrainConfig,
    ):
    model = AutoModelForMaskedLM.from_pretrained(config.model_name)
    tokenizer = dataset.tokenizer.base_tokenizer

    tokenizer.save_pretrained(config.trainer.output_dir)
    model = resize_model_embeddings(model, tokenizer)

    train_data, val_data = split_dataset(dataset.data)

    train_dataset = create_hf_dataset(train_data)
    val_dataset = create_hf_dataset(val_data)

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

    trainer.save_model()
    tokenizer.save_pretrained(config.trainer.output_dir)
    wandb.finish()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = "nugpt"
    os.environ["WANDB_LOG_MODEL"] = "end"
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config = NubertPreTrainConfig.from_yaml(args.config_path)
    full_dataset = NuDataset.from_config(
        config=config,
    )
    set_seed(config.trainer.seed)
    train_model(full_dataset, config)

if __name__ == "__main__":
    main()