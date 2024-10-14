import os
import argparse
import logging

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

from nugpt.datasets import AmountDataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model on NuDataset")
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased", help="Model identifier from huggingface.co/models")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer", help="Path to the tokenizer")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_transactions", type=int, default=5, help="Number of transactions per sequence")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=1.5, help="Total number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--bf16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--test_model", action="store_true", help="Whether to test the model after training")
    return parser.parse_args()

def split_dataset(dataset, test_size=0.1, val_size=0.1, seed=42):
    train_val, test = train_test_split(dataset, test_size=test_size, random_state=seed)    
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)
    
    return train, val, test

def create_hf_dataset(data):
    input_ids = [example["text"] for example in data]
    labels = [example["label"] for example in data]
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})


def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = "nubert-predictor"
    os.environ["WANDB_LOG_MODEL"] = "end"

    dataset_path = args.dataset_path if args.dataset_path else "/notebooks/nubank/"
    
    set_seed(args.seed)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    args.dataset_path = "/notebooks/nubank"
    args.model_name = "/notebooks/nuvank/output"
    args.tokenizer_path = "/notebooks/nuvank/output"
    args.per_device_train_batch_size = 16
    args.per_device_eval_batch_size = 16
    args.bf16 = True
    args.num_train_epochs = 1
    args.test_model = True
    

    num_amount_bins = 100
    num_timestamp_bins = 100
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_timestamp_bins,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    full_dataset = AmountDataset.from_raw_data(
        root=dataset_path,
        fname="amount_dataset_raw_2",
        vocab_dir=f"{dataset_path}/data/vocab",
        num_amount_bins=num_amount_bins,
        num_timestamp_bins=num_timestamp_bins,
        model_name=args.model_name,
        num_transaction_sequences=args.num_transactions,
        max_seq_len=args.max_length
    )
    print(f"max label found: {max(full_dataset.labels)}")

    summary = full_dataset.get_summary(verbose=True)

    train_data, val_data, test_data = split_dataset(full_dataset)

    train_dataset = create_hf_dataset(train_data)
    val_dataset = create_hf_dataset(val_data)
    test_dataset = create_hf_dataset(test_data)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=1,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"nubert-distil-predictor",
        save_strategy = "epoch",
        load_best_model_at_end=True,
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

    if args.test_model:
        logger.info("Testing the model...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")

    logger.info("Training completed. Model saved.")

if __name__ == "__main__":
    main()