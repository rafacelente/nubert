import os
import argparse
import logging

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

from nugpt import NuDataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model on NuDataset")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", help="Model identifier from huggingface.co/models")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--num_transactions", type=int, default=10, help="Number of transactions per sequence")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=1.5, help="Total number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--bf16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--test_model", action="store_true", help="Whether to test the model after training")
    parser.add_argument("--from_pretrained", action="store_true", help="Whether to load the model from a pretrained checkpoint")
    return parser.parse_args()

def split_dataset(dataset, test_size=0.1, val_size=0.1, seed=42):
    train_val, test = train_test_split(dataset, test_size=test_size, random_state=seed)    
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)
    
    return train, val, test

def create_hf_dataset(data):
    return Dataset.from_dict({"input_ids": data})

def resize_model_embeddings(model, tokenizer):
    """Resize the model's embeddings to match the tokenizer's vocabulary size."""
    model.resize_token_embeddings(len(tokenizer))
    return model

def generate_text(model, tokenizer, prompt=None, max_length=2048):
    if prompt is None:
        prompt = tokenizer.special_tokens_map["bos_token"]
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = "nugpt"
    os.environ["WANDB_LOG_MODEL"] = "end"

    dataset_path = args.dataset_path if args.dataset_path else "/notebooks/nubank/"
    
    set_seed(args.seed)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    full_dataset = NuDataset.from_raw_data(
        root=dataset_path,
        fname="raw_dataset",
        vocab_dir=f"{dataset_path}/data/vocab",
        num_bins=100,
        model_name=args.model_name,
        num_transaction_sequences=args.num_transactions,
        max_seq_len=args.max_length,
    )

    summary = full_dataset.get_summary(verbose=True)

    tokenizer = full_dataset.tokenizer.base_tokenizer
    # save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    model = resize_model_embeddings(model, tokenizer)

    train_data, val_data, test_data = split_dataset(full_dataset.data)

    train_dataset = create_hf_dataset(train_data)
    val_dataset = create_hf_dataset(val_data)
    test_dataset = create_hf_dataset(test_data)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"{args.model_name}-nugpt",
        save_strategy = "no",
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
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training completed. Model saved.")

    if args.test_model:
        logger.info("Testing the model...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")

if __name__ == "__main__":
    main()