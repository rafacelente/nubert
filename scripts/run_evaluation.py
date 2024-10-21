import os
from typing import Tuple, List
import argparse
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import Dataset
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from nubert.datasets import AmountDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Amount Prediction Model on Unseen Data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the unseen data CSV file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--agency_name", type=str, required=False, help="Name of the agency to run the evaluation on")
    parser.add_argument("--num_bins", type=int, default=15, help="Number of bins to discretize the amount")
    parser.add_argument("--num_transaction_sequences", type=int, default=5, help="Number of transaction sequences to consider")
    return parser.parse_args()

def load_model_and_tokenizer(
        model_path: str,
        tokenizer_path: str,
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def create_hf_dataset(data: pd.DataFrame) -> Dataset:
    input_ids = [example["input_ids"] for example in data]
    labels = [example["label"] for example in data]
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})

def prepare_data(
        data_path: str,
        tokenizer: AutoTokenizer,
        agency_name: str,
        max_length: int = 512,
        num_bins: int = 15,
        num_transaction_sequences: int = 5,
    ) -> Tuple[Dataset, pd.DataFrame]:
    assert os.path.exists(data_path), f"Data file {data_path} does not exist"
    assert data_path.endswith(".csv"), "Data file must be in CSV format"
    data_path, file_name = os.path.split(data_path)
    dataset = AmountDataset.from_raw_data(
        root=data_path,
        fname=file_name,
        filter_list=[agency_name],
        num_bins=num_bins,
        model_name=tokenizer.name_or_path,
        num_transaction_sequences=num_transaction_sequences,
        max_seq_len=max_length,
        stride=1,
    )
    table = dataset.trans_table
    dataset = create_hf_dataset(dataset)
    return dataset, table


def predict(
        model: AutoModelForSequenceClassification,
        dataset: Dataset,
        table_size: int,
        num_transaction_sequences: int,
        stride: int = 1,
    ) -> Tuple[list, list]:
    model.eval()
    model.to("cuda")
    predictions = [0] * table_size
    ground_truth = [0] * table_size

    for i in range(num_transaction_sequences - 1):
        predictions[i] = None
        ground_truth[i] = None
    if stride == 2:
        predictions[num_transaction_sequences - 1] = None
        ground_truth[num_transaction_sequences - 1] = None
        predictions[num_transaction_sequences + 1] = None
        ground_truth[num_transaction_sequences + 1] = None
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset), desc="Predicting"):
            input_ids = torch.Tensor(batch['input_ids']).to(torch.int64).unsqueeze(0).to("cuda")  # Add batch dimension
            outputs = model(input_ids, attention_mask=torch.ones(input_ids.shape).to(torch.int64).to("cuda"))
            predicted_class = outputs.logits.softmax(dim=-1).argmax(dim=-1).item()
            predictions[i] = predicted_class
            ground_truth[i] = batch['labels']
    
    return predictions, ground_truth

def compute_metrics(predictions: list, ground_truth: list) -> Tuple[float, float, list]:
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted')
    conf_matrix = confusion_matrix(ground_truth, predictions)
    return accuracy, f1, conf_matrix

def plot_confusion_matrix(conf_matrix, output_dir: str, num_transaction_sequences: int, num_bins: int):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{output_dir}/confusion_matrix-transactions-{num_transaction_sequences}-bins-{num_bins}.png")
    plt.close()

def plot_time_series(
        df: pd.DataFrame,
        output_dir: str,
        num_transaction_sequences: int,
        num_bins: int,
    ):
    agency_name = df['Agency Name'].iloc[0]
    agency_data = [(row['Timestamp'], row['ground_truth'], row['predictions']) 
                   for i, row in df.iterrows()]
    
    timestamps, ground_truth, preds = zip(*sorted(agency_data))
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, ground_truth, label='Ground Truth', marker='o')
    plt.plot(timestamps, preds, label='Predicted', marker='x')
    plt.title(f'Amount Prediction Over Time for Agency {agency_name}')
    plt.xlabel('Timestamp')
    plt.ylabel('Amount Bin')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_agency_{agency_name}-{num_transaction_sequences}-bins-{num_bins}.png")
    plt.close()

def main():
    args = parse_args()
    
    model_path = args.model_path
    data_path = args.data_path
    output_dir = args.output_dir
    agency_name = args.agency_name
    num_bins = args.num_bins
    num_transaction_sequences = args.num_transaction_sequences
    tokenizer_path = model_path

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    dataset, table = prepare_data(
        data_path=data_path,
        tokenizer=tokenizer,
        agency_name=agency_name,
        num_bins=num_bins,
        num_transaction_sequences=num_transaction_sequences,
    )

    predictions, ground_truth = predict(model, dataset, table_size=len(table), num_transaction_sequences=num_transaction_sequences)

    table['predictions'] = predictions
    table['ground_truth'] = ground_truth

    table.to_csv(f"{output_dir}/predictions-transactions-{num_transaction_sequences}-bins-{num_bins}.csv", index=False)

    accuracy, f1, conf_matrix = compute_metrics(predictions, ground_truth)
    
    print(f"transactions-{num_transaction_sequences}-bins-{num_bins}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_confusion_matrix(conf_matrix, output_dir, num_transaction_sequences, num_bins)
    plot_time_series(table, output_dir, num_transaction_sequences, num_bins)

    with open(f"{output_dir}/metrics-transactions-{num_transaction_sequences}-bins-{num_bins}.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()