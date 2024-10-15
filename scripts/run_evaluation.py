from typing import Tuple
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

from nugpt.datasets import AmountDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Amount Prediction Model on Unseen Data")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the trained model")
    parser.add_argument("--tokenizer_path", type=str, required=False, help="Path to the tokenizer")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the unseen data CSV file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--agency_name", type=str, required=False, help="Name of the agency to run the evaluation on")
    return parser.parse_args()

def load_model_and_tokenizer(
        model_path: str,
        tokenizer_path: str,
    ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def prepare_data(
        data_path: str,
        tokenizer: AutoTokenizer,
        agency_name: str,
        max_length: int = 512,
    ) -> Tuple[Dataset, pd.DataFrame]:
    dataset = AmountDataset.from_raw_data(
        root="./",
        fname=data_path,
        filter_list=[agency_name],
        vocab_dir="./",
        num_timestamp_bins=52,
        num_amount_bins=20,
        model_name=tokenizer.name_or_path,
        num_transaction_sequences=5,
        max_seq_len=max_length,
    )
    table = dataset.trans_table
    dataset = create_hf_dataset(dataset)
    return dataset, table


def create_hf_dataset(data: pd.DataFrame) -> Dataset:
    input_ids = [example["input_ids"] for example in data]
    labels = [example["label"] for example in data]
    return Dataset.from_dict({"input_ids": input_ids, "labels": labels})

def predict(model: AutoModelForSequenceClassification, dataset: Dataset) -> Tuple[list, list]:
    model.eval()
    model.to("cuda")
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Predicting"):
            input_ids = torch.Tensor(batch['input_ids']).to(torch.int64).unsqueeze(0).to("cuda")  # Add batch dimension
            outputs = model(input_ids, attention_mask=torch.ones(input_ids.shape).to(torch.int64).to("cuda"))
            predicted_class = outputs.logits.softmax(dim=-1).argmax(dim=-1).item()
            print(f"predicted: {predicted_class} | ground truth: {batch['labels']}")
            predictions.append(predicted_class)
            ground_truth.append(batch['labels'])
    
    return predictions, ground_truth

def compute_metrics(predictions: list, ground_truth: list) -> Tuple[float, float, list]:
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted')
    conf_matrix = confusion_matrix(ground_truth, predictions)
    return accuracy, f1, conf_matrix

def plot_confusion_matrix(conf_matrix, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

def plot_time_series(
        df: pd.DataFrame,
        output_dir: str,
    ):
    agency_name = df['AgencyName'].iloc[0]
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
    plt.savefig(f"{output_dir}/time_series_agency_{agency_name}.png")
    plt.close()

def main():
    args = parse_args()
    
    args.model_path = "/notebooks/nuvank/nubert-predictor/"
    args.tokenizer_path = "/notebooks/nuvank/output"
    args.data_path = "/notebooks/nubank/evaluation_dataset_2"
    args.output_dir = "/notebooks/nuvank/images/"
    args.agency_name = "OKLAHOMA STATE UNIVERSITY"
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    dataset, table = prepare_data(
        args.data_path,
        tokenizer,
        agency_name=args.agency_name,
    )
    
    predictions, ground_truth = predict(model, dataset)
    
    table['predictions'] = predictions
    table['ground_truth'] = ground_truth

    table.to_csv(f"{args.output_dir}/predictions.csv", index=False)

    accuracy, f1, conf_matrix = compute_metrics(predictions, ground_truth)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plot_confusion_matrix(conf_matrix, args.output_dir)
    plot_time_series(dataset, predictions, args.agency_id, args.output_dir)
    
    # Save metrics to a file
    with open(f"{args.output_dir}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()