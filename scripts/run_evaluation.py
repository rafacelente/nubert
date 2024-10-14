import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from nugpt.datasets import AmountDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Amount Prediction Model on Unseen Data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the unseen data CSV file")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--agency_id", type=str, required=True, help="ID of the agency to plot time series for")
    return parser.parse_args()

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def prepare_data(data_path, tokenizer, max_length=512):
    dataset = AmountDataset.from_raw_data(
        root="./",
        fname=data_path,
        vocab_dir="./",
        num_bins=100,
        model_name=tokenizer.name_or_path,
        num_transaction_sequences=5,
        max_seq_len=max_length
    )
    return dataset

def predict(model, dataset):
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for item in tqdm(dataset, desc="Predicting"):
            input_ids = item['text'].unsqueeze(0)  # Add batch dimension
            outputs = model(input_ids)
            predicted_class = outputs.logits.argmax(dim=-1).item()
            predictions.append(predicted_class)
            ground_truth.append(item['label'])
    
    return predictions, ground_truth

def compute_metrics(predictions, ground_truth):
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

def plot_time_series(dataset, predictions, agency_id, output_dir):
    agency_data = [(item['timestamp'], item['label'], pred) 
                   for item, pred in zip(dataset, predictions) 
                   if item['agency_id'] == agency_id]
    
    if not agency_data:
        print(f"No data found for agency {agency_id}")
        return
    
    timestamps, ground_truth, preds = zip(*sorted(agency_data))
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, ground_truth, label='Ground Truth', marker='o')
    plt.plot(timestamps, preds, label='Predicted', marker='x')
    plt.title(f'Amount Prediction Over Time for Agency {agency_id}')
    plt.xlabel('Timestamp')
    plt.ylabel('Amount Bin')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series_agency_{agency_id}.png")
    plt.close()

def main():
    args = parse_args()
    
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    dataset = prepare_data(args.data_path, tokenizer)
    
    predictions, ground_truth = predict(model, dataset)
    
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