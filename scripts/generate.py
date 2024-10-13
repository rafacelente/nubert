import argparse
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model on NuDataset")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    return parser.parse_args()

def generate_text(model, tokenizer, prompt=None, max_length=2048):
    if prompt is None:
        prompt = tokenizer.special_tokens_map["bos_token"]
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    args = parse_args()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    text = generate_text(model, tokenizer)
    print(text)    

if __name__ == "__main__":
    main()