from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from accelerate import Accelerator
import numpy as np
import argparse
import glob
import os
import torch
import json
from tqdm import tqdm
import re

from torch.nn import CrossEntropyLoss


def remove_emojis(text):
    # This regex pattern targets common emoji ranges in Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="facebook/flores",
        help="Name or path to the HF dataset.",
    )
    parser.add_argument("--config", type=str, help="Config for HF datasets.")
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Name of the split to consider e.g. 'train'",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="google/gemma-2b", help=""
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="sentence",
        help="Name of the column of interest.",
    )
    parser.add_argument(
        "--request_batch_size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (number of tokens)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000000,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=122, help="seed")
    parser.add_argument(
        "--output_filename", type=str, help="Name or path to the output file."
    )
    parser.add_argument(
        "--add_start_token",
        action="store_true",
        help="Whether to add a bos token during the perplexity computation.",
    )
    return parser.parse_args()


def _per_token_loss(inputs, logits):
    """
    https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    return a loss of size (batch size, sequence length)
    """
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fn = CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_logits.size(0), shift_logits.size(1))
    """
    loss = loss_fn(shift_logits.transpose(1, 2), shift_labels)
    """
    return loss


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.dataset_name_or_path, args.config, split=args.split)
    samples = [example[args.column_name] for example in dataset]
    samples = [remove_emojis(sample) for sample in samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": Accelerator().process_index},
        torch_dtype=torch.float16,
    )
    if args.output_filename:
        output_filename = args.output_filename
    else:
        output_filename = f"{args.dataset_name_or_path.split('/')[-1]}_{args.config}_{args.split}_seed_{args.seed}.jsonl"
    start = 0
    if os.path.exists(output_filename):
        with open(output_filename, "r") as fin:
            for line in fin:
                start += 1
    print("Start ...")
    device = model.device
    for i in tqdm(range(start, len(samples), args.request_batch_size)):
        prompts = samples[i : i + args.request_batch_size]
        # print("\n".join([f"{k}: {e}" for k, e in enumerate(prompts)]))
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        inputs = inputs.to(device)
        if args.add_start_token:
            bos_tokens_tensor = torch.tensor(
                [[tokenizer.bos_token_id]] * inputs["input_ids"].size(dim=0)
            ).to(device)
            inputs["input_ids"] = torch.cat(
                [bos_tokens_tensor, inputs["input_ids"]], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [
                    torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device),
                    inputs["attention_mask"],
                ],
                dim=1,
            )

        # labels = inputs["input_ids"]
        # outputs = model(**inputs, labels=labels)
        with torch.no_grad():
            outputs = model(**inputs)
        per_token_loss = _per_token_loss(inputs["input_ids"], outputs.logits)
        shift_attention_mask = inputs["attention_mask"][..., 1:].contiguous()
        per_sample_loss = per_token_loss * shift_attention_mask
        per_sample_loss = per_sample_loss.sum(-1) / shift_attention_mask.sum(-1)
        perplexity = torch.exp(per_sample_loss)
        # print(f"First sentence perplexity: {perplexity[0].item()}")
        with open(output_filename, "a") as fout:
            for j in range(len(prompts)):
                fout.write(
                    json.dumps(
                        {"sample": prompts[j], "perplexity": perplexity[j].item()}
                    )
                    + "\n"
                )
