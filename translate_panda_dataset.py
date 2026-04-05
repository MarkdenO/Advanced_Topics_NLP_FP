"""
translate_panda_perturbed.py
----------------------------
Translates only the `perturbed` column of the facebook/panda dataset
from English to Russian using Helsinki-NLP/opus-mt-en-ru (local, batched).

Intended to run on a GPU cluster (single node, one or more GPUs).
The translated dataset is saved in HuggingFace Arrow format and can be
loaded directly into the debiasing supervised finetuning notebook.

Requirements:
    pip install datasets transformers sentencepiece torch accelerate
"""

import os
import torch
import argparse
import tqdm
from datasets import load_dataset, DatasetDict
from transformers import MarianMTModel, MarianTokenizer

def create_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", "--model", type=str, default="Helsinki-NLP/opus-mt-en-ru")
    arg_parser.add_argument("-b", "--batch_size", type=int, default=128)
    arg_parser.add_argument("-n", "--number", type=int, default=50000, help="Number of examples to process")
    arg_parser.add_argument("-s", "--source_len", type=int, default=512)
    arg_parser.add_argument("-o", "--output_dir", type=str, default="./data")
    return arg_parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"  {torch.cuda.device_count()} GPUs available — using DataParallel")
    else:
        device = torch.device("cpu")
        print("No GPU found, falling back to CPU (will be slow)")
    return device


def load_model(device, model_name):
    print(f"Loading model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    return tokenizer, model


def translate_batch(texts: list[str], tokenizer, model, device, source_len) -> list[str]:
    """Translate a list of English strings to Russian."""
    # MarianMT expects >>ROMANCE<< prefix for some models, but opus-mt-en-ru does not.
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=source_len,
    ).to(device)

    with torch.no_grad():
        # Unwrap DataParallel for generation
        raw_model = model.module if hasattr(model, "module") else model
        generated = raw_model.generate(
            **encoded,
            num_beams=4,
            max_length=source_len,
            early_stopping=True,
        )

    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return translations


def translate_split(dataset_split, tokenizer, model, device, n, batch_size, source_len, split_name: str):
    dataset_split = dataset_split.select(range(min(n, len(dataset_split))))
    texts = dataset_split["perturbed"]
    translations = []

    print(f"\nTranslating split '{split_name}' ({len(texts)} examples) in batches of {batch_size}...")
    
    batches = range(0, len(texts), batch_size)
    for start in tqdm.tqdm(batches, desc=split_name, unit="batch"):
        batch = texts[start : start + batch_size]
        translated = translate_batch(batch, tokenizer, model, device, source_len)
        translations.extend(translated)

    return dataset_split.add_column("perturbed_ru", translations)


def main():
    args = create_arg_parser()
    device = get_device()
    tokenizer, model = load_model(device, args.model)
    splits = ["train", "validation"]

    print("\nLoading facebook/panda dataset...")
    raw_datasets = load_dataset("facebook/panda")

    translated_splits = {}
    for split in splits:
        if split not in raw_datasets:
            print(f"Split '{split}' not found, skipping.")
            continue
        translated_splits[split] = translate_split(
            raw_datasets[split], tokenizer, model, device, args.number, args.batch_size, args.source_len, split
        )

    translated = DatasetDict(translated_splits)

    print(f"\nSaving translated dataset to: {args.output_dir}")
    translated.save_to_disk(args.output_dir)

    print("\nDone! Sample check:")
    ex = translated["train"][0]
    print(ex)

if __name__ == "__main__":
    main()
