"""
Translates a JSONL dataset into a target language using Google Translate
(deep-translator) and saves the result as JSONL.

Usage:
    python translate_pandas_dataset.py --lang nl --output pandas_nl.jsonl
    python translate_pandas_dataset.py --lang de --input-file data/train.jsonl --output pandas_de.jsonl
    python translate_pandas_dataset.py --lang fr --max-rows 100

Requirements:
    pip install deep-translator tqdm
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError, TooManyRequests
from tqdm import tqdm

TEXT_FIELDS = ("query", "answer", "context", "original", "perturbed", "selected_word")


def get_text_fields(example: dict) -> list[str]:
    """Return the subset of TEXT_FIELDS that actually exist in this record."""
    return [f for f in TEXT_FIELDS if f in example and isinstance(example[f], str)]


def translate_batch(
    translator: GoogleTranslator,
    texts: list[str],
    retries: int = 3,
    backoff: float = 2.0,
) -> list[str]:
    """Translate a list of strings, retrying on transient errors."""
    if not texts:
        return []

    for attempt in range(1, retries + 1):
        try:
            translated = translator.translate_batch(texts)
            if isinstance(translated, str):
                return [translated]
            return list(translated)
        except (TooManyRequests, RequestError) as exc:
            if attempt == retries:
                raise
            wait = backoff * attempt
            print(f"\n  Translation error ({exc}). Waiting {wait}s before retry {attempt}/{retries}…")
            time.sleep(wait)

    return texts


def translate_example(
    example: dict,
    translator: GoogleTranslator,
    fields: list[str],
    cache: dict[str, str],
) -> dict:
    """
    Return a copy of *example* with the specified fields translated.
    Non-text fields are kept as-is.
    """
    translated = dict(example)

    pending_texts: list[str] = []
    for field in fields:
        raw = example.get(field, "")
        if isinstance(raw, str) and raw.strip() and raw not in cache:
            pending_texts.append(raw)

    if pending_texts:
        unique_pending = list(dict.fromkeys(pending_texts))
        translated_pending = translate_batch(translator, unique_pending)
        for src, tgt in zip(unique_pending, translated_pending):
            cache[src] = tgt

    for field in fields:
        raw = example.get(field, "")
        if isinstance(raw, str) and raw.strip():
            translated[field] = cache.get(raw, raw)

    return translated


def translate_dataset(
    language: str,
    input_path: str,
    output_path: str,
    max_rows: Optional[int] = None,
    delay: float = 0.1,
) -> None:
    """
    Main routine: load local JSONL → translate → write JSONL.

    Args:
        language:    Target language code recognised by Google Translate
                     (e.g. 'nl', 'de', 'fr', 'es', 'zh-CN').
        input_path:  Source JSONL file containing records to translate.
        output_path: Destination file path (will be overwritten).
        max_rows:    If set, only translate this many rows (useful for testing).
        delay:       Seconds to sleep between API calls to stay under rate limits.
    """
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    print(f"Loading JSONL from '{source}'…")

    total = 0
    with source.open("r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                total += 1
                if max_rows and total >= max_rows:
                    break

    print(f"  {total} examples to translate → {language}\n")

    translator = GoogleTranslator(source="auto", target=language)

    sample_fields: list[str] = []

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    errors = 0
    cache: dict[str, str] = {}

    with source.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, raw_line in enumerate(
            tqdm(fin, total=total if total > 0 else None, desc="Translating", unit="row"),
            start=1,
        ):
            if max_rows and idx > max_rows:
                break
            if not raw_line.strip():
                continue

            try:
                example = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors += 1
                tqdm.write(f"  ERROR parsing JSON on row {written + errors}: {exc}")
                continue

            fields = get_text_fields(example)
            if not sample_fields and fields:
                sample_fields = fields

            try:
                translated = translate_example(example, translator, fields, cache)
                fout.write(json.dumps(translated, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                errors += 1
                tqdm.write(f"  ERROR on row {written + errors}: {exc}")
                # Write the original row so we don't lose it
                fout.write(json.dumps(dict(example), ensure_ascii=False) + "\n")

            if delay > 0 and fields:
                time.sleep(delay)

    if total > 0 and not sample_fields:
        print(
            "WARNING: None of the expected text fields "
            f"({list(TEXT_FIELDS)}) were found in the dataset. "
            "Check the column names and update TEXT_FIELDS at the top of this script."
        )

    print(f"\nDone. {written} rows written to '{out_path}' ({errors} errors).")
    print(f"Translation cache size: {len(cache)} unique strings.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a local JSONL dataset via Google Translate."
    )
    parser.add_argument(
        "--lang",
        required=True,
        help=(
            "Target language code for Google Translate "
            "(e.g. nl, de, fr, es, zh-CN, ja, ar). "
            "Full list: https://cloud.google.com/translate/docs/languages"
        ),
    )
    parser.add_argument(
        "--input-file",
        default="data/train.jsonl",
        help="Input JSONL file path (default: data/train.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL file path. "
            "Defaults to 'pandas_<lang>.jsonl' in the current directory."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of rows translated (useful for testing).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (default: 0.0). Increase if rate-limited.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or f"pandas_{args.lang}.jsonl"

    try:
        translate_dataset(
            language=args.lang,
            input_path=args.input_file,
            output_path=output,
            max_rows=args.max_rows,
            delay=args.delay,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()