import json
import argparse
import os
import unicodedata
import re
# import spacy

from collections import defaultdict
from pathlib import Path

from evaluation_config import comparison_prompt, ALIASES

# ----------------- Do this if running the script inside exp2/------------------ #
import sys

# Adding parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
# ---------------------------------------------------------------------------------- #

# Spacy model to be used to identify nouns
# nlp = spacy.load("en_core_web_sm")

# Negative words for is_negated_match()
NEG_CUES = {"no", "not", "without", "isnt", "isn't", "arent", "aren't", "none", "missing", "absent"}


# Normalization
def normalize_text(s: str) -> str:
    if not s:
        return ""
    # Unicode normalize + lowercase
    s = unicodedata.normalize("NFKC", s).lower()

    # Convert hyphens/underscores to spaces
    s = re.sub(r"[-_]+", " ", s)

    # Replace punctuation with spaces (keeps word boundaries sane)
    s = re.sub(r"[^\w\s]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_into_item_spans(text: str):
    """Split model output into candidate item descriptions."""
    t = text.replace("**", " ").replace("\r", "\n")

    # Step 1: split by numbered list markers if present; otherwise treat as one chunk.
    if re.search(r"(?:^|\n)\s*\d+\.\s+", t):
        chunks = re.split(r"(?:^|\n)\s*\d+\.\s+", t)
    else:
        chunks = [t]

    chunks = [c.strip() for c in chunks if c.strip()]

    # Step 2: further split each chunk by newlines and sentence endings.
    # This splits on:
    #  - one or more newlines, OR
    #  - whitespace following ., !, or ? (end-of-sentence punctuation),
    #    so the punctuation stays attached to the sentence.
    splitter = re.compile(r"(?:\n+|(?<=[.!?,])\s+)")

    parts: list[str] = []
    for chunk in chunks:
        for piece in splitter.split(chunk):
            piece = piece.strip()
            if piece:
                parts.append(piece)

    return parts


def clean_np(np_text: str) -> str:
    s = np_text.strip().lower()
    s = re.sub(r"^\s*(a|an|the)\s+", "", s)     # drop determiners
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!-")
    return s


# def contains_noun(text):
#     doc = nlp(text)
#     return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)


# Gets alternative candidates of object labels
def get_candidates(label: str) -> list[str]:
    """
    Return normalized candidate phrases to search for a label:
    - the label itself
    - any configured aliases
    """
    label_n = normalize_text(label)
    cands = [label_n]
    for alt in ALIASES.get(label_n, []):
        cands.append(normalize_text(alt))
    # de-dupe while preserving order
    seen = set()
    out = []
    for c in cands:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


# Searching for phrase in the generation
def phrase_in_generation(phrase: str, text: str) -> bool:
    """
    Checks if phrase appears as whole words in text.
    Allows flexible whitespace between words (e.g., "digital   clock").
    Avoids substring collisions (e.g., "can" in "candy").
    """
    if not phrase or not text:
        return False
    # Convert spaces in phrase to \s+ so multiword phrases match robustly
    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


# Negation guard
def is_negated_match(phrase: str, text: str, window_words: int = 4) -> bool:
    """
    If phrase occurs near a negation cue, treat it as NOT present.
    Heuristic: checks up to N words before the match start.
    """
    if not phrase or not text:
        return False

    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    m = re.search(pattern, text)
    if not m:
        return False

    # Look at a window of words before the match
    prefix = text[:m.start()].strip()
    if not prefix:
        return False
    words = prefix.split()
    context = words[-window_words:] if len(words) > window_words else words
    return any(w in NEG_CUES for w in context)


# Checking if the object label in present in any form in the generation
def label_present(label: str, text: str, use_negation_guard: bool = True) -> bool:
    """
    True if any candidate (label or alias) is found in text,
    and not negated (optional).
    """
    for cand in get_candidates(label):
        if phrase_in_generation(cand, text):
            if use_negation_guard and is_negated_match(cand, text):
                continue
            return True
    return False


def calculate_accuracy(model_name, out_path):
    generations_file = out_path / f"{model_name}_generations.json"
    results_file = out_path / f"{model_name}_results.json"

    if not os.path.exists(generations_file):
        print(f"Error: {generations_file} not found.")
        return

    with open(generations_file, "r") as f:
        data = json.load(f)

    results = {}

    occlusion_types = [
        "no_occlusion",
        "low_occlusion",
        "medium_occlusion",
        "high_occlusion"
    ]

    total = len(data)
    print(f"\nTotal number of scenes: {total}\n")

    individual_results = {}
    object_types = set(["front_obj", "back_obj", "clutter"])

    for key, entry in data.items():
        print(f"\nScene id: {key}")
        individual_results[key] = {}

        # Parse the single generation into objects
        generation_raw = entry.get("generation", "")
        spans = split_into_item_spans(generation_raw)
        cleaned_spans = [clean_np(span) for span in spans if not re.search(r"\btable\b", span)]
        generation_objects = [span for span in cleaned_spans if span]

        print(f"Generation objects: {generation_objects}")
        individual_results[key]["generation"] = generation_raw,
        individual_results[key]["generation_objects"] = generation_objects

        for oc_type in occlusion_types:
            objects = entry.get(f"{oc_type}_objects", [])
            print(f"\n  [{oc_type}] Objects: {objects}")

            matched_objects = set()
            matched_gen_indices = set()

            for obj in objects:
                candidates = get_candidates(obj)
                obj_matched = False
                for gen_idx, gen_item in enumerate(generation_objects):
                    gen_item_norm = normalize_text(gen_item)
                    for cand in candidates:
                        if phrase_in_generation(cand, gen_item_norm):
                            if not is_negated_match(cand, gen_item_norm):
                                obj_matched = True
                                matched_gen_indices.add(gen_idx)
                                break
                    if obj_matched:
                        break
                if obj_matched:
                    matched_objects.add(obj)

            true_positives  = list(matched_objects)
            print(f"\n  True Positives  ({len(true_positives)}): {true_positives}")


            individual_results[key][oc_type] = {
                "objects": objects,
                "num_objects": len(objects), 
                "true_positives":  true_positives,
                "tp": len(true_positives),
            }

    # Aggregate totals per occlusion type
    overall_results = {}

    for oc_type in occlusion_types:
        total_tp = sum(individual_results[key][oc_type]["tp"] for key in individual_results)
        total_num_objects = sum(individual_results[key][oc_type]["num_objects"] for key in individual_results)

        pct_correct = (total_tp / total_num_objects * 100) if total_num_objects != 0 else 0.00

        overall_results[oc_type] = {
            "total_tp": total_tp,
            "total_num_objects": total_num_objects,
            "pct_correct": round(pct_correct, 4)
        }

        print(f"\n{oc_type}:")
        print(f"  TP={total_tp}, Total num objects={total_num_objects}")
        print(f"  Percentage of correct predctions={overall_results[oc_type]['pct_correct']}")

    # Update and save results
    results["overall"] = overall_results
    results["individual"] = individual_results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {results_file}")
    return


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy for a given model's generations.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["Qwen2.5-VL", "InternVL3.5", "Molmo2", "Molmo", "FT-Qwen2.5-VL", 
                  "Gemma3n", "Gemma3", "Llama-3.2", "FT-Llama-3.2", "FT-Gemma3", 
                  "Qwen3-VL", "FT-Qwen3-VL", "COCO-FT-Llama-3.2", "COCO-FT-Gemma3", "COCO-FT-Qwen2.5-VL", 
                "COCO-FT-Qwen3-VL"], # "FT-Gemma3n"
        help="Name of the VLM."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./runs/",
        help=(
            "Path to output directory where the model outputs should be stored."
        )
    )
    args = parser.parse_args()

    calculate_accuracy(args.model_name, Path(args.out_path))

if __name__ == "__main__":
    main()