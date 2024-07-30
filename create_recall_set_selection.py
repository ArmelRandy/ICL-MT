# https://github.com/sweta20/inContextMT/blob/main/create_recall_set_selection.py
import os
import glob
import sys
from tqdm import tqdm
import pickle
import argparse
import numpy as np
import math
from nltk.util import ngrams
from utils import *
import sacrebleu
from sacrebleu.metrics.helpers import extract_all_word_ngrams
from sacrebleu.tokenizers import tokenizer_13a
import re
from collections import Counter, defaultdict
from nltk.util import ngrams
from typing import List, Tuple

max_ngram_order = 4
tok_13a = tokenizer_13a.Tokenizer13a()


def my_log(num: float) -> float:
    if num == 0.0:
        return -9999999999
    return math.log(num)


def _preprocess(sent, ignore_whitespace):
    sent = sent.rstrip()
    if ignore_whitespace:
        sent = re.sub(r"\s+", "", sent)
    else:
        sent = tok_13a(sent)
    return sent


def _compute_score_from_stats(correct, total, effective_order=True):
    scores = [0.0 for x in range(max_ngram_order)]
    smooth_mteval = 1.0
    eff_order = max_ngram_order
    if not any(correct):
        return 0.0
    for n in range(1, len(scores) + 1):
        if total[n - 1] == 0:
            break
        if effective_order:
            eff_order = n
        if correct[n - 1] == 0:
            smooth_mteval *= 2
            scores[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
        else:
            scores[n - 1] = 100.0 * correct[n - 1] / total[n - 1]
    score = math.exp(sum([my_log(p) for p in scores[:eff_order]]) / eff_order)
    return score


def get_ngram_overlap_count(ref_ngrams, hyp_ngrams):
    correct = [0 for i in range(max_ngram_order)]
    total = correct[:]
    for hyp_ngram, hyp_count in hyp_ngrams.items():
        n = len(hyp_ngram) - 1
        total[n] += hyp_count
        if hyp_ngram in ref_ngrams:
            correct[n] += min(hyp_count, ref_ngrams[hyp_ngram])
    return _compute_score_from_stats(correct, total)


def get_recall_overlap_score(hyp_ngrams, ref_ngrams, beta=3.0, epsilon=1e-16):
    overlap_ngrams = ref_ngrams & hyp_ngrams
    tp = sum(overlap_ngrams.values())  # True positives.
    tpfp = sum(hyp_ngrams.values())  # True positives + False positives.
    tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

    try:
        prec = tp / tpfp  # precision
        rec = tp / tpfn  # recall
        factor = beta ** 2
        fscore = (1 + factor) * (prec * rec) / (factor * prec + rec)
    except ZeroDivisionError:
        prec = rec = fscore = epsilon
    return rec


def select_prompt_set(
    source, prompts, weight=0.1, ignore_whitespace=False, min_bleu_threshold=1
):
    ref_ngrams, ref_len = extract_all_word_ngrams(
        _preprocess(source, ignore_whitespace), 1, max_ngram_order
    )
    hyp_ngrams_list = {}
    for i, pr_src in enumerate(prompts):
        hyp_ngrams_list[i] = extract_all_word_ngrams(
            _preprocess(pr_src, ignore_whitespace), 1, max_ngram_order
        )[0]

    # print(ref_ngrams, hyp_ngrams_list[i])
    is_continue = True
    selected_prompts = []
    while is_continue:
        overlap_scores = []
        for i in hyp_ngrams_list:
            overlap_score = get_ngram_overlap_count(hyp_ngrams_list[i], ref_ngrams)
            overlap_scores.append(overlap_score)

        top_1 = np.argmax(overlap_scores)

        if overlap_scores[top_1] < min_bleu_threshold:
            break

        if top_1 not in selected_prompts:
            selected_prompts.append(top_1)

        # find intersecting ngrams
        hyp_top_1 = hyp_ngrams_list[top_1]
        intersect = ref_ngrams & hyp_top_1

        # downweight found ngrams
        for ngram, k in ref_ngrams.items():
            if ngram in hyp_top_1:
                ref_ngrams[ngram] *= weight

        # set ngrams of top_1 to 0
        for ngram, k in hyp_ngrams_list[top_1].items():
            hyp_ngrams_list[top_1][ngram] = 0.0

    return selected_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--input-prompt-file", type=str, required=True)
    parser.add_argument("--input-source-file", type=str, required=True)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--output-prompt-file", type=str, required=True)
    parser.add_argument("--ignore-whitespace", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--min-bleu-threshold", type=float, default=1.0)

    args = parser.parse_args()

    eval_samples = read_file(f"{args.input_source_file}")

    with open(args.input_prompt_file, "rb") as f:
        pool_prompts = pickle.load(f)

    prompts = {}
    number_of_prompts = []
    for i, source in tqdm(enumerate(eval_samples)):
        prompt_src = [pr.data["src"] for pr in pool_prompts[i]]
        selected_indices = select_prompt_set(
            source,
            prompt_src,
            weight=args.weight,
            ignore_whitespace=args.ignore_whitespace,
            min_bleu_threshold=args.min_bleu_threshold,
        )
        prompts[i] = [pool_prompts[i][j] for j in selected_indices]
        number_of_prompts.append(len(selected_indices))

    print("Maximum Number of prompts", max(number_of_prompts))

    with open(args.output_prompt_file, "wb") as f:
        pickle.dump(prompts, f)


if __name__ == "__main__":
    main()
