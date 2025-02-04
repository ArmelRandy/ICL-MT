from transformers import AutoTokenizer
from accelerate import Accelerator

from vllm import LLM, SamplingParams
from utils import MAPPING_LANG_TO_KEY, SUPPORTED_EMBEDDINGS
from templates import get_template
import numpy as np
import os

from sklearn.metrics import pairwise_distances
from datasets import load_dataset
from tqdm import tqdm
import argparse
import warnings
import json

from rouge_score import rouge_scorer
# from multiprocessing import Pool
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="Name or path of the model we use.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Name or path of the tokenizer of the model we use.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/flores",
        help="Path to the folders where the embedding are stored by language name.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        choices=["flores", "tico", "ood"],
        default="flores",
        help="Name of the dataset of interest e.g. flores, tico, ood."
    )
    parser.add_argument("--src", type=str, default="English", help="Source language.")
    parser.add_argument("--tgt", type=str, default="French", help="Target language.")
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of demonstrations for in-context learning.",
    )
    parser.add_argument(
        "--template_key",
        type=int,
        default=4,
        help="Name of the template we use for ICL.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Selection criterion = alpha sim(x, x_i) + (1 - alpha) sim(x, y_i)",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Whether or not to keep the most similar example the closest to the query.",
    )
    parser.add_argument(
        "--augment_pool",
        action="store_true",
        help="Whether or not to augment flores `dev` set with more examples.",
    )
    parser.add_argument(
        "--pool_name",
        type=str,
        help="Name of the embedding file of the pool instances (e.g. `pool.bin` -> `pool`).",
    )
    parser.add_argument(
        "--pool_dataset_name_or_path",
        type=str,
        help="Name or path of the pool data used to augment flores `dev` set.",
    )
    parser.add_argument(
        "--pool_size", type=int, help="Number of elements to consider in the pool."
    )
    parser.add_argument(
        "--use_euclidean",
        action="store_true",
        help="Use euclidean distance instead of cosine similarity.",
    )
    parser.add_argument("--seed", type=int, default=122)
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=2,
        help="Number of generation to perform in parallel.",
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of the generation."
    )
    parser.add_argument(
        "--top_p", type=float, help="Top_p parameter, for nucleus sampling."
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams, for beam search."
    )
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty.")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=75)
    parser.add_argument(
        "--max_samples",
        type=int,
        help="For debugging purpose, maximum number batch of sentences to translate.",
    )
    parser.add_argument(
        "--strategy",
        default="Laser",
        type=str,
        help="How to choose the example in context.",
    )
    parser.add_argument(
        "--output_path", type=str, default="./", help="path to the output folder."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="s2s",
        help="Which format for the search: src-to-tgt, tgt-to-tgt, src-to-tgt, tgt-to-src or mix.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of processes to run in case of parallel computing.",
    )
    parser.add_argument("--use_vllm", action="store_true", help="Whether to use vllm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    rng = np.random.default_rng(args.seed)
    if args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    k = args.k
    data_path = (
        args.data_path
        if args.data_path
        else os.path.join(os.path.dirname(__file__), "data", args.dataset_name_or_path)
    )
    src = args.src
    tgt = args.tgt
    if args.dataset_name_or_path == "flores":
        print("Using flores ...")
        ds_src = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[src])
        ds_tgt = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[tgt])
    elif args.dataset_name_or_path == "tico":
        print("Using tico ...")
        from tico import get_datasets
        assert src == "English", "This dataset only supports translation from English."
        ds_src, ds_tgt = get_datasets(tgt)
    elif args.dataset_name_or_path == "ood":
        print("Using ood ...")
        from tico import get_datasets
        assert src == "English", "This dataset only supports translation from English."
        ds_src, ds_tgt = get_datasets(tgt)
        ds_src_flores = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[src])
        ds_tgt_flores = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[tgt])
        ds_src["dev"] = ds_src_flores["dev"]
        ds_tgt["dev"] = ds_tgt_flores["dev"]
    
    template = get_template(args.template_key, src, tgt)
    stop_words = [tokenizer.eos_token]
    if template.prefix.strip() != "":
        stop_words.append(template.prefix.strip())
    if template.suffix.strip() != "":
        stop_words.append(template.suffix.strip())

    try:
        os.mkdir(args.output_path)
    except OSError as error:
        # print(error)
        print(f"{args.output_path} already exists.")

    output_path = args.output_path
    left = MAPPING_LANG_TO_KEY[args.src].split("_")[0].capitalize()
    right = MAPPING_LANG_TO_KEY[args.tgt].split("_")[0].capitalize()
    output_path = os.path.join(output_path, f"{left}_to_{right}")
    os.makedirs(output_path, exist_ok=True)

    strategy = args.strategy

    if args.augment_pool:
        from datasets import Dataset, concatenate_datasets

        print(
            "We are going to augment the `dev` set with more examples in order to create a bigger pool."
        )
        pool = load_dataset(args.pool_dataset_name_or_path)["train"]
        ds_src_pool = Dataset.from_dict(
            {
                "sentence": [
                    example[MAPPING_LANG_TO_KEY[src].split("_")[0][:2]]
                    for example in pool
                ]
            }
        )
        ds_tgt_pool = Dataset.from_dict(
            {
                "sentence": [
                    example[MAPPING_LANG_TO_KEY[tgt].split("_")[0][:2]]
                    for example in pool
                ]
            }
        )
        # Updating the `dev` splits
        ds_src["dev"] = concatenate_datasets(
            [
                ds_src["dev"].remove_columns(
                    [
                        col
                        for col in ds_src["dev"].column_names
                        if col not in ["sentence"]
                    ]
                ),
                ds_src_pool,
            ]
        )
        ds_tgt["dev"] = concatenate_datasets(
            [
                ds_tgt["dev"].remove_columns(
                    [
                        col
                        for col in ds_tgt["dev"].column_names
                        if col not in ["sentence"]
                    ]
                ),
                ds_tgt_pool,
            ]
        )

    if strategy in SUPPORTED_EMBEDDINGS:
        # Embedding of the sentences to translate i.e. `devtest`
        X_src_devtest = np.fromfile(
            os.path.join(
                data_path,
                f"{MAPPING_LANG_TO_KEY[src].split('_')[0]}/{strategy}/devtest.bin",
            ),
            dtype=float if "Cohere" in strategy else np.float32,
            count=-1,
        ).reshape(len(ds_src["devtest"]), -1)
        # Embedding of the sentences in the source language that will be used as demonstrations
        # for ICL i.e. `dev`
        X_src_dev = np.fromfile(
            os.path.join(
                data_path,
                f"{MAPPING_LANG_TO_KEY[src].split('_')[0]}/{strategy}/dev.bin",
            ),
            dtype=float if "Cohere" in strategy else np.float32,
            count=-1,
        ).reshape(len(ds_src["dev"]), -1)
        # Embedding of the sentences in the target language that will be used as demonstrations
        # for ICL i.e. `dev`
        if args.format in ["s2t", "t2t", "t2s"]:
            X_tgt_dev = np.fromfile(
                os.path.join(
                    data_path,
                    f"{MAPPING_LANG_TO_KEY[tgt].split('_')[0]}/{strategy}/dev.bin",
                ),
                dtype=float if "Cohere" in strategy else np.float32,
                count=-1,
            ).reshape(len(ds_tgt["dev"]), -1)
            # Embedding of the translation of our sentences of interest
            X_tgt_devtest = np.fromfile(
                os.path.join(
                    data_path,
                    f"{MAPPING_LANG_TO_KEY[tgt].split('_')[0]}/{strategy}/devtest.bin",
                ),
                dtype=float if "Cohere" in strategy else np.float32,
                count=-1,
            ).reshape(len(ds_tgt["devtest"]), -1)
        # If there is a complementary pool
        if args.augment_pool:
            # Embeddding of the sentences of the pool written in the source language
            print("Vector concatenation.")
            X_src_pool = np.fromfile(
                os.path.join(
                    data_path,
                    f"{MAPPING_LANG_TO_KEY[src].split('_')[0]}/{strategy}/{args.pool_name}.bin",
                ),
                dtype=float if "Cohere" in strategy else np.float32,
                count=-1,
            ).reshape(-1, X_src_dev.shape[-1])
            # Embedding of the sentences of the pool written in the target language
            X_tgt_pool = np.fromfile(
                os.path.join(
                    data_path,
                    f"{MAPPING_LANG_TO_KEY[tgt].split('_')[0]}/{strategy}/{args.pool_name}.bin",
                ),
                dtype=float if "Cohere" in strategy else np.float32,
                count=-1,
            ).reshape(-1, X_tgt_dev.shape[-1])
            # Concatenate the `dev` split and the `pool`
            X_src_dev = np.concatenate((X_src_dev, X_src_pool), axis=0)
            X_tgt_dev = np.concatenate((X_tgt_dev, X_tgt_pool), axis=0)
        # `D` is a similarity matrix, i.e. 1 - distance matrix
        if args.format == "s2s":
            D = 1 - pairwise_distances(X_src_devtest, X_src_dev, metric="cosine")
        elif args.format == "t2t":
            D = 1 - pairwise_distances(X_tgt_devtest, X_tgt_dev, metric="cosine")
        elif args.format == "s2t":
            D = 1 - pairwise_distances(X_src_devtest, X_tgt_dev, metric="cosine")
        elif args.format == "t2s":
            D = 1 - pairwise_distances(X_tgt_devtest, X_src_dev, metric="cosine")
        elif args.format == "mix":
            D1 = 1 - pairwise_distances(X_src_devtest, X_src_dev, metric="cosine")
            D2 = 1 - pairwise_distances(X_src_devtest, X_tgt_dev, metric="cosine")
            D = args.alpha * D1 + (1 - args.alpha) * D2
        else:
            warnings.warn(
                f"""The format {args.format} is not one of ("s2s", "s2t", "t2s", "t2t").\
                We are going to use the `s2s` format. Ignore this warning if you don't mind."""
            )
            D = 1 - pairwise_distances(X_src_devtest, X_src_dev, metric="cosine")
    elif strategy == "RoBERTa":
        from transformers import AutoModel
        import torch

        roberta = AutoModel.from_pretrained("FacebookAI/roberta-large")
        tok = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
        dev = [ex["sentence"] for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [ex["sentence"] for ex in ds_src["devtest"]]

        L_dev = []
        L_devtest = []
        inputs_dev = tok(dev, padding=True, truncation=True, return_tensors="pt")
        inputs_devtest = tok(
            devtest, padding=True, truncation=True, return_tensors="pt"
        )
        print("Start embedding...")
        with torch.no_grad():
            dev_outputs = roberta(**inputs_dev)
            devtest_outputs = roberta(**inputs_devtest)
        X_src_dev = dev_outputs.pooler_output.detach().numpy()
        X_src_devtest = devtest_outputs.pooler_output.detach().numpy()
        print("End embedding.")
        D = 1 - pairwise_distances(
            X_src_devtest,
            X_src_dev,
            metric="euclidean" * args.use_euclidean
            + "cosine" * (1 - args.use_euclidean),
        )
        print(f"Shape: {D.shape}")
    elif strategy == "bm25":
        from rank_bm25 import BM25Okapi

        dev = [ex["sentence"] for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [ex["sentence"] for ex in ds_src["devtest"]]

        tokenized_dev = [doc.split(" ") for doc in dev]
        bm25 = BM25Okapi(tokenized_dev)

        def f(example):
            return bm25.get_scores(example.split(" "))

        import multiprocess as mp

        p = mp.Pool(args.num_workers)

        bm25_scores = p.map(f, devtest)
        D = np.array([score for score in bm25_scores]).reshape(len(devtest), len(dev))
    elif strategy == "Rouge":
        import multiprocess as mp

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        dev = [ex["sentence"] for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        dev = [scorer._tokenizer.tokenize(ex) for ex in dev]
        devtest = [
            scorer._tokenizer.tokenize(ex["sentence"]) for ex in ds_src["devtest"]
        ]
        input = itertools.product(devtest, dev)
        p = mp.Pool(args.num_workers)

        def f(example):
            x, y = example
            return rouge_scorer._score_lcs(x, y)

        rouge_scores = p.map(f, list(input))
        D = np.array([score.fmeasure for score in rouge_scores]).reshape(
            len(devtest), len(dev)
        )
    elif strategy == "Pos":
        import multiprocess as mp
        import spacy

        nlp = spacy.load("en_core_web_sm")
        dev = [nlp(ex["sentence"]) for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [nlp(ex["sentence"]) for ex in ds_src["devtest"]]

        dev = [[token.pos_ for token in doc] for doc in dev]
        devtest = [[token.pos_ for token in doc] for doc in devtest]

        def lcs(a, b):
            N = len(a)
            M = len(b)
            assert N != 0 and M != 0
            dp = [[0] * M for _ in range(N)]
            for j in range(M):
                if a[0] == b[j]:
                    for k in range(j, M):
                        dp[0][k] = 1
                    break
            for i in range(N):
                if a[i] == b[0]:
                    for k in range(i, N):
                        dp[k][0] = 1
                    break

            for i in range(1, N):
                for j in range(1, M):
                    if a[i] == b[j]:
                        dp[i][j] = 1 + dp[i - 1][j - 1]
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[N - 1][M - 1]

        def f(example):
            x, y = example
            return lcs(x, y)

        input = itertools.product(devtest, dev)
        p = mp.Pool(args.num_workers)
        lcs_scores = p.map(f, list(input))
        D = np.array(lcs_scores).reshape(len(devtest), len(dev))
        D = D / (D.sum(axis=1).reshape(-1, 1) + D.sum(axis=0).reshape(1, -1))

    elif strategy == "Grakel":
        from grakel import Graph
        from grakel.kernels import ShortestPath
        import spacy

        nlp = spacy.load("en_core_web_sm")

        dev = [nlp(ex["sentence"]) for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [nlp(ex["sentence"]) for ex in ds_src["devtest"]]

        def build_graph(doc):
            node_labels = {token.i: token.pos_ for token in doc}
            edges = {}
            edge_labels = {}
            for token in doc:
                for child in token.children:
                    edges[(token.i, child.i)] = 1
                    edge_labels[(token.i, child.i)] = child.dep_
            G = Graph(edges, edge_labels=edge_labels, node_labels=node_labels)
            return G

        sp_kernel = ShortestPath()
        K_dev = sp_kernel.fit_transform([build_graph(doc) for doc in dev])
        D = sp_kernel.transform([build_graph(doc) for doc in devtest])

    elif strategy == "BLEU_pos":
        import multiprocess as mp
        from sacrebleu.metrics import BLEU

        bleu = BLEU(tokenize="none")

        import spacy

        nlp = spacy.load("en_core_web_sm")

        dev = [nlp(ex["sentence"]) for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [nlp(ex["sentence"]) for ex in ds_src["devtest"]]

        dev = [" ".join([token.pos_ for token in doc]) for doc in dev]
        devtest = [" ".join([token.pos_ for token in doc]) for doc in devtest]

        input = itertools.product(devtest, dev)
        p = mp.Pool(args.num_workers)

        def f(example):
            x, y = example
            return bleu.corpus_score([x], [[y]]).score

        bleu_scores = p.map(f, list(input))
        D = np.array([score for score in bleu_scores]).reshape(len(devtest), len(dev))

    elif strategy == "BLEU":
        import multiprocess as mp
        from sacrebleu.metrics import BLEU

        bleu = BLEU(tokenize="flores200")

        import spacy

        nlp = spacy.load("en_core_web_sm")

        dev = [ex["sentence"] for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [ex["sentence"] for ex in ds_src["devtest"]]

        input = itertools.product(devtest, dev)
        p = mp.Pool(args.num_workers)

        def f(example):
            x, y = example
            return bleu.corpus_score([x], [[y]]).score

        bleu_scores = p.map(f, list(input))
        D = np.array([score for score in bleu_scores]).reshape(len(devtest), len(dev))

    elif strategy == "RBM25":
        from sacremoses import MosesTokenizer
        from rank_bm25 import BM25Okapi

        mt_tok = MosesTokenizer(lang="en")
        devtest = [ex["sentence"] for ex in ds_src["devtest"]]
        dev = [ex["sentence"] for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]

        tokenized_dev = [mt_tok.tokenize(doc) for doc in dev]
        bm25 = BM25Okapi(tokenized_dev)

        def f(example):
            return bm25.get_scores(mt_tok.tokenize(example))

        import multiprocess as mp

        p = mp.Pool(8)

        bm25_scores = p.map(f, devtest)
        D_bm25 = np.array([score for score in bm25_scores]).reshape(
            len(devtest), len(dev)
        )

        from create_recall_set_selection import select_prompt_set

        weight = 0.1  # Lambda
        ignore_whitespace = False  # True
        min_bleu_threshold = 0.01  # Threshold

        def g(i):
            source = devtest[i]
            top_n_indices = np.argsort(D_bm25[i])[-100:]
            prompt_src = [dev[int(idx)] for idx in top_n_indices]
            selected_indices = select_prompt_set(
                source,
                prompt_src,
                weight=weight,
                ignore_whitespace=ignore_whitespace,
                min_bleu_threshold=min_bleu_threshold,
            )
            return selected_indices

        p = mp.Pool(8)
        R = p.map(g, [i for i in range(len(devtest))])
        assert (
            min([len(element) for element in R]) >= args.k
        ), f"The minimum number of select indices is not greater than {args.k}."
        indices = [element[: args.k] for element in R]
        print(
            f"Sanity check : Min ({min([len(element) for element in indices])}), Max({max([len(element) for element in indices])}), k ({args.k})"
        )

    elif "+" in strategy:  # "SONAR + Bm25"
        strategy = strategy.split("+")[0].strip()
        print(f"Loading {strategy} embeddings.")
        # Start by getting the `strategy` (SONAR) embeddings.
        # Embedding of the sentences to translate i.e. `devtest`
        X_src_devtest = np.fromfile(
            os.path.join(
                data_path,
                f"{MAPPING_LANG_TO_KEY[src].split('_')[0]}/{strategy}/devtest.bin",
            ),
            dtype=float if "Cohere" in strategy else np.float32,
            count=-1,
        ).reshape(len(ds_src["devtest"]), -1)
        # Embedding of the sentences in the source language that will be used as demonstrations
        # for ICL i.e. `dev`
        X_src_dev = np.fromfile(
            os.path.join(
                data_path,
                f"{MAPPING_LANG_TO_KEY[src].split('_')[0]}/{strategy}/dev.bin",
            ),
            dtype=float if "Cohere" in strategy else np.float32,
            count=-1,
        ).reshape(len(ds_src["dev"]), -1)
        # If there is a complementary pool
        if args.augment_pool:
            # Embeddding of the sentences of the pool written in the source language
            print("Vector concatenation.")
            X_src_pool = np.fromfile(
                os.path.join(
                    data_path,
                    f"{MAPPING_LANG_TO_KEY[src].split('_')[0]}/{strategy}/{args.pool_name}.bin",
                ),
                dtype=np.float32,
                count=-1,
            ).reshape(-1, X_src_dev.shape[-1])
            # Concatenate the `dev` split and the `pool`
            X_src_dev = np.concatenate((X_src_dev, X_src_pool), axis=0)
        # `D` is a similarity matrix, i.e. 1 - distance matrix
        D1 = 1 - pairwise_distances(X_src_devtest, X_src_dev, metric="cosine")
        if args.pool_size is not None:
            D1 = D1[:, : args.pool_size]

        from rank_bm25 import BM25Okapi

        dev = [ex["sentence"] for ex in ds_src["dev"]]
        if args.pool_size is not None:
            dev = dev[: args.pool_size]
        devtest = [ex["sentence"] for ex in ds_src["devtest"]]

        tokenized_dev = [doc.split(" ") for doc in dev]
        bm25 = BM25Okapi(tokenized_dev)

        def f(example):
            return bm25.get_scores(example.split(" "))

        import multiprocess as mp

        p = mp.Pool(args.num_workers)

        bm25_scores = p.map(f, devtest)
        D2 = np.array([score for score in bm25_scores]).reshape(len(devtest), len(dev))
        indices = []
        for i in range(D1.shape[0]):
            R1 = D1[i].argsort()[-k:]
            R2 = D2[i].argsort()[-k:]
            intersection = list(set(R1) & set(R2))
            # Give an advantage to those who are in both sets
            fusion = {}
            for element in intersection:
                fusion[element] = 1
            for j in range(k):
                if R2[j] in fusion:
                    fusion[R2[j]] += j
                else:
                    fusion[R2[j]] = j

            for j in range(k):
                if R1[j] in fusion:
                    fusion[R1[j]] += j
                else:
                    fusion[R1[j]] = j
            L = [(key, v) for (key, v) in fusion.items()]
            L = [(key, v, j) for j, (key, v) in enumerate(L)]
            L = sorted(L, key=lambda x: (x[1], x[2]))
            indices.append([a for (a, _, _) in L][-k:])
        strategy = f"{strategy}+Bm25"
        print(f"The strategy of interest is {strategy}.")
    elif strategy == "Random":
        pass
    else:
        raise KeyError("You provided a `strategy` that is not supported!")
    # Consider the `k` sentences which have the highest similarity with the input sequence.
    if args.pool_size is not None:
        print(
            f"Reducing the size of the pool. We consider only {args.pool_size} examples!"
        )
        ds_src["dev"] = ds_src["dev"].select([j for j in range(args.pool_size)])
        ds_tgt["dev"] = ds_tgt["dev"].select([j for j in range(args.pool_size)])

    if strategy == "Random":
        indices = [
            rng.choice(len(ds_src["dev"]), size=k, replace=False).tolist()
            for _ in range(len(ds_src["devtest"]))
        ]
    else:
        if strategy in ["RBM25"] or "+" in strategy:
            # Indices are already defined above
            pass
        else:
            if args.pool_size is not None:
                print(f"shape before : {D.shape}")
                D = D[:, : args.pool_size]
                print(f"shape after : {D.shape}")
            indices = D.argsort(axis=-1)[:, -k:]
        if args.reverse:
            print(
                "We are going to change the prompting ordering. The closest example in terms of similarity will be the furthest to the query in the prompt fed to the model"
            )
            indices = indices[:, ::-1]

    inputs = []
    for i in range(len(ds_src["devtest"])):
        a = ds_src["dev"].select(indices[i])
        b = ds_tgt["dev"].select(indices[i])
        # Do not modify the ordering of the demonstrations, the most similar should be closest to the new query
        demonstrations = [(a[j]["sentence"], b[j]["sentence"]) for j in range(k)]
        prompt = template.get_prompt(demonstrations, ds_src["devtest"][i]["sentence"])
        inputs.append(prompt)

    request_batch_size = args.request_batch_size
    if request_batch_size % accelerator.num_processes != 0:
        request_batch_size = (
            1 + args.request_batch_size // accelerator.num_processes
        ) * accelerator.num_processes
        warnings.warn(
            f"Your request batch size ({args.request_batch_size}) is can not be divided by the number of processes. We'll pad it to {request_batch_size}."
        )
    output_filename = f"{src}_to_{tgt}_{k}_shot_seed_{args.seed}_template_{args.template_key}_{strategy}_{args.format}.jsonl"
    # Resume where we stopped the last time
    start = 0
    if os.path.exists(os.path.join(output_path, output_filename)):
        with open(os.path.join(output_path, output_filename), "r") as fin:
            for line in fin:
                start += 1
    if args.use_vllm:
        if ("AWQ" in args.model_name_or_path) or ("GPTQ" in args.model_name_or_path):
            model = LLM(
                model=args.model_name_or_path,
                quantization="AWQ" if "AWQ" in args.model_name_or_path else "GPTQ",
                dtype="half",
                max_model_len=2048
                if any([name in args.model_name_or_path for name in ["bloom", "OLMo"]])
                else 4096,
                enforce_eager=True,
                tensor_parallel_size=accelerator.num_processes,
            )
        else:
            model = LLM(
                model=args.model_name_or_path,
                dtype="half",
                max_model_len=2048
                if any([name in args.model_name_or_path for name in ["bloom", "OLMo"]])
                # else 8192,
                else 4096,
                enforce_eager=True,
                tensor_parallel_size=accelerator.num_processes,
            )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            best_of=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            use_beam_search=not args.do_sample,
            skip_special_tokens=True,
            stop=["\n###"]
            # stop_token_ids=[tokenizer.eos_token_id],
        )
    else:
        from transformers import AutoModelForCausalLM
        from inference import hf_generate

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            #load_in_8bit=True,
            device_map={"": accelerator.process_index},
            trust_remote_code=True,
        )

    for i in tqdm(range(start, len(inputs), request_batch_size)):
        if args.max_samples is not None and i >= args.max_samples:
            break
        prompts = inputs[i : i + request_batch_size]
        number_of_elements = len(prompts)
        if number_of_elements % accelerator.num_processes != 0:
            padded_length = accelerator.num_processes * (
                1 + number_of_elements // accelerator.num_processes
            )
            prompts = prompts + [prompts[-1]] * (padded_length - number_of_elements)

        if args.use_vllm:
            response = model.generate(prompts, sampling_params)
        else:
            response = hf_generate(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_words=[],
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=1,
                do_sample=args.do_sample,
                forced_bos_token_id=None,
                #batch_size=min(4, args.request_batch_size),
                #verbose=True
            )
        outputs = []
        # I/O sanity check
        assert len(response) == len(
            prompts
        ), f"The size of the input ({len(prompts)}) does not match the size of the output ({len(response)})"
        response = response[:number_of_elements]
        for j, r in enumerate(response):
            # post process the answer to get the translation of the last sentence
            if args.use_vllm:
                output = r.outputs[0].text
            else:
                output = r["answer"]
                assert output.startswith(
                    prompts[j]
                ), f"This output\n\n{output}\n\nDoes not start with the prompt\n\n{prompts[j]}\n"
                output = output[len(prompts[j]) :]
            output = output.lstrip()
            print(f"{i+j+1}-> {output}\n")
            if k == 0:
                output = output.split("\n")[0]
            end = output.find(template.suffix)
            if end == -1:
                pass
            else:
                output = output[:end]
            min_index = None
            for stop_word in stop_words:
                idx = output.find(stop_word)
                if idx != -1:
                    if min_index is None:
                        min_index = idx
                    else:
                        min_index = min(idx, min_index)
            if min_index is not None:
                output = output[0:min_index]
            output = output.strip()
            outputs.append(output)
        # Save the predictions to an output file
        if accelerator.is_main_process:
            with open(os.path.join(output_path, output_filename), "a") as fout:
                for output in outputs:
                    fout.write(json.dumps({"translation": output.strip()}) + "\n")