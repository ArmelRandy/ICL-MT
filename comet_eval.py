from datasets import load_dataset
from comet import load_from_checkpoint, download_model
from utils import MAPPING_LANG_TO_KEY
import argparse
import json
import os
import numpy as np

import itertools
from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

ds_dict = {}

MP = {
    "eng": "English",
    "fra": "French",
    "wol": "Wolof",
    "swh": "Swahili",
    "deu": "German",
}

MP.update(
    {
        "hau": "Hausa",
        "jav": "Javanese",
        "som": "Somali",
        "tel": "Telugu",
        "urd": "Urdu",
        "xho": "Xhosa",
        "zul": "Zulu",
        "npi": "Nepali"
    }
)

METHODS = [
    "Random",
    "Laser",
    "Laser2",
    "LaBSE",
    "Cohere",
    "SONAR",
    "E5",
    "BLOOM_one",
    "BLOOM_middle",
    "BLOOM_last",
    "BLOOM_one_avg",
    "BLOOM_middle_avg",
    "BLOOM_last_avg",
]

METHODS += ["bm25", "BLEU", "BLEU_pos", "Rouge", "Pos", "Grakel"]
METHODS += ["RBM25", "RoBERTa", "SONAR+Bm25"]

DIRECTIONS = [
    "Eng_to_Fra",
    "Eng_to_Deu",
    "Eng_to_Swh",
    "Eng_to_Wol",
    "Fra_to_Eng",
    "Deu_to_Eng",
    "Swh_to_Eng",
    "Wol_to_Eng",
]

DIRECTIONS += ["Eng_to_Hau", "Eng_to_Npi", "Eng_to_Som", "Eng_to_Urd"]
DIRECTIONS += ["Eng_to_Swh", "Eng_to_Fra", "Eng_to_Deu"]
DIRECTIONS += [
    f"{a}_to_{b}"
    for (a, b) in itertools.product(
        ["Eng", "Fra", "Deu"], ["Hau", "Jav", "Som", "Swh", "Tel", "Urd", "Xho", "Zul"]
    )
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Unbabel/wmt22-comet-da",
        help="Name or path of the evaluation model (e.g. Unbabel/wmt22-comet-da)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size of the evaluation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./generations/bloom",
        help="Path to the folder where the generation are stored with the format (`direction`/`file.jsonl`)",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        choices=["flores", "tico", "ood"],
        default="flores",
        help="Name of the dataset of interest e.g. flores, tico, ood.",
    )
    parser.add_argument(
        "--language_identifier_name_or_path",
        type=str,
        help="Name or path of the language identifier (e.g. facebook/fasttext-language-identification)",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Name or path of the output folder."
    )
    parser.add_argument(
        "--empty",
        action="store_true",
        help="whether to set the score of empty sequence to 0",
    )
    return parser.parse_args()


def main(args):
    batch_size = args.batch_size
    data_dir = args.data_dir
    model_name_or_path = args.model_name_or_path

    model_path = download_model(model_name_or_path)
    model = load_from_checkpoint(model_path)

    language_aware = False
    if args.language_identifier_name_or_path:
        import fasttext
        from huggingface_hub import hf_hub_download

        print("Loading the identifier ...")
        identifier_path = hf_hub_download(
            repo_id=args.language_identifier_name_or_path, filename="model.bin"
        )
        identifier = fasttext.load_model(identifier_path)
        language_aware = True
        print("Identifier loaded!")
    
    os.makedirs(args.output_dir, exist_ok=True)
    for direction in DIRECTIONS:
        print(f"{direction}")
        if language_aware:
            if os.path.exists(
                os.path.join(args.output_dir, f"{direction}_scores.json")
            ):
                # continue
                pass
        else:
            if os.path.exists(
                os.path.join(args.output_dir, f"{direction}_scores.json")
            ):
                # continue
                pass
        d_comet = {}
        d_bleu = {}
        d_chrf = {}
        d_raw = {}

        left, right = direction.split("_")[0].lower(), direction.split("_")[-1].lower()
        src_lang, tgt_lang = MP[left], MP[right]

        if left not in ds_dict or right not in ds_dict:
            if args.dataset_name_or_path == "flores":
                print("Using flores ...")
                ds_src = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[src_lang])
                ds_tgt = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[tgt_lang])
            elif args.dataset_name_or_path == "tico":
                print("Using tico ...")
                from tico import get_datasets

                assert (
                    src == "English"
                ), "This dataset only supports translation from English."
                ds_src, ds_tgt = get_datasets(tgt_lang)
            elif args.dataset_name_or_path == "ood":
                print("Using ood ...")
                from tico import get_datasets

                assert (
                    src == "English"
                ), "This dataset only supports translation from English."
                ds_src, ds_tgt = get_datasets(tgt_lang)
                ds_src_flores = load_dataset(
                    "facebook/flores", MAPPING_LANG_TO_KEY[src_lang]
                )
                ds_tgt_flores = load_dataset(
                    "facebook/flores", MAPPING_LANG_TO_KEY[tgt_lang]
                )
                ds_src["dev"] = ds_src_flores["dev"]
                ds_tgt["dev"] = ds_tgt_flores["dev"]

            ds_dict[left] = ds_src
            ds_dict[right] = ds_tgt

        sources = [example["sentence"] for example in ds_dict[left]["devtest"]]
        targets = [example["sentence"] for example in ds_dict[right]["devtest"]]
        for method in METHODS:
            print("dir: " + f'{data_dir}/{method}/{direction}')
            if (
                not os.path.exists(f"{data_dir}/{method}/{direction}")
                or len(os.listdir(f"{data_dir}/{method}/{direction}")) == 0
            ):
                continue
            # print("dir: " + f'{data_dir}/{method}/{direction}')
            d_comet[method] = {}
            d_bleu[method] = {}
            d_chrf[method] = {}
            d_raw[method] = {}
            for filename in os.listdir(f"{data_dir}/{method}/{direction}"):
                features = filename.split(".")[0].split("_")
                src = features[0]
                tgt = features[2]
                k = int(features[3])
                strategy = features[-1]  # format, s2s, s2t, t2t
                seed = features[6]
                if tgt != MP[direction.split("_")[-1].lower()]:
                    continue
                if src != MP[direction.split("_")[0].lower()]:
                    continue
                # COMET
                if strategy in d_comet[method]:
                    pass
                else:
                    d_comet[method][strategy] = {}
                    d_bleu[method][strategy] = {}
                    d_chrf[method][strategy] = {}
                    d_raw[method][strategy] = {}

                predictions = []
                languages = [0] * len(targets)  # 1 = right language, 0 = wrong language
                with open(
                    os.path.join(f"{data_dir}/{method}/{direction}", filename), "r"
                ) as fin:
                    for j, line in enumerate(fin):
                        prediction = json.loads(line)["translation"]
                        predictions.append(prediction)
                        if language_aware:
                            if args.empty and len(prediction.strip()) == 0:
                                print("Empty sequence")
                                languages[j] = 0
                                continue
                            label, probability = identifier.predict(
                                prediction.split("\n")[0]
                            )
                            label = label[0]
                            languages[j] = MAPPING_LANG_TO_KEY[tgt] in label
                data = [
                    {"src": sources[i], "mt": predictions[i], "ref": targets[i]}
                    for i in range(len(predictions))
                ]
                model_output = model.predict(data, batch_size=batch_size, gpus=1)
                if language_aware:
                    print(
                        f"Translating from {src.lower()} to {tgt.lower()}. There are {sum(languages)} translations in the right language."
                    )
                    score = np.mean(np.array(model_output.scores) * np.array(languages))
                else:
                    score = model_output.system_score
                b = bleu.corpus_score(predictions, [targets]).score
                c = chrf.corpus_score(predictions, [targets]).score
                raw_score = model_output.system_score
                if method != "Random":
                    d_comet[method][strategy][k] = score
                    d_bleu[method][strategy][k] = b
                    d_chrf[method][strategy][k] = c
                    d_raw[method][strategy][k] = raw_score
                else:
                    if k not in d_comet[method][strategy]:
                        d_comet[method][strategy][k] = {seed: score}
                        d_bleu[method][strategy][k] = {seed: b}
                        d_chrf[method][strategy][k] = {seed: c}
                        d_raw[method][strategy][k] = {seed: raw_score}
                    else:
                        d_comet[method][strategy][k][seed] = score
                        d_bleu[method][strategy][k][seed] = b
                        d_chrf[method][strategy][k][seed] = c
                        d_raw[method][strategy][k][seed] = raw_score
                print(
                    f"{filename}\nBLEU: {b}\nchrF++: {c}\nCOMET: {raw_score}\nlaCOMET: {score}"
                )
        print(d_comet)
        print(d_bleu)
        print(d_chrf)
        print(d_raw)
        with open(
            os.path.join(args.output_dir, f"{direction}_scores.json"), "w"
        ) as fout:
            json.dump(d_comet, fout)
        with open(
            os.path.join(args.output_dir, f"{direction}_bleu_scores.json"), "w"
        ) as fout:
            json.dump(d_bleu, fout)
        with open(
            os.path.join(args.output_dir, f"{direction}_chrf_scores.json"), "w"
        ) as fout:
            json.dump(d_chrf, fout)
        with open(
            os.path.join(args.output_dir, f"{direction}_raw_scores.json"), "w"
        ) as fout:
            json.dump(d_raw, fout)
    print("END")


if __name__ == "__main__":
    args = parse_args()
    main(args)
