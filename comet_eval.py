from datasets import load_dataset
from comet import load_from_checkpoint, download_model
from utils import MAPPING_LANG_TO_KEY
import argparse
import json
import os
import numpy as np

ds_dict = {
    "eng": load_dataset("facebook/flores", "eng_Latn"),
    "deu": load_dataset("facebook/flores", "deu_Latn"),
    "fra": load_dataset("facebook/flores", "fra_Latn"),
    "swh": load_dataset("facebook/flores", "swh_Latn"),
    "wol": load_dataset("facebook/flores", "wol_Latn"),
}

MP = {
    "eng": "English",
    "fra": "French",
    "wol": "Wolof",
    "swh": "Swahili",
    "deu": "German",
}

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
        default="home/azebazed/experiments/ICL-MT/generations",
        help="Path to the folder where the generation are stored with the format (`direction`/`file.jsonl`)",
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
    os.makedirs(args.output_dir, exist_ok=True)
    batch_size = args.batch_size
    data_dir = args.data_dir
    model_name_or_path = args.model_name_or_path

    model_path = download_model(model_name_or_path)
    model = load_from_checkpoint(model_path)

    language_aware = False
    if args.language_identifier_name_or_path:
        import fasttext
        from huggingface_hub import hf_hub_download

        identifier_path = hf_hub_download(
            repo_id=args.language_identifier_name_or_path, filename="model.bin"
        )
        identifier = fasttext.load_model(identifier_path)
        language_aware = True

    for direction in [
        "Eng_to_Fra",
        "Eng_to_Deu",
        "Eng_to_Swh",
        "Eng_to_Wol",
        "Fra_to_Eng",
        "Deu_to_Eng",
        "Swh_to_Eng",
        "Wol_to_Eng",
    ]:
        print(f"{direction}")
        if language_aware:
            if os.path.exists(
                os.path.join(args.output_dir, f"{direction}_scores.json")
            ):
                continue
        else:
            if os.path.exists(
                os.path.join(args.output_dir, f"{direction}_scores.json")
            ):
                continue
        d_comet = {}
        sources = [
            example["sentence"]
            for example in ds_dict[direction.split("_")[0].lower()]["devtest"]
        ]
        targets = [
            example["sentence"]
            for example in ds_dict[direction.split("_")[-1].lower()]["devtest"]
        ]
        for method in METHODS:
            if (
                not os.path.exists(f"{data_dir}/{method}/{direction}")
                or len(os.listdir(f"{data_dir}/{method}/{direction}")) == 0
            ):
                continue
            d_comet[method] = {}
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
                if method != "Random":
                    d_comet[method][strategy][k] = score
                else:
                    if k not in d_comet[method][strategy]:
                        d_comet[method][strategy][k] = {seed: score}
                    else:
                        d_comet[method][strategy][k][seed] = score

        print(d_comet)
        if language_aware:
            with open(
                os.path.join(args.output_dir, f"{direction}_scores.json"), "w"
            ) as fout:
                json.dump(d_comet, fout)
        else:
            with open(
                os.path.join(args.output_dir, f"{direction}_scores.json"), "w"
            ) as fout:
                json.dump(d_comet, fout)


if __name__ == "__main__":
    args = parse_args()
    main(args)
