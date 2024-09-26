import os
import json
import numpy as np
from comet import load_from_checkpoint, download_model
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from scipy import stats

bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

ds_dict = {
    "eng": load_dataset("facebook/flores", "eng_Latn"),
    "deu": load_dataset("facebook/flores", "deu_Latn"),
    "fra": load_dataset("facebook/flores", "fra_Latn"),
    "swh": load_dataset("facebook/flores", "swh_Latn"),
    "wol": load_dataset("facebook/flores", "wol_Latn"),
    "hau": load_dataset("facebook/flores", "hau_Latn"),
    "jav": load_dataset("facebook/flores", "jav_Latn"),
    "som": load_dataset("facebook/flores", "som_Latn"),
    "tel": load_dataset("facebook/flores", "tel_Telu"),
    "urd": load_dataset("facebook/flores", "urd_Arab"),
    "xho": load_dataset("facebook/flores", "xho_Latn"),
    "yor": load_dataset("facebook/flores", "yor_Latn"),
    "zul": load_dataset("facebook/flores", "zul_Latn"),
}

MAPPING_LANG_TO_KEY = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Swahili": "swh_Latn",
    "Wolof": "wol_Latn",
    "Hausa": "hau_Latn",
    "Javanese": "jav_Latn",
    "Somali": "som_Latn",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab",
    "Yoruba": "yor_Latn",
    "Xhosa": "xho_Latn",
    "Zulu": "zul_Latn",
}

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

import fasttext
from huggingface_hub import hf_hub_download

identifier_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin"
)
identifier = fasttext.load_model(identifier_path)


def selection():
    for name in ["bloom", "mistral-v0.1", "llama-2-7b"]:
        data_dir = f"./generations/prompt/{name}/Random"
        for direction in ["Eng_to_Fra", "Eng_to_Deu", "Eng_to_Swh", "Eng_to_Wol"]:
            # for direction in [f"Eng_to_{col}" for col in ["Hau", "Jav", "Som", "Tel", "Urd", "Xho", "Yor", "Zul"]]:
            sources = [
                example["sentence"]
                for example in ds_dict[direction.split("_")[0].lower()]["devtest"]
            ]
            targets = [
                example["sentence"]
                for example in ds_dict[direction.split("_")[-1].lower()]["devtest"]
            ]
            if not os.path.exists(data_dir):
                continue
            directory = os.path.join(data_dir, direction)
            if not os.path.exists(directory):
                continue
            dico = {}
            for filename in os.listdir(directory):
                features = filename.split(".")[0].split("_")
                src = features[0]
                tgt = features[2]
                k = int(features[3])
                seed = int(features[6])
                template = int(features[8])
                if k in dico:
                    pass
                else:
                    dico[k] = {}
                predictions = []
                languages = [0] * len(targets)  # 1 = right language, 0 = wrong language
                with open(os.path.join(directory, filename), "r") as fin:
                    for j, line in enumerate(fin):
                        prediction = json.loads(line)["translation"]
                        predictions.append(prediction)
                        label, probability = identifier.predict(
                            prediction.split("\n")[0]
                        )
                        if len(prediction.strip()) == 0:
                            languages[j] = 0
                            continue
                        label = label[0]
                        languages[j] = MAPPING_LANG_TO_KEY[tgt] in label
                data = [
                    {"src": sources[i], "mt": predictions[i], "ref": targets[i]}
                    for i in range(len(predictions))
                ]
                model_output = model.predict(data, batch_size=8, gpus=1)
                comet_score = 100 * model_output.system_score
                b = bleu.corpus_score(predictions, [targets]).score
                c = chrf.corpus_score(predictions, [targets]).score
                lacomet_score = 100 * np.mean(
                    np.array(model_output.scores) * np.array(languages)
                )
                print(
                    f"{template}: {filename}\nBLEU = {b}\nchrF++ = {c}\nCOMET = {comet_score}\nlaCOMET = {lacomet_score}\n"
                )
                if template in dico[k]:
                    dico[k][template]["bleu"].append(b)
                    dico[k][template]["chrf++"].append(c)
                    dico[k][template]["comet"].append(comet_score)
                    dico[k][template]["lacomet"].append(lacomet_score)
                else:
                    dico[k][template] = {
                        "bleu": [b],
                        "chrf++": [c],
                        "comet": [comet_score],
                        "lacomet": [lacomet_score],
                    }
            print(dico)
            for k in dico:
                prompt = f"Name = {name}, K = {k}, direction = {direction}\n\n"
                b, c, co, la = "", "", "", ""
                for template in [7, 8, 9, 10, 11, 12]:
                    if template not in dico[k]:
                        print(f"{template} is not a key.")
                        continue
                    co += f"{template} comet: {round(np.mean(dico[k][template]['comet']), 1)} "
                    b += f"{template} bleu: {round(np.mean(dico[k][template]['bleu']), 1)} "
                    c += f"{template} chrf++: {round(np.mean(dico[k][template]['chrf++']), 1)} "
                    la += f"{template} lacomet: {round(np.mean(dico[k][template]['lacomet']), 1)} "
                prompt += f"{co}\n{b}\n{c}\n{la}\n\n"
                print(prompt)

"""
if __name__ == "__main__":
    selection()
"""

# """
if __name__ == "__main__":
    directions = [f"Eng_to_{col.capitalize()}" for col in ds_dict]
    print(f"directions: {directions}")
    names = ["bloom", "mistral-v0.1", "llama-2-7b", "gemma"]
    retrievers = ["Random", "SONAR", "Cohere", "E5", "Laser2", "LaBSE"]
    ngrams = ["BLEU", "bm25"]
    test = ["RBM25", "RoBERTa"]
    methods = retrievers + ngrams + test
    idx = 0
    name = names[idx]
    alternative = "two-sided"  # "greater"
    for direction in directions:
        dico = {}
        for m in methods:
            if m in retrievers:
                p = f"./generations/{name}/{m}/{direction}"
            elif m in ngrams:
                p = f"./generations/{name}/ngram/{m}/{direction}"
            elif m in test:
                p = f"./generations/{name}/test/{m}/{direction}"
            else:
                raise ValueError(f"Unsupported retrieval method: {m}")
            if not os.path.exists(p):
                continue
            if m not in dico:
                dico[m] = {}
            # print(f"L: {len(os.listdir(p))}")
            for filename in os.listdir(p):
                # print(f"filename: {filename}")
                features = filename.split(".")[0].split("_")
                src = features[0]
                tgt = features[2]
                k = int(features[3])
                seed = int(features[6])
                template = int(features[8])
                method = features[9]
                format = features[10]
                if format != "s2s":
                    continue

                if k in [0, 2, 20]:
                    continue

                if k in dico[m]:
                    pass
                else:
                    dico[m][k] = {}
                if seed in dico[m][k]:
                    pass
                else:
                    dico[m][k][seed] = {}

                sources = [
                    example["sentence"]
                    for example in ds_dict[direction.split("_")[0].lower()]["devtest"]
                ]
                targets = [
                    example["sentence"]
                    for example in ds_dict[direction.split("_")[-1].lower()]["devtest"]
                ]
                predictions = []
                languages = [0] * len(targets)  # 1 = right language, 0 = wrong language
                with open(os.path.join(p, filename), "r") as fin:
                    for j, line in enumerate(fin):
                        prediction = json.loads(line)["translation"]
                        predictions.append(prediction)
                        label, probability = identifier.predict(
                            prediction.split("\n")[0]
                        )
                        if len(prediction.strip()) == 0:
                            languages[j] = 0
                            continue
                        label = label[0]
                        languages[j] = MAPPING_LANG_TO_KEY[tgt] in label
                data = [
                    {"src": sources[i], "mt": predictions[i], "ref": targets[i]}
                    for i in range(len(predictions))
                ]
                model_output = model.predict(data, batch_size=8, gpus=1)
                comet_scores = 100 * np.array(model_output.scores)
                lacomet_scores = comet_scores * np.array(languages)
                dico[m][k][seed]["comet_scores"] = comet_scores
                dico[m][k][seed]["lacomet_scores"] = lacomet_scores
                print(
                    f"{src} to {tgt} in {k}-shot with seed = {seed}\nCOMET: {round(np.mean(comet_scores), 2)} - Language-Aware COMET: {round(np.mean(lacomet_scores), 2)}"
                )
        # Statistical signifance between SONAR and Random (seed in [122, 42, 13]) in k-shot (k in [1, 2, 5, 10, 20])
        rng = np.random.default_rng(122)
        n_bootstrap = 300
        sample_test_set_size = 500
        threshold = 0.05
        test_set_size = 1012  # len(sources)
        bootstrap_indices = [
            list(rng.choice(test_set_size, size=sample_test_set_size, replace=True))
            for _ in range(n_bootstrap)
        ]
        K = [1, 2, 5, 10, 20]
        for method_of_interest in [method for method in methods if method != "Random"]:
            if method_of_interest not in dico:
                continue
            for k in K:
                if k not in dico[method_of_interest]:
                    continue
                comet_scores_1 = dico[method_of_interest][k][122]["comet_scores"]
                lacomet_scores_1 = dico[method_of_interest][k][122]["lacomet_scores"]
                bootstrap_comet_scores_1 = []
                bootstrap_lacomet_scores_1 = []
                for indices in bootstrap_indices:
                    bootstrap_comet_scores_1.append(
                        np.mean([comet_scores_1[j] for j in indices])
                    )
                    bootstrap_lacomet_scores_1.append(
                        np.mean([lacomet_scores_1[j] for j in indices])
                    )
                for seed in [122, 42, 13]:
                    comet_scores_2 = dico["Random"][k][seed]["comet_scores"]
                    lacomet_scores_2 = dico["Random"][k][seed]["lacomet_scores"]
                    bootstrap_comet_scores_2 = []
                    bootstrap_lacomet_scores_2 = []
                    for indices in bootstrap_indices:
                        bootstrap_comet_scores_2.append(
                            np.mean([comet_scores_2[j] for j in indices])
                        )
                        bootstrap_lacomet_scores_2.append(
                            np.mean([lacomet_scores_2[j] for j in indices])
                        )
                    comet_output = stats.ttest_rel(
                        bootstrap_comet_scores_1,
                        bootstrap_comet_scores_2,
                        alternative=alternative,  # greater
                    )
                    lacomet_output = stats.ttest_rel(
                        bootstrap_lacomet_scores_1,
                        bootstrap_lacomet_scores_2,
                        alternative=alternative,
                    )
                    comet_pvalue = comet_output.pvalue
                    lacomet_pvalue = lacomet_output.pvalue
                    print(
                        f"===\nModel = {name}, {method_of_interest} vs Random for K = {k}, direction = {direction}\nSeed = {seed}\n===\nCOMET = {comet_output}\nLanguage-Aware COMET = {lacomet_output}\n==="
                    )
                    if lacomet_pvalue >= threshold:
                        print("IMPOSSIBLE TO KNOW")
    print("END")
# """
