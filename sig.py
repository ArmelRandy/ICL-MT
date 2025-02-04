import os
import json
import numpy as np
from comet import load_from_checkpoint, download_model
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from scipy import stats

bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

ds_dict = {}

MP = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "swh": "Swahili",
    "wol": "Wolof",
    "hau": "Hausa",
    "jav": "Javanese",
    "som": "Somali",
    "tel": "Telugu",
    "urd": "Urdu",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
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

if __name__ == "__main__":
    directions = [f"Eng_to_{col.capitalize()}" for col in MP]
    print(f"directions: {directions}")
    names = ["llama-3-8b", "bloom", "mistral-v0.1", "llama-2-7b", "gemma"]
    retrievers = ["Random", "SONAR", "Cohere", "E5", "Laser2", "LaBSE"]
    ngrams = ["BLEU", "bm25"]
    test = ["RBM25", "RoBERTa"]
    
    methods = retrievers + ngrams + test
    idx = 0
    name = names[idx]
    alternative = "greater" # "two-sided"
    dataset_name_or_path = "flores"
    header = os.path.join(os.path.dirname(__file__), "generations")
    for direction in directions:
        dico = {}
        for m in methods:
            if m in retrievers:
                p = os.path.join(header, f"{name}/{m}/{direction}")
            elif m in ngrams:
                p = os.path.join(header, f"/{name}/ngram/{m}/{direction}")
            elif m in test:
                p = os.path.join(header, f"{name}/test/{m}/{direction}")
            else:
                raise ValueError(f"Unsupported retrieval method: {m}")
            if not os.path.exists(p):
                continue
            
            if not os.path.exists(p):
                continue
            if m not in dico:
                dico[m] = {}
            
            for filename in os.listdir(p):
                #print(f"filename: {filename}")
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

                left, right = direction.split("_")[0].lower(), direction.split("_")[-1].lower()
                src_lang, tgt_lang = MP[left], MP[right]

                if left not in ds_dict or right not in ds_dict:
                    if dataset_name_or_path == "flores":
                        print("Using flores ...")
                        ds_src = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[src_lang])
                        ds_tgt = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[tgt_lang])
                    elif dataset_name_or_path == "tico":
                        print("Using tico ...")
                        from tico import get_datasets

                        assert (
                            src == "English"
                        ), "This dataset only supports translation from English."
                        ds_src, ds_tgt = get_datasets(tgt_lang)
                    elif dataset_name_or_path == "ood":
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
                print(f'{src} to {tgt} in {k}-shot with seed = {seed}\nCOMET: {round(np.mean(comet_scores), 2)} - Language-Aware COMET: {round(np.mean(lacomet_scores), 2)}')
        
        # Statistical signifance between SONAR and Random (seed in [122, 42, 13]) in k-shot (k in [1, 2, 5, 10, 20])
        rng = np.random.default_rng(122)
        n_bootstrap = 300
        sample_test_set_size = 500
        threshold = 0.05
        test_set_size = 1012 if dataset_name_or_path == "flores" else 2100
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
                        bootstrap_comet_scores_1, bootstrap_comet_scores_2, alternative=alternative # greater
                    )
                    lacomet_output = stats.ttest_rel(
                        bootstrap_lacomet_scores_1, bootstrap_lacomet_scores_2, alternative=alternative
                    )
                    comet_pvalue = comet_output.pvalue
                    lacomet_pvalue = lacomet_output.pvalue
                    print(f"===\nModel = {name}, {method_of_interest} vs Random for K = {k}, direction = {direction}\nSeed = {seed}\n===\nCOMET = {comet_output}\nLanguage-Aware COMET = {lacomet_output}\n===")
                    if lacomet_pvalue >= threshold:
                        print("IMPOSSIBLE TO KNOW")
    
    print("END")