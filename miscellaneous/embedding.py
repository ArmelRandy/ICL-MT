from transformers import BertModel, BertTokenizerFast, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import torch
import cohere
import os


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


languages = ["eng_Latn", "fra_Latn", "swh_Latn", "deu_Latn", "wol_Latn"]
print("Load dataset ...")
ds_dict = {lang: load_dataset("facebook/flores", lang) for lang in languages}


MODELS = [
    ("LaBSE", "sentence-transformers/LaBSE"),
    ("Cohere", "embed-multilingual-v3.0"),
    ("E5", "intfloat/multilingual-e5-large"),
    ("SONAR", "text_sonar_basic_encoder"),
    ("Laser2", ""),
    ("MiniLM", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("LEALLA", "setu4993/LEALLA-large"),
    ("BLOOM", "bigscience/bloom-7b1"),
]

data_dir = os.path.join(os.path.dirname(__file__), "..", "data/flores")

for name, model_name_or_path in MODELS:
    print(f"Name: {name}")
    if "Cohere" in model_name_or_path or "Cohere" in name:
        cohere_key = os.environ.get(
            "COHERE_API_KEY"
        )  # Get your API key from www.cohere.com
        co = cohere.Client(cohere_key)
        # Encode your documents with input type 'search_document'
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            dev_emb = co.embed(
                L_dev, input_type="search_document", model=model_name_or_path
            ).embeddings
            dev_emb = np.asarray(dev_emb)

            devtest_emb = co.embed(
                L_devtest, input_type="search_document", model=model_name_or_path
            ).embeddings
            devtest_emb = np.asarray(devtest_emb)
            dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
            devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))
    elif "LEALLA" in model_name_or_path:
        tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
        model = BertModel.from_pretrained(model_name_or_path)
        model = model.eval()
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            dev_tokenized = tokenizer(L_dev, return_tensors="pt", padding=True)
            devtest_tokenized = tokenizer(L_devtest, return_tensors="pt", padding=True)
            with torch.no_grad():
                dev_outputs = model(**dev_tokenized)
                devtest_outputs = model(**devtest_tokenized)
            dev_emb = dev_outputs.pooler_output
            devtest_emb = devtest_outputs.pooler_output

            dev_emb = dev_emb.detach().numpy()
            devtest_emb = devtest_emb.detach().numpy()
            dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
            devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))
    elif "e5" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            # Tokenize the input texts
            batch_dev = tokenizer(
                L_dev,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_devtest = tokenizer(
                L_devtest,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                dev_outputs = model(**batch_dev)
                devtest_outputs = model(**batch_devtest)
            embeddings_dev = average_pool(
                dev_outputs.last_hidden_state, batch_dev["attention_mask"]
            )
            embeddings_devtest = average_pool(
                devtest_outputs.last_hidden_state, batch_devtest["attention_mask"]
            )

            dev_emb = embeddings_dev.detach().numpy()
            devtest_emb = embeddings_devtest.detach().numpy()
            print(f"{dev_emb.shape}, {devtest_emb.shape}")
            dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
            devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))
    elif name == "SONAR":
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        t2vec_model = TextToEmbeddingModelPipeline(
            encoder=model_name_or_path, tokenizer=model_name_or_path
        )
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            dev_emb = t2vec_model.predict(L_dev, source_lang=lang)
            devtest_emb = t2vec_model.predict(L_devtest, source_lang=lang)

            dev_emb = dev_emb.detach().numpy()
            devtest_emb = devtest_emb.detach().numpy()
            dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
            devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))

    elif "BLOOM" in name:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(
            model_name_or_path, device_map={"": 0}
        )  # 30 Layers
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            # Tokenize the input texts
            batch_dev = tokenizer(
                L_dev,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_devtest = tokenizer(
                L_devtest,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            for key in batch_dev:
                batch_dev[key] = batch_dev[key].to("cuda")
                batch_devtest[key] = batch_devtest[key].to("cuda")
            # Handle it with a dataloader
            batch_size = 8
            dico_dev = {
                key: []
                for key in [
                    f"{name}_one",
                    f"{name}_middle",
                    f"{name}_last",
                    f"{name}_one_avg",
                    f"{name}_middle_avg",
                    f"{name}_last_avg",
                ]
            }
            for i in range(0, batch_dev["input_ids"].shape[0], batch_size):
                print(f"=== batch dev {1 + i//batch_size} ===")
                input_ids = batch_dev["input_ids"][i : i + batch_size, :]
                attention_mask = batch_dev["attention_mask"][i : i + batch_size, :]
                # Get the model's outputs
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                # Get the embeddings
                dico_dev[f"{name}_one"].append(outputs.hidden_states[1][:, -1, :])
                dico_dev[f"{name}_one_avg"].append(
                    average_pool(outputs.hidden_states[1], attention_mask)
                )
                dico_dev[f"{name}_middle"].append(outputs.hidden_states[15][:, -1, :])
                dico_dev[f"{name}_middle_avg"].append(
                    average_pool(outputs.hidden_states[15], attention_mask)
                )
                dico_dev[f"{name}_last"].append(outputs.hidden_states[30][:, -1, :])
                dico_dev[f"{name}_last_avg"].append(
                    average_pool(outputs.hidden_states[30], attention_mask)
                )

            for key in dico_dev:
                dico_dev[key] = torch.cat(dico_dev[key])
                dico_dev[key] = dico_dev[key].cpu().detach().numpy()
                print(f"dev set, key = {key}, shape = {dico_dev[key].shape}")

            dico_devtest = {
                key: []
                for key in [
                    f"{name}_one",
                    f"{name}_middle",
                    f"{name}_last",
                    f"{name}_one_avg",
                    f"{name}_middle_avg",
                    f"{name}_last_avg",
                ]
            }
            for i in range(0, batch_devtest["input_ids"].shape[0], batch_size):
                print(f"=== batch devtest {1 + i//batch_size} ===")
                input_ids = batch_devtest["input_ids"][i : i + batch_size, :]
                attention_mask = batch_devtest["attention_mask"][i : i + batch_size, :]
                # Get the model's outputs
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                # Get the embeddings
                dico_devtest[f"{name}_one"].append(outputs.hidden_states[1][:, -1, :])
                dico_devtest[f"{name}_one_avg"].append(
                    average_pool(outputs.hidden_states[1], attention_mask)
                )
                dico_devtest[f"{name}_middle"].append(
                    outputs.hidden_states[15][:, -1, :]
                )
                dico_devtest[f"{name}_middle_avg"].append(
                    average_pool(outputs.hidden_states[15], attention_mask)
                )
                dico_devtest[f"{name}_last"].append(outputs.hidden_states[30][:, -1, :])
                dico_devtest[f"{name}_last_avg"].append(
                    average_pool(outputs.hidden_states[30], attention_mask)
                )

            for key in dico_devtest:
                dico_devtest[key] = torch.cat(dico_devtest[key])
                dico_devtest[key] = dico_devtest[key].cpu().detach().numpy()
                print(f"devtest set, key = {key}, shape = {dico_devtest[key].shape}")

            # saving
            for key in dico_dev:
                dico_dev[key].tofile(os.path.join(output_path, f"{key}/dev.bin"))
                dico_devtest[key].tofile(
                    os.path.join(output_path, f"{key}/devtest.bin")
                )

    elif name == "Laser2":
        from laser_encoders import LaserEncoderPipeline

        # Initialize the LASER encoder pipeline
        encoder = LaserEncoderPipeline(lang=lang)
        # Encode sentences into embeddings
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            dev_emb = encoder.encode_sentences(L_dev)
            devtest_emb = encoder.encode_sentences(L_devtest)
            # If you want the output embeddings to be L2-normalized, set normalize_embeddings to True
            # normalized_embeddings = encoder.encode_sentences(["nnọọ, kedu ka ị mere"], normalize_embeddings=True)
            dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
            devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))
    else:
        model = SentenceTransformer(model_name_or_path)
        for lang in tqdm(languages):
            print(f"Language = {lang}")
            ds = ds_dict[lang]
            filename = lang.split("_")[0]
            L_dev = [example["sentence"] for example in ds["dev"]]
            L_devtest = [example["sentence"] for example in ds["devtest"]]
            output_path = os.path.join(data_dir, filename)
            os.makedirs(os.path.join(output_path, name), exist_ok=True)
            if all(
                [
                    os.path.exists(os.path.join(output_path, f"{name}/{col}.bin"))
                    for col in ["dev", "devtest"]
                ]
            ):
                print(
                    f"{os.path.join(output_path, f'{name}/dev.bin')} and {os.path.join(output_path, f'{name}/devtest.bin')} already exist!"
                )
                continue
            dev_emb = model.encode(L_dev)
            devtest_emb = model.encode(L_devtest)

        dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
        devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))
