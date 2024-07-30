from datasets import load_dataset
import os

if __name__ == "__main__":
    for lang in ["eng_Latn", "fra_Latn", "deu_Latn", "swh_Latn", "wol_Latn"]:
        print(f"Language = {lang}")
        ds = load_dataset("facebook/flores", lang)
        filename = lang.split("_")[0]
        save_dir = os.path.join(os.path.dirname(__file__), "..", f"data/{filename}")
        os.makedirs(save_dir, exist_ok=True)
        L_dev = [example["sentence"] for example in ds["dev"]]
        L_devtest = [example["sentence"] for example in ds["devtest"]]
        with open(os.path.join(save_dir, "dev.txt"), "w") as f:
            f.write("\n".join(L_dev))
        with open(os.path.join(save_dir, "devtest.txt"), "w") as f:
            f.write("\n".join(L_devtest))
        print("Done!")
