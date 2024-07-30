from datasets import load_dataset, DatasetDict, Dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=str, help="Flores200 code of the source language (e.g. eng_Latn)"
    )
    parser.add_argument(
        "--tgt", type=str, help="FLores200 code of the target language (e.g. swh_Latn)"
    )
    parser.add_argument("--M", type=int, help="Size of the final dataset.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the newly created dataset on the hub.",
    )
    return parser.parse_args()


def kde4(verbose=False):
    dataset = load_dataset("kde4", lang1="en", lang2="fr")
    special_characters = [
        '"',
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "-",
        "+",
        "?",
        "_",
        "=",
        ",",
        "<",
        ">",
        "/",
        "\\",
    ]
    count = 0
    indices = []
    for i, example in enumerate(dataset["train"]):
        source = example["translation"]["en"]
        target = example["translation"]["fr"]
        if any([source.startswith(char) for char in special_characters]) or any(
            [target.startswith(char) for char in special_characters]
        ):
            continue
        if len(source) >= 100 and len(target) >= 100:
            indices.append(i)
            count += 1
            if verbose:
                print(f"source: {source}\ntarget: {target}")
                print("-" * 50)
    print(f"The total number of elements is {count}, this should be equal to 20058.")
    ds = Dataset.from_dict(
        {
            "en": [dataset["train"][index]["translation"]["en"] for index in indices],
            "fr": [dataset["train"][index]["translation"]["fr"] for index in indices],
        }
    )
    if args.push_to_hub:
        ds.push_to_hub("ArmelRandy/kde4")


def main(args):
    src = args.src
    tgt = args.tgt
    M = args.M
    dataset = load_dataset("allenai/nllb", f"{src}-{tgt}", streaming=True)
    elements = []
    for i, example in enumerate(dataset["train"]):
        if i >= M:
            break
        elements.append(example)

    col_1, col_2 = src.split("_")[0][0:2], tgt.split("_")[0][0:2]

    ds = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    col_1: [element["translation"][src] for element in elements[0:M]],
                    col_2: [element["translation"][tgt] for element in elements[0:M]],
                }
            )
        }
    )

    if args.push_to_hub:
        r = int(M / 1000)
        ds.push_to_hub(f"ArmelRandy/nllb_{col_1}_{col_2}_{r}K")


if __name__ == "__main__":
    args = parse_args()
    main(args)
