import json
import os

MAPPING_LANG_TO_KEY = {}
with open(
    os.path.join(os.path.dirname(__file__), "data", "flores200.jsonl"), "r"
) as fin:
    for line in fin:
        for k, v in json.loads(line).items():
            MAPPING_LANG_TO_KEY[k] = v

SUPPORTED_EMBEDDINGS = [
    "Laser2",
    "Cohere",
    "LaBSE",
    "E5",
    "SONAR",
    "BLOOM_one",
    "BLOOM_middle",
    "BLOOM_last",
    "BLOOM_one_avg",
    "BLOOM_middle_avg",
    "BLOOM_last_avg",
]
