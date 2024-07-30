# Description

- ab.py

Derive the SONAR embeddings for [nllb_en_sw_20K](https://huggingface.co/datasets/ArmelRandy/nllb_en_sw_20K) and [nllb_en_fr_20K](https://huggingface.co/datasets/ArmelRandy/nllb_en_fr_20K).

- embedding.py

Derive the embeddings of FLORES-200 dev and devtest sets for English, French, German, Swahili and Wolof with multiple sentence embeddings. The resulting matrices are stored in `.bin` files.

- save.py

Save the FLORES-200 dev and devtest sets for English, French, German, Swahili and Wolof as `.txt` files.

- scaling_datasets.py

It indicates how `nllb_en_sw_20K` and `nllb_en_fr_20K` were derived from [NLLB](https://huggingface.co/datasets/allenai/nllb).