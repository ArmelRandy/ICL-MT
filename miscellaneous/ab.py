import os
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

if __name__ == "__main__":
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder"
    )

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data/flores')

    for src, tgt in [("eng_Latn", "fra_Latn"), ("eng_Latn", "swh_Latn")] :
        source = src.split("_")[0][0:2]
        target = tgt.split("_")[0][0:2]
        
        ds = load_dataset(f"ArmelRandy/nllb_{source}_{target}_20K")["train"]

        L_src = [example[source] for example in ds]
        L_tgt = [example[target] for example in ds]

        print(f"Here is an example sample:\n{src}: {L_src[33]}\n{tgt}: {L_tgt[33]}")

        src_emb = t2vec_model.predict(L_src, source_lang=src)
        tgt_emb = t2vec_model.predict(L_tgt, source_lang=tgt)

        src_emb = src_emb.detach().numpy()
        tgt_emb = tgt_emb.detach().numpy()

        src_emb.tofile(
            os.path.join(
                output_dir,
                f"{src.split('_')[0]}/SONAR/pool_nllb_{source}_to_{target}.bin"
            )
        )

        tgt_emb.tofile(
            os.path.join(
                output_dir,
                f"{tgt.split('_')[0]}/SONAR/pool_nllb_{source}_to_{target}.bin"
            )
        )