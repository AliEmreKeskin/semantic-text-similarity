from sentence_transformers import SentenceTransformer, models
from lib import train
from sklearn.decomposition import PCA

if __name__ == '__main__':
    word_embedding_model = models.Transformer(model_name_or_path="bert-base-uncased", max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_dimension=512)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda")
    model = train(model, "sts_data/sts-train.csv", "sts_data/sts-dev.csv", validation_show_progress_bar=True,
                       evaluation_steps=500, batch_size=16, epochs=100, warmup_steps=100, show_progress_bar=True,
                       output_path="best_model_pooling_512", pca=None)
