from sentence_transformers import SentenceTransformer
from lib import train_pca, save_pca
from sklearn.decomposition import PCA

if __name__ == '__main__':
    model = SentenceTransformer(model_name_or_path="best_model", device="cuda")
    pca = PCA(n_components=512)
    pca = train_pca(pca=pca, model=model, train_dataset_path="sts_data/sts-train.csv")
    save_pca(pca=pca, path="pca_orj_768_to_512.pkl")
