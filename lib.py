import pathlib

import sentence_transformers.util
from sentence_transformers import SentenceTransformer, losses, evaluation
from sts_dataset import StsDataset
from torch.utils.data import DataLoader
import numpy as np
from typing import List
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import pickle as pk
from pathlib import Path


def create_evaluation_data(annotation_file_path: str, classification_threshold: float = 0.5):
    dataset = StsDataset(annotation_file_path)
    sentences1 = []
    sentences2 = []
    scores = []
    labels = []
    for i in dataset:
        sentences1.append(i.texts[0])
        sentences2.append(i.texts[1])
        scores.append(i.label)
        labels.append(0 if i.label < classification_threshold else 1)
    return sentences1, sentences2, scores, labels


def create_evaluator(annotation_file_path: str, batch_size: int, show_progress_bar: bool):
    sentences1, sentences2, scores, _ = create_evaluation_data(annotation_file_path=annotation_file_path)
    return evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, batch_size=batch_size,
                                                   show_progress_bar=show_progress_bar)


def create_evaluator_binary_classification(annotation_file_path: str, batch_size: int, show_progress_bar: bool):
    sentences1, sentences2, _, labels = create_evaluation_data(annotation_file_path=annotation_file_path)
    return evaluation.BinaryClassificationEvaluator(sentences1=sentences1, sentences2=sentences2, labels=labels,
                                                    batch_size=batch_size, show_progress_bar=show_progress_bar)


def train(model: SentenceTransformer, train_dataset_path: str, validation_dataset_path: str,
          validation_show_progress_bar: bool,
          evaluation_steps: int, batch_size: int, epochs: int, warmup_steps: int, show_progress_bar: bool,
          output_path: str = None, pca: PCA = None):
    train_dataset = StsDataset(train_dataset_path)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = create_evaluator(validation_dataset_path, batch_size=batch_size,
                                 show_progress_bar=validation_show_progress_bar)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps,
              evaluator=evaluator, evaluation_steps=evaluation_steps, output_path=output_path,
              show_progress_bar=show_progress_bar)

    if pca is None:
        return model

    pca = train_pca(pca=pca, model=model, train_dataset_path=train_dataset_path, output_path=output_path)
    return model, pca


def train_pca(pca: PCA, model: SentenceTransformer, train_dataset_path: str, output_path: str = None):
    print("train_pca : started")
    sentences1, sentences2, _, _ = create_evaluation_data(annotation_file_path=train_dataset_path)
    vectors1 = model.encode(sentences1)
    vectors2 = model.encode(sentences2)
    vector = vectors1 + vectors2
    pca.fit(vector)
    if output_path is not None:
        save_pca(pca=pca, path=output_path + "/pca.pkl")
        print("train_pca : saved to " + output_path + "/pca.pkl")
    print("train_pca : finished")
    return pca


def test(model_name_or_path, dataset_path: str, device: str, evaluator_type: str, output_path: str):
    model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)
    evaluator = None
    if evaluator_type == "EmbeddingSimilarityEvaluator":
        evaluator = create_evaluator(annotation_file_path=dataset_path, batch_size=1, show_progress_bar=True)
    elif evaluator_type == "BinaryClassificationEvaluator":
        evaluator = create_evaluator_binary_classification(annotation_file_path=dataset_path, batch_size=1,
                                                           show_progress_bar=True)
    pathlib.Path(output_path).mkdir(exist_ok=True)
    evaluator(model=model, output_path=output_path)


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def euclidean_distance(u, v):
    return np.linalg.norm(u - v)


def index_of_most_similar(db, query):
    return np.argmax(np.dot(query, db.T) / (np.linalg.norm(db) * np.linalg.norm(query)))


class SemanticSentenceSimilarity:
    def __init__(self, model_name_or_path: str, device: str):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.db_sentences = []
        self.db_vectors: np.ndarray = None

    def compare(self, sentence_one: str, sentence_two: str) -> float:
        sentence_one_vec = self.model.encode([sentence_one])[0]
        sentence_two_vec = self.model.encode([sentence_two])[0]
        return cosine_similarity(sentence_one_vec, sentence_two_vec)

    def extend_db(self, new_sentences: List[str]) -> None:
        self.db_sentences.extend(new_sentences)
        if self.db_vectors is None:
            self.db_vectors = self.model.encode(new_sentences)
        else:
            self.db_vectors = np.append(self.db_vectors, self.model.encode(new_sentences), axis=0)

    def search_db(self, query_sentence: str) -> str:
        query_vector = self.model.encode([query_sentence])[0]
        return self.db_sentences[index_of_most_similar(self.db_vectors, query_vector)]


def apply_pca(data, n_components: int):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def threshold_classification(value, threshold):
    return 0 if value < threshold else 1


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def evaluate(model_name_or_path, dataset_path: str, device: str, output_path: str,
             classification_threshold: float = 0.5, plot_conf_mat: bool = False):
    model = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)
    sentences1, sentences2, scores, labels = create_evaluation_data(annotation_file_path=dataset_path,
                                                                    classification_threshold=classification_threshold)

    vectors1 = model.encode(sentences=sentences1)
    vectors2 = model.encode(sentences=sentences2)

    pca_path = model_name_or_path + "/pca.pkl"
    if Path(pca_path).exists():
        pca = load_pca(path=pca_path)
        vectors1 = pca.transform(vectors1)
        vectors2 = pca.transform(vectors2)

    sim = [cosine_similarity(x, y) for x, y in zip(vectors1, vectors2)]
    classification_predictions = [threshold_classification(value=x, threshold=classification_threshold) for x in sim]

    accuracy = accuracy_score(y_true=labels, y_pred=classification_predictions)
    f1 = f1_score(y_true=labels, y_pred=classification_predictions)
    precision = precision_score(y_true=labels, y_pred=classification_predictions)
    recall = recall_score(y_true=labels, y_pred=classification_predictions)
    conf_matrix = confusion_matrix(y_true=labels, y_pred=classification_predictions)

    if plot_conf_mat:
        plot_confusion_matrix(conf_matrix=conf_matrix)

    print(f"evaluate : accuracy {accuracy}, f1 {f1}, precision {precision}, recall {recall}")
    return accuracy, f1, precision, recall, conf_matrix


def save_pca(pca: PCA, path: str):
    pk.dump(obj=pca, file=open(file=path, mode="wb"))


def load_pca(path: str):
    pca: PCA = pk.load(file=open(file=path, mode="rb"))
    return pca


class BinaryClassificationEvaluator(sentence_transformers.evaluation.SentenceEvaluator):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
