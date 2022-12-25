import argparse
from lib import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type=str, default="best_model_pooling_512")
    parser.add_argument('-m', '--model', type=str, default="best_model_pooling_768_pca_512")
    parser.add_argument('-d', '--dataset', type=str, default="sts_data/sts-test.csv")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    evaluate(model_name_or_path=args.model, dataset_path=args.dataset, device=args.device, classification_threshold=0.5,
             output_path="evaluation_results", plot_conf_mat=True)
