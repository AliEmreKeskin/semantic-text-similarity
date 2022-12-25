from torch.utils.data import Dataset
from sentence_transformers import InputExample

class StsDataset(Dataset):
    def __init__(self, annotations_file_path: str):
        self.annotations_lines = None
        with open(annotations_file_path) as annotations_file:
            self.annotations_lines = annotations_file.readlines()

    def __len__(self):
        return len(self.annotations_lines)

    def __getitem__(self, idx):
        return self.__parse_line(self.annotations_lines[idx])

    def __parse_line(self, line: str):
        splitted = line.split("\t")
        return InputExample(texts=[splitted[5], splitted[6].strip()], label=float(splitted[4])/5)


if __name__ == '__main__':
    sts = StsDataset("sts_data/sts-train.csv")
    for i in sts:
        print(i)
