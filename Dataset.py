from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, sents_padded, labels):
        self.sents_padded = sents_padded
        self.labels = labels

    def __len__(self):
        return len(self.sents_padded)

    def __getitem__(self, index):
        x = self.sents_padded[index]
        y = self.labels[index]
        return x, y
