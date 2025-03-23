from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.label_to_index = {label: idx for idx, label in enumerate(set(labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = self.label_to_index[self.labels[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
