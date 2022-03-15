import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        datum = self.data[idx]
        sample = {"Data sample": datum, "Class": label}

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        datum, label = sample['Data sample'], sample['Class']
        return {'Data sample': torch.Tensor(datum),
                'Class': torch.Tensor(label)}