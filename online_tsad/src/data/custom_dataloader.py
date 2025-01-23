import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    """
    A custom dataset class that inherits from torch.utils.data.Dataset and can include metadata
    Metadata can be e.g. the position of the anomaly, level, ...
    """

    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]


def get_dataloaders(train, val, test, batch_size, num_workers=0):
    """
    Creates dataloaders for training, validation, and testing with potential metadata
    """
    # Unpack data
    X_train = train[0]
    X_val = val[0]
    X_test, y_test = test

    train_idx, val_idx = train_test_split(range(len(X_train)), train_size=0.9, random_state=0)

    # Create TensorDatasets
    train_dataset = CustomDataset(
        torch.tensor(X_train.values[train_idx], dtype=torch.float32).unsqueeze(1),
    )
    trainval_dataset = CustomDataset(
        torch.tensor(X_train.values[val_idx], dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = CustomDataset(
        torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1),
    )
    test_dataset = CustomDataset(
        torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1),
    )
    # Create a dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    trainval_dataloader = DataLoader(
        trainval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataloader, trainval_dataloader, val_dataloader, test_dataloader
