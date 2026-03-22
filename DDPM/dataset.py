import torch
from pathlib import Path
from torch.utils.data import Dataset


class DDPMDataset(Dataset):
    def __init__(
        self,
        data_path,
    ):
        super().__init__()
        self.data_path = data_path


    def __len__(self):
        pass


    def __getitem__(self, index):
        pass



def collate_fn():
    pass