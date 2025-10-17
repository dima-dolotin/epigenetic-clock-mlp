import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np


class EpigeneDataset(Dataset):
    def __init__(self, path):
        
        df = pd.read_csv(path, index_col = 0)
        inpt = df.iloc[:, :-1].to_numpy()
        label = df.iloc[:, -1].to_numpy()

        mean = inpt.mean(axis = 0)
        std = inpt.std(axis = 0)
        std[std==0]=1e-6
        inpt = (inpt - mean)/std

        self.inpt = torch.tensor(inpt, dtype = torch.float32)
        self.label = torch.tensor(label, dtype = torch.float32).unsqueeze(1)


    def __len__(self):
        return self.inpt.shape[0]
    
    
    def __getitem__(self, idx):
        return self.inpt[idx], self.label[idx]
    

def split_dataset(dataset, testset_n, seed = 32526):
    trainset_n = len(dataset) - testset_n
    generator = torch.Generator().manual_seed(seed) 
    trainset, testset = random_split(dataset, [trainset_n, testset_n], generator = generator)

    # save testset for further use
    test_inputs = testset.dataset.inpt[testset.indices]
    test_labels = testset.dataset.label[testset.indices]
    torch.save({"test_inputs": test_inputs, "test_labels": test_labels}, "testset.pt")

    return trainset, testset


def load_data(trainset, batch_size):
    train_loader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

    return train_loader
