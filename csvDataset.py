from torch.utils.data import Dataset
import os
import pandas as pd



class csvDataset(Dataset):
    def __init__(self, path, x_name="article", y_name="highlights", transform=None):
        """
        path is path to csv file
        x_name, y_name is dt row names
        """
        p = os.path.join(os.getcwd(), path)
        
        self.x_name = x_name
        self.y_name = y_name

        self.data = pd.read_csv(p)
        self.transform = transform

    def __len__(self):
        return len(self.data[self.x_name])

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 1]
        summary = self.data.iloc[idx, 2]

        sample = {"text" : text, "summary" : summary}

        if self.transform:
            sample = self.transform(sample)

        return sample
