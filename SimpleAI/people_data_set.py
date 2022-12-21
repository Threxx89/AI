import numpy as np
import torch as T


class PeopleDataset(T.utils.data.Dataset):
    def __init__(self, src_file, device, num_rows=None):
        # Load All data from file
        x_tmp = np.loadtxt(src_file, max_rows=num_rows,
                           usecols=range(0, 7), delimiter=",",
                           skiprows=0, dtype=np.float32)
        # Loads Political Stance
        y_tmp = np.loadtxt(src_file, max_rows=num_rows,
                           usecols=7, delimiter=",", skiprows=0,
                           dtype=np.long)

        self.x_data = T.tensor(x_tmp,
                               dtype=T.float32).to(device)
        self.y_data = T.tensor(y_tmp,
                               dtype=T.long).to(device)

    def __len__(self):
        return len(self.x_data)  # required

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx, 0:7]
        pol = self.y_data[idx]
        sample = \
            {'predictors': preds, 'political': pol}
        return sample

# ---------------------------------------------------
