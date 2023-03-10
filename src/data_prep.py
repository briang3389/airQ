from constants import *
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


def disp_ds_info(dl, count):
    inp_seq = next(iter(dl))[0]
    out_seq = next(iter(dl))[1]
    print(" train seq:", inp_seq.shape)
    print(" test seq", out_seq.shape)
    for i in range(count):
        print(" Input seq:", inp_seq[i])
        print(" Out seq:", out_seq[i])


class air_quality_ds(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.data_len = len(x)

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        train_ex = torch.from_numpy(self.x[i]).float()
        test_ex = torch.from_numpy(self.y[i]).float()
        return (train_ex, test_ex)

'''
def get_data_from(file_path):
    # TODO: NEED TO IMPLEMENT
    return None, None
'''

def get_data_loaders():
    
    train_x = np.load(X_TRAIN_PATH,allow_pickle=True)
    train_y = np.load(X_TEST_PATH,allow_pickle=True)
    valid_x = np.load(Y_TRAIN_PATH,allow_pickle=True)
    valid_y = np.load(Y_TEST_PATH,allow_pickle=True)
    print(train_x.shape)
    print(train_y.shape)
    print(valid_x.shape)
    print(valid_y.shape)
    valid_ds = air_quality_ds(valid_x, valid_y)
    print("validation batches:", len(valid_ds))

    train_ds = air_quality_ds(train_x, train_y)
    print("train batches:", len(train_ds))

    # create dataloaders
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"batches in valid_dl: {len(valid_dl)}")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    print(f"batches in train_dl: {len(train_dl)}")

    print("====Train DS====")
    disp_ds_info(train_dl, 2)
    print("====Validation DS====")
    disp_ds_info(valid_dl, 4)

    return train_dl, valid_dl
