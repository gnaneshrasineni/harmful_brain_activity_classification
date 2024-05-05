import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as albu

df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]

class EEGSpectrogramDataset(Dataset):
    """Handles loading and augmentation of EEG spectrogram data."""
    
    def __init__(self, data, augment=False, mode='train', specs=None, eeg_specs=None):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.__getitems__([index])

    def __getitems__(self, indices):
        X, y = self._generate_data(indices)
        if self.augment:
            X = self._augment(X)
        return list(zip(X, y)) if self.mode == 'train' else X

    def _generate_data(self, indexes):
        """Combines pre-existing and newly generated EEG spectrogram data into feature and target arrays."""
        X = np.zeros((len(indexes), 128, 256, 8), dtype='float32')
        y = np.zeros((len(indexes), 6), dtype='float32')
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            r = 0 if self.mode == 'test' else int((row['min'] + row['max'])//4)
            for k in range(4):
                img = self.specs[row.spec_id][r:r+300, k*100:(k+1)*100].T
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + 1e-6)
                img = np.nan_to_num(img, nan=0.0)
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img
            if self.mode != 'test':
                y[j,] = row[TARGETS]
        return X, y

    def _random_transform(self, img):
        return albu.Compose([albu.HorizontalFlip(p=0.5)])(image=img)['image']
        
    def _augment(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])
        return img_batch
