# Main.
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from dataloader import EEGSpectrogramDataset
from train import EEGEffnetB0
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, GroupKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Configuration settings
    num_folds = 5
    batch_size = 32
    num_workers = 3
    epochs = 10
    load_models_from = None  # Set path if loading pre-trained models

    # Load training dataset
    train_data = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
    target_columns = train_data.columns[-6:]  # Update according to the specific dataset
    train_data[target_columns] = train_data[target_columns].div(train_data[target_columns].sum(axis=1), axis=0)

    # Set up cross-validation
    group_kfold = GroupKFold(n_splits=num_folds)
    split_indices = group_kfold.split(train_data, groups=train_data['patient_id'])

    # Training and validation
    for fold_number, (train_index, valid_index) in enumerate(split_indices):
        print(f'### Training Fold {fold_number+1}')
        train_subset = train_data.iloc[train_index]
        valid_subset = train_data.iloc[valid_index]

        train_dataset = EEGSpectrogramDataset(data=train_subset, augment=True, mode='train')
        valid_dataset = EEGSpectrogramDataset(data=valid_subset, mode='valid')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # Model setup
        model = EEGEffnetB0()
        trainer = Trainer(max_epochs=epochs, gpus=1 if torch.cuda.is_available() else 0)
        
        if load_models_from is None:
            trainer.fit(model, train_loader, valid_loader)
            trainer.save_checkpoint(f'EffNet_Fold_{fold_number+1}.ckpt')
        else:
            checkpoint_path = f'{load_models_from}/EffNet_Fold_{fold_number+1}.ckpt'
            model = EEGEffnetB0.load_from_checkpoint(checkpoint_path)


    # INFER ON TEST
    test_data_path = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
    spectrograms2 = {int(f.split('.')[0]): np.load(os.path.join(test_data_path, f)) for f in os.listdir(test_data_path) if f.endswith('.npy')}
    all_eegs2 = np.load('/kaggle/input/brain-eeg-spectrograms/eeg_specs_test.npy', allow_pickle=True).item()

    preds = []
    test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
    test_ds = EEGSpectrogramDataset(test, mode='test', specs=spectrograms2, eeg_specs=all_eegs2)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=3)

    for i in range(5):
        print(f'### Testing Fold {i+1}')

        ckpt_file = f'EffNet_v2_f{i}.ckpt' if load_models_from is None else f'{load_models_from}/EffNet_v2_f{i}.ckpt'
        model = EEGEffnetB0.load_from_checkpoint(ckpt_file)
        model.to(device).eval()
        fold_preds = []

        with torch.inference_mode():
            for test_batch in test_loader:
                test_batch = test_batch.to(device)
                pred = torch.softmax(model(test_batch), dim=1).cpu().numpy()
                fold_preds.append(pred)
            fold_preds = np.concatenate(fold_preds)

        preds.append(fold_preds)

    pred = np.mean(preds, axis=0)
    print('Test preds shape:', pred.shape)

    test_res = pd.DataFrame({'eeg_id': test.eeg_id.values})
    test_res[target_columns] = pred
    test_res.to_csv('report_result.csv', index=False)
    print('result shape:', test_res.shape)
    test_res.head()

if __name__ == "__main__":
    main()
