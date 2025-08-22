import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from utils import normalize, padding


class SimpleDataset(Dataset):
    def __init__(
            self,
            dataset_file: str,
            inp_feat_max_len: int,
            padding: bool = True,
    ):
        if dataset_file.endswith('.parquet'):
            self.df = pd.read_parquet(dataset_file)
        else:
            raise ValueError(f'{dataset_file} is not supported.')

        self.inp_feat_max_len = inp_feat_max_len
        self.padding = padding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # get mel-spectrogram file path
        mel_spec_path = self.df.loc[index, 'mel_spec_path']

        mel_spec = np.load(mel_spec_path, allow_pickle=True).item()['logS'].T.astype(np.float32)

        # padding
        mel_spec = padding(mel_spec, pad_value=mel_spec.max() - 80., max_length=self.inp_feat_max_len)
        # norm
        mel_spec = normalize(mel_spec)
        text = self.df.loc[index, 'text']

        return mel_spec, text


def collate_fn(batch, tokenizer):
    mel_spec, text = list(zip(*batch))
    mel_spec = torch.from_numpy(np.concatenate([np.expand_dims(x, axis=0) for x in mel_spec], axis=0))
    encoded_text_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return mel_spec, encoded_text_inputs
