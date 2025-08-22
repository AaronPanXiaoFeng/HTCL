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
        """
        trigger_song_id
        song_id
        trigger_song_id_mel_spec_path
        song_id_mel_spec_path
        trigger_song_id_text
        song_id_text
        """
        # get mel-spectrogram file path
        mel_spec_path1 = self.df.loc[index, 'trigger_song_id_mel_spec_path']
        mel_spec_path2 = self.df.loc[index, 'song_id_mel_spec_path']

        mel_spec1 = np.load(mel_spec_path1, allow_pickle=True).item()['logS'].T.astype(np.float32)
        mel_spec2 = np.load(mel_spec_path2, allow_pickle=True).item()['logS'].T.astype(np.float32)

        # padding
        mel_spec1 = padding(mel_spec1, pad_value=mel_spec1.max() - 80., max_length=self.inp_feat_max_len)
        mel_spec2 = padding(mel_spec2, pad_value=mel_spec2.max() - 80., max_length=self.inp_feat_max_len)

        # norm
        mel_spec1 = normalize(mel_spec1)
        mel_spec2 = normalize(mel_spec2)

        text1 = self.df.loc[index, 'trigger_song_id_text']
        text2 = self.df.loc[index, 'song_id_text']

        return mel_spec1, mel_spec2, text1, text2


def collate_fn(batch, tokenizer):
    mel_spec1, mel_spec2, text1, text2 = list(zip(*batch))

    mel_spec1 = torch.from_numpy(np.concatenate([np.expand_dims(x, axis=0) for x in mel_spec1], axis=0))
    mel_spec2 = torch.from_numpy(np.concatenate([np.expand_dims(x, axis=0) for x in mel_spec2], axis=0))

    encoded_text_inputs1 = tokenizer(text1, padding=True, truncation=True, return_tensors="pt")
    encoded_text_inputs2 = tokenizer(text2, padding=True, truncation=True, return_tensors="pt")

    return mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2
