import pandas as pd
import numpy as np
import os
import yaml
import ast
import datetime
from yaml import SafeLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from collections import defaultdict
from tqdm import tqdm

from miditok.tokenizations import REMI
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile

class MusicDataModule(LightningDataModule):
    def __init__(self,
            batch_size: int = 32,
            num_workers: int = 0,
            max_samples: int = None):
        super().__init__()
        self.dataset_path = os.getenv('DATASET_PATH')
        csv_path = os.path.join(self.dataset_path, 'maestro-v3.0.0.csv')

        # Save hyperparemeters
        self.save_hyperparameters(logger=False)

        # Read config file
        self.read_config()

        # Get tokenizer
        pitch_range = range(21, 109)
        beat_res = {(0, 4): 8, (4, 12): 4}
        nb_velocities = 32
        additional_tokens = {'Chord': False, 'Rest': True, 'Tempo': True,
                             'Program': False, 'TimeSignature': False,
                             'rest_range': (2, 8),  # (half, 8 beats)
                             'nb_tempos': 32,  # nb of tempo bins
                             'tempo_range': (40, 250)}  # (min, max)

        self.tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens)
        self.vocab_len = self.tokenizer.len

        # Read csv
        read_csv = pd.read_csv(csv_path)

        # Get split sets
        self.read_train_df = read_csv[read_csv['split'] == 'train']
        if max_samples is not None:
            self.read_train_df = self.read_train_df.iloc[:max_samples]

        self.read_val_df = read_csv[read_csv['split'] == 'validation'].iloc[:1]

        self.read_test_df = read_csv[read_csv['split'] == 'test']

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']

        self.input_seq_len = dataset_params['input_seq_len']
        self.num_predict_steps = dataset_params['num_predict_steps']

    def setup(self, stage: str = None):
        """
        Build data dictionaries for training or prediction.
        :param stage: 'fit' for training, 'predict' for prediction
        :return: None
        """
        if stage == 'fit':
            train_dict = self.get_sequences(self.read_train_df, "train")
            val_dict = self.get_sequences(self.read_val_df, "val")

            self.data_train, self.data_val = train_dict, val_dict

        elif stage == 'predict':
            predict_dict = None

            self.data_predict = predict_dict

    def train_dataloader(self):
        """
        Uses train dictionary (output of format_X) to return train DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        """
        Uses val dictionary (output of format_X) to return val DataLoader, that will be fed to pytorch lightning's
        Trainer.
        :return: train DataLoader
        """
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        """
        Uses predict dictionary (output of format_X) to return predict DataLoader, that will be fed to pytorch
        lightning's Trainer.
        :return: predict DataLoader
        """
        return DataLoader(dataset=self.data_predict,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def get_sequences(self, df: pd.DataFrame, type: str = "train"):
        """
        Formats the .csv read into an initial dataframe with base features.
        :param df: dataframe read from a csv file
        :param type: whether we are treating a training or test set
        :return: correctly formatted dataframe
        """
        midi_dict = defaultdict()
        i = 0
        print(f"Loading {type} dataset...")
        for path in tqdm(df['midi_filename']):
            midi = MidiFile(os.path.join(self.dataset_path, path))
            tokens = self.tokenizer(midi)

            for j in range(0, len(tokens[0]), self.num_predict_steps):
                midi_dict[i] = defaultdict()

                # Associate sequence to midi
                midi_dict[i]['midi_filename'] = path

                # Get current sequence
                midi_dict[i]['src'] = np.array(tokens[0].ids[j:j + self.input_seq_len]).astype(int)
                # Pad sequence if too short
                if len(midi_dict[i]['src']) < self.input_seq_len:
                    midi_dict[i]['src'] = np.concatenate((midi_dict[i]['src'],
                                                         [0] * (self.input_seq_len - len(midi_dict[i]['src'])))).astype(int)

                midi_dict[i]['tgt'] = np.array(tokens[0].ids[j + self.num_predict_steps:
                                                         j + self.input_seq_len + self.num_predict_steps]).astype(int)
                if len(midi_dict[i]['tgt']) < self.input_seq_len:
                    midi_dict[i]['tgt'] = np.concatenate((midi_dict[i]['tgt'],
                                                         [0] * (self.input_seq_len - len(midi_dict[i]['tgt'])))).astype(int)
                    i += 1
                    break

                i += 1

        return midi_dict