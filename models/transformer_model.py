import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
from collections import OrderedDict

from models.nn_blocks import *

class TransformerModel(LightningModule):

    def __init__(self, vocab_size):
        super(TransformerModel, self).__init__()
        self.read_config()

        self.build_model(vocab_size)

    def read_config(self):
        """
        Read configuration file with hyperparameters.
        :return: None
        """
        config_path = os.path.join(os.getcwd(), './config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']
        model_params = params['ModelParams']

        self.embedding_dims = model_params['embedding_dims']
        self.num_predict_steps = dataset_params['num_predict_steps']

    def build_model(self, vocab_size: int):
        """
        Build model's layers.
        :return: None
        """
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dims)
        self.positional_enc = PositionalEncoding(d_model=self.embedding_dims)
        self.transformer = nn.Transformer(d_model=self.embedding_dims,
                                          nhead=4,
                                          num_encoder_layers=3,
                                          num_decoder_layers=3,
                                          dim_feedforward=512)
        self.linear = nn.Linear(self.embedding_dims, vocab_size)

    def training_step(self, batch, batch_idx):
        """
        Perform train step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform validation step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Perform test step.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :param batch_idx: index of current batch, non applicable here
        :return: mean loss
        """
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        return loss

    def _shared_step(self, batch):
        """
        Get predictions, calculate loss and eventually useful metrics (here the only metric is MAE which is the same as
        the loss function).
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :return: loss: tensor of shape (batch_size), metrics: dictionary with metrics
        """
        logits = self.forward(batch)

        loss = self.calc_loss(logits, batch['input_seq'][:, self.num_predict_steps:])

        metrics = self.calc_metrics(logits, batch)

        return loss, metrics

    def forward(self, batch):
        """
        Pass text embedding through convolutional layers. Concatenate result with base features and pass through final
        MLP to get predictions of a batch.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :return: predictions: tensor of shape (batch_size)
        """
        x_embedded = self.embed_layer(batch['input_seq'])
        input = self.positional_enc(x_embedded)
        src = input[:, :-self.num_predict_steps]
        tgt = input[:, self.num_predict_steps:]
        out = self.transformer(src, tgt)
        logits = torch.transpose(self.linear(out), 1, 2)

        return logits

    def calc_loss(self, logits, target):
        """
        Calculate L1 loss.
        :param prediction: tensor of predictions (batch_size)
        :param target: tensor of ground truths (batch_size)
        :return: tensor of losses (batch_size)
        """
        loss_func = nn.CrossEntropyLoss(reduction='none')

        loss = loss_func(logits, target.long())[:, -self.num_predict_steps:]

        return loss

    def configure_optimizers(self):
        """
        Selection of gradient descent algorithm and learning rate scheduler.
        :return: optimizer algorithm, learning rate scheduler
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=72, eta_min=5e-5)

        return [optimizer], []

    def calc_metrics(self, prediction, target):
        """
        Calculate useful metrics. Not applicable here (MAE is already loss).
        :param prediction: tensor of predictions (batch_size)
        :param target: tensor of ground truths (batch_size)
        :return: metric dictionary
        """
        metrics = {}

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        """
        Log metrics on Tensorboard.
        :param metrics: metric dictionary
        :param type: check if training or validation metrics
        :return: None
        """
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
