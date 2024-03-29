import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
from collections import OrderedDict

from models.nn_blocks import *

class TorchTransformerModel(LightningModule):

    def __init__(self, vocab_size):
        super(TorchTransformerModel, self).__init__()
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
                                          dim_feedforward=256,
                                          batch_first=True)
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

        loss = self.calc_loss(logits, batch['tgt'][:, self.num_predict_steps:])

        metrics = self.calc_metrics(logits, batch)

        return loss, metrics

    def forward(self, batch):
        """
        Pass text embedding through convolutional layers. Concatenate result with base features and pass through final
        MLP to get predictions of a batch.
        :param batch: tuple (X, y), where the shape of X is (batch_size, 23) and of y is (batch_size)
        :return: predictions: tensor of shape (batch_size)
        """
        src = batch['src']
        tgt = batch['tgt']

        tgt_input = tgt[:, :-self.num_predict_steps]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

        src_emb = self.embed_layer(src)
        src_emb = self.positional_enc(src_emb)

        tgt_emb = self.embed_layer(tgt_input)
        tgt_emb = self.positional_enc(tgt_emb)

        out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, src_padding_mask)
        #out = self.transformer(src_emb, tgt_emb)
        logits = torch.transpose(self.linear(out), 1, 2)

        return logits

    def calc_loss(self, logits, target):
        """
        Calculate L1 loss.
        :param prediction: tensor of predictions (batch_size)
        :param target: tensor of ground truths (batch_size)
        :return: tensor of losses (batch_size)
        """
        loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        loss = loss_fn(logits, target.long())

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

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_enc(self.embed_layer(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_enc(self.embed_layer(tgt)), memory, tgt_mask)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        return src_mask.to(self.device), tgt_mask.to(self.device),\
               src_padding_mask.to(self.device), tgt_padding_mask.to(self.device)