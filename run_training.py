import argparse
import os
import yaml

from models.transformer_model import TransformerModel
from dataset import MusicDataModule

import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--weights_path', '-w', default=None)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    training_params = params['TrainingParams']

    # Initialize data module
    data_module = MusicDataModule(
        batch_size=8,
        num_workers=0,
        max_samples=None
    )
    data_module.setup(stage='fit')

    # Initialize model
    model = TransformerModel(vocab_size=data_module.vocab_len)

    if args.weights_path is not None:
        model = model.load_from_checkpoint(args.weights_path)

    # Loggers and checkpoints
    version = args.version
    logger = TensorBoardLogger('.', version=version)
    model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{version}/checkpoints',
                                 save_top_k=0,
                                 save_weights_only=True)
    lr_monitor = LearningRateMonitor()

    # Trainer
    trainer = Trainer(accelerator='auto',
                      devices=1 if torch.cuda.is_available() else None,
                      max_epochs=72,
                      val_check_interval=3000,
                      callbacks=[model_ckpt, lr_monitor],
                      logger=logger)
    trainer.fit(model, data_module)
