import argparse
import os
import yaml

from models.transformer_model import TorchTransformerModel
#from models.fast_transformer_model import FastTransformerModel
from dataset import MusicDataModule

import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument("--model", "-m", choices=["torch", "fast"])
    parser.add_argument('--weights_path', '-w', default=None)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    training_params = params['TrainingParams']

    # Initialize data module
    data_module = MusicDataModule(
        batch_size=16,
        num_workers=0,
        max_samples=30
    )
    data_module.setup(stage='fit')

    # Initialize model
    model = None
    if args.model == "torch":
        model = TorchTransformerModel(vocab_size=data_module.vocab_len)
    #elif args.model == "fast":
    #    model = FastTransformerModel(vocab_size=data_module.vocab_len)

    if args.weights_path is not None:
        model = model.load_from_checkpoint(args.weights_path)

    # Loggers and checkpoints
    version = args.version
    logger = TensorBoardLogger('.', version=version)
    model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{version}/checkpoints',
                                 save_top_k=2,
                                 monitor='loss_train',
                                 save_weights_only=True)
    lr_monitor = LearningRateMonitor()

    # Trainer
    trainer = Trainer(accelerator='auto',
                      devices=1,
                      max_epochs=15,
                      val_check_interval=1000,
                      callbacks=[model_ckpt, lr_monitor],
                      logger=logger)
    trainer.fit(model, data_module)
