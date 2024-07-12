import os
import json
import urllib
import urllib.request
import torch
import pytorch_lightning as pl
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

from cog import Input, Path

from util import unpack, print_files

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config
        
def train(
    dataset_url: Path = Input(description="HTTPS URL of a file containg training data"),
    dataset_config: Path = Input(description="HTTPS URL of a file containing data config JSON."),
    model_checkpoint: Path = Input(description="HTTPS URL of the model checkpoint."),
    model_config: Path = Input(description="HTTPS URL of the model config JSON."),
    batch_size: int = Input(description="Batch size.", default=8, ge=1),
    num_workers: int = Input(description="Number of workers.", default=1, ge=1),
    checkpoint_every: int = Input(description="Save a checkpoint after this many epochs.", default=1000, ge=100),
    debug: bool = Input(description="Print debugging information.", default=False)
) -> Path:
    
    # system setup
    SRC_DIR = os.path.join('src')
    DATA_DIR = os.path.join(SRC_DIR, 'dataset')
    CKPT_DIR = os.path.join(SRC_DIR, 'checkpoints')

    dirs = [SRC_DIR, DATA_DIR, CKPT_DIR]
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)
            
    num_gpus = torch.cuda.device_count()

    # get data from url
    if dataset_url:
        print('unpacking dataset:')
        unpack(dataset_url, DATA_DIR)

    with open(model_config, 'r') as f:
        model_config = json.load(f)

    with open(dataset_config, 'r') as f:
        dataset_config = json.load(f)
        
    if debug:
        print_files('.')
        
    # create dataloader
    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=batch_size, 
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )
    
    # create model
    model = create_model_from_config(model_config)
    
    # use checkpoint for model weights
    if model_checkpoint:
        model_checkpoint = str(model_checkpoint)
        copy_state_dict(model, load_ckpt_state_dict(model_checkpoint))
    
    training_wrapper = create_training_wrapper_from_config(model_config, model)
    
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=checkpoint_every, dirpath=CKPT_DIR, save_top_k=-1)
    
    exc_callback = ExceptionCallback()
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
    
    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    strategy = 'ddp_find_unused_parameters_true' if num_gpus > 1 else "auto" 

    trainer = pl.Trainer(
            devices=num_gpus,
            accelerator="gpu",
            num_nodes = 1,
            strategy=strategy,
            precision="16-mixed",
            accumulate_grad_batches=1, 
            callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
            logger=None,
            log_every_n_steps=1,
            max_epochs=10000000,
            default_root_dir=SRC_DIR,
            gradient_clip_val=0.0,
            reload_dataloaders_every_n_epochs = 0
        )
    # trainer.fit(training_wrapper, train_dl, ckpt_path=CKPT_DIR if CKPT_DIR else None)
    trainer.fit(training_wrapper, train_dl, ckpt_path=None)
    
    return Path(CKPT_DIR)

if __name__ == "__main__":
    print("Let's do some debuggin':")
    train(dataset_url=None,
          dataset_config="dataset.json",
          model_checkpoint=None,
          model_config="model_config.json",
          batch_size=4,
          num_workers=1,
          checkpoint_every=1000,
          debug=False)