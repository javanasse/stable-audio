import os
import wandb
import json
import torch
import pytorch_lightning as pl
from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

from cog import Input, Path

from util import unpack

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config
        
DATA_PATH = os.path.join('src', 'dataset')

paths = [DATA_PATH]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(paths)

def train(
    data_url: Path = Input(description="HTTPS URL of a file containg training data"),
    checkpoint: Path = Input(description="HTTPS URL of the model checkpoint."),
    learning_rate: float = Input(description="learning rate, for learning!", default=1e-4, ge=0),
    seed: int = Input(description="random seed to use for training", default=None)
) -> str:
    # get data from url
    # unpack(data_url, DATA_PATH)
    return "hehe"

# from getdata import repopulate_dataset

# os.environ["WANDB__SERVICE_WAIT"] = "300"

# def train():
    
#     print("downloading data:")
#     repopulate_dataset("dataset")
    
#     name = "test-training"
#     path_dataset_config = 'dataset.json'
#     path_model_config = 'model_config.json'
#     path_model_ckpt_path = 'model.ckpt'
#     batch_size = 1
#     num_workers = 1
#     path_save_dir = "chkpts"
#     checkpoint_every = 1000
#     num_gpus = 2
#     ckpt_path = None

#     #Get JSON config from args.model_config
#     with open(path_model_config) as f:
#         model_config = json.load(f)

#     with open(path_dataset_config) as f:
#         dataset_config = json.load(f)

#     train_dl = create_dataloader_from_config(
#         dataset_config, 
#         batch_size=batch_size, 
#         num_workers=num_workers,
#         sample_rate=model_config["sample_rate"],
#         sample_size=model_config["sample_size"],
#         audio_channels=model_config.get("audio_channels", 2),
#     )

#     model = create_model_from_config(model_config)

#     copy_state_dict(model, load_ckpt_state_dict(path_model_ckpt_path))

#     training_wrapper = create_training_wrapper_from_config(model_config, model)

#     wandb_logger = pl.loggers.WandbLogger(project=name)
#     wandb_logger.watch(training_wrapper)

#     exc_callback = ExceptionCallback()

#     if path_save_dir and isinstance(wandb_logger.experiment.id, str):
#         checkpoint_dir = os.path.join(path_save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
#     else:
#         checkpoint_dir = None

#     ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
#     save_model_config_callback = ModelConfigEmbedderCallback(model_config)

#     demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

#     strategy = 'ddp_find_unused_parameters_true' if num_gpus > 1 else "auto" 

#     trainer = pl.Trainer(
#             devices=num_gpus,
#             accelerator="gpu",
#             num_nodes = 1,
#             strategy=strategy,
#             precision="16-mixed",
#             accumulate_grad_batches=1, 
#             callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback],
#             logger=wandb_logger,
#             log_every_n_steps=1,
#             max_epochs=10000000,
#             default_root_dir=path_save_dir,
#             gradient_clip_val=0.0,
#             reload_dataloaders_every_n_epochs = 0
#         )
#     trainer.fit(training_wrapper, train_dl, ckpt_path=ckpt_path if ckpt_path else None)