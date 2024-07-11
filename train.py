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

os.environ["WANDB__SERVICE_WAIT"] = "300"

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

name = "test-training"
path_dataset_config = 'dataset.json'
path_model_config = 'model_config.json'
path_model_ckpt_path = 'model.ckpt'
batch_size = 8
num_workers = 4
path_save_dir = "chkpts"
checkpoint_every = 1000
num_gpus = 1
ckpt_path = None

#Get JSON config from args.model_config
with open(path_model_config) as f:
    model_config = json.load(f)

with open(path_dataset_config) as f:
    dataset_config = json.load(f)

train_dl = create_dataloader_from_config(
    dataset_config, 
    batch_size=batch_size, 
    num_workers=num_workers,
    sample_rate=model_config["sample_rate"],
    sample_size=model_config["sample_size"],
    audio_channels=model_config.get("audio_channels", 2),
)

model = create_model_from_config(model_config)

copy_state_dict(model, load_ckpt_state_dict(path_model_ckpt_path))

training_wrapper = create_training_wrapper_from_config(model_config, model)

wandb_logger = pl.loggers.WandbLogger(project=name)
wandb_logger.watch(training_wrapper)

exc_callback = ExceptionCallback()

if path_save_dir and isinstance(wandb_logger.experiment.id, str):
    checkpoint_dir = os.path.join(path_save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
else:
    checkpoint_dir = None

ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
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
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=path_save_dir,
        gradient_clip_val=0.0,
        reload_dataloaders_every_n_epochs = 0
    )
trainer.fit(training_wrapper, train_dl, ckpt_path=ckpt_path if ckpt_path else None)

''''''
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# path_model_config = 'model_config.json'
# path_model_ckpt_path = 'model.ckpt'

# with open(path_model_config) as f:
#     model_config = json.load(f)
# model = stable_audio_tools.create_model_from_config(model_config)
# stable_audio_tools.copy_state_dict(model, stable_audio_tools.load_ckpt_state_dict(path_model_ckpt_path))

# sample_rate = model_config["sample_rate"]
# sample_size = model_config["sample_size"]

# model = model.to(device)

# conditioning = [{
#     "prompt": "128 BPM tech house drum loop",
#     "seconds_start": 0, 
#     "seconds_total": 180
# }]

# # Generate stereo audio
# output = stable_audio_tools.generate_diffusion_cond(
#     model,
#     steps=100,
#     cfg_scale=7,
#     conditioning=conditioning,
#     sample_size=sample_size,
#     sigma_min=0.3,
#     sigma_max=500,
#     sampler_type="dpmpp-3m-sde",
#     device=device
# )

''''''


# args = [
#     "python3", "stable-audio-tools/train.py",
#     "--dataset-config", "dataset.json",
#     "--model-config", "model_config.json",
#     "--name", "stable_audio_open_finetune",
#     "--save-dir", "checkpoints",
#     "--checkpoint-every", "1000",
#     "--batch-size", "32",
#     "--num-gpus", "2",
#     "--precision", "16-mixed",
#     "--seed", "128",
#     "--pretrained-ckpt-path", "model.ckpt",
# ]

# subprocess.run(args=args)
# python3 stable-audio-tools/train.py --dataset-config dataset.json --model-config model_config.json --name test