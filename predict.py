import os
import json
import torch
import torchaudio
import einops
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

path_model_config = 'model_config.json'
path_model_ckpt_path = 'model.ckpt'

with open(path_model_config) as f:
    model_config = json.load(f)
model = create_model_from_config(model_config)
copy_state_dict(model, load_ckpt_state_dict(path_model_ckpt_path))

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

conditioning = [{
    "prompt": "128 BPM tech house drum loop",
    "seconds_start": 0, 
    "seconds_total": 180
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = einops.rearrange(output, "b d n -> d (b n)")
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)