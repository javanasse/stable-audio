import os
import json
import urllib.request
import torch
import torchaudio
import einops

from typing import Optional, Any
from cog import BasePredictor, Input, Path

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.utils import copy_state_dict

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print('we setup:')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        path_model_config = 'model_config.json'
        path_model_ckpt = 'model.ckpt'
        
        if not os.path.exists(path_model_ckpt):
            print(f'model at {path_model_ckpt} needed:')
            urllib.request.urlretrieve("https://storage.googleapis.com/stable_audio_public/model.ckpt", "model.ckpt")
            print(f"model downloaded.")

        with open(path_model_config) as f:
            model_config = json.load(f)
        self.model = create_model_from_config(model_config)
        copy_state_dict(self.model, load_ckpt_state_dict(path_model_ckpt))

        self.sample_rate = model_config["sample_rate"]
        self.sample_size = model_config["sample_size"]

        self.model = self.model.to(self.device)

    def predict(self,
            prompt: str = Input(description="text prompt"),
    ) -> Path:
        """Run a single prediction on the model"""
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0, 
            "seconds_total": 180
        }]

        # Generate stereo audio
        output = generate_diffusion_cond(
            self.model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=self.device
        )
        
        output = einops.rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save("output.wav", output, self.sample_rate)
        
        return Path("output.wav")