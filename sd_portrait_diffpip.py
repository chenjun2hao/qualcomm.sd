from diffusers import DiffusionPipeline
import torch

import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

from torch import Generator
from utils import *


path = '/data1/chenjun/huf/portrait-finetuned' # Path to the appropriate model-type
# Insert your prompt below.
prompt = "Faceshot Portrait of pretty young (18-year-old) Caucasian wearing a high neck sweater, sharp focus, BREAK epicrealism"
negative_prompt = ""


torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# Below code will run on gpu, please pass cpu everywhere as the device and set 'dtype' to torch.float32 for cpu inference.
with torch.inference_mode():
    pipe = DiffusionPipeline.from_pretrained(path, safety_checker=None, requires_safety_checker=False)
    pipe.to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    img = pipe(prompt=prompt,negative_prompt=negative_prompt, num_inference_steps=10, guidance_scale=7.5).images[0]
    img.save("output/image.png")

    t = 1
