from diffusers import DiffusionPipeline
from torch import autocast

pipe = DiffusionPipeline.from_pretrained("Ojimi/anime-kawai-diffusion")
# pipe = pipe.to("cuda")
device = "cuda"

prompt = "man who is smiling and jacked with a sword"
with autocast(device):
    out = pipe(prompt).images[0]
    # image = pipe(prompt, negative_prompt="lowres, bad anatomy").images[0]
    out.save("test2.png") 