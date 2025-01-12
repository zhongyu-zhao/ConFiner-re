# ConFiner-re
Reproduce the Confiner framework for training-free video generation enhancement.

## Recommend Environment
- Ubuntu 20.04
- Python 3.8
- Pytorch 2.1.0
- CUDA 11.8

## Installation
1. create a virtual environment using python 3.8
```bash
conda create --name confiner python=3.8 -y
conda activate confiner
```
2. install the depencdencies of video diffusion models
for animatediff-lightning, stablediffusion v1.5 and modelscope:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```
3. fix the bugs of the environment
we first update environment variables:
```bash
export LD_LIBRARY_PATH=/home/zhongyu.zhao/miniconda3/envs/confiner/lib/python3.8/site-packages/torch/lib:LD_LIBRARY_PATH
```
and check if torch can support FP8:
```python
python -c "
import torch; 
print('FP8 TYPE I :', torch.float8_e4m3fn); 
print('FP8 TYPE II:', torch.float8_e5m2)
"
```
then install some dependencies to fix the possible bugs of lavie:
```
pip uninstall xformers
pip install ninja
# for H100, H800 series GPUs
TORCH_CUDA_ARCH_LIST='9.0' pip install -v -U git+https://github.com/facebookresearch/xformers.git@v0.0.20#egg=xformers
```

## Environment Check
### Quick Start
according to the documents from these video diffusion model repos, we can easily use the models to generate videos.
- Stable Diffusion v1.5
```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
```
- Animatediff-Lightning
```python
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda"
dtype = torch.float16

step = 4  # Options: [1,2,4,8]
repo = "ByteDance/AnimateDiff-Lightning"
ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
base = "emilianJR/epiCRealism"  # Choose to your favorite base model.

adapter = MotionAdapter().to(device, dtype)
adapter.load_state_dict(load_file(hf_hub_download(repo ,ckpt), device=device))
pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

output = pipe(prompt="A girl smiling", guidance_scale=1.0, num_inference_steps=step)
export_to_gif(output.frames[0], "animation.gif")
```
- LaVie
see repo docs for more details.
- ModelScope
```python
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
test_text = {
        'text': 'A panda eating bamboo on a rock.',
    }
output_video_path = p(test_text, output_video='./output.mp4')[OutputKeys.OUTPUT_VIDEO]
print('output_video_path:', output_video_path)
```
## Acknowledgement
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Animatediff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning)
- [LaVie](https://github.com/Vchitect/LaVie)
- [ModelScope](https://modelscope.cn/models/iic/text-to-video-synthesis/summary)