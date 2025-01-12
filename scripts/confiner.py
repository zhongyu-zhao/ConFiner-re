import torch
from diffusers import StableDiffusionPipeline, AnimateDiffPipeline
from diffusers import MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


CONTROL_CLS  = AnimateDiffPipeline
SPATIAL_CLS  = StableDiffusionPipeline
TEMPORAL_CLS = pipeline

class ConFiner:
    def __init__(
            self, 
            control_expert  = "ByteDance/AnimateDiff-Lightning", 
            control_base_model = "emilianJR/epiCRealism", 
            spatial_expert  = "sd-legacy/stable-diffusion-v1-5",
            temporal_expert = "damo/text-to-video-synthesis",
            dtype = torch.float16,
            device = "cuda",
            step = 4,  # Options: [1,2,4,8]
        ):
        self.step = step
        self.device = device
        self.dtype = dtype

        self.motion_adapter = MotionAdapter().to(device, dtype)
        self.motion_adapter.load_state_dict(
            load_file(hf_hub_download(
                control_expert, 
                f"animatediff_lightning_{step}step_diffusers.safetensors"
                ))
            )
        self.control_expert = CONTROL_CLS.from_pretrained(
            control_base_model, 
            motion_adapter=self.motion_adapter, 
            torch_dtype=dtype
        ).to(device)

        self.control_expert.scheduler = EulerDiscreteScheduler.from_config(
            self.control_expert.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        self.spatial_expert = SPATIAL_CLS.from_pretrained(
            spatial_expert,
            torch_dtype=dtype
        ).to(device)
        self.temporal_expert = TEMPORAL_CLS(
            task="text-to-video-synthesis",
            model=temporal_expert,
            device=device,
        )

    def generate_coarse_video(self, prompt):
        """
        使用control expert生成coarse video
        :param prompt: 输入的文本提示
        :param num_steps: 生成的步数
        :return: 生成的coarse video
        """
        self.coarse_video = self.control_expert(
            prompt = prompt, 
            guidance_scale = 1.0,
            num_inference_steps = self.step
        )
        return self.coarse_video

    def generate_video_structure(self, coarse_video, noisy_steps=50):
        """
        使用control expert生成video structure
        :param coarse_video: 输入的coarse video
        :param num_steps: 生成的步数
        :return: 生成的video structure
        """
        self.video_structure = self.control_expert.add_noise(
            coarse_video, 
            noisy_steps
        )
        return self.video_structure

    def refine_video(
            self, 
            video_structure, 
            prompt, 
            num_refinements=3, 
            coordinate_flag=True
        ):
        """
        执行K次优化, 根据flag选择refinement方式
        :param video_structure: 输入的video structure
        :param num_refinements: 优化的次数
        :param flag: 选择refinement方式的标志
        :return: 优化后的video
        """
        video = video_structure
        for step in range(num_refinements):
            if coordinate_flag:
                # coordinated denoising
                video = self.spatial_expert.denoise( video, step, prompt)
                video = self.temporal_expert.add_noise(video, step)
                video = self.temporal_expert.denoise(video, step, prompt)
                video = self.spatial_expert.add_noise( video, step)
                video = self.spatial_expert.denoise( video, step, prompt)
            else:
                # standard denoising
                video = self.spatial_expert.denoise(video, step)
        return video


if __name__ == '__main__':

    # 创建流程实例
    confiner = ConFiner()

    # 生成coarse video
    prompt = "A beautiful sunset over the mountains"
    coarse_video = confiner.generate_coarse_video(prompt)
    # 生成video structure
    video_structure = confiner.generate_video_structure(coarse_video)
    # 执行优化
    refined_video = confiner.refine_video(video_structure, prompt, num_refinements=5, coordinate_flag=True)
