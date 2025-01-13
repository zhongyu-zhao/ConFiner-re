import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, pipelines
from diffusers import MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
import modelscope
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


CONTROL_CLS  = AnimateDiffPipeline
SPATIAL_CLS  = StableDiffusionPipeline
TEMPORAL_CLS = pipeline
OUTVIDEO_CLS = pipelines.animatediff.pipeline_output.AnimateDiffPipelineOutput
TEMPORAL_MODEL_CLS = modelscope.models.multi_modal.video_synthesis.text_to_video_synthesis_model.TextToVideoSynthesis

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
            guidance_scale = 1.,
        ):
        self.step = step
        self.device = device
        self.dtype = dtype
        self.guidance_scale = guidance_scale

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

    def generate_video_structure(self, coarse_video=None, noisy_steps=4):
        """
        使用control expert生成video structure
        :param coarse_video: 输入的coarse video
        :param num_steps: 生成的步数
        :return: 生成的video structure
        """
        assert noisy_steps <= self.step, "noisy_steps should be less than or equal to step"
        num_steps = self.control_expert.scheduler.timesteps[self.step-noisy_steps]

        assert coarse_video is None or isinstance(coarse_video, OUTVIDEO_CLS), \
            "invalid coarse video type: None or PIL Image"
        if coarse_video is None:
            coarse_video = self.coarse_video

        # transform PIL.Image.Image to torch.Tensor        
        coarse_video = [transforms.ToTensor()(pil).unsqueeze(0) for pil in coarse_video.frames[0]]
        # concatenate along the time dimension
        coarse_video = torch.cat(coarse_video, dim=0)

        # sample normal noise
        noise = torch.randn_like(coarse_video)

        # add noise to the coarse video
        self.video_structure = self.control_expert.scheduler.add_noise(
            coarse_video, 
            noise,
            num_steps
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
                video = self.spatial_expert(
                    prompt = prompt, 
                    guidance_scale = self.guidance_scale, 
                    num_inference_steps = step
                )
                # add-denoise loop - temporal
                video = self._q_sample(
                    model = self.temporal_expert.model, 
                    x_start = video, 
                    timestep = step
                )
                video = self.temporal_expert(prompt)
                # add-denoise loop - spatial
                video = self._q_sample(
                    model = self.spatial_expert,
                    x_start = video,
                    timestep = step
                )
                video = self.spatial_expert(
                    prompt = prompt, 
                    guidance_scale = self.guidance_scale, 
                    num_inference_steps = step
                )
            else:
                # standard denoising
                video = self.spatial_expert(
                    prompt = prompt, 
                    guidance_scale = self.guidance_scale, 
                    num_inference_steps = step
                )
        return video
    
    def _q_sample(self, model, x_start, timestep, noise=None):
        """
        执行q-sample
        :param x_start: 输入的video
        :param timestep: 输入的时间步
        :param noise: 输入的噪声
        :return: q-sample后的video
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if isinstance(model, TEMPORAL_MODEL_CLS):
            # get noise schedule
            noise_schedule = model.diffusion
            # get efficient parameters
            sqrt_alphas_cumprod = noise_schedule.sqrt_alphas_cumprod[timestep]
            sqrt_one_minus_alphas_cumprod = noise_schedule.sqrt_one_minus_alphas_cumprod[timestep]
            # add noise
            x_start_hat = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
            return x_start_hat
        elif isinstance(model, SPATIAL_CLS):
            num_step = model.scheduler._timesteps[timestep]
            # add noise to the coarse video
            self.video_structure = self.control_expert.scheduler.add_noise(
                x_start, 
                noise,
                num_step
            )
            return self.video_structure
        else:
            raise NotImplementedError("only support temporal diffusion model")


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
