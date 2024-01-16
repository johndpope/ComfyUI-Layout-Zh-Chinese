import torch
import os
import folder_paths
from diffusers import EulerDiscreteScheduler
from .pipeline import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from .style_template import styles

class PhotoMakerNode:

    STYLE_NAMES = list(styles.keys())
    DEFAULT_STYLE_NAME = "Photographic (Default)"

    @staticmethod
    def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[PhotoMakerNode.DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + ' ' + negative

    def __init__(self, base_model_path, ref_images_path):
        self.base_model_path = base_model_path
        self.ref_images_path = ref_images_path
        self.photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
        self.device = "cuda"
        self.pipe = self.initialize_model()
        self.input_id_images = self.load_ref_images()

    def load_ref_images(self):
        image_basename_list = os.listdir(self.ref_images_path)
        image_path_list = [
            os.path.join(ref_images_path, basename) 
            for basename in image_basename_list
            if not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))  # 只包括有效的图像文件
        ]

        input_id_images = []
        for image_path in image_path_list:
            input_id_images.append(load_image(image_path))

        return input_id_images

    def initialize_model(self):
        # Load base model
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            self.base_model_path, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True, 
            variant="fp16",
        ).to(self.device)

        # Load PhotoMaker checkpoint
        pipe.load_photomaker_adapter(
            os.path.dirname(self.photomaker_path),
            subfolder="",
            weight_name=os.path.basename(self.photomaker_path),
            trigger_word="img"
        )     
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()

        return pipe

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "SG161222/RealVisXL_V3.0", "multiline": False}),
                "ref_images_path": ("STRING", {"default": "./examples/newton_man"}),
                "prompt": ("STRING", {"default": "Enter your prompt here"}),
                "negative_prompt": ("STRING", {"default": "Enter negative prompt here"}),
                "style_name": ("SELECT", {"options": PhotoMakerNode.STYLE_NAMES, "default": PhotoMakerNode.DEFAULT_STYLE_NAME}),
                "style_strength_ratio": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0}),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider"}),
                "seed": ("INT", {"default": 42, "min": 0})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_images"
    CATEGORY = "📷PhotoMaker"

    def generate_images(self, base_model_path, ref_images_path, prompt, negative_prompt, style_name, style_strength_ratio, num_steps, guidance_scale, seed):
        self.base_model_path = base_model_path
        self.ref_images_path = ref_images_path
        self.input_id_images = self.load_ref_images()

        positive, negative = self.apply_style(style_name, prompt, negative_prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
        if start_merge_step > 30:
            start_merge_step = 30

        images = self.pipe(
            prompt=positive,
            input_id_images=self.input_id_images,
            negative_prompt=negative,
            num_images_per_prompt=1,
            num_inference_steps=num_steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images

        return images

NODE_CLASS_MAPPINGS = {
    "PhotoMakerNode": PhotoMakerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoMakerNode": "Photo Maker Node"
}
