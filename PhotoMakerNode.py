
import torch
import os
import folder_paths
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from .pipeline import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from .style_template import styles
from PIL import Image
import numpy as np

# global variable
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
device = "cuda" if torch.cuda.is_available() else "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + ' ' + negative

#batch
class PhotoMaker_Batch_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "SG161222/RealVisXL_V3.0", "multiline": False}),
                "ref_images_path": ("STRING", {"default": "./examples/newton_man"}),
                "prompt": ("STRING", {"default": "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth", "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.1, "max": 10.0, "step": 0.1, "display": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_images"
    CATEGORY = "📷PhotoMaker"

    def process_images(self, base_model_path, ref_images_path, prompt, negative_prompt, style_name, style_strength_ratio, steps, guidance_scale, seed):

        # Load base model
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)

        # Load PhotoMaker checkpoint
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()
        
        # Process images
        image_basename_list = os.listdir(ref_images_path)
        #image_path_list = sorted([os.path.join(ref_images_path, basename) for basename in image_basename_list])
        image_path_list = [
            os.path.join(ref_images_path, basename) 
            for basename in image_basename_list
            if not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))  # 只包括有效的图像文件
        ]

        input_id_images = [load_image(image_path) for image_path in image_path_list]
      
        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
      
        start_merge_step = int(float(style_strength_ratio) / 100 * steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(seed)

        output = pipe(
            prompt=prompt,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
        )

        # 检查 output 是单个图像对象还是图像列表
        if isinstance(output, Image.Image):
            # 如果是单个 PIL.Image 对象，则将其转换为 numpy.ndarray 并放入列表
            images = [np.array(output)]
        elif isinstance(output, list) and all(isinstance(img, Image.Image) for img in output):
            # 如果是 PIL.Image 对象的列表，则转换每个图像
            images = [np.array(img) for img in output]
        else:
            # 否则假设 output 已经是 numpy.ndarray 或 torch.Tensor 的列表
            images = output

        return images

# Dictionary to export the node
NODE_CLASS_MAPPINGS = {
    "PhotoMaker_Batch_Zho": PhotoMaker_Batch_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoMaker_Batch_Zho": "📷PhotoMaker"
}
