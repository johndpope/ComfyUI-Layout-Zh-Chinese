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
#photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
device = "cuda" if torch.cuda.is_available() else "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"


def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + ' ' + negative


class BaseModelLoaderNode_Zho:
    def __init__(self, base_model_path):
        self.base_model_path = base_model_path

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model_path": ("STRING", {"default": "SG161222/RealVisXL_V3.0"})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "📷PhotoMaker"
  
    def load_model(self):
        # Code to load the base model
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        return pipe


class PhotoMakerAdapterLoaderNode_Zho:
    def __init__(self, repo_id, filename, pipe):
        self.repo_id = repo_id
        self.filename = filename
        self.pipe = pipe

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": "TencentARC/PhotoMaker"}),
                "filename": ("STRING", {"default": "photomaker-v1.bin"}),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_photomaker_adapter"
    CATEGORY = "📷PhotoMaker"

    def load_photomaker_adapter(self):
        # 使用hf_hub_download方法获取PhotoMaker文件的路径
        photomaker_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type="model"
        )

        # 加载PhotoMaker检查点
        self.pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.fuse_lora()
        return self.pipe


class ImagePreprocessingNode_Zho:
    def __init__(self, ref_image=None, ref_images_path=None, mode="single"):
        self.ref_image = ref_image
        self.ref_images_path = ref_images_path
        self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_images_path": ("STRING", {"default": "path/to/images"}),  # 图像文件夹路径
                "mode": (["single", "multiple"], {"default": "multiple"})  # 选择模式
            },
            "optional": {
                "ref_image": ("IMAGE",)  # 单张图像（可选）
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image"
    CATEGORY = "📷PhotoMaker"
  
    def preprocess_image(self, ref_image=None, ref_images_path=None, mode="single"):
        # 使用传入的参数更新类属性
        self.ref_image = ref_image if ref_image is not None else self.ref_image
        self.ref_images_path = ref_images_path if ref_images_path is not None else self.ref_images_path
        self.mode = mode

        if self.mode == "single" and self.ref_image is not None:
            # 单张图像处理
            image_np = (255. * self.ref_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            return [pil_image]
        elif self.mode == "multiple":
            # 多张图像路径处理
            image_basename_list = os.listdir(self.ref_images_path)
            image_path_list = [
                os.path.join(self.ref_images_path, basename) 
                for basename in image_basename_list
                if not basename.startswith('.') and basename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
            ]
            return [load_image(image_path) for image_path in image_path_list]
        else:
            raise ValueError("Invalid mode. Choose 'single' or 'multiple'.")


class CompositeImageGenerationNode_Zho:
    def __init__(self, style_name, style_strength_ratio, steps, seed, prompt, negative_prompt, guidance_scale, pil_image, pipe):
        self.style_name = style_name
        self.style_strength_ratio = style_strength_ratio
        self.steps = steps
        self.seed = seed
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance_scale = guidance_scale
        self.pil_image = pil_image
        self.pipe = pipe

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "sci-fi, closeup portrait photo of a man img wearing the sunglasses in Iron man suit, face, slim body, high quality, film grain", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth", "multiline": True}),
                "style_name": (STYLE_NAMES, {"default": DEFAULT_STYLE_NAME}),
                "style_strength_ratio": ("INT", {"default": 20, "min": 1, "max": 50, "display": "slider"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "pil_image": ("IMAGE",),
                "pipe": ("MODEL",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "📷PhotoMaker"

    def generate_image(self):
        # Code for the remaining process including style template application, merge step calculation, etc.
        prompt, negative_prompt = apply_style(self.style_name, self.prompt, self.negative_prompt)
        
        start_merge_step = int(float(self.style_strength_ratio) / 100 * self.steps)
        if start_merge_step > 30:
            start_merge_step = 30

        generator = torch.Generator(device=device).manual_seed(self.seed)

        output = self.pipe(
            prompt=self.prompt,
            input_id_images=[self.pil_image],
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=self.steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=self.guidance_scale,
            return_dict=False
        )
      
        # 检查输出类型并相应处理
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是 StableDiffusionXLPipelineOutput，需要从中提取图像
            images_list = output.images

        # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        return images_tensors


NODE_CLASS_MAPPINGS = {
    "BaseModel_Loader": BaseModelLoaderNode_Zho,
    "PhotoMakerAdapter_Loader": PhotoMakerAdapterLoaderNode_Zho,
    "Ref_Image_Preprocessing": ImagePreprocessingNode_Zho,
    "PhotoMaker_Generation": CompositeImageGenerationNode_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BaseModel_Loader": "📷Base Model Loader",
    "PhotoMakerAdapter_Loader": "📷PhotoMaker Adapter Loader",
    "Ref_Image_Preprocessing": "📷Ref Image Preprocessing",
    "PhotoMaker_Generation": "📷PhotoMaker Generation"
}
