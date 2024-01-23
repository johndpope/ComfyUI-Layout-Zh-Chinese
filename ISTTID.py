import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import os
import cv2
import torch
import numpy as np
from PIL import Image
import folder_paths

from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from .pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


current_directory = os.path.dirname(os.path.abspath(__file__))

class FaceAnalysisImageGeneration:

    def __init__(self):
        # 初始化模型路径和加载模型
        self.checkpoints_dir = "./checkpoints"
        self.model_name = 'antelopev2'
        self.controlnet_path = f'{self.checkpoints_dir}/ControlNetModel'
        self.face_adapter = f'{self.checkpoints_dir}/ip-adapter.bin'

        # 下载所需的文件
        hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir=self.checkpoints_dir)
        hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir=self.checkpoints_dir)
        hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir=self.checkpoints_dir)

        # 加载模型
        self.app = FaceAnalysis(name=self.model_name, root=current_directory, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=self.controlnet, torch_dtype=torch.float16
        )
        self.pipe.cuda()
        self.pipe.load_ip_adapter_instantid(self.face_adapter)


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "enter path"}),
                "prompt": ("STRING", {"default": "enter prompt"}),
                "negative_prompt": ("STRING", {"default": "enter negative prompt"}),
                "scale": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "📷InstantID"
  
    def process_image(self, image_path, prompt, negative_prompt, scale=0.8):
        # 处理图像并生成新图像
        image = load_image(image_path)
        face_info = self.app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(image, face_info['kps'])

        self.pipe.set_ip_adapter_scale(scale)
        generated_image = self.pipe(
            prompt, image_embeds=face_emb, image=face_kps, controlnet_conditioning_scale=scale
        ).images[0]

        return generated_image

NODE_CLASS_MAPPINGS = {
    "FaceAnalysisImageGeneration": FaceAnalysisImageGeneration
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAnalysisImageGeneration": "InstantID"
}

