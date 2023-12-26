import glob
import os
from nodes import ArtGallery_Zho
import folder_paths
from server import PromptServer
from folder_paths import get_directory_by_type
from aiohttp import web
import shutil


@PromptServer.instance.routes.get("/zho/view/{name}")
async def view(request):
    name = request.match_info["name"]
    image_path = get_img_path(name)
  
    if not os.path.exists(image_path):
        return web.Response(status=404)
    
    return web.FileResponse(image_path)

def get_img_path(template_name):
    p = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(p, 'img_lists/artists/')
    image_filename = f"{template_name}.jpg"
    full_image_path = os.path.join(image_path, image_filename)

    return full_image_path

def populate_items(names, type):
    for idx, item_name in enumerate(names):
        # 使用 get_img_path 函数来获取图像路径
        image_path = get_img_path(item_name)

        # 检查图像文件是否存在
        has_image = os.path.isfile(image_path)

        # 更新 names 列表，包含内容名称和图像路径（如果存在）
        names[idx] = {
            "content": item_name,
            "image": image_path if has_image else None,
        }
    # 根据内容名称对列表进行排序
    names.sort(key=lambda i: i["content"].lower())
  

class ArtGalleryWithImages_Zho(ArtGallery_Zho):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        names = types["required"]["artist"][0]
        populate_items(names, "artists")
        return types

    def artgallery(self, **kwargs):
        kwargs["artist"] = kwargs["artist"]["content"]
        return super().artgallery(**kwargs)


NODE_CLASS_MAPPINGS = {
    "ArtGalleryWithImages_Zho": ArtGalleryWithImages_Zho,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtGalleryWithImages_Zho": "🖌️ ArtGallery_Zho",
}
