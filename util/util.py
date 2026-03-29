from typing import Any, List, Sequence, Union
from PIL import Image

def toliststr(s: Union[str, Sequence[Any]]) -> List[str]:
    """
        字符串形式的列表 转换为 真实列表
    """
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

def resize_image(images , max_size , merged_patch_size):
    if max_size == -1:
        return images
    processed_images = []
    for image in images:
        width, height = image.size
        long_side = max(width, height)
        scale = max_size / long_side
        short_side = min(width, height)
        new_short = short_side * scale
        new_short = (int(new_short) // merged_patch_size) * merged_patch_size
        new_short = max(new_short, merged_patch_size)  # 防止短边为 0
        new_size = (max_size , new_short) if  width >= height else (new_short , max_size)
        processed_image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
        processed_images.append(processed_image)
    return processed_images