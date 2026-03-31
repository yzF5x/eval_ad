from __future__ import annotations

import re
from typing import Any, Optional

import requests

from .loader import get_config

# Constants
ANSWER_TAG_PATTERN = r"<answer>(.*?)</answer>"
YES_NO_PATTERN = r"\b(yes|no)\b"

DEFAULT_API_PROMPT = (
    "Determine whether there is an anomaly or defect by the semantics of the following paragraph. "
    "If yes, answer \"yes\", otherwise answer \"no\". "
    "No other words are allowed. No punctuation is allowed. The paragraph is: "
)

DEFAULT_BORDER_RATIO = 0.2
DEFAULT_GRID_SIZE = 15
DEFAULT_PATCH_SIZE = 14
DEFAULT_MERGE_SIZE = 2
DEFAULT_TOP_K_PERCENT = 10
DEFAULT_EXPONENT = 2.0
DEFAULT_SUM_QUANTILE = 0.6
DEFAULT_MINMAX_LOW = 0.5
DEFAULT_MINMAX_HIGH = 1.0
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.7
DEFAULT_SPATIAL_ENTROPY_QUANTILE = 0.95
DEFAULT_SPATIAL_ENTROPY_MULTIPLIER = 2.0
DEFAULT_THRESHOLD_SE_FALLBACK = 10.0
DEFAULT_PAR_THRESHOLD = 0.5
DEFAULT_VISION_TOKEN_ID = 151655
DEFAULT_SC_TOP_N = 3
DEFAULT_SC_FALLBACK_N = 2

QUESTION_WITH_TAG = (
    "Are there any defects or anomalies in this image? Answer with the word 'Yes' or 'No'. "
    "Before answering, perform a structured visual assessment in the following order:\n"
    "Overview: Briefly describe the overall content, context, and general appearance of the image.\n"
    "Analysis: Examine the image closely for any defects, irregularities, or anomalies. If none are present, "
    "explain why the image appears normal and consistent with expected standards.\n"
    "Conclusion: Provide a one-sentence summary that states the final judgment on whether any defects or anomalies exist.\n"
    "Output your response strictly in this format without any additional text or tags outside the specified structure:\n"
    "<overview>...</overview><analyze>...</analyze><conclusion>...</conclusion><answer>Yes/No</answer>"
)

_ANSWER_TAG_RE = re.compile(ANSWER_TAG_PATTERN, re.IGNORECASE | re.DOTALL)
_YES_NO_RE = re.compile(YES_NO_PATTERN, re.IGNORECASE)


# Prompt helpers

def build_messages(image_path: str, question: str, with_tag: bool) -> list:
    text = QUESTION_WITH_TAG if with_tag else question
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    ]


# Parsing helpers

def extract_tagged_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    match = _ANSWER_TAG_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def normalize_yes_no(text: Optional[str]) -> str:
    if not text:
        return ""
    match = _YES_NO_RE.search(text)
    if not match:
        return ""
    return match.group(1).lower()


# API client

def send2api(prediction: str, prompt: str | None = None, timeout: int = 30) -> str:
    config = get_config()
    url = config.api_base
    ssh_key = config.OPENROUTER_API_KEY
    model = config.api_model
    if not ssh_key:
        return "Something Wrong with API"

    prompt_text = prompt or DEFAULT_API_PROMPT
    headers = {"Authorization": f"Bearer {ssh_key}"}
    text = {"type": "text", "text": f"{prompt_text} '{prediction}' "}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [text],
            }
        ],
    }
    try:
        response = requests.post(url=url, headers=headers, json=payload, timeout=timeout)
    except Exception:
        return "Something Wrong with API"

    if response.status_code != 200:
        return "Something Wrong with API"

    json_llm_answer = response.json()
    choices = json_llm_answer.get("choices", [])
    if not choices:
        return "Something Wrong with API"
    message = choices[0].get("message", {})
    content = message.get("content", "")
    return content


# Torch helpers

def move_to_cpu(obj):
    """Recursively move all torch.Tensors in obj to CPU."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(v) for v in obj)
    return obj


# CLI helpers

def apply_config_overrides(args: Any, config: Any) -> Any:
    """Apply AppConfig values onto argparse namespace when attributes exist."""
    for key, value in config.__dict__.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


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