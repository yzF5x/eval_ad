import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from .loader import get_config
from .core import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_BORDER_RATIO,
    DEFAULT_EXPONENT,
    DEFAULT_GRID_SIZE,
    DEFAULT_MERGE_SIZE,
    DEFAULT_MINMAX_HIGH,
    DEFAULT_MINMAX_LOW,
    DEFAULT_PAR_THRESHOLD,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SC_FALLBACK_N,
    DEFAULT_SC_TOP_N,
    DEFAULT_SPATIAL_ENTROPY_MULTIPLIER,
    DEFAULT_SPATIAL_ENTROPY_QUANTILE,
    DEFAULT_SUM_QUANTILE,
    DEFAULT_THRESHOLD_SE_FALLBACK,
    DEFAULT_TOP_K_PERCENT,
    DEFAULT_VISION_TOKEN_ID,
)

matplotlib.use("Agg")


def elbow_chord(values: Sequence[float]) -> float:
    if len(values) <= 2:
        return min(values) if values else 0.0
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return y[0]
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    return float(y[int(np.argmax(d))])


def combined_weights(values: torch.Tensor, exponent: float = DEFAULT_EXPONENT) -> torch.Tensor:
    normalized_values = (values - values.min()) / (values.max() - values.min())
    weights = torch.exp(exponent * normalized_values)
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    return weights


def get_threshold_and_weight_from_sum(c: torch.Tensor, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.zeros(c.shape[0], dtype=torch.float32)
    summed = c[:, start] + c[:, end]
    threshold = torch.quantile(summed, DEFAULT_SUM_QUANTILE)
    valid_indices = summed > threshold
    summed_weights = combined_weights(summed)
    index[valid_indices] = summed_weights[valid_indices]
    return index, threshold, summed, summed_weights


def get_periphery_mask(h: int, w: int, border_ratio: float = DEFAULT_BORDER_RATIO) -> torch.Tensor:
    mask = torch.zeros((h, w))
    border_h = int(h * border_ratio)
    border_w = int(w * border_ratio)
    mask[:border_h, :] = 1
    mask[-border_h:, :] = 1
    mask[:, :border_w] = 1
    mask[:, -border_w:] = 1
    return mask


def periphery_attention_ratio(attn_map: torch.Tensor, border_ratio: float = DEFAULT_BORDER_RATIO) -> torch.Tensor:
    h, w = attn_map.shape
    mask = get_periphery_mask(h, w, border_ratio).to(attn_map.device)
    return (attn_map * mask).sum() / attn_map.sum()


def get_par_from_attention(
    c: torch.Tensor,
    border_ratio: float = DEFAULT_BORDER_RATIO,
    grid_height: int = DEFAULT_GRID_SIZE,
    grid_width: int = DEFAULT_GRID_SIZE,
) -> torch.Tensor:
    par_info = torch.zeros(c.shape[0], dtype=torch.float32)
    for i in range(c.shape[0]):
        visual_tensor = c[i].reshape(grid_height, grid_width)
        par_info[i] = periphery_attention_ratio(visual_tensor, border_ratio)
    return par_info


def aggregate_cross_attentions(cross_attentions: torch.Tensor, token_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    t = cross_attentions.shape[0]
    device = cross_attentions.device
    dtype = cross_attentions.dtype
    weights = token_weights.to(device=device).float()
    if weights.ndim != 1 or weights.shape[0] != t:
        weights = weights.view(t)
    weights_view = weights.view(t, *([1] * (cross_attentions.ndim - 1)))
    weighted = cross_attentions.float() * weights_view
    agg = weighted.sum(dim=0)
    denom = weights.sum()
    if abs(float(denom)) < eps:
        denom = torch.tensor(eps, device=device, dtype=weights.dtype)
    agg = agg / denom
    if agg.dtype != dtype:
        agg = agg.to(dtype)
    return agg


def minmax_norm_torch_scaled(
    x: Union[torch.Tensor, Sequence[float], np.ndarray],
    low: float = DEFAULT_MINMAX_LOW,
    high: float = DEFAULT_MINMAX_HIGH,
    invert: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.to(dtype=torch.float32)
    xmin = x.min()
    xmax = x.max()
    span = (xmax - xmin).abs()
    if span < eps:
        mid = 0.5 * (low + high)
        return torch.full_like(x, fill_value=mid)
    norm01 = (x - xmin) / (span + eps)
    if invert:
        norm01 = 1.0 - norm01
    return low + (high - low) * norm01


def get_weight_with_indices(
    se_list: torch.Tensor,
    sum_list: torch.Tensor,
    valid_index: Optional[Union[torch.Tensor, Sequence[int], Sequence[bool]]] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    low: float = DEFAULT_MINMAX_LOW,
    high: float = DEFAULT_MINMAX_HIGH,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = se_list.device
    m = se_list.numel()
    if valid_index is None:
        valid_mask = torch.ones(m, dtype=torch.bool, device=device)
    else:
        vi = torch.as_tensor(valid_index, device=device)
        if vi.dtype == torch.bool:
            valid_mask = vi
        else:
            valid_mask = torch.zeros(m, dtype=torch.bool, device=device)
            valid_mask[vi.long()] = True
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    k = valid_indices.numel()
    if k == 0:
        full_weights = torch.zeros(m, device=device, dtype=torch.float32)
        return full_weights, valid_indices, torch.zeros(0, device=device), torch.zeros(0, dtype=torch.long, device=device)
    se_valid = se_list[valid_indices].to(device=device).float()
    sum_valid = sum_list[valid_indices].to(device=device).float()
    sum_norm = minmax_norm_torch_scaled(sum_valid, low=low, high=high, invert=False, eps=eps)
    se_norm = minmax_norm_torch_scaled(se_valid, low=low, high=high, invert=True, eps=eps)
    raw = alpha * sum_norm + beta * se_norm
    if k < 3:
        token_weights = torch.ones_like(raw, device=device, dtype=torch.float32)
    else:
        rmin = raw.min()
        rmax = raw.max()
        if (rmax - rmin).abs() < eps:
            token_weights = torch.ones_like(raw, device=device, dtype=torch.float32)
        else:
            token_weights = (raw - rmin) / (rmax - rmin + eps)
    s = token_weights.sum()
    if s.abs() < eps:
        token_weights = torch.ones_like(token_weights, device=device, dtype=torch.float32) / float(k)
    else:
        token_weights = token_weights / (s + eps)
    full_weights = torch.zeros(m, dtype=torch.float32, device=device)
    full_weights[valid_indices] = token_weights
    sorted_order = torch.argsort(token_weights, descending=True)
    sorted_valid_indices = valid_indices[sorted_order]
    return full_weights, valid_indices, token_weights, sorted_valid_indices


def compute_spatial_consistency(
    cross_attentions: torch.Tensor,
    grid_height: int = DEFAULT_GRID_SIZE,
    grid_width: int = DEFAULT_GRID_SIZE,
    top_k_percent: int = DEFAULT_TOP_K_PERCENT,
) -> float:
    num_patches = grid_height * grid_width
    top_k = max(1, int(num_patches * top_k_percent / 100))
    patch_sets = []
    for t in range(cross_attentions.shape[0]):
        attn = cross_attentions[t]
        top_indices = np.argsort(attn)[-top_k:]
        patch_sets.append(set(top_indices.tolist()))
    if not patch_sets:
        return 0.0
    intersection = patch_sets[0]
    union = patch_sets[0]
    for s in patch_sets[1:]:
        intersection = intersection & s
        union = union | s
    return float(len(intersection) / (len(union) + 1e-8))


def spatial_entropy(attn_map_2d: torch.Tensor) -> Dict[str, Any]:
    s = attn_map_2d
    mean_val = torch.mean(s)
    b = torch.relu(s - mean_val * DEFAULT_SPATIAL_ENTROPY_MULTIPLIER)
    threshold = torch.quantile(b, DEFAULT_SPATIAL_ENTROPY_QUANTILE).item()
    b_np = b.detach().cpu().to(torch.float32).numpy()
    binary = (b_np > threshold).astype(np.int32)
    labeled, num = label(binary, structure=np.ones((3, 3)))
    total = float(b.sum().item())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}
    probs = []
    for i in range(1, num + 1):
        comp_sum = b_np[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)
    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}


def get_spatial_entropy_from_attention(
    c: torch.Tensor,
    grid_height: int = DEFAULT_GRID_SIZE,
    grid_width: int = DEFAULT_GRID_SIZE,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], float, torch.Tensor]:
    se_info = torch.zeros(c.shape[0], dtype=torch.float32)
    se_info_list = []
    for i in range(c.shape[0]):
        visual_tensor = c[i].reshape(grid_height, grid_width)
        se = spatial_entropy(visual_tensor)
        se_value = se["spatial_entropy"]
        if math.isinf(se_value) or math.isnan(se_value) or se_value <= 0:
            se_info[i] = float("inf")
        else:
            se_info[i] = se_value
        se_info_list.append(se)
    valid_se_info = se_info[torch.isfinite(se_info)]
    threshold = elbow_chord(valid_se_info.numpy())
    valid_indices = se_info < threshold
    return se_info, se_info_list, threshold, valid_indices


def row_normalize(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    row_sums = a.sum(dim=1, keepdim=True)
    return a / (row_sums + eps)


def get_token_indices_by_pos_and_words(
    text: str,
    tokenizer: Any,
    lang_model: str = "en_core_web_sm",
    keep_pos: Set[str] = {"NOUN", "ADJ"},
    remove_pos: Set[str] = {"PUNCT", "DET", "CCONJ", "ADV", "X", "SPACE", "ADP"},
    explicit_keep_words: Mapping[str, Any] = {},
    explicit_remove_words: Set[str] = {"think", "answer", "addCriterion", "begin_of_box", "end_of_box"},
) -> Tuple[List[int], List[str]]:
    nlp = spacy.load(lang_model)
    doc = nlp(text)
    if keep_pos:
        selected_words = {token.text for token in doc if token.pos_ in keep_pos}
    elif remove_pos:
        selected_words = {token.text for token in doc if token.pos_ not in remove_pos}
    else:
        selected_words = {token.text for token in doc}
    tokens = tokenizer.tokenize(text)
    keep_indices = []
    keep_tokens = []
    for i, tok in enumerate(tokens):
        clean_tok = tok.lstrip("Ġ▁")
        if clean_tok in explicit_keep_words:
            keep_indices.append(i)
            keep_tokens.append(clean_tok)
        elif clean_tok in selected_words and clean_tok not in explicit_remove_words:
            keep_indices.append(i)
            keep_tokens.append(clean_tok)
    return keep_indices, keep_tokens


def aggregate_llm_attention(attn: Sequence[torch.Tensor]) -> torch.Tensor:
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((torch.tensor([0.0]), attns_per_head[-1][1:].cpu(), torch.tensor([0.0])))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def heterogenous_stack(vecs: Sequence[torch.Tensor]) -> torch.Tensor:
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([torch.concat((v, torch.zeros(max_length - v.shape[0]))) for v in vecs])


def normalize_heatmap(
    vlm_attn_weights: torch.Tensor,
    grid_height: int,
    height: int,
    width: int,
    gamma_factor: float = 1.0,
    grid_width: int = DEFAULT_GRID_SIZE,
) -> np.ndarray:
    vlm_attn_image = vlm_attn_weights.reshape((grid_height, grid_width)).to(torch.float32)
    vlm_attn_image = F.interpolate(vlm_attn_image.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bicubic").squeeze()
    return np.power(vlm_attn_image.numpy(), gamma_factor)


def custom_weighted_sum(filtered_vlm_attn: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    weights = index.view(-1, 1).float()
    num_to_count = weights.sum()
    weighted_attn = filtered_vlm_attn * weights
    result = weighted_attn.sum(dim=0)
    if num_to_count > 0:
        result = result / num_to_count
    result.clamp_(min=0)
    return result


def heatmap_visual(attn_over_image_np: np.ndarray, image: Any, title: str = "Attention heatmap overlay", save_name: str = "image.png") -> Any:
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    ax.imshow(attn_over_image_np, cmap="jet", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_name)
    plt.close()
    return fig


def visual_attn_token2image(
    keep_tokens: Sequence[str],
    filtered_vlm_attn: torch.Tensor,
    save_name: str,
    grid_height: int,
    grid_width: int,
    height: int,
    width: int,
    image: Any,
    summed: Optional[torch.Tensor] = None,
    se_info: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    threshold_se: Optional[float] = None,
    par_info: Optional[torch.Tensor] = None,
    weight_info: Optional[torch.Tensor] = None,
) -> None:
    additional_info = not (summed is None and se_info is None)
    num_tokens = len(keep_tokens)
    grid_cols = 5
    grid_rows = (num_tokens // grid_cols) + (1 if num_tokens % grid_cols != 0 else 0)
    fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    if additional_info:
        fig.suptitle(f"sum-threshold: {threshold*100:.2f}  se-threshold:{threshold_se:.2f}")
    axs = axs.flatten()
    for idx, token in enumerate(keep_tokens):
        attn_weights_over_vis_tokens = filtered_vlm_attn[idx]
        attn_over_image = normalize_heatmap(attn_weights_over_vis_tokens, grid_height, height, width, gamma_factor=1, grid_width=grid_width)
        axs[idx].imshow(image)
        axs[idx].imshow(attn_over_image, cmap="jet", alpha=0.5)
        fontdict = {"fontsize": 8, "color": "red", "weight": "bold"}
        if additional_info:
            if weight_info is None:
                title = f"{token} sum:{summed[idx]*100:.2f} se:{se_info[idx]:.2f}"
                if summed[idx] >= threshold and se_info[idx] <= threshold_se:
                    if par_info[idx] <= DEFAULT_PAR_THRESHOLD:
                        title += "|par"
                    axs[idx].set_title(title, fontdict=fontdict)
                else:
                    axs[idx].set_title(title, fontsize=8)
            else:
                mean = 1 / weight_info.shape[0]
                title = f"{token} sum:{summed[idx]*100:.2f} se:{se_info[idx]:.2f} w: {weight_info[idx]:.2f}"
                if weight_info[idx] > mean:
                    axs[idx].set_title(title, fontdict=fontdict)
                else:
                    axs[idx].set_title(title, fontsize=8)
        else:
            axs[idx].set_title(token, fontsize=8)
        axs[idx].axis("off")
    if additional_info:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_name, format="png", dpi=300)
    plt.close()


def get_saved_attention(output_ids: Mapping[str, Any]) -> Tuple[torch.Tensor, int]:
    aggregated_prompt_attention = []
    for layer in output_ids["attentions"][0]:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.0
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
    llm_attn_matrix = heterogenous_stack([torch.tensor([1])] + list(aggregated_prompt_attention) + list(map(aggregate_llm_attention, output_ids["attentions"])))
    output_token_len = len(output_ids["attentions"])
    return llm_attn_matrix, output_token_len

def get_image_info(image):
    pass

def _resolve_attention_context(
    tokenizer: Any,
    llm_attn_matrix: torch.Tensor,
    sequences: torch.Tensor,
    input_token_len: int,
    output_token_len: int,
    processed_image: Sequence[Any],
    patch_size: int,
    merge_size: int,
    vision_token_id: int,
) -> Tuple[Any, int, int, int, int, int, int, int, List[str], str, str, torch.Tensor]:
    image = processed_image[-1]
    width, height = image.size
    grid_width = int(width / (patch_size * merge_size))
    grid_height = int(height / (patch_size * merge_size))
    num_patches = int(grid_width * grid_height)
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len
    flat_ids = sequences[0, :output_token_start].view(-1)
    vision_token_start = torch.where(flat_ids == vision_token_id)[0][0].item()
    vision_token_end = int(vision_token_start + num_patches)
    token_list = sequences[0, output_token_start:output_token_end]
    token_list_decoded = tokenizer.batch_decode(token_list, skip_special_tokens=True)
    input_text = tokenizer.decode(sequences[0, vision_token_end:output_token_start])
    output_text = tokenizer.decode(token_list, skip_special_tokens=True)
    vlm_attn = llm_attn_matrix[output_token_start:output_token_end, vision_token_start:vision_token_end]
    vlm_attn = row_normalize(vlm_attn)
    return image, width, height, grid_width, grid_height, vision_token_end, output_token_start, output_token_end, token_list_decoded, input_text, output_text, vlm_attn


def _build_masks_and_scores(
    llm_attn_matrix: torch.Tensor,
    vlm_attn: torch.Tensor,
    keep_indices: Sequence[int],
    keep_indices_i: Sequence[int],
    output_token_start: int,
    vision_token_end: int,
    with_tag: bool,
    grid_height: int,
    grid_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config = get_config()
    prompt2text_attn_all = llm_attn_matrix[output_token_start : output_token_start + vlm_attn.shape[0], vision_token_end:output_token_start]
    filtered_prompt2output_text = prompt2text_attn_all[:, keep_indices_i]
    row_idx, col_idx = torch.meshgrid(torch.tensor(keep_indices), torch.tensor(keep_indices_i), indexing="ij")
    _ = prompt2text_attn_all[row_idx, col_idx]
    if with_tag:
        _, _, summed_all, _ = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text,
            0,1
        )
    else:
        _, _, summed_all, _ = get_threshold_and_weight_from_sum(
            filtered_prompt2output_text,
            5,6
        )
    threshold = summed_all[keep_indices].median()
    valid_sum_index_all = summed_all >= threshold
    par_info_all = get_par_from_attention(vlm_attn, config.attention_threshold, grid_height, grid_width)
    valid_par_index_all = par_info_all <= 0.5
    valid_filtered_token = torch.zeros(summed_all.shape[0], dtype=torch.bool)
    valid_filtered_token[keep_indices] = True
    se_info_all, _, _, _ = get_spatial_entropy_from_attention(vlm_attn, grid_height=grid_height, grid_width=grid_width)
    valid_se_isfinite = torch.isfinite(se_info_all)
    se_info_valid_indices = valid_sum_index_all & valid_filtered_token & valid_par_index_all & valid_se_isfinite
    se_info = se_info_all[se_info_valid_indices]
    try:
        threshold_se = elbow_chord(se_info.cpu().numpy())
    except Exception:
        threshold_se = DEFAULT_THRESHOLD_SE_FALLBACK
    valid_se_index_all = se_info_all < threshold_se
    return summed_all, threshold, threshold_se, par_info_all, valid_sum_index_all, valid_par_index_all, valid_filtered_token, se_info_all, valid_se_isfinite, valid_se_index_all


def _progressive_reasoning_mask(valid_filtered_token: torch.Tensor, conditions: Sequence[torch.Tensor]) -> torch.Tensor:
    final_valid_index_reasoning = valid_filtered_token.clone()
    for cond in conditions:
        candidate = final_valid_index_reasoning & cond
        if candidate.sum().item() >= 3:
            final_valid_index_reasoning = candidate
        else:
            break
    return final_valid_index_reasoning


def _select_sc_mask(full_weights: torch.Tensor, sorted_valid_indices: torch.Tensor, final_valid_index_reasoning: torch.Tensor, output_text: str) -> torch.Tensor:
    valid_sc = torch.zeros(final_valid_index_reasoning.shape[0], dtype=torch.bool)
    pred_answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    pred_answer = pred_answer_match.group(1).strip() if pred_answer_match else output_text.strip()
    pred_yes_no_match = re.search(r"(yes|no|Yes|No)", pred_answer)
    pred_yes_no = pred_yes_no_match.group(1) if pred_yes_no_match else ""
    pred_has_anomaly = "yes" in pred_yes_no.lower()
    if pred_has_anomaly:
        if sorted_valid_indices.shape[0] >= DEFAULT_SC_TOP_N:
            valid_sc[sorted_valid_indices[:DEFAULT_SC_TOP_N]] = True
        else:
            valid_sc[sorted_valid_indices[:DEFAULT_SC_FALLBACK_N]] = True
        return valid_sc
    return final_valid_index_reasoning


def get_attention_from_saved_new(
    tokenizer: Any,
    llm_attn_matrix: torch.Tensor,
    sequences: torch.Tensor,
    input_token_len: int,
    output_token_len: int,
    processed_image: Sequence[Any],
    processed_prompt: str,
    return_aggreagate: bool = False,
    patch_size: int = DEFAULT_PATCH_SIZE,
    merge_size: int = DEFAULT_MERGE_SIZE,
    save_name: str = "global_attn_heatmap",
    save_fig: bool = False,
    with_tag: bool = True,
    vision_token_id: int = DEFAULT_VISION_TOKEN_ID,
) -> Tuple[np.ndarray, float]:
    image, width, height, grid_width, grid_height, vision_token_end, output_token_start, output_token_end, token_list_decoded, input_text, output_text, vlm_attn = _resolve_attention_context(
        tokenizer,
        llm_attn_matrix,
        sequences,
        input_token_len,
        output_token_len,
        processed_image,
        patch_size,
        merge_size,
        vision_token_id,
    )
    keep_indices_i, _ = get_token_indices_by_pos_and_words(input_text, tokenizer)
    keep_indices, keep_tokens = get_token_indices_by_pos_and_words(
        output_text,
        tokenizer,
        keep_pos={"NOUN"},
        explicit_remove_words={"defect", "defects", "anomaly", "anomalies", "image", "overview", "analyze", "conclusion", "answer", "think", "Yes", "No"},
    )
    valid_all = torch.ones(vlm_attn.shape[0], dtype=torch.bool)
    global_map = normalize_heatmap(custom_weighted_sum(vlm_attn, valid_all), grid_height, height, width, gamma_factor=1, grid_width=grid_width)
    to_change = "." + save_name.split(".")[-1]
    try:
        summed_all, threshold, threshold_se, par_info_all, valid_sum_index_all, valid_par_index_all, valid_filtered_token, se_info_all, valid_se_isfinite, valid_se_index_all = _build_masks_and_scores(
            llm_attn_matrix,
            vlm_attn,
            keep_indices,
            keep_indices_i,
            output_token_start,
            vision_token_end,
            with_tag,
            grid_height,
            grid_width,
        )
    except Exception:
        if save_fig:
            save_path = save_name.replace(f"{to_change}", "_global_attention.png")
            heatmap_visual(global_map, image, title="original_global_attention", save_name=save_path)
        return global_map, 1.0
    final_valid_index_reasoning = _progressive_reasoning_mask(
        valid_filtered_token,
        [valid_sum_index_all, valid_se_isfinite, valid_par_index_all, valid_se_index_all],
    )
    filtered_map = normalize_heatmap(
        custom_weighted_sum(vlm_attn, final_valid_index_reasoning.to(torch.int)),
        grid_height,
        height,
        width,
        grid_width=grid_width,
    )
    final_index = final_valid_index_reasoning.nonzero(as_tuple=True)[0]
    final_keep_tokens = [token_list_decoded[i] for i in final_index.tolist()]
    full_weights, _, final_token_weights, sorted_valid_indices = get_weight_with_indices(se_info_all, summed_all, final_valid_index_reasoning)
    try:
        valid_sc = _select_sc_mask(full_weights, sorted_valid_indices, final_valid_index_reasoning, output_text)
    except Exception:
        valid_sc = final_valid_index_reasoning
    sc_new = compute_spatial_consistency(vlm_attn[valid_sc][:], grid_height, grid_width, top_k_percent=DEFAULT_TOP_K_PERCENT)
    aggregated_map = normalize_heatmap(
        aggregate_cross_attentions(vlm_attn[final_valid_index_reasoning][:], final_token_weights),
        grid_height,
        height,
        width,
        grid_width=grid_width,
    )
    if save_fig:
        filtered_vlm_attn = vlm_attn[keep_indices][:]
        summed = summed_all[keep_indices]
        se_info = se_info_all[keep_indices]
        visual_attn_token2image(keep_tokens, filtered_vlm_attn, save_name.replace(".png", "_filtered_tokens_attention.png"), grid_height, grid_width, height, width, image, summed, se_info, threshold, threshold_se, par_info_all[keep_indices])
        visual_attn_token2image(final_keep_tokens, vlm_attn[final_valid_index_reasoning][:], save_name.replace(".png", "_final_filtered_attention.png"), grid_height, grid_width, height, width, image, summed_all[final_valid_index_reasoning], se_info_all[final_valid_index_reasoning], threshold, threshold_se, par_info_all[final_valid_index_reasoning])
        if return_aggreagate:
            visual_attn_token2image(final_keep_tokens, vlm_attn[final_valid_index_reasoning][:], save_name.replace(".png", "_final_aggreated_attention.png"), grid_height, grid_width, height, width, image, summed_all[final_valid_index_reasoning], se_info_all[final_valid_index_reasoning], threshold, threshold_se, par_info_all[final_valid_index_reasoning], final_token_weights)
            save_path = save_name.replace(f"{to_change}", "_final_aggreated_image.png")
            heatmap_visual(aggregated_map, image, title=f"final valid image SC: {sc_new:.2f}\n {output_text}", save_name=save_path)
        else:
            save_path = save_name.replace(f"{to_change}", "_final_valid_image.png")
            heatmap_visual(filtered_map, image, title=f"final valid image SC: {sc_new:.2f}", save_name=save_path)
    if return_aggreagate:
        return aggregated_map, sc_new
    return filtered_map, sc_new
