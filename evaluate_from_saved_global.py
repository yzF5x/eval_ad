#!/usr/bin/env python3
"""
evaluate_from_saved_global.py
Second pass: read saved attention files, rebuild processed inputs, and compute segmentation metrics.
"""

import os
import numpy as np
from qwen_vl_utils import process_vision_info
from util import get_attention_from_saved_new, compute_seg_metrics, load_model, resize_image, load_args_from_cli
from util.core import extract_tagged_answer, normalize_yes_no, build_messages
from util.path_builders import PathBuilder
from PIL import Image
import pickle


def _should_save_fig(args, paths: PathBuilder, save_name: str, return_aggregate: bool) -> bool:
    if not args.global_save_fig:
        return False
    visualization_check_path = paths.visualization_check_path(save_name, return_aggregate)
    if os.path.exists(visualization_check_path) and not args.overwrite:
        return False
    return True


def main(args):
    model, processor, tokenizer = load_model(args.model_path)
    paths = PathBuilder(base_dir=args.generated_dir, model_path=args.model_path, replace_path=args.replace_path)
    save_dir = paths.model_output_dir()
    out_path = paths.attention_dir()
    out_img_path = paths.images_dir()
    os.makedirs(out_img_path, exist_ok=True)
    if not os.path.isdir(out_path):
        raise FileNotFoundError(f"Generated dir not found: {out_path}")

    pixel_dct_median = {}
    pixel_dct_median_new = {}
    pixel_dct_median_zero = {}

    files = paths.pth_file_paths(os.listdir(out_path))
    files.sort()
    for fpath in files:
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        llm_attn_matrix = data['llm_attn_matrix']
        sequences = data['sequence']
        meta = data["meta"]
        img_path = meta['image_path']
        question = meta['question']
        category = meta.get('category', '')
        gt_image = meta.get('gt_image', '')
        input_token_len = int(meta.get('input_token_len', 0))
        output_token_len = int(meta.get('output_token_len', 0))
        output_text = meta.get('output_text', '')

        messages = build_messages(img_path, question, args.with_tag)
        image_relative_path = paths.image_relative_path(img_path)
        save_name = paths.visualization_base_path(image_relative_path)
        args.save_fig = _should_save_fig(args, paths, save_name, args.return_aggregate)
        directory = paths.parent_dir(save_name)
        os.makedirs(directory, exist_ok=True)
        processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        merged_patch_size = args.merged_patch_size
        max_size = args.max_size
        processed_image = resize_image(processed_image, max_size, merged_patch_size)
        width, height = processed_image[0].size

        pred_tagged = extract_tagged_answer(output_text)
        pred_yes_or_no = normalize_yes_no(pred_tagged or output_text)
        pred_has_anomaly = pred_yes_or_no == "yes"

        pred_mask_median, sc = get_attention_from_saved_new(
            tokenizer=tokenizer,
            llm_attn_matrix=llm_attn_matrix,
            sequences=sequences,
            input_token_len=input_token_len,
            output_token_len=output_token_len,
            processed_image=processed_image,
            processed_prompt=processed_text,
            save_name=save_name,
            save_fig=args.save_fig,
            with_tag=args.with_tag,
            return_aggreagate=args.return_aggregate,
        )

        try:
            gt_img = Image.open(gt_image)
            gt_img = gt_img.convert('L')
            gt_img = gt_img.resize((width, height),resample=Image.NEAREST)
            gt_mask = (np.array(gt_img) > 128).astype(int)
        except Exception:
            gt_mask = np.zeros((height, width)).astype(int)

        gt_flat = gt_mask.flatten().tolist()
        pred_mask_median_new = pred_mask_median
        pred_mask_median_zero = pred_mask_median
        if pred_has_anomaly:
            if sc >= args.sc_high_threshold and pred_mask_median.max() != 0:
                pred_mask_median_new = pred_mask_median / pred_mask_median.max()
        else:
            if sc < args.sc_low_threshold:
                pred_mask_median_new = np.zeros((height, width), dtype=int)
            if args.normal_set_zero:
                pred_mask_median_zero = np.zeros((height, width), dtype=int)

        pred_flat_median = pred_mask_median.flatten().tolist()
        pred_flat_median_zero = pred_mask_median_zero.flatten().tolist()
        pred_flat_median_new = pred_mask_median_new.flatten().tolist()

        if category not in pixel_dct_median:
            pixel_dct_median[category] = {"pred": [], "true": []}
        if category not in pixel_dct_median_new:
            pixel_dct_median_new[category] = {"pred": [], "true": []}
        if category not in pixel_dct_median_zero:
            pixel_dct_median_zero[category] = {"pred": [], "true": []}
        pixel_dct_median[category]["pred"].append(pred_flat_median)
        pixel_dct_median[category]["true"].append(gt_flat)
        pixel_dct_median_new[category]["pred"].append(pred_flat_median_new)
        pixel_dct_median_new[category]["true"].append(gt_flat)
        pixel_dct_median_zero[category]["pred"].append(pred_flat_median_zero)
        pixel_dct_median_zero[category]["true"].append(gt_flat)

        print(img_path, " finished.")

    seg_metrics_median = compute_seg_metrics(pixel_dct_median)
    seg_metrics_median_zero = compute_seg_metrics(pixel_dct_median_zero)
    seg_metrics_median_new = compute_seg_metrics(pixel_dct_median_new)

    out_model_dir = paths.results_dir()
    os.makedirs(out_model_dir, exist_ok=True)
    metric_paths = paths.segmentation_metric_paths(args.return_aggregate)
    if args.return_aggregate:
        seg_metrics_median.to_excel(metric_paths["median"], index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(metric_paths["median_zero"], index=False, float_format="%.3f")
        seg_metrics_median_new.to_excel(metric_paths["median_new"], index=False, float_format="%.3f")
    else:
        seg_metrics_median.to_excel(metric_paths["median"], index=False, float_format="%.3f")
        seg_metrics_median_zero.to_excel(metric_paths["median_zero"], index=False, float_format="%.3f")
    print("Evaluation complete. Results saved to:", out_model_dir)


if __name__ == "__main__":
    args = load_args_from_cli(section="evaluate_from_saved_global")
    main(args)
