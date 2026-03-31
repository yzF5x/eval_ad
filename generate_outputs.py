#!/usr/bin/env python3
"""
generate_outputs.py
First pass: iterate images, run model.generate(...), and save outputs and metadata to disk.
"""

import os
import json
from qwen_vl_utils import process_vision_info
from util import (
    load_args_from_cli,
    resize_image,
    toliststr,
    load_dataset,
    use_monkey_patch_qwen2_5vl_qkvfp32_eager_visionattn,
    use_monkey_patch_qwen2_5vl_qkvfp32_eager_encoderselfattn,
    load_model,
    get_saved_attention,
)
from util.core import build_messages, move_to_cpu
from util.path_builders import PathBuilder
import torch
import pickle

# output_path = "../results/1013-median/MVTecAD_seg_0shot/"


def main(args):
    ret = {}
    if args.use_qkvfp32_monkey_patch:
        use_monkey_patch_qwen2_5vl_qkvfp32_eager_visionattn()
        use_monkey_patch_qwen2_5vl_qkvfp32_eager_encoderselfattn()

    eval_dataset = load_dataset(args.dataset_path)
    model, processor, _ = load_model(args.model_path)
    paths = PathBuilder(base_dir=args.save_dir, model_path=args.model_path, replace_path=args.replace_path)
    save_dir = paths.model_output_dir()
    os.makedirs(save_dir, exist_ok=True)
    out_path = paths.attention_dir()
    os.makedirs(out_path, exist_ok=True)
    for _, data in eval_dataset.iterrows():
        img_path = toliststr(data["image_path"])[0]
        messages = build_messages(img_path, data["question"], args.with_tag)
        processed_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed_image, _ = process_vision_info(messages)
        merged_patch_size = args.merged_patch_size
        max_size = args.max_size
        processed_image = resize_image(processed_image, max_size, merged_patch_size)

        inputs = processor(
            text=[processed_text],
            images=processed_image,
            return_tensors="pt",
        ).to(model.device)

        # generate
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=args.save_attentions,  # if we want attentions, set True
            output_attentions=args.save_attentions,
        )

        sequences = generated["sequences"]
        input_token_len = len(inputs.input_ids[0])
        trimmed = sequences[0][input_token_len:]
        output_text = processor.tokenizer.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        image_relative_path = paths.image_relative_path(img_path)
        save_path = paths.attention_file_path(image_relative_path)
        ret[image_relative_path] = {
            "id": image_relative_path,
            "category": data['category'],
            "pred_reasoning": output_text,
            "gt_reasoning": data['answer'],
        }
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Skipping existing {save_path}")
            continue
        else:
            llm_attn_matrix, output_token_len = get_saved_attention(output_ids=generated)
        meta = {
            "image_path": img_path,
            "question": data["question"],
            "category": data.get("category", ""),
            "gt_image": data.get("gt_image", ""),
            "input_token_len": input_token_len,
            "output_text": output_text,
            'output_token_len': output_token_len,
        }

        save_dict = {
            "llm_attn_matrix": llm_attn_matrix,
            "sequence": sequences,
            "meta": meta,
        }
        save_dict_cpu = move_to_cpu(save_dict)
        with open(save_path, "wb") as f:
            pickle.dump(save_dict_cpu, f)
    results_dir = paths.results_dir()
    os.makedirs(results_dir, exist_ok=True)
    json_path = paths.result_json_path()
    with open(json_path, 'w') as f:
        json.dump(ret, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = load_args_from_cli(section="generate_outputs")
    main(args)
