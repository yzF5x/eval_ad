import json

from util.core import extract_tagged_answer, normalize_yes_no, send2api
from util.metrics import compute_classify_matrics
from util import load_args_from_cli
from util.path_builders import PathBuilder


if __name__ == '__main__':
    args = load_args_from_cli(section="calculate_metrics")
    paths = PathBuilder(base_dir=args.save_dir, model_path=args.model_path)
    args.orig_result_path = paths.result_json_path()
    orig_result_path = args.orig_result_path
    with open(orig_result_path, 'r') as f:
        results = json.load(f)
    anomaly_dct = {}
    idx = 0
    for _, v in results.items():
        gt_answer = 0 if 'good' in v['id'] else 1
        if not anomaly_dct.get(v['category']):
            anomaly_dct[v['category']] = {"pred": [], "true": []}

        pred_tagged = extract_tagged_answer(v.get("pred_reasoning", ""))
        pred_yes_or_no = normalize_yes_no(pred_tagged or v.get("pred_reasoning", ""))
        if not pred_yes_or_no:
            pred_yes_or_no = normalize_yes_no(send2api(v.get("pred_reasoning", "")).strip())
            idx += 1
            print(pred_yes_or_no)
        pred_answer = 1 if pred_yes_or_no == 'yes' else 0

        anomaly_dct[v['category']]["pred"].append(pred_answer)
        anomaly_dct[v['category']]["true"].append(gt_answer)

    anomaly_scores = compute_classify_matrics(anomaly_dct)
    path = paths.anomaly_score_path(orig_result_path)
    anomaly_scores.to_excel(path, index=False, float_format='%.3f')
    print(idx)
