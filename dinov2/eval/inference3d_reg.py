import os
import sys
import json
import torch
from functools import partial
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from pathlib import Path
from monai.inferers import sliding_window_inference

from dinov2.eval.linear3d_class import remove_ddp_wrapper
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate_dict, MultiChannelFeatureModel, ViTAdapterFeatureWrapper
from dinov2.eval.metrics import build_metric, MetricType
from dinov2.data import make_regression_dataset_3d, make_data_loader, SamplerType
from dinov2.data.transforms import make_regression_transform_3d
from dinov2.eval.setup import setup_and_build_model_3d
from dinov2.eval.linear3d_reg import LinearRegressor, AllRegressors, LinearPostprocessor, create_linear_input
from dinov2.eval.linear3d_reg import str2bool
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
import argparse
import json
import torch.nn.functional as F


def load_subject_ids_from_json(dataset_json_path, split="test"):
    with open(dataset_json_path, "r") as f:
        data = json.load(f)
    # return ["_".join(Path(entry["label"]).stem.split("_")[1:]) for entry in data[split]]
    return [Path(entry["label"]).stem.split("_")[1] for entry in data[split]]


def run_inference(args):
    # === 1. Setup model
    model, autocast_dtype = setup_and_build_model_3d(args)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    # === 2. Dataset and transforms
    _, val_transform = make_regression_transform_3d(args.dataset_name, args.image_size, min_int=-1.0, resize_scale=args.resize_scale)
    _, val_dataset, test_dataset, input_channels, num_outputs = make_regression_dataset_3d(
        dataset_name=args.dataset_name,
        dataset_percent=args.dataset_percent,
        base_directory=args.base_data_dir,
        train_transforms=None,
        val_transforms=val_transform,
        cache_path=args.cache_dir,
        dataset_seed=args.dataset_seed,
    )

    if args.use_validation:
        print(f"[!] Using validation set for inference: {len(val_dataset)} samples.")
        test_dataset = val_dataset
    
    test_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )

    # === 3. Load feature extractor
    feature_model = ViTAdapterFeatureWrapper(
        vit_model=model,
        input_channels=input_channels,
        n_last_blocks=4,
        autocast_ctx=autocast_ctx,
    )
    feature_model.eval()
    feature_model.cuda()

    # === 4. Create and load regressors
    sample_output = feature_model(test_dataset[0]['image'].unsqueeze(0).cuda())
    linear_regressors = AllRegressors(torch.nn.ModuleDict())  # Dummy container for Checkpointer
    optimizer = torch.optim.SGD([torch.tensor(0.0, requires_grad=True)], lr=1e-3)  # Dummy optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)  # Dummy scheduler
    checkpointer = Checkpointer(linear_regressors, args.output_dir, optimizer=optimizer, scheduler=scheduler)
    ckpt = checkpointer.resume_or_load(f"{args.output_dir}/best_val.pth", resume=False)
    # json_filename_results = os.path.join(f"{args.output_dir}, results_eval_regression.json)

    # Try to read best classifier from checkpoint
    best_regressor_name = ckpt.get("iteration_metadata", {}).get("best_regressor_name")
    # assert best_regressor_name, "Best regressor name not found in checkpoint metadata."    

    # Fallback: Read from JSON file
    if not best_regressor_name:
        try:
            metrics_path = os.path.join(args.output_dir, "results_eval_regression.json")
            with open(metrics_path, "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.strip().startswith("{\"best_regressor\""):
                        best_regressor_name = json.loads(line)["best_regressor"]["name"]
                        break
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve best regressor from JSON: {e}")

    assert best_regressor_name, "Best regressor name not found in checkpoint or metrics file."
    print(f"Using best regressor: {best_regressor_name}")

    # Now load the actual regressor
    n_blocks = 1 if "1_blocks" in best_regressor_name else 4
    avgpool = "avgpool_True" in best_regressor_name
    out_dim = create_linear_input(sample_output, use_n_blocks=n_blocks, use_avgpool=avgpool).shape[1]
    regressor = LinearRegressor(out_dim, n_blocks, avgpool, num_outputs).cuda()    
    regressor_ckpt = torch.load(f"{args.output_dir}/best_val.pth", map_location="cuda")

    regressor.load_state_dict({
        k.replace(f"regressors_dict.{best_regressor_name}.", ""): v
        for k, v in regressor_ckpt["model"].items()
        if k.startswith(f"regressors_dict.{best_regressor_name}.")
    })    
    # regressor.load_state_dict(regressor_ckpt["model"]["regressor." + best_regressor_name])

    postprocessor = LinearPostprocessor(regressor.eval().cuda())

    # === 5. Run inference
    # postprocessors = {k: LinearPostprocessor(v) for k, v in linear_regressors.regressors_dict.items()}
    # metrics = {k: metric.clone() for k in linear_regressors.regressors_dict}
    # metric = build_metric(MetricType.MEAN_ACCURACY, num_classes=num_outputs, ks=(1,))
    metric = build_metric(MetricType.MEAN_ABSOLUTE_ERROR)
    _, _, pred_dict = evaluate_dict(
        feature_model,
        test_loader,
        {best_regressor_name: postprocessor},  # postprocessors
        {best_regressor_name: metric},  # metrics
        torch.cuda.current_device(),
        return_preds=True
    )
    preds = pred_dict[best_regressor_name]

    # === 6. Save predictions
    output_dir = os.path.join(args.output_dir, "predictions_eval_format")
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_json_path = os.path.join(args.base_data_dir, f"{args.dataset_name}.json")
    subject_ids = load_subject_ids_from_json(dataset_json_path, split="validation" if args.use_validation else "test")


    for i, (score, label) in enumerate(zip(preds["preds"], preds["target"])):
        subject_id = subject_ids[i]

        subject_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        # Apply softmax to get probabilities
        # probabilities = F.softmax(score, dim=-1)

        # prob. of positive class
        # pred_score = probabilities[1].item() if len(probabilities) > 1 else probabilities[0].item()
        # Just a float
        pred_score = score.item() if isinstance(score, torch.Tensor) else score
        with open(os.path.join(subject_dir, "prediction.txt"), "w") as f:
            f.write(f"{pred_score:.6f}")

        # Label
        label = label.item() if isinstance(label, torch.Tensor) else label
        with open(os.path.join(subject_dir, "label.txt"), "w") as f:
            f.write(f"{label:.6f}\n")

    print(f"[✓] Inference complete. Predictions saved to: {output_dir}")


def get_args():
    parents = []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]

    parser = argparse.ArgumentParser(parents=parents)
    # parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--dataset-percent", type=int, default=100, help="Percentage of the dataset to use")
    parser.add_argument("--base-data-dir", type=str, required=True, help="Base directory for the dataset")
    parser.add_argument("--cache-dir", type=str, required=True, help="Directory to cache dataset files")
    parser.add_argument("--image-size", type=int, default=112, help="Size of the input images")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--dataset-seed", type=int, default=0, help="Seed for dataset shuffling")
    parser.add_argument("--resize-scale", type=float, default=1.0, help="Resize scale for images")
    # parser.add_argument("--pretrained-weights", type=str, required=True, help="Path to the pretrained model weights")
    parser.add_argument("--use_validation", type=str2bool, default=False, help="Use validation set for inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()    
    run_inference(args)
