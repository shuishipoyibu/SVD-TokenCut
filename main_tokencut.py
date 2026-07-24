"""
Main experiment file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import os
import argparse
import random
import pickle
import subprocess
import sys

import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_img, visualize_eigvec, visualize_predictions, visualize_predictions_gt 
from object_discovery import ncut 
from pca import center_features, project_features_pca, project_features_svd
from cosine_svd import project_features_cosine_svd
from random_projection import project_features_random
from diagnostics import (
    CoreDiagnosticsAccumulator,
    compare_cosine_graphs,
    compute_cosine_similarity,
    compute_energy_retention,
)
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "moco_vit_small",
            "moco_vit_base",
            "mae_vit_base",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    
    parser.add_argument(
        "--save-feat-dir",
        type=str,
        default=None,
        help="if save-feat-dir is not None, only computing features and save it into save-feat-dir",
    )
    
    parser.add_argument(
        "--load-feat-dir",
        type=str,
        default=None,
        help="load cached features from this directory and skip DINO feature extraction",
    )
    
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="apply per-image centered PCA before TokenCut",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=None,
        help="target centered PCA dimension; requires --use-pca",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="run a centered PCA sweep, for example --pca-dims 8 16 32 64",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help="subtract the per-image patch mean without reducing dimensions",
    )
    parser.add_argument(
        "--use-svd",
        action="store_true",
        help="apply per-image uncentered SVD before TokenCut",
    )
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=None,
        help="target uncentered SVD dimension; requires --use-svd",
    )
    parser.add_argument(
        "--svd-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="run an uncentered SVD sweep, for example --svd-dims 8 16 32 64",
    )
    
    parser.add_argument(
        "--use-cosine-svd",
        action="store_true",
        help="apply per-image SVD to L2-normalized patch keys",
    )
    parser.add_argument(
        "--cosine-svd-dim",
        type=int,
        default=None,
        help="target Cosine-SVD dimension; requires --use-cosine-svd",
    )
    parser.add_argument(
        "--cosine-svd-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="run a Cosine-SVD sweep, for example --cosine-svd-dims 8 16 32 64",
    )
    parser.add_argument(
        "--use-random-projection",
        action="store_true",
        help="apply a seeded Gaussian random projection before TokenCut",
    )
    parser.add_argument(
        "--random-projection-dim",
        type=int,
        default=None,
        help="target random-projection dimension; requires --use-random-projection",
    )
    parser.add_argument(
        "--random-projection-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help=(
            "run a Gaussian random-projection sweep, for example "
            "--random-projection-dims 8 16 32 64"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="seed for the Gaussian random projection matrix",
    )
    
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and 
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory to store predictions and visualizations."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="optional subdirectory used to isolate one run or reduction sweep",
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["attn", "pred", "all", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # TokenCut parameters
    parser.add_argument(
        "--which_features",
        type=str,
        default="k",
        choices=["k", "q", "v"],
        help="Which features to use",
    )
    parser.add_argument(
        "--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered."
    )
    parser.add_argument("--resize", type=int, default=None, help="Resize input image to fix size")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="only process the first N images from the dataset",
    )
    parser.add_argument("--tau", type=float, default=0.2, help="Tau for seperating the Graph.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps for defining the Graph.")
    parser.add_argument("--no-binary-graph", action="store_true", default=False, help="Generate a binary graph where edge of the Graph will binary. Or using similarity score as edge weight.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    args = parser.parse_args()

    
    if args.run_name is not None:
        if (
            not args.run_name
            or args.run_name in {".", ".."}
            or os.path.basename(args.run_name) != args.run_name
        ):
            raise ValueError("--run-name must be a single directory name")

    if args.cosine_svd_dims is not None:
        conflicting = (
            args.center_only
            or args.use_pca
            or args.pca_dim is not None
            or args.pca_dims is not None
            or args.use_svd
            or args.svd_dim is not None
            or args.svd_dims is not None
            or args.use_cosine_svd
            or args.cosine_svd_dim is not None
            or args.use_random_projection
            or args.random_projection_dim is not None
            or args.random_projection_dims is not None
        )
        if conflicting:
            raise ValueError(
                "--cosine-svd-dims cannot be combined with another reduction mode"
            )
        if args.dinoseg:
            raise ValueError("--cosine-svd-dims cannot be used with --dinoseg")
        if args.save_feat_dir is not None:
            raise ValueError(
                "--cosine-svd-dims cannot be used with --save-feat-dir"
            )
        if any(dim <= 0 for dim in args.cosine_svd_dims):
            raise ValueError(
                "every value passed to --cosine-svd-dims must be positive"
            )
        if len(set(args.cosine_svd_dims)) != len(args.cosine_svd_dims):
            raise ValueError("--cosine-svd-dims cannot contain duplicate dimensions")

        sweep_name = args.run_name or datetime.datetime.now().strftime(
            "cosine_svd_sweep_%Y%m%d-%H%M%S-%f"
        )
        forwarded_args = []
        index = 1
        while index < len(sys.argv):
            if sys.argv[index] == "--cosine-svd-dims":
                index += 1
                while index < len(sys.argv) and not sys.argv[index].startswith("--"):
                    index += 1
                continue
            forwarded_args.append(sys.argv[index])
            index += 1
        if args.run_name is None:
            forwarded_args.extend(["--run-name", sweep_name])

        print(
            f"Running Cosine-SVD sweep {sweep_name}: {args.cosine_svd_dims}",
            flush=True,
        )
        for position, dim in enumerate(args.cosine_svd_dims, start=1):
            print(
                f"\n[Cosine-SVD sweep {position}/{len(args.cosine_svd_dims)}] "
                f"dimension={dim}",
                flush=True,
            )
            command = [
                sys.executable,
                os.path.abspath(__file__),
                *forwarded_args,
                "--use-cosine-svd",
                "--cosine-svd-dim",
                str(dim),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
        print(f"Cosine-SVD sweep completed: {sweep_name}", flush=True)
        raise SystemExit(0)

    if args.random_projection_dims is not None:
        conflicting = (
            args.center_only
            or args.use_pca
            or args.pca_dim is not None
            or args.pca_dims is not None
            or args.use_svd
            or args.svd_dim is not None
            or args.svd_dims is not None
            or args.use_cosine_svd
            or args.cosine_svd_dim is not None
            or args.cosine_svd_dims is not None
            or args.use_random_projection
            or args.random_projection_dim is not None
        )
        if conflicting:
            raise ValueError(
                "--random-projection-dims cannot be combined with another "
                "reduction mode"
            )
        if args.dinoseg:
            raise ValueError(
                "--random-projection-dims cannot be used with --dinoseg"
            )
        if args.save_feat_dir is not None:
            raise ValueError(
                "--random-projection-dims cannot be used with --save-feat-dir"
            )
        if any(dim <= 0 for dim in args.random_projection_dims):
            raise ValueError(
                "every value passed to --random-projection-dims must be positive"
            )
        if len(set(args.random_projection_dims)) != len(
            args.random_projection_dims
        ):
            raise ValueError(
                "--random-projection-dims cannot contain duplicate dimensions"
            )

        sweep_name = args.run_name or datetime.datetime.now().strftime(
            "random_projection_sweep_%Y%m%d-%H%M%S-%f"
        )
        forwarded_args = []
        index = 1
        while index < len(sys.argv):
            if sys.argv[index] == "--random-projection-dims":
                index += 1
                while index < len(sys.argv) and not sys.argv[index].startswith("--"):
                    index += 1
                continue
            forwarded_args.append(sys.argv[index])
            index += 1
        if args.run_name is None:
            forwarded_args.extend(["--run-name", sweep_name])

        print(
            "Running Gaussian random-projection sweep "
            f"{sweep_name}: {args.random_projection_dims}, seed={args.random_seed}",
            flush=True,
        )
        for position, dim in enumerate(args.random_projection_dims, start=1):
            print(
                f"\n[Random-projection sweep "
                f"{position}/{len(args.random_projection_dims)}] dimension={dim}",
                flush=True,
            )
            command = [
                sys.executable,
                os.path.abspath(__file__),
                *forwarded_args,
                "--use-random-projection",
                "--random-projection-dim",
                str(dim),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
        print(f"Random-projection sweep completed: {sweep_name}", flush=True)
        raise SystemExit(0)

    if args.svd_dims is not None:
        if (
            args.center_only
            or args.use_pca
            or args.pca_dim is not None
            or args.pca_dims is not None
            or args.use_svd
            or args.svd_dim is not None
            or args.use_cosine_svd
            or args.cosine_svd_dim is not None
            or args.cosine_svd_dims is not None
            or args.use_random_projection
            or args.random_projection_dim is not None
            or args.random_projection_dims is not None
        ):
            raise ValueError(
                "--svd-dims cannot be combined with center-only, PCA, "
                "--use-svd, or --svd-dim"
            )
        if args.dinoseg:
            raise ValueError("--svd-dims cannot be used with --dinoseg")
        if args.save_feat_dir is not None:
            raise ValueError("--svd-dims cannot be used with --save-feat-dir")
        if any(dim <= 0 for dim in args.svd_dims):
            raise ValueError("every value passed to --svd-dims must be positive")
        if len(set(args.svd_dims)) != len(args.svd_dims):
            raise ValueError("--svd-dims cannot contain duplicate dimensions")

        sweep_name = args.run_name or datetime.datetime.now().strftime(
            "svd_sweep_%Y%m%d-%H%M%S-%f"
        )
        forwarded_args = []
        index = 1
        while index < len(sys.argv):
            if sys.argv[index] == "--svd-dims":
                index += 1
                while index < len(sys.argv) and not sys.argv[index].startswith("--"):
                    index += 1
                continue
            forwarded_args.append(sys.argv[index])
            index += 1
        if args.run_name is None:
            forwarded_args.extend(["--run-name", sweep_name])

        print(f"Running uncentered SVD sweep {sweep_name}: {args.svd_dims}", flush=True)
        for position, dim in enumerate(args.svd_dims, start=1):
            print(
                f"\n[SVD sweep {position}/{len(args.svd_dims)}] dimension={dim}",
                flush=True,
            )
            command = [
                sys.executable,
                os.path.abspath(__file__),
                *forwarded_args,
                "--use-svd",
                "--svd-dim",
                str(dim),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
        print(f"Uncentered SVD sweep completed: {sweep_name}", flush=True)
        raise SystemExit(0)

    if args.pca_dims is not None:
        if (
            args.center_only
            or args.use_pca
            or args.pca_dim is not None
            or args.use_svd
            or args.svd_dim is not None
            or args.svd_dims is not None
            or args.use_cosine_svd
            or args.cosine_svd_dim is not None
            or args.cosine_svd_dims is not None
            or args.use_random_projection
            or args.random_projection_dim is not None
            or args.random_projection_dims is not None
        ):
            raise ValueError(
                "--pca-dims cannot be combined with center-only, SVD, "
                "--use-pca, or --pca-dim"
            )
        if args.dinoseg:
            raise ValueError("--pca-dims cannot be used with --dinoseg")
        if args.save_feat_dir is not None:
            raise ValueError("--pca-dims cannot be used with --save-feat-dir")
        if any(dim <= 0 for dim in args.pca_dims):
            raise ValueError("every value passed to --pca-dims must be positive")
        if len(set(args.pca_dims)) != len(args.pca_dims):
            raise ValueError("--pca-dims cannot contain duplicate dimensions")

        sweep_name = args.run_name or datetime.datetime.now().strftime(
            "pca_sweep_%Y%m%d-%H%M%S-%f"
        )
        forwarded_args = []
        index = 1
        while index < len(sys.argv):
            if sys.argv[index] == "--pca-dims":
                index += 1
                while index < len(sys.argv) and not sys.argv[index].startswith("--"):
                    index += 1
                continue
            forwarded_args.append(sys.argv[index])
            index += 1
        if args.run_name is None:
            forwarded_args.extend(["--run-name", sweep_name])

        print(f"Running PCA sweep {sweep_name}: {args.pca_dims}", flush=True)
        for position, dim in enumerate(args.pca_dims, start=1):
            print(
                f"\n[PCA sweep {position}/{len(args.pca_dims)}] dimension={dim}",
                flush=True,
            )
            command = [
                sys.executable,
                os.path.abspath(__file__),
                *forwarded_args,
                "--use-pca",
                "--pca-dim",
                str(dim),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
        print(f"PCA sweep completed: {sweep_name}", flush=True)
        raise SystemExit(0)

    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be a positive integer")
    if args.save_feat_dir is not None and args.load_feat_dir is not None:
        raise ValueError("--save-feat-dir and --load-feat-dir cannot be used together")
    if args.load_feat_dir is not None and args.dinoseg:
        raise ValueError("--load-feat-dir can only be used with TokenCut features, not --dinoseg")
    if args.use_pca and args.pca_dim is None:
        raise ValueError("--use-pca requires --pca-dim")
    if args.pca_dim is not None and not args.use_pca:
        raise ValueError("--pca-dim requires --use-pca")
    if args.pca_dim is not None and args.pca_dim <= 0:
        raise ValueError("--pca-dim must be a positive integer")
    if args.use_svd and args.svd_dim is None:
        raise ValueError("--use-svd requires --svd-dim")
    if args.svd_dim is not None and not args.use_svd:
        raise ValueError("--svd-dim requires --use-svd")
    if args.svd_dim is not None and args.svd_dim <= 0:
        raise ValueError("--svd-dim must be a positive integer")
    if args.use_cosine_svd and args.cosine_svd_dim is None:
        raise ValueError("--use-cosine-svd requires --cosine-svd-dim")
    if args.cosine_svd_dim is not None and not args.use_cosine_svd:
        raise ValueError("--cosine-svd-dim requires --use-cosine-svd")
    if args.cosine_svd_dim is not None and args.cosine_svd_dim <= 0:
        raise ValueError("--cosine-svd-dim must be a positive integer")
    if (
        args.use_random_projection
        and args.random_projection_dim is None
    ):
        raise ValueError(
            "--use-random-projection requires --random-projection-dim"
        )
    if (
        args.random_projection_dim is not None
        and not args.use_random_projection
    ):
        raise ValueError(
            "--random-projection-dim requires --use-random-projection"
        )
    if (
        args.random_projection_dim is not None
        and args.random_projection_dim <= 0
    ):
        raise ValueError("--random-projection-dim must be a positive integer")
    active_reductions = (
        args.center_only,
        args.use_pca,
        args.use_svd,
        args.use_cosine_svd,
        args.use_random_projection,
    )
    if sum(active_reductions) > 1:
        raise ValueError("feature reduction modes are mutually exclusive")
    if any(active_reductions) and args.dinoseg:
        raise ValueError("Feature reduction modes are only supported for TokenCut")
    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path, args.resize)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device('cuda') 
    model = None if args.load_feat_dir is not None else get_model(args.arch, args.patch_size, device)

    # Register once so every forward pass invokes only one qkv hook.
    feat_out = {}
    qkv_hook_handle = None
    if args.load_feat_dir is None and not args.dinoseg and "vit" in args.arch:
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output

        qkv_module = model._modules["blocks"][-1]._modules["attn"]._modules["qkv"]
        qkv_hook_handle = qkv_module.register_forward_hook(hook_fn_forward_qkv)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.run_name is not None:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with TokenCut 
        exp_name = f"TokenCut-{args.arch}"
        if "vit" in args.arch:
            exp_name += f"{args.patch_size}_{args.which_features}"
        if args.center_only:
            exp_name += "_center"
        elif args.use_pca:
            exp_name += f"_pca{args.pca_dim}"
        elif args.use_svd:
            exp_name += f"_svd{args.svd_dim}"
        elif args.use_cosine_svd:
            exp_name += f"_cosine_svd{args.cosine_svd_dim}"
        elif args.use_random_projection:
            exp_name += (
                f"_random_projection{args.random_projection_dim}"
                f"_seed{args.random_seed}"
            )

    print(f"Running TokenCut on the dataset {dataset.name} (exp: {exp_name})")

    # Visualization 
    if args.visualize:
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        
    if args.save_feat_dir is not None : 
        os.makedirs(args.save_feat_dir, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    total_images = len(dataset.dataloader)
    if args.max_images is not None:
        total_images = min(args.max_images, total_images)
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(total_images)
    
    segmentation_time_seconds = 0.0
    timed_images = 0
    skipped_images = 0
    core_diagnostics = CoreDiagnosticsAccumulator()
    pbar = tqdm(range(total_images))
    for im_id in pbar:
        inp = dataset.dataloader[im_id]

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]

        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1], im_id)
        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # # Move to gpu
        if device == torch.device('cuda'):
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        
        feat_name = im_name.replace(".jpg", ".npy").replace(".jpeg", ".npy").replace(".png", ".npy")
# ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no usable gt annotations.
                if gt_bbxs.shape[0] == 0:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():

            # ------------ FORWARD PASS -------------------------------------------
            if args.load_feat_dir is not None:
                feat_path = os.path.join(args.load_feat_dir, feat_name)
                if not os.path.isfile(feat_path):
                    raise FileNotFoundError(
                        f"Cached feature not found for {im_name}: {feat_path}"
                    )
                feats = torch.from_numpy(np.load(feat_path, allow_pickle=False)).to(device)
                scales = [args.patch_size, args.patch_size]
            elif "vit"  in args.arch:
                # Discard the preceding image's captured qkv output.
                if not args.dinoseg:
                    feat_out.clear()

                # Forward pass in the model
                attentions = model.get_last_selfattention(img[None, :, :, :])

                # Scaling factor
                scales = [args.patch_size, args.patch_size]

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Baseline: compute DINO segmentation technique proposed in the DINO paper
                # and select the biggest component
                if args.dinoseg:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    segmentation_start = time.perf_counter()
                    pred = dino_seg(attentions, (w_featmap, h_featmap), args.patch_size, head=args.dinoseg_head)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    segmentation_time_seconds += (
                        time.perf_counter() - segmentation_start
                    )
                    timed_images += 1
                    pred = np.asarray(pred)
                else:
                    # Extract the qkv features of the last attention layer
                    qkv = (
                        feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
                    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

                    # Modality selection
                    if args.which_features == "k":
                        #feats = k[:, 1:, :]
                        feats = k
                    elif args.which_features == "q":
                        #feats = q[:, 1:, :]
                        feats = q
                    elif args.which_features == "v":
                        #feats = v[:, 1:, :]
                        feats = v
                        
                    if args.save_feat_dir is not None : 
                        np.save(os.path.join(args.save_feat_dir, feat_name), feats.cpu().numpy())
                        continue

            else:
                raise ValueError("Unknown model.")

        # ------------ Apply TokenCut ------------------------------------------- 
        if not args.dinoseg:
            original_feats = feats
            singular_values = None
            energy_target_dim = None
            total_energy = None
            # Start after DINO features are available. This includes any optional
            # feature transformation and the complete TokenCut segmentation.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            segmentation_start = time.perf_counter()

            try:
                if args.center_only:
                    feats = center_features(feats)
                elif args.use_pca:
                    patches = original_feats[:, 1:, :]
                    centered = patches - patches.mean(dim=1, keepdim=True)
                    total_energy = centered.square().sum()
                    feats, singular_values = project_features_pca(
                        feats, args.pca_dim, return_singular_values=True
                    )
                    energy_target_dim = args.pca_dim
                elif args.use_svd:
                    total_energy = original_feats[:, 1:, :].square().sum()
                    feats, singular_values = project_features_svd(
                        feats, args.svd_dim, return_singular_values=True
                    )
                    energy_target_dim = args.svd_dim
                elif args.use_cosine_svd:
                    normalized = F.normalize(original_feats[:, 1:, :], p=2, dim=2)
                    total_energy = normalized.square().sum()
                    feats, singular_values = project_features_cosine_svd(
                        feats,
                        args.cosine_svd_dim,
                        return_singular_values=True,
                    )
                    energy_target_dim = args.cosine_svd_dim
                elif args.use_random_projection:
                    feats = project_features_random(
                        feats,
                        args.random_projection_dim,
                        seed=args.random_seed,
                    )
            except ValueError as exc:
                message = str(exc)
                if "supports dimensions from" not in message:
                    raise
                skipped_images += 1
                pbar.write(f"Skipping {im_name}: {message}")
                continue

            pred, objects, foreground, seed, bins, eigenvector, ncut_diagnostics = ncut(
                feats,
                [w_featmap, h_featmap],
                scales,
                init_image_size,
                args.tau,
                args.eps,
                im_name=im_name,
                no_binary_graph=args.no_binary_graph,
                return_diagnostics=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            segmentation_time_seconds += time.perf_counter() - segmentation_start
            timed_images += 1

            if not any(
                (
                    args.center_only,
                    args.use_pca,
                    args.use_svd,
                    args.use_cosine_svd,
                    args.use_random_projection,
                )
            ):
                energy_retention = 1.0
                e_cos = 0.0
                q_r = 1.0
            else:
                reference_similarity = compute_cosine_similarity(original_feats)
                e_cos, q_r = compare_cosine_graphs(
                    reference_similarity,
                    ncut_diagnostics["similarity"],
                    args.tau,
                )
                energy_retention = (
                    compute_energy_retention(singular_values, energy_target_dim, total_energy)
                    if singular_values is not None
                    else None
                )
            core_diagnostics.add(
                energy_retention,
                e_cos,
                q_r,
                ncut_diagnostics["lambda2"],
                ncut_diagnostics["spectral_gap"],
            )
            
            if args.visualize == "pred" and args.no_evaluation :
                image = dataset.load_image(im_name, size_im)
                visualize_predictions(image, pred, vis_folder, im_name)
            if args.visualize == "attn" and args.no_evaluation:
                visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
            if args.visualize == "all" and args.no_evaluation:
                image = dataset.load_image(im_name, size_im)
                visualize_predictions(image, pred, vis_folder, im_name)
                visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)
                        
        # ------------ Visualizations -------------------------------------------
        # Save the prediction
        preds_dict[im_name] = pred

        # Evaluation
        if args.no_evaluation:
            continue

        # Compare prediction to GT boxes
        ious = bbox_iou(torch.from_numpy(pred), torch.from_numpy(gt_bbxs))
        
        if torch.any(ious >= 0.5):
            corloc[im_id] = 1
        vis_folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)
        image = dataset.load_image(im_name)
        #visualize_predictions(image, pred, vis_folder, im_name)
        #visualize_eigvec(eigenvector, vis_folder, im_name, [w_featmap, h_featmap], scales)

        cnt += 1
        if cnt % 50 == 0:
            pbar.set_description(f"Found {int(np.sum(corloc))}/{cnt}")

    if qkv_hook_handle is not None:
        qkv_hook_handle.remove()
    average_time_seconds = (
        segmentation_time_seconds / timed_images if timed_images else 0.0
    )
    formatted_time = str(
        datetime.timedelta(milliseconds=int(segmentation_time_seconds * 1000))
    )
    print(
        f"Segmentation time (features ready -> segmentation complete): "
        f"{formatted_time} ({segmentation_time_seconds:.6f} seconds, "
        f"{average_time_seconds:.6f} seconds/image)"
    )

    folder = f"{args.output_dir}/{exp_name}"
    os.makedirs(folder, exist_ok=True)

    # Save predicted bounding boxes
    if args.save_predictions:
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    result_lines = [
        f"segmentation_time_seconds,{segmentation_time_seconds:.6f},,\n",
        f"timed_images,{timed_images},,\n",
        f"skipped_images,{skipped_images},,\n",
        f"segmentation_time_seconds_per_image,{average_time_seconds:.6f},,\n",
    ]
    result_lines.extend(core_diagnostics.result_lines())

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_lines.insert(0, "corloc,%.1f,,\n" % (100*np.sum(corloc)/cnt))

    result_file = os.path.join(folder, "results.txt")
    with open(result_file, "w") as f:
        f.writelines(result_lines)
    print("File saved at %s" % result_file)
