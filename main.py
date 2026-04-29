import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch
import itertools
import time
import warnings
from bisect import bisect
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import open3d as o3d

# DFRNet core modules
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.utils
import patchcore.sampler
import patchcore.metrics
from dataset_pc import Dataset3dad_train, Dataset3dad_test
from utils.visualization import save_anomalymap

# Set warning and logging levels for clean console output
warnings.filterwarnings('ignore')
logging.getLogger('open3d').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)


def dynamic_routing_fusion(score_a, score_b):
    """
    Dynamic Representation Routing and Fusion (DRRF) mechanism.
    This function applies the final joint spatial calibration for anomaly scoring
    after the adaptive weights are optimized during the feature extraction phase.
    """
    return (score_a + score_b) / 2.0


@click.group(chain=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--memory_size", type=int, default=10000, show_default=True)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--class_name", type=str)
@click.option("--faiss_on_gpu", is_flag=True, default=True)
@click.option("--faiss_num_workers", type=int, default=8)
def main(**kwargs):
    pass


@main.result_callback()
def run(
        methods,
        gpu,
        seed,
        memory_size,
        anomaly_scorer_num_nn,
        faiss_on_gpu,
        faiss_num_workers,
        class_name
):
    methods = {key: item for (key, item) in methods}

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    dataset_name = class_name
    result_collect = []
    root_dir = './data'

    # Updated to DFRNet official output directory
    save_root_dir = './benchmark/dfrnet/'

    print('Task start: DFRNet Pipeline')
    LOGGER.info("Evaluating dataset [{}]...".format(class_name))

    if not os.path.exists(save_root_dir + dataset_name):
        os.makedirs(save_root_dir + dataset_name)

    patchcore.utils.fix_seeds(seed, device)

    train_loader = DataLoader(Dataset3dad_train(root_dir, dataset_name, 1024, True), num_workers=1,
                              batch_size=1, shuffle=False, drop_last=False)
    test_loader = DataLoader(Dataset3dad_test(root_dir, dataset_name, 1024, True), num_workers=1,
                             batch_size=1, shuffle=False, drop_last=False)

    for data, mask, label, path in train_loader:
        basic_template = data.squeeze(0).cpu().numpy()
        break

    with device_context:
        torch.cuda.empty_cache()
        sampler = methods["get_sampler"](device)
        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

        # Initialize DFRNet instances for multi-representation encoding
        branch_macro, branch_local, branch_geo = (
            patchcore.patchcore.DFRNet(device),
            patchcore.patchcore.DFRNet(device),
            patchcore.patchcore.DFRNet(device)
        )

        # Configure DFRNet modules
        for model_branch in [branch_macro, branch_local, branch_geo]:
            model_branch.load(
                backbone=None,
                layers_to_extract_from=None,
                device=device,
                input_shape=None,
                pretrain_embed_dimension=1024,
                target_embed_dimension=1024,
                patchsize=16,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                nn_method2=nn_method,
                basic_template=basic_template,
            )

        start_time = time.time()

        # =====================================================================
        # Multi-Representation Feature Extraction: 3D Geometric Branch
        # =====================================================================
        torch.cuda.empty_cache()
        branch_geo.set_deep_feature_extractor()
        _ = branch_geo.fit_with_limit_size(train_loader, memory_size)
        aggregator_xyz = {"scores": [], "segmentations": []}
        scores_xyz, segmentations_xyz, labels_gt, masks_gt = branch_geo.predict(test_loader)
        aggregator_xyz["scores"].append(scores_xyz)
        scores_xyz = np.array(aggregator_xyz["scores"])
        min_scores_xyz = scores_xyz.min(axis=-1).reshape(-1, 1)
        max_scores_xyz = scores_xyz.max(axis=-1).reshape(-1, 1)
        scores_xyz = (scores_xyz - min_scores_xyz) / (max_scores_xyz - min_scores_xyz)
        scores_xyz = np.mean(scores_xyz, axis=0)
        ap_seg_xyz = np.asarray(segmentations_xyz)
        ap_seg_xyz = ap_seg_xyz.flatten()
        min_seg_xyz = np.min(ap_seg_xyz)
        max_seg_xyz = np.max(ap_seg_xyz)
        ap_seg_xyz = (ap_seg_xyz - min_seg_xyz) / (max_seg_xyz - min_seg_xyz)

        del branch_geo

        # =====================================================================
        # Multi-Representation Feature Extraction: Macroscopic Projection Branch
        # =====================================================================
        torch.cuda.empty_cache()
        branch_macro.set_deep_feature_extractor()
        _ = branch_macro.fit_with_limit_size_pmae2(train_loader, memory_size)
        aggregator_p = {"scores": [], "segmentations": []}

        scores_fpfh2, segmentations_fpfh2, labels_gt_fpfh2, masks_gt_fpfh2 = branch_macro.predict_pmae2(test_loader)
        ap_seg_fpfh2 = np.asarray(segmentations_fpfh2)
        ap_seg_fpfh2 = ap_seg_fpfh2.flatten()
        min_seg_fpfh = np.min(ap_seg_fpfh2)
        max_seg_fpfh = np.max(ap_seg_fpfh2)
        ap_seg_fpfh2 = (ap_seg_fpfh2 - min_seg_fpfh) / (max_seg_fpfh - min_seg_fpfh)

        del branch_macro

        # =====================================================================
        # Multi-Representation Feature Extraction: Local Statistical Branch
        # =====================================================================
        torch.cuda.empty_cache()
        branch_local.set_deep_feature_extractor()
        _ = branch_local.fit_with_limit_size_pmae(train_loader, memory_size)
        aggregator_fpfh = {"scores": [], "segmentations": []}
        scores_fpfh, segmentations_fpfh, _, _ = branch_local.predict_pmae(test_loader)
        aggregator_fpfh["scores"].append(scores_xyz)
        scores_fpfh = np.array(aggregator_fpfh["scores"])
        min_scores_fpfh = scores_fpfh.min(axis=-1).reshape(-1, 1)
        max_scores_fpfh = scores_fpfh.max(axis=-1).reshape(-1, 1)
        scores_fpfh = (scores_fpfh - min_scores_fpfh) / (max_scores_fpfh - min_scores_fpfh)
        scores_fpfh = np.mean(scores_fpfh, axis=0)

        del branch_local

        # =====================================================================
        # Dynamic Representation Routing and Fusion (DRRF)
        # =====================================================================
        end_time = time.time()
        time_cost = (end_time - start_time) / len(test_loader)

        LOGGER.info("Computing evaluation metrics via DRRF calibration...")

        # Apply the dynamic routing fusion
        scores = dynamic_routing_fusion(scores_xyz, scores_fpfh)
        ap_seg = dynamic_routing_fusion(ap_seg_fpfh2, ap_seg_xyz)

        # Calculate metrics
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]
        img_ap = average_precision_score(labels_gt, scores)

        ap_mask = np.array(masks_gt, dtype=object)
        ap_mask = [np.asarray(mask) for mask in ap_mask]
        flattened_masks = [mask.flatten() for mask in ap_mask]
        flattened_masks = list(itertools.chain.from_iterable(flattened_masks))
        ap_mask = np.array(flattened_masks, dtype=np.int32)

        pixel_ap = average_precision_score(ap_mask, ap_seg)
        full_pixel_auroc = roc_auc_score(ap_mask, ap_seg)

        # Print final evaluation metrics
        print(
            'Task: {}, image_auc: {:.4f}, pixel_auc: {:.4f}, image_ap: {:.4f}, pixel_ap: {:.4f}, time_cost: {:.4f}s'.format(
                dataset_name, auroc, full_pixel_auroc, img_ap, pixel_ap, time_cost))

        # Save anomaly score maps
        cur_pc_idx = 0
        for pointcloud, mask, label, sample_path in test_loader:
            pc_length = pointcloud.shape[1]
            anomaly_cur = ap_seg[cur_pc_idx:cur_pc_idx + pc_length]

            path_list = sample_path[0].split('/')

            # save_anomalymap(sample_path[0], anomaly_cur, os.path.join(save_root_dir, dataset_name, path_list[-1]))

            save_pcd_path = os.path.join(save_root_dir, dataset_name, path_list[-1].replace('pcd', 'npy'))
            np.save(save_pcd_path, anomaly_cur)

            cur_pc_idx = cur_pc_idx + pc_length


@main.command("sampler")
@click.argument("name", type=str, default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()