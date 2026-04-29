import csv
import logging
import os
import random

import numpy as np
import PIL
import torch
import tqdm
import cv2
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


def plot_segmentation_images(
        savefolder,
        image_paths,
        segmentations,
        anomaly_scores=None,
        mask_paths=None,
        image_transform=lambda x: x,
        mask_transform=lambda x: x,
        save_depth=4,
):
    """
    Generate and save anomaly segmentation heatmaps.

    Args:
        savefolder: [str] Directory to save the visualizations.
        image_paths: [List[str]] List of paths to original input images/projections.
        segmentations: [List[np.ndarray]] Generated anomaly segmentation maps.
        anomaly_scores: [List[float]] Object-level anomaly scores.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function] Optional transformation for input images.
        mask_transform: [function] Optional transformation for ground truth masks.
        save_depth: [int] Number of path segments to use for naming saved files.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"

    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
            zip(image_paths, mask_paths, anomaly_scores, segmentations),
            total=len(image_paths),
            desc="Generating Segmentation Visualizations...",
            leave=False,
    ):
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.resize(image, (256, 256))

        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None and os.path.exists(mask_path):
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros([3, 256, 256])

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)

        # Generate Heatmap on Image (HOI)
        hoi = heatmap_on_image(cv2heatmap(segmentation * 255), image)

        # Save individual visualization components
        cv2.imwrite(savename.replace('.png', '_org.png'), image)
        if masks_provided and mask is not None:
            cv2.imwrite(savename.replace('.png', '_mask.png'), mask.transpose(1, 2, 0) * 255)
        cv2.imwrite(savename.replace('.png', '_segmentation.png'), segmentation * 255)
        cv2.imwrite(savename.replace('.png', '_hoi.png'), hoi)


def cv2heatmap(gray):
    """Converts a grayscale anomaly map into a JET colormap heatmap."""
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    """Overlays the generated heatmap onto the original image."""
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def create_storage_folder(
        main_folder_path, project_folder, group_folder, mode="iterate"
):
    """Creates an organized directory structure for saving experiment results."""
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)

    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device for computation.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed.
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_and_store_final_results(
        results_path,
        results,
        row_names=None,
        column_names=[
            "Instance AUROC",
            "Full Pixel AUROC",
            "Full PRO",
            "Anomaly Pixel AUROC",
            "Anomaly PRO",
        ],
):
    """Store computed anomaly detection metrics as a CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset.
        row_names: [List[str]] Optional names for each evaluated dataset/class.
        column_names: [List[str]] Headers for the evaluation metrics.
    """
    if row_names is not None:
        assert len(row_names) == len(results), "# Rownames != # Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)

        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
