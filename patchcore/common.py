import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
            self,
            n_nearest_neighbours,
            query_features: np.ndarray,
            index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,
            64,
            8,
        )
        return self._index_to_gpu(index)


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class Real3DADOptimizedScorer(object):
    """
    Geometry-Density-Aware Scoring (GDAS) mechanism.
    Optimized for 3D point cloud anomaly detection by explicitly compensating
    for measurement biases induced by sensor noise and non-uniform sampling.
    """

    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        Initializes the GDAS anomaly scorer.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous points.
            nn_method: Nearest neighbour search method (e.g., FAISS).
        """
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

        # Key hyperparameters for 3D anomaly detection
        self.geometric_weight = 0.5
        self.spatial_threshold = 0.12
        self.density_factor = 0.3
        self.stability_alpha = 0.9
        self.local_global_balance = 0.6
        self.outlier_percentile = 92

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """
        Fits the detection features to the memory bank.

        Args:
            detection_features: [list of np.arrays]
                List of feature vectors for all training point clouds.
        """
        self.detection_features = self.feature_merger.merge(detection_features)
        self.nn_method.fit(self.detection_features)

        # Precompute 3D spatial statistics for density estimation
        self._precompute_3d_statistics()

    def _precompute_3d_statistics(self):
        """Precomputes 3D point cloud statistics for adaptive scoring."""
        train_distances, _ = self.imagelevel_nn(self.detection_features)

        # Robust statistics for distance distribution
        self.train_distance_median = np.median(train_distances, axis=1)
        self.train_distance_std = np.std(train_distances, axis=1)
        self.train_distance_iqr = np.percentile(train_distances, 75, axis=1) - np.percentile(train_distances, 25, axis=1)

        # Compute trimmed mean to mitigate outlier influence
        self.train_distance_trimmed_mean = self._compute_trimmed_mean(train_distances, trim_percent=10)

        # Global distribution statistics
        self.global_median = np.median(self.train_distance_median)
        self.global_std = np.std(self.train_distance_median)
        self.global_iqr = np.percentile(self.train_distance_median, 75) - np.percentile(self.train_distance_median, 25)
        self.global_trimmed_mean = np.mean(self.train_distance_trimmed_mean)

        self._compute_geometric_baseline()

    def _compute_trimmed_mean(self, data, trim_percent=10):
        """Computes trimmed mean to reduce sensitivity to extreme anomalies."""
        sorted_data = np.sort(data, axis=1)
        n = sorted_data.shape[1]
        trim_count = int(n * trim_percent / 100)
        if trim_count > 0:
            trimmed = sorted_data[:, trim_count:-trim_count]
        else:
            trimmed = sorted_data
        return np.mean(trimmed, axis=1)

    def _compute_geometric_baseline(self):
        """Computes geometric consistency baseline from normal data."""
        k_geo = min(5, self.n_nearest_neighbours)
        geo_distances, _ = self.nn_method.run(k_geo, self.detection_features)

        self.geometric_consistency = []
        self.geometric_stability = []

        for distances in geo_distances:
            if len(distances) > 1:
                mean_dist = np.mean(distances)
                cv = np.std(distances) / (mean_dist + 1e-8)
                self.geometric_consistency.append(cv)

                if len(distances) >= 2:
                    stability = distances[0] / (distances[1] + 1e-8)
                    self.geometric_stability.append(stability)
                else:
                    self.geometric_stability.append(1.0)
            else:
                self.geometric_consistency.append(0.0)
                self.geometric_stability.append(1.0)

        self.geometric_consistency = np.array(self.geometric_consistency)
        self.geometric_stability = np.array(self.geometric_stability)
        self.geo_consistency_threshold = np.percentile(self.geometric_consistency, 80)
        self.geo_stability_threshold = np.percentile(self.geometric_stability, 80)

    def _compute_spatial_weights(self, query_distances):
        """Computes 3D spatial-aware weights based on geometric variations."""
        batch_size = query_distances.shape[0]

        # 1. Variance-based spatial weights
        distance_variance = np.var(query_distances, axis=1, keepdims=True)
        distance_iqr = (np.percentile(query_distances, 75, axis=1, keepdims=True) -
                        np.percentile(query_distances, 25, axis=1, keepdims=True))

        norm_variance = distance_variance / (self.global_std ** 2 + 1e-8)
        norm_iqr = distance_iqr / (self.global_iqr + 1e-8)

        # 2. Geometric consistency weights
        geometric_weights = np.ones_like(query_distances)
        for i in range(batch_size):
            distances = query_distances[i]
            if len(distances) > 1:
                mean_dist = np.mean(distances)
                cv = np.std(distances) / (mean_dist + 1e-8)

                if cv > self.geo_consistency_threshold:
                    weight_factor = 1.0 + self.geometric_weight * np.tanh(
                        (cv / self.geo_consistency_threshold - 1.0)
                    )
                    geometric_weights[i] *= weight_factor

        # 3. Aggregated spatial weights
        spatial_weights = 1.0 + self.spatial_threshold * (
                0.6 * norm_variance + 0.4 * norm_iqr
        ) * geometric_weights

        spatial_weights = spatial_weights / np.sum(spatial_weights, axis=1, keepdims=True)

        return spatial_weights

    def _compute_density_aware_scoring(self, query_distances):
        """Density-aware anomaly scoring to mitigate non-uniform sampling."""
        # 1. Multi-scale base scoring
        k_local = min(3, self.n_nearest_neighbours)
        k_mid = min(max(self.n_nearest_neighbours // 2, 1), self.n_nearest_neighbours)

        local_scores = np.median(query_distances[:, :k_local], axis=1)
        mid_scores = np.median(query_distances[:, :k_mid], axis=1)
        global_scores = np.median(query_distances, axis=1)

        # 2. Density-aware adjustments using harmonic mean
        k_density = min(5, self.n_nearest_neighbours)
        density_distances = query_distances[:, :k_density]
        harmonic_density = k_density / np.sum(1.0 / (density_distances + 1e-8), axis=1)
        relative_density = harmonic_density / (self.global_trimmed_mean + 1e-8)

        # 3. Adaptive multi-scale fusion based on density
        density_weights = np.clip(relative_density, 0.2, 0.9)
        adaptive_scores = (
                density_weights * (0.5 * local_scores + 0.5 * mid_scores) +
                (1 - density_weights) * global_scores
        )

        return adaptive_scores, relative_density

    def _stability_enhancement(self, raw_scores, query_distances):
        """Stability enhancement for score distribution."""
        # 1. Soft suppression of extreme outliers
        q_high = np.percentile(raw_scores, self.outlier_percentile)
        q_low = np.percentile(raw_scores, 25)
        iqr = q_high - q_low
        upper_bound = q_high + 1.2 * iqr

        outlier_mask = raw_scores > upper_bound
        if np.any(outlier_mask):
            excess = raw_scores[outlier_mask] - upper_bound
            raw_scores[outlier_mask] = upper_bound + np.sqrt(excess)

        # 2. Local smoothing based on distance consistency
        if len(raw_scores) > 3:
            distance_consistency = np.std(query_distances, axis=1)
            consistency_threshold = np.percentile(distance_consistency, 65)

            smooth_mask = distance_consistency < consistency_threshold
            if np.any(smooth_mask):
                window_size = min(3, len(raw_scores))
                smoothed_scores = np.copy(raw_scores)

                for i in range(len(raw_scores)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(raw_scores), i + window_size // 2 + 1)

                    window = raw_scores[start_idx:end_idx]
                    weights = np.exp(-0.5 * np.arange(len(window)) ** 2 / (window_size / 2) ** 2)
                    weights = weights / np.sum(weights)
                    smoothed_scores[i] = np.sum(window * weights)

                raw_scores[smooth_mask] = (
                        self.stability_alpha * raw_scores[smooth_mask] +
                        (1 - self.stability_alpha) * smoothed_scores[smooth_mask]
                )

        return raw_scores

    def predict(
            self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts anomaly scores based on GDAS mechanism.

        Args:
             query_features: [list of np.arrays] List of feature vectors
                 extracted from the test point clouds.
        """
        query_features = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)

        # 1. Spatial weight calibration
        spatial_weights = self._compute_spatial_weights(query_distances)
        weighted_distances = query_distances * spatial_weights
        weighted_scores = np.sum(weighted_distances, axis=1)

        # 2. Density-aware score derivation
        density_scores, relative_density = self._compute_density_aware_scoring(query_distances)

        # 3. Adaptive fusion strategy
        fusion_weight = np.clip(
            self.local_global_balance * (1 + np.tanh(relative_density - 1)),
            0.3, 0.8
        )
        fused_scores = fusion_weight * weighted_scores + (1 - fusion_weight) * density_scores

        # 4. Final stability enhancement
        final_scores = self._stability_enhancement(fused_scores, query_distances)

        return final_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "real3dad_optimized_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "real3dad_optimized_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
            self,
            save_folder: str,
            save_features_separately: bool = False,
            prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )


class NearestNeighbourScorer(object):
    """Legacy Nearest-Neighbourhood Anomaly Scorer."""
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    def predict(
            self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
            self,
            save_folder: str,
            save_features_separately: bool = False,
            prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )


class NearestNeighbourScorer2(NearestNeighbourScorer):
    """Secondary Nearest-Neighbourhood Anomaly Scorer for global metrics."""
    pass