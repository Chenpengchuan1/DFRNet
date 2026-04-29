import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2
import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
import open3d as o3d
from M3DM.cpu_knn import fill_missing_values
from feature_extractors.ransac_position import get_registration_np, get_registration_refine_np
from utils.utils import get_args_point_mae
from M3DM.models import Model1
import torchvision.models as models
from scipy.spatial.distance import cdist

LOGGER = logging.getLogger(__name__)


class NoiseAwareFeatureRefinement(torch.nn.Module):
    """Noise-Aware Feature Refinement (NAFR) Module - Focused on noise suppression and manifold smoothing."""

    def __init__(self, feature_dim=320, noise_threshold=0.1, smoothing_factor=0.05):
        super(NoiseAwareFeatureRefinement, self).__init__()
        self.feature_dim = feature_dim
        self.noise_threshold = noise_threshold  # Noise detection threshold
        self.smoothing_factor = smoothing_factor  # Smoothing intensity

        # Simple noise detection weights
        self.noise_detector = torch.nn.Linear(feature_dim, 1)
        torch.nn.init.xavier_uniform_(self.noise_detector.weight)
        torch.nn.init.zeros_(self.noise_detector.bias)

    def _detect_noise_features(self, X):
        """Detect noise feature points."""
        # Calculate feature variance as a noise indicator
        feature_var = torch.var(X, dim=1, keepdim=True)

        # Use simple threshold detection
        noise_scores = torch.sigmoid(self.noise_detector(X))

        # Combine variance and learned noise scores
        combined_noise = feature_var * noise_scores
        noise_mask = combined_noise > self.noise_threshold

        return noise_mask.squeeze(), combined_noise.squeeze()

    def _apply_gaussian_smoothing(self, X, noise_mask):
        """Apply Gaussian smoothing to noise features."""
        if not noise_mask.any():
            return X

        X_smoothed = X.clone()

        # Perform local smoothing only on noise features
        noisy_features = X[noise_mask]
        if len(noisy_features) > 0:
            # Use global mean for slight smoothing
            global_mean = torch.mean(X, dim=0, keepdim=True)
            smoothed = (1 - self.smoothing_factor) * noisy_features + self.smoothing_factor * global_mean
            X_smoothed[noise_mask] = smoothed

        return X_smoothed

    def forward(self, X):
        """Feature refinement workflow."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        device = X.device
        original_shape = X.shape

        # Ensure it's a 2D tensor
        if len(X.shape) == 1:
            X = X.unsqueeze(0)

        n, d = X.shape

        # Return directly for very small datasets
        if n <= 5:
            return X.squeeze(0) if len(original_shape) == 1 else X

        try:
            # Step 1: Detect noise features
            noise_mask, noise_scores = self._detect_noise_features(X)

            # Step 2: Apply smoothing to noise features
            X_filtered = self._apply_gaussian_smoothing(X, noise_mask)

            # Step 3: Retain original information (strong residual connection)
            alpha = 0.9  # Retain 90% of the original features
            X_final = alpha * X + (1 - alpha) * X_filtered

        except Exception:
            # Return original features on exception
            X_final = X

        # Restore original shape
        if len(original_shape) == 1:
            X_final = X_final.squeeze(0)

        return X_final


class DynamicRepresentationRoutingAndFusion(torch.nn.Module):
    """Dynamic Representation Routing and Fusion (DRRF) Module."""

    def __init__(self, input_dim=320, hidden_dim=256, num_heads=4):
        super(DynamicRepresentationRoutingAndFusion, self).__init__()

        # Modality-specific encoders
        self.modality_encoders = torch.nn.ModuleDict({
            'image': torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, input_dim),
                torch.nn.LayerNorm(input_dim)
            ),
            'geometry': torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, input_dim),
                torch.nn.LayerNorm(input_dim)
            ),
            'raw': torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden_dim, input_dim),
                torch.nn.LayerNorm(input_dim)
            )
        })

        # Cross-modal attention mechanism
        self.cross_modal_attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Enhanced routing network
        self.routing_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 3, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),

            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim // 2, 3)
        )

        # Adaptive weight adjustment
        self.adaptive_weights = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 4, 3),
            torch.nn.Sigmoid()
        )

        # Learnable temperature and balance parameters
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        self.balance_factor = torch.nn.Parameter(torch.ones(1) * 0.5)

        # Feature fusion layer
        self.feature_fusion = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, input_dim)
        )

        # Residual connection weights
        self.residual_gate = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim // 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 4, 1),
            torch.nn.Sigmoid()
        )

        # Final layer normalization
        self.final_norm = torch.nn.LayerNorm(input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, image_feat, geometry_feat, raw_feat):
        batch_size = image_feat.shape[0] if len(image_feat.shape) > 1 else 1

        # Ensure batch format
        if len(image_feat.shape) == 1:
            image_feat = image_feat.unsqueeze(0)
            geometry_feat = geometry_feat.unsqueeze(0)
            raw_feat = raw_feat.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Modality-specific encoding
        encoded_features = []
        encoded_features.append(self.modality_encoders['image'](image_feat))
        encoded_features.append(self.modality_encoders['geometry'](geometry_feat))
        encoded_features.append(self.modality_encoders['raw'](raw_feat))

        # Stack features for cross-modal attention
        stacked_features = torch.stack(encoded_features, dim=1)  # [B, 3, D]

        # Cross-modal attention
        attended_features, attention_weights = self.cross_modal_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Separate attended features
        image_attended = attended_features[:, 0, :]
        geometry_attended = attended_features[:, 1, :]
        raw_attended = attended_features[:, 2, :]

        # Calculate routing weights
        concat_attended = torch.cat([image_attended, geometry_attended, raw_attended], dim=-1)
        routing_logits = self.routing_network(concat_attended)

        # Softmax with temperature scaling
        modality_weights = torch.softmax(routing_logits / torch.clamp(self.temperature, min=0.1), dim=-1)

        # Adaptive weight adjustment
        mean_feature = torch.mean(torch.stack([image_attended, geometry_attended, raw_attended], dim=1), dim=1)
        adaptive_factors = self.adaptive_weights(mean_feature)
        adjusted_weights = modality_weights * adaptive_factors
        adjusted_weights = adjusted_weights / (torch.sum(adjusted_weights, dim=-1, keepdim=True) + 1e-8)

        # Weighted fusion
        w_image = adjusted_weights[:, 0:1].expand_as(image_attended)
        w_geometry = adjusted_weights[:, 1:2].expand_as(geometry_attended)
        w_raw = adjusted_weights[:, 2:3].expand_as(raw_attended)

        # Multi-fusion strategy
        # 1. Weighted average
        weighted_fusion = w_image * image_attended + w_geometry * geometry_attended + w_raw * raw_attended

        # 2. Max pooling fusion
        max_fusion = torch.max(torch.stack([
            w_image * image_attended,
            w_geometry * geometry_attended,
            w_raw * raw_attended
        ], dim=1), dim=1)[0]

        # Combine fusion methods
        fused_feat = self.balance_factor * weighted_fusion + (1 - self.balance_factor) * max_fusion

        # Feature fusion layer processing
        fused_feat = self.feature_fusion(fused_feat)

        # Dynamic residual connection
        residual_input = torch.cat([fused_feat, mean_feature], dim=-1)
        residual_weight = self.residual_gate(residual_input)
        fused_feat = fused_feat + residual_weight * mean_feature

        # Final layer normalization
        fused_feat = self.final_norm(fused_feat)

        if squeeze_output:
            fused_feat = fused_feat.squeeze(0)

        return fused_feat, adjusted_weights


class DFRNet(torch.nn.Module):
    def __init__(self, device):
        """DFRNet anomaly detection class with DRRF and NAFR Modules."""
        super(DFRNet, self).__init__()
        self.device = device

        # Initialize DRRF module
        self.drrf_module = DynamicRepresentationRoutingAndFusion(input_dim=320, hidden_dim=256, num_heads=4).to(device)

        # Initialize NAFR module
        self.nafr_module = NoiseAwareFeatureRefinement(
            feature_dim=320,
            noise_threshold=0.1,
            smoothing_factor=0.05
        ).to(device)

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            anomaly_score_num_nn=1,
            featuresampler=patchcore.sampler.IdentitySampler(),
            nn_method=patchcore.common.FaissNN(False, 4),
            nn_method2=None,
            basic_template=None,
            **kwargs,
    ):
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = 0.5
        self.target_embed_dimension = target_embed_dimension

        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # GDAS Scorer initialized for anomaly scoring
        self.anomaly_scorer = patchcore.common.Real3DADOptimizedScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        if nn_method2:
            self.global_anomaly_scorer = patchcore.common.Real3DADOptimizedScorer(
                n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method2
            )

        self.eff = models.efficientnet_b0(pretrained=True)
        self.eff.classifier = torch.nn.Identity()
        self.eff = self.eff.to(self.device)
        self.eff.eval()

        self.featuresampler = featuresampler
        self.dataloader_count = 0
        self.basic_template = basic_template
        self.deep_feature_extractor = None
        self.global_features_memory = []

    def set_deep_feature_extractor(self):
        self.deep_feature_extractor = Model1(
            device='cuda',
            rgb_backbone_name='vit_base_patch8_224_dino',
            xyz_backbone_name='Point_MAE',
            group_size=128,
            num_group=16384
        )
        self.deep_feature_extractor = self.deep_feature_extractor.cuda()

    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def _embed_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(), self.basic_template)
        reg_data = reg_data.astype(np.float32)
        return reg_data

    def _embed_fpfh(self, point_cloud, detach=True):
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            o3d_pc,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        fpfh = pcd_fpfh.data.T
        fpfh = fpfh.astype(np.float32)
        return fpfh

    def extract_mrpfe_features(self, point_cloud):
        """Multi-Representation Point Feature Encoding (MRPFE)."""
        max_points = 12000
        if point_cloud.shape[0] > max_points:
            indices = np.random.choice(point_cloud.shape[0], max_points, replace=False)
            point_cloud_sampled = point_cloud[indices]
        else:
            point_cloud_sampled = point_cloud

        min_coords = np.min(point_cloud_sampled, axis=0)
        max_coords = np.max(point_cloud_sampled, axis=0)
        mean_coords = np.mean(point_cloud_sampled, axis=0)
        std_coords = np.std(point_cloud_sampled, axis=0)

        z_range = max_coords[2] - min_coords[2]
        z_splits = [
            min_coords[2],
            min_coords[2] + z_range * 0.25,
            min_coords[2] + z_range * 0.5,
            min_coords[2] + z_range * 0.75,
            max_coords[2]
        ]

        def enhanced_project_points(points, z_min, z_max, axis=2, sigma=1.0):
            if len(points) == 0:
                return np.zeros((224, 224, 3), dtype=np.uint8)

            mask = (points[:, axis] >= z_min) & (points[:, axis] <= z_max)
            filtered_points = points[mask]

            if len(filtered_points) == 0:
                return np.zeros((224, 224, 3), dtype=np.uint8)

            distances = np.abs(filtered_points[:, axis] - (z_min + z_max) / 2)
            max_dist = np.max(distances) if np.max(distances) > 0 else 1
            density_weights = np.exp(-distances / (sigma * max_dist))

            if axis == 2:
                projected_points = filtered_points[:, :2]
                x_range = max_coords[0] - min_coords[0] if max_coords[0] - min_coords[0] > 0 else 1
                y_range = max_coords[1] - min_coords[1] if max_coords[1] - min_coords[1] > 0 else 1
            elif axis == 1:
                projected_points = np.column_stack((filtered_points[:, 0], filtered_points[:, 2]))
                x_range = max_coords[0] - min_coords[0] if max_coords[0] - min_coords[0] > 0 else 1
                y_range = max_coords[2] - min_coords[2] if max_coords[2] - min_coords[2] > 0 else 1
            else:
                projected_points = filtered_points[:, 1:3]
                x_range = max_coords[1] - min_coords[1] if max_coords[1] - min_coords[1] > 0 else 1
                y_range = max_coords[2] - min_coords[2] if max_coords[2] - min_coords[2] > 0 else 1

            image = np.zeros((224, 224, 3), dtype=np.float32)

            for i in range(len(projected_points)):
                if axis == 2:
                    x = int((projected_points[i, 0] - min_coords[0]) / x_range * 223)
                    y = int((projected_points[i, 1] - min_coords[1]) / y_range * 223)
                elif axis == 1:
                    x = int((projected_points[i, 0] - min_coords[0]) / x_range * 223)
                    y = int((projected_points[i, 1] - min_coords[2]) / y_range * 223)
                else:
                    x = int((projected_points[i, 0] - min_coords[1]) / x_range * 223)
                    y = int((projected_points[i, 1] - min_coords[2]) / y_range * 223)

                if 0 <= x < 224 and 0 <= y < 224:
                    intensity = density_weights[i] * 255
                    image[y, x] = [intensity, intensity, intensity]

            image = cv2.GaussianBlur(image, (3, 3), 0.5)
            return np.uint8(np.clip(image, 0, 255))

        # 1. Image Modality Features
        layer_images = []
        for i in range(len(z_splits) - 1):
            layer_img = enhanced_project_points(
                point_cloud_sampled, z_splits[i], z_splits[i + 1], axis=2, sigma=0.5
            )
            layer_images.append(layer_img)

        image_features = []
        for img in layer_images:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            img_tensor = img_tensor / 255.0
            with torch.no_grad():
                feat = self.eff.features[:-1](img_tensor).reshape(320, -1)
                feat = torch.max(feat, dim=1)[0]
                image_features.append(feat)
        image_feat = torch.stack(image_features).mean(dim=0)

        # 2. Geometric Modality Features
        xy_proj = enhanced_project_points(point_cloud_sampled, min_coords[2], max_coords[2], axis=2, sigma=0.3)
        xz_proj = enhanced_project_points(point_cloud_sampled, min_coords[1], max_coords[1], axis=1, sigma=0.3)
        yz_proj = enhanced_project_points(point_cloud_sampled, min_coords[0], max_coords[0], axis=0, sigma=0.3)

        geometry_features = []
        for proj_img in [xy_proj, xz_proj, yz_proj]:
            img_tensor = torch.from_numpy(proj_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            img_tensor = img_tensor / 255.0
            with torch.no_grad():
                feat = self.eff.features[:-1](img_tensor).reshape(320, -1)
                feat = torch.max(feat, dim=1)[0]
                geometry_features.append(feat)
        geometry_feat = torch.stack(geometry_features).mean(dim=0)

        # 3. Raw Point Cloud Features (Statistical Features)
        basic_stats = np.array([
            np.mean(point_cloud_sampled, axis=0),
            np.std(point_cloud_sampled, axis=0),
            np.min(point_cloud_sampled, axis=0),
            np.max(point_cloud_sampled, axis=0),
            np.median(point_cloud_sampled, axis=0),
        ]).flatten()

        percentiles = [10, 25, 75, 90]
        percentile_features = []
        for p in percentiles:
            percentile_features.extend(np.percentile(point_cloud_sampled, p, axis=0))

        distances_to_center = np.linalg.norm(point_cloud_sampled - mean_coords, axis=1)
        distance_stats = np.array([
            np.mean(distances_to_center),
            np.std(distances_to_center),
            np.max(distances_to_center),
            np.min(distances_to_center)
        ])

        x_mid, y_mid, z_mid = mean_coords
        density_features = []
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                for z_sign in [-1, 1]:
                    mask = ((point_cloud_sampled[:, 0] - x_mid) * x_sign >= 0) & \
                           ((point_cloud_sampled[:, 1] - y_mid) * y_sign >= 0) & \
                           ((point_cloud_sampled[:, 2] - z_mid) * z_sign >= 0)
                    density_features.append(np.sum(mask))
        density_features = np.array(density_features) / len(point_cloud_sampled)

        try:
            hull = np.array(point_cloud_sampled)
            hull_volume = 1.0
            hull_features = [hull_volume]
        except:
            hull_features = [1.0]

        curvature_features = [
            np.var(point_cloud_sampled, axis=0).mean(),
            np.sum(np.diff(point_cloud_sampled, axis=0) ** 2) / len(point_cloud_sampled)
        ]

        all_stats = np.concatenate([
            basic_stats,
            percentile_features,
            distance_stats,
            density_features,
            hull_features,
            curvature_features
        ])

        raw_feat = torch.zeros(320, device=self.device)
        stats_tensor = torch.from_numpy(all_stats).float().to(self.device)
        raw_feat[:len(stats_tensor)] = stats_tensor

        if len(stats_tensor) < 320:
            remaining_dims = 320 - len(stats_tensor)
            repeated_stats = stats_tensor.repeat((remaining_dims // len(stats_tensor)) + 1)[:remaining_dims]
            noise = torch.randn_like(repeated_stats) * 0.01
            raw_feat[len(stats_tensor):] = repeated_stats + noise

        return image_feat, geometry_feat, raw_feat

    def process_point_cloud(self, point_cloud):
        """Process point cloud with DRRF Module"""
        image_feat, geometry_feat, raw_feat = self.extract_mrpfe_features(point_cloud)

        if len(image_feat.shape) == 1:
            image_feat = image_feat.unsqueeze(0)
            geometry_feat = geometry_feat.unsqueeze(0)
            raw_feat = raw_feat.unsqueeze(0)

        with torch.no_grad():
            fused_feature, modality_weights = self.drrf_module(image_feat, geometry_feat, raw_feat)

        return fused_feature.squeeze(0)

    def _embed_pointmae(self, point_cloud, detach=True):
        """Optimized feature embedding with DRRF and NAFR"""
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(), self.basic_template)
        fpfh_features = self._embed_fpfh(reg_data)

        pointcloud_data = torch.from_numpy(reg_data).permute(1, 0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1, 0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        fpfh_features = fpfh_features[center_idx.cpu().numpy()].squeeze(0)

        # DRRF global feature extraction
        drrf_feature = self.process_point_cloud(reg_data)
        self.current_global_feature = drrf_feature.cpu().numpy().astype(np.float32)

        local_feature = np.concatenate((fpfh_features, pmae_features), axis=1)

        n = local_feature.shape[0]
        global_expanded = np.tile(self.current_global_feature, (n, 1))
        combined_feature = np.concatenate((global_expanded, local_feature), axis=1)

        # NAFR feature filtering
        combined_feature_tensor = torch.from_numpy(combined_feature).float().to(self.device)
        filtered_features = self.nafr_module(combined_feature_tensor)
        combined_feature = filtered_features.cpu().numpy().astype(np.float32)

        return combined_feature, center_idx

    def _embed_pointmae2(self, point_cloud, detach=True):
        """Feature embedding version 2 without explicit global features"""
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(), self.basic_template)
        fpfh_features = self._embed_fpfh(reg_data)

        pointcloud_data = torch.from_numpy(reg_data).permute(1, 0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1, 0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        fpfh_features = fpfh_features[center_idx.cpu().numpy()].squeeze(0)

        local_feature = np.concatenate((fpfh_features, pmae_features), axis=1)

        # NAFR feature filtering
        local_feature_tensor = torch.from_numpy(local_feature).float().to(self.device)
        filtered_features = self.nafr_module(local_feature_tensor)
        local_feature = filtered_features.cpu().numpy().astype(np.float32)

        return local_feature, center_idx

    def fit_with_limit_size(self, training_data, limit_size):
        """DFRNet memory bank construction."""
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)

    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                features = self._embed_xyz(input_pointcloud)
                features_tensor = torch.from_numpy(features).float().to(self.device)
                filtered_features = self.nafr_module(features_tensor)
                return filtered_features.cpu().numpy()

        features = []
        with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                torch.cuda.empty_cache()
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        return features

    def fit_with_limit_size_pmae(self, training_data, limit_size):
        """DFRNet training with localized representations."""
        return self._fill_memory_bank_with_limit_size_pmae(training_data, limit_size)

    def _fill_memory_bank_with_limit_size_pmae(self, input_data, limit_size):
        """Computes and sets the support features for localized branches."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx = self._embed_pointmae(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
                input_data, desc="Extracting localized representations...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                torch.cuda.empty_cache()
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        return features

    def fit_with_limit_size_pmae2(self, training_data, limit_size):
        """DFRNet training with localized representations (version 2)."""
        return self._fill_memory_bank_with_limit_size_pmae2(training_data, limit_size)

    def _fill_memory_bank_with_limit_size_pmae2(self, input_data, limit_size):
        """Computes and sets the support features for localized branches."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx = self._embed_pointmae2(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
                input_data, desc="Extracting localized representations (v2)...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                torch.cuda.empty_cache()
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        return features

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inference...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                torch.cuda.empty_cache()
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, input_pointcloud):
        """Infer score and mask for a batch."""
        with torch.no_grad():
            features = self._embed_xyz(input_pointcloud)
            features_tensor = torch.from_numpy(features).float().to(self.device)
            filtered_features = self.nafr_module(features_tensor)
            features = filtered_features.cpu().numpy()
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]

    def predict_pmae(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_pmae(data)
        return self._predict_pmae(data)

    def _predict_dataloader_pmae(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Localized inference...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                torch.cuda.empty_cache()
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_pmae(self, input_pointcloud):
        """Predict anomaly scores for localized features."""
        with torch.no_grad():
            features, sample_dix = self._embed_pointmae(input_pointcloud)
            features = np.asarray(features, order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(), :]
            full_scores = fill_missing_values(xyz_sampled, patch_scores, input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]

    def predict_pmae2(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_pmae2(data)
        return self._predict_pmae2(data)

    def _predict_dataloader_pmae2(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Localized inference (v2)...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                torch.cuda.empty_cache()
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae2(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_pmae2(self, input_pointcloud):
        """Predict anomaly scores for localized features (version 2)."""
        with torch.no_grad():
            features, sample_dix = self._embed_pointmae2(input_pointcloud)
            features = np.asarray(features, order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(), :]
            full_scores = fill_missing_values(xyz_sampled, patch_scores, input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]
