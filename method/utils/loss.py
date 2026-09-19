import torch
import torch.nn.functional as F
from torch import nn

ATTENTION_REGRESSION_SCALE = 10.0
criterion_reg = nn.MSELoss()


def gaussian_fixation_targets(coordinates, height=224, width=224, sigma=16.0):
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=coordinates.device, dtype=torch.float32),
        torch.arange(width, device=coordinates.device, dtype=torch.float32),
        indexing="ij",
    )
    center_x = coordinates[:, 0, None, None] * width
    center_y = coordinates[:, 1, None, None] * height
    squared_distance = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    return torch.exp(-squared_distance / (2 * sigma**2))


def focal_heatmap_loss(probabilities, targets, alpha=1.0, beta=4.0):
    probabilities = probabilities.clamp(1e-06, 1 - 1e-06)
    positive = targets.eq(1.0)
    negative = targets.lt(1.0)
    negative_weight = (1 - targets).pow(beta)
    positive_loss = torch.log(probabilities) * (1 - probabilities).pow(alpha) * positive
    negative_loss = (
        torch.log(1 - probabilities)
        * probabilities.pow(alpha)
        * negative_weight
        * negative
    )
    positive_count = positive.float().sum()
    if positive_count == 0:
        return -negative_loss.sum()
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count


class ClinicalTrajectoryConsistencyLoss(nn.Module):
    def __init__(
        self, lambda_geo=1.0, clinical_duration_sigma=1.0 / 7.0, epsilon=1e-08
    ):
        super().__init__()
        if lambda_geo < 0 or clinical_duration_sigma <= 0:
            raise ValueError(
                "lambda_geo must be non-negative and clinical_duration_sigma positive"
            )
        self.lambda_geo = lambda_geo
        self.clinical_duration_sigma = clinical_duration_sigma
        self.epsilon = epsilon

    def geometry_loss(self, pred_coordinate, previous_fixation, current_fixation):
        pred_delta = pred_coordinate - previous_fixation
        gt_delta = current_fixation - previous_fixation
        pred_norm = torch.linalg.vector_norm(pred_delta, dim=1)
        gt_norm = torch.linalg.vector_norm(gt_delta, dim=1)
        cosine = F.cosine_similarity(pred_delta, gt_delta, dim=1, eps=0.0001)
        direction_loss = 1.0 - cosine.clamp(-1.0, 1.0)
        direction_valid = (gt_norm > 1.0 / 224.0) & (pred_norm > 1.0 / 224.0)
        direction_loss = torch.where(
            direction_valid, direction_loss, torch.zeros_like(direction_loss)
        )
        amplitude_loss = torch.abs(pred_norm - gt_norm)
        return direction_loss + amplitude_loss

    def forward(
        self,
        pred_heatmap,
        pred_duration,
        gt_duration,
        clinical_attention,
        previous_fixation,
        current_fixation,
        valid_mask,
        gt_heatmap=None,
    ):
        valid_mask = valid_mask.bool()
        if not valid_mask.any():
            zero = pred_heatmap.sum() * 0.0
            return {
                "loss_energy": zero,
                "loss_clinical_value": zero,
                "loss_clin": zero,
                "loss_geo": zero,
                "loss_consistency": zero,
            }
        pred_heatmap = pred_heatmap[valid_mask]
        pred_duration = pred_duration[valid_mask]
        gt_duration = gt_duration[valid_mask]
        clinical_attention = clinical_attention[valid_mask]
        previous_fixation = previous_fixation[valid_mask]
        current_fixation = current_fixation[valid_mask]
        clinical_attention = clinical_attention.detach()
        pred_probability = F.softmax(pred_heatmap.flatten(1), dim=1).reshape_as(
            pred_heatmap
        )
        if gt_heatmap is None:
            gt_heatmap = self.build_gt_heatmap(
                current_fixation, pred_heatmap.shape[-2], pred_heatmap.shape[-1]
            )
        else:
            gt_heatmap = gt_heatmap[valid_mask]
        gt_probability = self.normalize_heatmap(gt_heatmap)
        clinical_size = clinical_attention.shape[-2:]
        pred_clinical = self.pool_probability(pred_probability, clinical_size)
        gt_clinical = self.pool_probability(gt_probability, clinical_size)
        energy_loss = self.spatial_energy_loss(pred_clinical, gt_clinical)
        grid_size = int(round(pred_clinical.shape[1] ** 0.5))
        if grid_size * grid_size != pred_clinical.shape[1]:
            raise ValueError("Clinical probability must come from a square grid")
        kernel = self.duration_kernel(grid_size, pred_clinical)
        pred_duration_field = pred_duration[:, None] * (pred_clinical @ kernel.T)
        gt_duration_field = gt_duration[:, None] * (gt_clinical @ kernel.T)
        clinical_prior = clinical_attention.flatten(1).clamp(0.0, 1.0)
        clinical_normalizer = clinical_prior.sum(dim=1).clamp_min(self.epsilon)
        pred_clinical_value = (clinical_prior * pred_duration_field).sum(
            dim=1
        ) / clinical_normalizer
        gt_clinical_value = (clinical_prior * gt_duration_field).sum(
            dim=1
        ) / clinical_normalizer
        clinical_value_loss = F.smooth_l1_loss(
            pred_clinical_value, gt_clinical_value, reduction="none"
        )
        loss_energy = energy_loss.mean()
        loss_clinical_value = clinical_value_loss.mean()
        loss_clin = loss_energy + loss_clinical_value
        pred_coordinate = self.soft_coordinate(pred_probability)
        loss_geo = self.geometry_loss(
            pred_coordinate, previous_fixation, current_fixation
        ).mean()
        return {
            "loss_energy": loss_energy,
            "loss_clinical_value": loss_clinical_value,
            "loss_clin": loss_clin,
            "loss_geo": loss_geo,
            "loss_consistency": loss_clin + self.lambda_geo * loss_geo,
        }


def compute_loss(
    outputs,
    batch,
    termination_pos_weight=1.0,
    attention_map_weight=0.5,
    lambda_geo=1.0,
    clinical_duration_sigma=1.0 / 7.0,
    duration_loss_weight=1.0,
    sed_grid_loss_weight=0.2,
    consistency_loss_weight=0.5,
):
    if (
        clinical_duration_sigma <= 0
        or duration_loss_weight < 0
        or sed_grid_loss_weight < 0
        or (attention_map_weight < 0)
        or (consistency_loss_weight < 0)
    ):
        raise ValueError(
            "clinical_duration_sigma must be positive and loss weights non-negative"
        )
    termination = batch["termination"].float()
    non_terminal = termination < 0.5
    if non_terminal.any():
        coordinate_logits = F.interpolate(
            outputs["coordinate_logits"][non_terminal].unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        coordinate_targets = gaussian_fixation_targets(
            batch["target_coordinate"][non_terminal]
        )
        coordinate_loss = focal_heatmap_loss(
            torch.sigmoid(coordinate_logits), coordinate_targets
        )
        duration_loss = F.smooth_l1_loss(
            outputs["pred_duration"][non_terminal],
            batch["target_duration"][non_terminal],
        )
        sed_grid_logits = F.avg_pool2d(
            coordinate_logits.unsqueeze(1), kernel_size=44, stride=44
        ).flatten(1)
        target_pixels = (
            torch.floor(batch["target_coordinate"][non_terminal] * 224.0)
            .long()
            .clamp(0, 223)
        )
        target_x = (target_pixels[:, 0] // 44).clamp(0, 4)
        target_y = (target_pixels[:, 1] // 44).clamp(0, 4)
        sed_grid_loss = F.cross_entropy(sed_grid_logits, target_y * 5 + target_x)
    else:
        coordinate_loss = outputs["coordinate_logits"].sum() * 0.0
        duration_loss = outputs["pred_duration"].sum() * 0.0
        sed_grid_loss = outputs["coordinate_logits"].sum() * 0.0
    pos_weight = torch.as_tensor(
        termination_pos_weight,
        dtype=outputs["pred_termination"].dtype,
        device=outputs["pred_termination"].device,
    )
    termination_logits = outputs["pred_termination"].float()
    termination_loss = (
        (1.0 - termination) * F.softplus(termination_logits)
        + termination * pos_weight.float() * F.softplus(-termination_logits)
    ).mean()
    predicted_attention = outputs["attention_map"]
    attention_logits = outputs.get("attention_map_logits")
    if attention_logits is None:
        attention_logits = torch.logit(predicted_attention.clamp(1e-06, 1.0 - 1e-06))
    target_attention = batch["attention_map"].squeeze(1).clamp(0.0, 1.0)
    if not torch.isfinite(target_attention).all():
        raise ValueError("attention_map target contains NaN or infinity")
    attention_mse_loss = criterion_reg(predicted_attention, target_attention)
    attention_regression_loss = criterion_reg(
        predicted_attention * ATTENTION_REGRESSION_SCALE,
        target_attention * ATTENTION_REGRESSION_SCALE,
    )
    predicted_log_distribution = F.log_softmax(attention_logits.flatten(1), dim=1)
    predicted_distribution = predicted_log_distribution.exp()
    target_distribution = target_attention.flatten(1)
    target_distribution = target_distribution / target_distribution.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-08)
    attention_kl_loss = F.kl_div(
        predicted_log_distribution, target_distribution, reduction="batchmean"
    )
    mixture = 0.5 * (predicted_distribution + target_distribution)
    attention_js_loss = 0.5 * (
        F.kl_div(
            mixture.clamp_min(1e-08).log(),
            predicted_distribution,
            reduction="batchmean",
        )
        + F.kl_div(
            mixture.clamp_min(1e-08).log(), target_distribution, reduction="batchmean"
        )
    )
    attention_map_loss = attention_regression_loss + attention_js_loss
    consistency_module = ClinicalTrajectoryConsistencyLoss(
        lambda_geo=lambda_geo, clinical_duration_sigma=clinical_duration_sigma
    )
    consistency = consistency_module(
        pred_heatmap=outputs["coordinate_logits"],
        pred_duration=outputs["pred_duration"],
        gt_duration=batch["target_duration"],
        clinical_attention=outputs["clinical_attention"],
        previous_fixation=batch["previous_coordinate"],
        current_fixation=batch["target_coordinate"],
        valid_mask=non_terminal,
    )
    total_loss = (
        coordinate_loss
        + duration_loss_weight * duration_loss
        + termination_loss
        + sed_grid_loss_weight * sed_grid_loss
        + attention_map_weight * attention_map_loss
        + consistency_loss_weight * consistency["loss_consistency"]
    )
    return {
        "loss": total_loss,
        "coordinate_loss": coordinate_loss,
        "duration_loss": duration_loss,
        "sed_grid_loss": sed_grid_loss,
        "termination_loss": termination_loss,
        "attention_map_loss": attention_map_loss,
        "attention_mse_loss": attention_mse_loss,
        "attention_regression_loss": attention_regression_loss,
        "attention_kl_loss": attention_kl_loss,
        "attention_js_loss": attention_js_loss,
        **consistency,
    }
