import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

IMAGE_SIZE = 224


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=3):
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        modules = []
        for index in range(layers):
            modules.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < layers - 1:
                modules.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*modules)

    def forward(self, values):
        return self.network(values)


def sine_position_encoding(height, width, channels, device):
    if channels % 4:
        raise ValueError("hidden_dim must be divisible by 4")
    quarter = channels // 4
    scale = torch.exp(
        torch.arange(quarter, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None] * scale[None]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None] * scale[None]
    y_encoding = torch.cat((y.sin(), y.cos()), dim=1)[:, None].expand(-1, width, -1)
    x_encoding = torch.cat((x.sin(), x.cos()), dim=1)[None].expand(height, -1, -1)
    return torch.cat((x_encoding, y_encoding), dim=-1).permute(2, 0, 1).unsqueeze(0)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, hidden_dim, weights=False, freeze=True):
        super().__init__()
        backbone_weights = ResNet50_Weights.DEFAULT if weights else None
        backbone = resnet50(weights=backbone_weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.high_projection = nn.Conv2d(256, hidden_dim, 1)
        self.low_projection = nn.Conv2d(2048, hidden_dim, 1)
        self.freeze_backbone = freeze
        if freeze:
            for module in (
                self.stem,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
            ):
                module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            for module in (
                self.stem,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
            ):
                module.eval()
        return self

    def forward(self, images):
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            features = self.stem(images)
            high_resolution = self.layer1(features)
            features = self.layer2(high_resolution)
            features = self.layer3(features)
            low_resolution = self.layer4(features)
        return (
            self.high_projection(high_resolution),
            self.low_projection(low_resolution),
        )


class TextProjector(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features):
        return self.layers(features)


class SemanticRelationEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        position = sine_position_encoding(7, 7, hidden_dim, torch.device("cpu"))
        self.register_buffer(
            "position", position.flatten(2).transpose(1, 2), persistent=False
        )
        rows, columns = torch.meshgrid(torch.arange(7), torch.arange(7), indexing="ij")
        grid = torch.stack((rows.flatten(), columns.flatten()), dim=1)
        local_mask = (grid[:, None] - grid[None]).abs().amax(dim=-1) > 1
        self.register_buffer("local_mask", local_mask, persistent=False)

    def forward(self, local_text, height, width):
        if (height, width) != (7, 7):
            raise ValueError(f"Expected a 7x7 local grid, got {height}x{width}")
        position = self.position.to(dtype=local_text.dtype)
        return self.encoder(local_text + position, mask=self.local_mask)


class LocalSemanticFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.text_to_visual = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.visual_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.channel_fusion = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        correspondence_mask = ~torch.eye(49, dtype=torch.bool)
        self.register_buffer(
            "correspondence_mask", correspondence_mask, persistent=False
        )

    def forward(self, visual, local_text, height, width):
        if (height, width) != (7, 7):
            raise ValueError(f"Expected a 7x7 local grid, got {height}x{width}")
        text_context, _ = self.text_to_visual(
            local_text,
            visual,
            visual,
            attn_mask=self.correspondence_mask,
            need_weights=False,
        )
        refined_text = self.text_norm(local_text + text_context)
        visual_context, _ = self.visual_to_text(
            visual,
            local_text,
            local_text,
            attn_mask=self.correspondence_mask,
            need_weights=False,
        )
        refined_visual = self.visual_norm(visual + visual_context)
        fused = torch.cat((refined_visual, refined_text), dim=-1)
        fused = fused.transpose(1, 2).reshape(visual.shape[0], -1, 7, 7)
        return self.channel_fusion(fused)


class GlobalQueryRefiner(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, task_query, global_text):
        context, _ = self.cross_attention(
            task_query, global_text, global_text, need_weights=False
        )
        context = F.layer_norm(context, context.shape[-1:])
        return self.norm(task_query + context)


class AttentionMapPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        attention_dim = hidden_dim // 8
        joint_dim = hidden_dim + attention_dim
        self.fused_feature_upsampler = nn.Sequential(
            nn.Conv2d(hidden_dim, 96, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, attention_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(attention_dim, attention_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.high_feature_norm = nn.GroupNorm(1, hidden_dim)
        self.fused_feature_norm = nn.GroupNorm(1, attention_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, joint_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(joint_dim),
        )
        self.attended_feature_norm = nn.GroupNorm(1, joint_dim)
        self.map_decoder = nn.Sequential(
            nn.ConvTranspose2d(joint_dim, 160, 3, stride=1, padding=1),
            nn.GroupNorm(8, 160),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(160, 80, 4, stride=2, padding=1),
            nn.GroupNorm(8, 80),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(80, 1, 4, stride=2, padding=1),
        )
        final_projection = self.map_decoder[-1]
        nn.init.normal_(final_projection.weight, mean=0.0, std=0.01)
        nn.init.constant_(final_projection.bias, -2.0)
        self.query_scale = joint_dim ** (-0.5)

    def forward(self, high_features, fused_features, task_query):
        if high_features.shape[-2:] != (56, 56):
            raise ValueError(
                f"Expected high-resolution features [B,C,56,56], got {tuple(high_features.shape)}"
            )
        upsampled_fused = self.fused_feature_upsampler(fused_features)
        if upsampled_fused.shape[-2:] != high_features.shape[-2:]:
            raise ValueError(
                "Upsampled fused features and high-resolution features must share a grid"
            )
        joint_features = torch.cat(
            (
                self.high_feature_norm(high_features),
                self.fused_feature_norm(upsampled_fused),
            ),
            dim=1,
        )
        query = self.query_projection(task_query.squeeze(1))
        spatial_scores = (
            torch.einsum("bc,bchw->bhw", query, joint_features) * self.query_scale
        )
        height, width = spatial_scores.shape[-2:]
        spatial_probability = F.softmax(spatial_scores.flatten(1), dim=1).reshape_as(
            spatial_scores
        )
        spatial_attention = -torch.expm1(-spatial_probability * (height * width))
        attended_features = self.attended_feature_norm(
            joint_features * spatial_attention.unsqueeze(1)
        )
        map_logits = self.map_decoder(attended_features).squeeze(1)
        if map_logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected attention logits [B,224,224], got {tuple(map_logits.shape)}"
            )
        return map_logits


class GazeSearchGlobalLocalModel(nn.Module):
    def __init__(
        self,
        num_tasks=13,
        hidden_dim=384,
        num_heads=4,
        encoder_layers=3,
        decoder_layers=6,
        max_history=6,
        dropout=0.0,
        weights=False,
        freeze_backbone=True,
        text_dim=128,
        attention_lambda1=1.0,
        attention_lambda2=-1.0,
        history_gaussian_sigma=0.75,
        attention_epsilon=1e-06,
        task_condition_scale=0.5,
    ):
        super().__init__()
        if hidden_dim < 8 or hidden_dim % 8:
            raise ValueError("hidden_dim must be divisible by 8")
        if attention_lambda1 <= 0 or attention_lambda2 >= 0:
            raise ValueError(
                "attention_lambda1 must be positive and attention_lambda2 negative"
            )
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_history = max_history
        self.text_dim = text_dim
        self.history_gaussian_sigma = history_gaussian_sigma
        self.attention_epsilon = attention_epsilon
        self.task_condition_scale = task_condition_scale
        self.image_encoder = ResNetFeatureExtractor(
            hidden_dim, weights, freeze_backbone
        )
        self.register_buffer(
            "high_position",
            sine_position_encoding(56, 56, hidden_dim, torch.device("cpu")),
            persistent=False,
        )
        self.register_buffer(
            "low_position",
            sine_position_encoding(7, 7, hidden_dim, torch.device("cpu")),
            persistent=False,
        )
        history_y, history_x = torch.meshgrid(
            torch.arange(7), torch.arange(7), indexing="ij"
        )
        self.register_buffer("history_x", history_x.float(), persistent=False)
        self.register_buffer("history_y", history_y.float(), persistent=False)
        self.register_buffer("time_index", torch.arange(max_history), persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.memory_encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.temporal_embedding = nn.Embedding(max_history, hidden_dim)
        self.memory_type_embedding = nn.Embedding(2, hidden_dim)
        self.local_text_projector = TextProjector(text_dim, hidden_dim)
        self.global_text_projector = TextProjector(text_dim, hidden_dim)
        self.semantic_relation_encoder = SemanticRelationEncoder(
            hidden_dim, num_heads, dropout
        )
        self.local_semantic_fusion = LocalSemanticFusion(hidden_dim, num_heads, dropout)
        self.global_query_refiner = GlobalQueryRefiner(hidden_dim, num_heads, dropout)
        self.attention_map_predictor = AttentionMapPredictor(hidden_dim)
        self.raw_attention_lambda1 = nn.Parameter(
            torch.tensor(math.log(math.expm1(attention_lambda1)), dtype=torch.float32)
        )
        self.raw_attention_lambda2 = nn.Parameter(
            torch.tensor(math.log(math.expm1(-attention_lambda2)), dtype=torch.float32)
        )
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim,
            num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.fixation_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.decoder_head = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.duration_head = MLP(hidden_dim + 1, hidden_dim, 1)
        self.termination_head = MLP(hidden_dim + 1, hidden_dim, 1)
        self.coordinate_head = nn.Conv2d(1, 1, 3, padding=1, padding_mode="replicate")

    @property
    def attention_lambda1(self):
        return F.softplus(self.raw_attention_lambda1)

    @property
    def attention_lambda2(self):
        return -F.softplus(self.raw_attention_lambda2)

    def encode_image_text(self, images, global_text_feature, local_text_features):
        if (
            global_text_feature.ndim != 2
            or global_text_feature.shape[-1] != self.text_dim
        ):
            raise ValueError(
                f"global_text_feature must be [B,{self.text_dim}], got {tuple(global_text_feature.shape)}"
            )
        high_visual, low_visual = self.image_encoder(images)
        batch_size, _, low_h, low_w = low_visual.shape
        if high_visual.shape[-2:] != (56, 56) or (low_h, low_w) != (7, 7):
            raise ValueError("Expected encoder feature grids of 56x56 and 7x7")
        if local_text_features.shape != (batch_size, low_h * low_w, self.text_dim):
            raise ValueError(
                f"local_text_features must be [B,{low_h * low_w},{self.text_dim}], got {tuple(local_text_features.shape)}"
            )
        high_features = high_visual + self.high_position.to(dtype=high_visual.dtype)
        low_features = low_visual + self.low_position.to(dtype=low_visual.dtype)
        local_text = self.local_text_projector(local_text_features)
        local_text = self.semantic_relation_encoder(local_text, low_h, low_w)
        low_tokens = low_features.flatten(2).transpose(1, 2)
        fused_features = self.local_semantic_fusion(
            low_tokens, local_text, low_h, low_w
        )
        global_text = self.global_text_projector(global_text_feature).unsqueeze(1)
        return (high_features, fused_features, global_text)

    def _history_attention_map(self, history, padding_mask, height, width):
        if (height, width) != (7, 7):
            raise ValueError(f"Expected a 7x7 history grid, got {height}x{width}")
        x_grid = self.history_x.to(dtype=history.dtype)
        y_grid = self.history_y.to(dtype=history.dtype)
        center_x = history[..., 0, None, None] * 6
        center_y = history[..., 1, None, None] * 6
        squared_distance = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        gaussian = torch.exp(-squared_distance / (2 * self.history_gaussian_sigma**2))
        gaussian = gaussian * (~padding_mask)[..., None, None]
        history_map = gaussian.sum(dim=1)
        maximum = history_map.flatten(1).amax(dim=1).clamp_min(self.attention_epsilon)
        return history_map / maximum[:, None, None]

    def _decode_with_global_context(
        self,
        high_features,
        fused_features,
        global_text,
        task_ids,
        history,
        padding_mask,
    ):
        batch_size, _, high_h, high_w = high_features.shape
        low_h, low_w = fused_features.shape[-2:]
        x_index = (history[..., 0] * high_w).long().clamp(0, high_w - 1)
        y_index = (history[..., 1] * high_h).long().clamp(0, high_h - 1)
        batch_index = torch.arange(batch_size, device=history.device)[:, None]
        fixation_features = high_features.permute(0, 2, 3, 1)[
            batch_index, y_index, x_index
        ]
        fixation_features = (
            fixation_features + self.temporal_embedding(self.time_index)[None]
        )
        fixation_features = fixation_features + self.memory_type_embedding.weight[1]
        low_tokens = fused_features.flatten(2).transpose(1, 2)
        low_tokens = low_tokens + self.memory_type_embedding.weight[0]
        memory = torch.cat((low_tokens, fixation_features), dim=1)
        memory_padding = torch.cat(
            (padding_mask.new_zeros(batch_size, 49), padding_mask), dim=1
        )
        memory = self.memory_encoder(memory, src_key_padding_mask=memory_padding)
        memory = memory[:, :49]
        raw_task_query = self.task_embedding(task_ids).unsqueeze(1)
        task_query = self.global_query_refiner(raw_task_query, global_text)
        attention_map_logits = self.attention_map_predictor(
            high_features, fused_features, task_query
        )
        attention_map = torch.sigmoid(attention_map_logits)
        attention_prior = F.adaptive_avg_pool2d(
            attention_map.unsqueeze(1), (low_h, low_w)
        ).squeeze(1)
        history_prior = self._history_attention_map(history, padding_mask, low_h, low_w)
        attention_bias = (
            self.attention_lambda1 * torch.log(attention_prior + self.attention_epsilon)
            + self.attention_lambda2 * history_prior
        )
        attention_bias = attention_bias.flatten(1).unsqueeze(1)
        attention_bias = attention_bias.repeat_interleave(self.num_heads, dim=0)
        decoded = self.fixation_decoder(
            task_query, memory, memory_mask=attention_bias
        ).squeeze(1)
        conditioned_decoded = (
            decoded + self.task_condition_scale * raw_task_query.squeeze(1)
        )
        history_length = (~padding_mask).sum(dim=1, dtype=decoded.dtype).unsqueeze(1)
        head_input = torch.cat((conditioned_decoded, history_length), dim=1)
        coordinate_embedding = self.decoder_head(conditioned_decoded)
        coordinate_logits = torch.einsum(
            "bc,bchw->bhw", coordinate_embedding, high_features
        )
        coordinate_logits = self.coordinate_head(
            coordinate_logits.unsqueeze(1)
        ).squeeze(1)
        return {
            "coordinate_logits": coordinate_logits,
            "pred_duration": F.softplus(self.duration_head(head_input).squeeze(1)),
            "pred_termination": self.termination_head(head_input).squeeze(1),
            "attention_map": attention_map,
            "attention_map_logits": attention_map_logits,
            "clinical_attention": attention_prior,
            "attention_lambda1": self.attention_lambda1,
            "attention_lambda2": self.attention_lambda2,
            "task_query": task_query.squeeze(1),
            "raw_task_query": raw_task_query.squeeze(1),
        }

    def forward(
        self,
        images,
        task_ids,
        history,
        padding_mask,
        global_text_feature,
        local_text_features,
    ):
        high_features, fused_features, global_text = self.encode_image_text(
            images, global_text_feature, local_text_features
        )
        outputs = self._decode_with_global_context(
            high_features, fused_features, global_text, task_ids, history, padding_mask
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        images,
        task_ids,
        global_text_feature,
        local_text_features,
        max_steps=6,
        termination_threshold=0.5,
        sample=False,
        enforce_ior=True,
        ior_radius=16.0,
        edge_margin=2,
    ):
        self.eval()
        high_features, fused_features, global_text = self.encode_image_text(
            images, global_text_feature, local_text_features
        )
        batch_size = len(images)
        histories = torch.zeros(batch_size, self.max_history, 2, device=images.device)
        padding_mask = torch.ones(
            batch_size, self.max_history, dtype=torch.bool, device=images.device
        )
        histories[:, 0] = 0.5
        padding_mask[:, 0] = False
        coordinates = [
            [torch.tensor([IMAGE_SIZE / 2, IMAGE_SIZE / 2])] for _ in range(batch_size)
        ]
        durations = [[torch.tensor(0.3)] for _ in range(batch_size)]
        attention_maps = None
        active = torch.ones(batch_size, dtype=torch.bool, device=images.device)
        yy, xx = torch.meshgrid(
            torch.arange(IMAGE_SIZE, device=images.device),
            torch.arange(IMAGE_SIZE, device=images.device),
            indexing="ij",
        )
        for step in range(min(max_steps, self.max_history)):
            outputs = self._decode_with_global_context(
                high_features,
                fused_features,
                global_text,
                task_ids,
                histories,
                padding_mask,
            )
            if attention_maps is None:
                attention_maps = outputs["attention_map"].detach().cpu()
            stop = torch.sigmoid(outputs["pred_termination"]) > termination_threshold
            coordinate_scores = F.interpolate(
                outputs["coordinate_logits"].unsqueeze(1),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            if enforce_ior:
                for batch_index in range(batch_size):
                    for point in coordinates[batch_index]:
                        visited = (xx - point[0].to(images.device)) ** 2 + (
                            yy - point[1].to(images.device)
                        ) ** 2
                        coordinate_scores[batch_index][
                            visited <= ior_radius**2
                        ] = -torch.inf
            if edge_margin > 0:
                coordinate_scores[:, :edge_margin] = -torch.inf
                coordinate_scores[:, -edge_margin:] = -torch.inf
                coordinate_scores[:, :, :edge_margin] = -torch.inf
                coordinate_scores[:, :, -edge_margin:] = -torch.inf
            flat = coordinate_scores.flatten(1)
            if sample:
                probabilities = F.softmax(flat, dim=1)
                next_index = torch.multinomial(probabilities, 1).squeeze(1)
            else:
                next_index = flat.argmax(dim=1)
            next_xy = torch.stack(
                (next_index % IMAGE_SIZE, next_index // IMAGE_SIZE), dim=1
            ).float()
            append = active & ~stop
            for batch_index in append.nonzero(as_tuple=False).flatten().tolist():
                coordinates[batch_index].append(next_xy[batch_index].cpu())
                durations[batch_index].append(
                    outputs["pred_duration"][batch_index].cpu()
                )
            active &= ~stop
            if not active.any():
                break
            next_slot = step + 1
            if next_slot < self.max_history:
                histories[append, next_slot] = next_xy[append] / IMAGE_SIZE
                padding_mask[append, next_slot] = False
        if attention_maps is None:
            raise RuntimeError("Generation produced no attention maps")
        return [
            {
                "coordinates": torch.stack(points).float(),
                "durations": torch.stack(times).float(),
                "attention_map": attention_maps[index],
            }
            for index, (points, times) in enumerate(zip(coordinates, durations))
        ]
