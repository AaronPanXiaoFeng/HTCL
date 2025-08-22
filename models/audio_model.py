from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat


class FeatureExtractor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 conv_filters: Union[Tuple[int], List[int]],
                 conv_strides: Union[Tuple[int], List[int]],
                 conv_kernels: Union[Tuple[int], List[int]],
                 conv_bias: bool = False,
                 layer_norm_eps: float = 1e-05,
                 activation: str = 'gelu'
                 ):
        assert len(conv_filters) == len(conv_strides) == len(conv_kernels)

        super().__init__()
        self.layers = []
        conv_filters = [input_dim] + list(conv_filters)
        for i in range(1, len(conv_filters)):
            self.layers.append(
                nn.Sequential(
                    Rearrange('b t c -> b c t'),
                    nn.Conv1d(conv_filters[i - 1], conv_filters[i], kernel_size=conv_kernels[i - 1],
                              stride=conv_strides[i - 1], bias=conv_bias),
                    Rearrange('b c t -> b t c'),
                    nn.LayerNorm(conv_filters[i], eps=layer_norm_eps),
                    {'relu': nn.ReLU(), 'gelu': nn.GELU()}[activation]
                )
            )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FeatureProjection(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, layer_norm_eps: float = 1e-05, dropout: float = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.projection = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PositionalConvEmbedding(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, groups: int, layer_norm_eps: float = 1e-05,
                 activation: str = 'gelu'):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups
        )

        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = {'relu': nn.ReLU(), 'gelu': nn.GELU()}[activation]
        self.num_pad_remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.conv(hidden_states)
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class AudioModel(nn.Module):

    def __init__(self,
                 input_dim=128,
                 conv_filters=(512, 512, 512),
                 conv_strides=(2, 2, 2),
                 conv_kernels=(5, 3, 3),
                 conv_bias=False,
                 layer_norm_eps=1e-05,
                 feat_extract_activation='gelu',
                 hidden_size=768,
                 feat_proj_dropout=0.0,
                 kernel_size_pos_conv_emb=31,
                 groups_pos_conv_emb=16,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout=0.1,
                 ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(
            input_dim, conv_filters, conv_strides, conv_kernels, conv_bias, layer_norm_eps, feat_extract_activation
        )

        self.feature_projection = FeatureProjection(
            conv_filters[-1], hidden_size, layer_norm_eps=layer_norm_eps, dropout=feat_proj_dropout
        )

        self.pos_conv_embed = PositionalConvEmbedding(
            hidden_size, kernel_size_pos_conv_emb, groups_pos_conv_emb, layer_norm_eps=layer_norm_eps,
            activation=feat_extract_activation
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=intermediate_size, dropout=hidden_dropout,
            activation=hidden_act, layer_norm_eps=layer_norm_eps, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_hidden_layers)

        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, inp, attention_mask: Optional[torch.Tensor] = None):
        extract_features = self.feature_extractor(inp)
        hidden_states = self.feature_projection(extract_features)

        if attention_mask is not None:
            hidden_states[attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        b, t, d = hidden_states.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        if attention_mask is not None:
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=attention_mask.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)

        hidden_states = self.transformer_encoder(hidden_states, src_key_padding_mask=attention_mask)
        return hidden_states


if __name__ == '__main__':
    print(torch.__version__)

    model = AudioModel(
        input_dim=128,
        conv_filters=(512, 512, 512),
        conv_strides=(3, 2, 2),
        conv_kernels=(5, 3, 2),
        conv_bias=False,
        layer_norm_eps=1e-08,
        feat_extract_activation='gelu',
        hidden_size=768,
        feat_proj_dropout=0.0,
        kernel_size_pos_conv_emb=15,
        groups_pos_conv_emb=16,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act='gelu',
        hidden_dropout=0.1,
    )

    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num_params: {num_params}')
    # num_params: 87648527

    x = torch.randn(4, 17, 128)  # minimum length: 17

    y = model(x)
    print(y.shape)  # torch.Size([4, 2, 768])
    print(y.requires_grad)  # True

    representation = y[:, 0]
    print(representation.shape)  # torch.Size([4, 768])
