import torch
import torch.nn as nn

from model.houseganpp_models import (
    Discriminator,
    add_pool,
    compute_gradient_penalty,
)


MODEL_NAME = "transformer_gen"


def _infer_graph_ids(num_nodes, edges, device):
    if edges is None or edges.numel() == 0:
        return torch.arange(num_nodes, device=device, dtype=torch.long)

    parent = torch.arange(num_nodes, device=device, dtype=torch.long)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for src, _, dst in edges.view(-1, 3).long():
        src_root = find(int(src.item()))
        dst_root = find(int(dst.item()))
        if src_root != dst_root:
            parent[dst_root] = src_root

    roots = torch.empty(num_nodes, device=device, dtype=torch.long)
    for idx in range(num_nodes):
        roots[idx] = find(idx)

    _, inverse = torch.unique(roots, return_inverse=True)
    return inverse


class MaskEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.cnn(x)).flatten(1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.num_heads = num_heads

    def forward(self, x, key_pad_mask=None, attn_bias=None):
        normed = self.norm1(x)
        if attn_bias is not None:
            B, L, _ = attn_bias.shape
            attn_mask = attn_bias.unsqueeze(1).expand(B, self.num_heads, L, L)
            attn_mask = attn_mask.reshape(B * self.num_heads, L, L)
        else:
            attn_mask = None
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_pad_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class RoomTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.pos_edge_bias = nn.Parameter(torch.zeros(1))
        self.neg_edge_bias = nn.Parameter(torch.zeros(1))

    def _build_global_bias(self, num_nodes, edges, device):
        bias = torch.zeros(num_nodes, num_nodes, device=device)
        if edges is None or edges.numel() == 0:
            return bias
        e = edges.view(-1, 3).long()

        def scatter_bias(mask, scalar):
            if not mask.any():
                return 0
            src = e[mask, 0]
            dst = e[mask, 2]
            all_s = torch.cat([src, dst])
            all_d = torch.cat([dst, src])
            flat = all_s * num_nodes + all_d
            indicator = torch.zeros(num_nodes * num_nodes, device=device)
            indicator = indicator.scatter_add(
                0, flat, torch.ones(flat.numel(), dtype=torch.float, device=device)
            )
            return indicator.view(num_nodes, num_nodes) * scalar

        bias = bias + scatter_bias(e[:, 1] > 0, self.pos_edge_bias)
        bias = bias + scatter_bias(e[:, 1] < 0, self.neg_edge_bias)
        return bias

    def forward(self, tokens, edges):
        num_nodes = tokens.size(0)
        d_model = tokens.size(-1)
        graph_ids = _infer_graph_ids(num_nodes, edges, tokens.device)
        num_graphs = int(graph_ids.max().item()) + 1
        counts = torch.bincount(graph_ids, minlength=num_graphs)
        max_nodes = int(counts.max().item())

        global_bias = self._build_global_bias(num_nodes, edges, tokens.device)

        node_indices = []
        padded = tokens.new_zeros(num_graphs, max_nodes, d_model)
        key_pad_mask = torch.ones(num_graphs, max_nodes, dtype=torch.bool, device=tokens.device)
        for g in range(num_graphs):
            idx = torch.nonzero(graph_ids == g, as_tuple=False).flatten()
            n = idx.numel()
            padded[g, :n] = tokens[idx]
            key_pad_mask[g, :n] = False
            node_indices.append(idx)

        attn_bias = tokens.new_zeros(num_graphs, max_nodes, max_nodes)
        for g in range(num_graphs):
            idx = node_indices[g]
            n = idx.numel()
            attn_bias[g, :n, :n] = global_bias[idx][:, idx]

        for layer in self.layers:
            padded = layer(padded, key_pad_mask, attn_bias)

        out = tokens.new_empty(tokens.shape)
        for g in range(num_graphs):
            idx = node_indices[g]
            n = idx.numel()
            out[idx] = padded[g, :n]
        return out


class SpatialDecoder(nn.Module):
    def __init__(self, d_model, channels=32):
        super().__init__()
        self.channels = channels
        self.init_fc = nn.Linear(d_model, channels * 8 * 8)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, tokens):
        x = self.init_fc(tokens).view(-1, self.channels, 8, 8)
        return self.upsample(x).view(-1, 64, 64)


class Generator(nn.Module):
    def __init__(self, noise_dim=128, cond_dim=18, d_model=128, num_heads=4, num_layers=4, dec_channels=32):
        super().__init__()
        self.noise_proj = nn.Linear(noise_dim, d_model)
        self.type_proj = nn.Linear(cond_dim, d_model)
        self.mask_enc = MaskEncoder(d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.transformer = RoomTransformerEncoder(d_model, num_heads, num_layers)
        self.decoder = SpatialDecoder(d_model, channels=dec_channels)

    def forward(self, z, given_m=None, given_y=None, given_w=None, given_v=None):
        z = z.view(-1, z.shape[-1])
        y = given_y.view(-1, given_y.shape[-1])

        token = self.noise_proj(z) + self.type_proj(y) + self.mask_enc(given_m)
        token = self.norm_in(token)

        token = self.transformer(token, given_w)
        return self.decoder(token)


__all__ = [
    "MODEL_NAME",
    "MaskEncoder",
    "TransformerBlock",
    "RoomTransformerEncoder",
    "SpatialDecoder",
    "Generator",
    "Discriminator",
    "add_pool",
    "compute_gradient_penalty",
]
