import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


MODEL_NAME = "graphgps_hybrid"


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = int(torch.max(nd_to_sample).item()) + 1
    pooled_x = torch.zeros(batch_size, x.shape[-1], dtype=dtype, device=device)
    pool_to = nd_to_sample.view(-1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x


def compute_gradient_penalty(
    D,
    x,
    x_fake,
    given_y=None,
    given_w=None,
    nd_to_sample=None,
    data_parallel=None,
    ed_to_sample=None,
):
    batch_size = int(torch.max(nd_to_sample).item()) + 1
    device = x.device
    u = torch.empty(x.shape[0], 1, 1, device=device).uniform_(0, 1)
    x_both = x.data * u + x_fake.data * (1 - u)
    x_both = x_both.requires_grad_(True)
    grad_outputs = torch.ones(batch_size, 1, device=device)
    out = D(x_both, given_y, given_w, nd_to_sample)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=x_both,
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty


def conv_block(
    in_channels,
    out_channels,
    k,
    s,
    p,
    act=None,
    upsample=False,
    spec_norm=False,
    batch_norm=False,
):
    block = []
    if upsample:
        layer = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=True
        )
    else:
        layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=True
        )
    block.append(spectral_norm(layer) if spec_norm else layer)
    if batch_norm:
        block.append(nn.InstanceNorm2d(out_channels, affine=True))
    if act == "leaky":
        block.append(nn.LeakyReLU(0.1, inplace=True))
    elif act == "relu":
        block.append(nn.ReLU(inplace=True))
    elif act == "tanh":
        block.append(nn.Tanh())
    return block


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            *conv_block(channels, channels, 3, 1, 1, act="leaky", batch_norm=True),
            *conv_block(channels, channels, 3, 1, 1, act=None, batch_norm=True),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            *conv_block(channels, channels, 4, 2, 1, act="leaky", upsample=True)
        )

    def forward(self, x):
        return self.net(x)


def _pool_neighbor_features(feats, edges, positive=True):
    dtype, device = feats.dtype, feats.device
    node_count = feats.size(0)
    channels, height, width = feats.shape[-3:]
    pooled = torch.zeros(node_count, channels, height, width, dtype=dtype, device=device)

    if edges is None or edges.numel() == 0:
        return pooled

    edges = edges.view(-1, 3)
    edge_mask = edges[:, 1] > 0 if positive else edges[:, 1] < 0
    edge_inds = torch.where(edge_mask)[0]
    if edge_inds.numel() == 0:
        return pooled

    src = torch.cat([edges[edge_inds, 0], edges[edge_inds, 2]]).long()
    dst = torch.cat([edges[edge_inds, 2], edges[edge_inds, 0]]).long()
    src_feats = feats[src]
    dst = dst.view(-1, 1, 1, 1).expand_as(src_feats)
    pooled.scatter_add_(0, dst, src_feats)
    return pooled


class SpatialCMP(nn.Module):
    """HouseGAN++-style spatial message passing over node feature maps."""

    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            *conv_block(3 * channels, 3 * channels, 3, 1, 1, act="leaky", batch_norm=True),
            ResidualBlock(3 * channels),
            ResidualBlock(3 * channels),
            *conv_block(3 * channels, channels, 3, 1, 1, act="leaky", batch_norm=True),
        )

    def forward(self, feats, edges=None):
        pos = _pool_neighbor_features(feats, edges, positive=True)
        neg = _pool_neighbor_features(feats, edges, positive=False)
        return self.encoder(torch.cat([feats, pos, neg], dim=1))


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


def _batched_attention(tokens, graph_ids, attn_module):
    num_graphs = int(graph_ids.max().item()) + 1
    counts = torch.bincount(graph_ids, minlength=num_graphs)
    max_nodes = int(counts.max().item())
    channels = tokens.size(-1)
    padded = tokens.new_zeros(num_graphs, max_nodes, channels)
    mask = torch.ones(num_graphs, max_nodes, dtype=torch.bool, device=tokens.device)

    for graph_idx in range(num_graphs):
        node_idx = torch.nonzero(graph_ids == graph_idx, as_tuple=False).flatten()
        n_nodes = node_idx.numel()
        padded[graph_idx, :n_nodes] = tokens[node_idx]
        mask[graph_idx, :n_nodes] = False

    attended, _ = attn_module(
        padded, padded, padded, key_padding_mask=mask, need_weights=False
    )

    out = tokens.new_empty(tokens.shape)
    for graph_idx in range(num_graphs):
        node_idx = torch.nonzero(graph_ids == graph_idx, as_tuple=False).flatten()
        n_nodes = node_idx.numel()
        out[node_idx] = attended[graph_idx, :n_nodes]
    return out


class HybridGraphGPSBlock(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.local_spatial = SpatialCMP(channels)
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden_dim = channels * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.spatial_refine = nn.Sequential(
            ResidualBlock(channels),
            *conv_block(channels, channels, 3, 1, 1, act="leaky", batch_norm=True),
        )
        self.local_scale = nn.Parameter(torch.tensor(1.0))
        self.global_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edges):
        local_delta = self.local_spatial(x, edges)

        pooled = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        graph_ids = _infer_graph_ids(x.size(0), edges, x.device)
        attn_out = _batched_attention(self.norm1(pooled), graph_ids, self.attn)
        tokens = pooled + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        global_delta = (tokens - pooled).unsqueeze(-1).unsqueeze(-1)

        x = x + self.local_scale * local_delta + self.global_scale * global_delta
        x = x + self.spatial_refine(x)
        return x


class Generator(nn.Module):
    def __init__(self, channels=16, noise_dim=128, cond_dim=18):
        super().__init__()
        self.init_size = 32 // 4
        self.channels = channels
        self.noise_dim = noise_dim
        self.cond_dim = cond_dim

        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, channels * self.init_size**2)
        )
        self.enc_1 = nn.Sequential(
            *conv_block(2, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, channels, 3, 2, 1, act="leaky"),
        )
        self.enc_2 = nn.Sequential(
            *conv_block(2 * channels, 2 * channels, 3, 1, 1, act="leaky"),
            *conv_block(2 * channels, channels, 3, 1, 1, act="leaky"),
        )

        self.gps_1 = HybridGraphGPSBlock(channels)
        self.upsample_1 = UpsampleBlock(channels)
        self.gps_2 = HybridGraphGPSBlock(channels)
        self.upsample_2 = UpsampleBlock(channels)
        self.gps_3 = HybridGraphGPSBlock(channels)
        self.upsample_3 = UpsampleBlock(channels)
        self.gps_4 = HybridGraphGPSBlock(channels)

        self.decoder = nn.Sequential(
            ResidualBlock(channels),
            *conv_block(channels, 256, 3, 1, 1, act="leaky"),
            *conv_block(256, 128, 3, 1, 1, act="leaky"),
            *conv_block(128, 1, 3, 1, 1, act="tanh"),
        )

    def forward(self, z, given_m=None, given_y=None, given_w=None, given_v=None):
        z = z.view(-1, self.noise_dim)
        y = given_y.view(-1, self.cond_dim)
        z = torch.cat([z, y], 1)

        x = self.l1(z)
        f = x.view(-1, self.channels, self.init_size, self.init_size)

        m = self.enc_1(given_m)
        f = torch.cat([f, m], 1)
        f = self.enc_2(f)

        x = self.gps_1(f, given_w)
        x = self.upsample_1(x)
        x = self.gps_2(x, given_w)
        x = self.upsample_2(x)
        x = self.gps_3(x, given_w)
        x = self.upsample_3(x)
        x = self.gps_4(x, given_w)
        x = self.decoder(x)
        x = x.view(-1, x.shape[2], x.shape[3])
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            *conv_block(9, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
            *conv_block(16, 16, 3, 1, 1, act="leaky"),
        )
        self.l1 = nn.Sequential(nn.Linear(18, 8 * 64**2))
        self.cmp_1 = SpatialCMP(16)
        self.downsample_1 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_2 = SpatialCMP(16)
        self.downsample_2 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_3 = SpatialCMP(16)
        self.downsample_3 = nn.Sequential(*conv_block(16, 16, 3, 2, 1, act="leaky"))
        self.cmp_4 = SpatialCMP(16)
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 2, 1, act="leaky"),
            *conv_block(256, 128, 3, 2, 1, act="leaky"),
            *conv_block(128, 128, 3, 2, 1, act="leaky"),
        )
        self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
        x = x.view(-1, 1, 64, 64)
        y = self.l1(given_y)
        y = y.view(-1, 8, 64, 64)
        x = torch.cat([x, y], 1)
        x = self.encoder(x)
        x = self.cmp_1(x, given_w)
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w)
        x = self.downsample_2(x)
        x = self.cmp_3(x, given_w)
        x = self.downsample_3(x)
        x = self.cmp_4(x, given_w)
        x = self.decoder(x)
        x = x.view(-1, x.shape[1])
        x_g = add_pool(x, nd_to_sample)
        return self.fc_layer_global(x_g)


__all__ = [
    "MODEL_NAME",
    "add_pool",
    "compute_gradient_penalty",
    "conv_block",
    "ResidualBlock",
    "SpatialCMP",
    "HybridGraphGPSBlock",
    "Generator",
    "Discriminator",
]
