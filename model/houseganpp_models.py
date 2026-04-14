import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


MODEL_NAME = "houseganpp"


class MeanPoolConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_input, n_output, k_size, stride=1,
                      padding=(k_size - 1) // 2, bias=True)
        )

    def forward(self, x):
        out = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] +
               x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4.0
        return self.model(out)


class ConvMeanPool(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_input, n_output, k_size, stride=1,
                      padding=(k_size - 1) // 2, bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return (out[:, :, ::2, ::2] + out[:, :, 1::2, ::2] +
                out[:, :, ::2, 1::2] + out[:, :, 1::2, 1::2]) / 4.0


class UpsampleConv(nn.Module):
    def __init__(self, n_input, n_output, k_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(n_input, n_output, k_size, stride=1,
                      padding=(k_size - 1) // 2, bias=True)
        )

    def forward(self, x):
        x = x.repeat(1, 4, 1, 1)
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size, resample='up',
                 bn=True, spatial_dim=None):
        super().__init__()
        self.resample = resample

        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size,
                                   padding=(k_size - 1) // 2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size,
                                   padding=(k_size - 1) // 2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size,
                                   padding=(k_size - 1) // 2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size,
                                   padding=(k_size - 1) // 2)
            self.conv_shortcut = None
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]

        norm = nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims)
        out_norm = nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims)
        self.model = nn.Sequential(
            norm, nn.ReLU(inplace=True), self.conv1,
            out_norm, nn.ReLU(inplace=True), self.conv2,
        )

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        return self.conv_shortcut(x) + self.model(x)


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = int(torch.max(nd_to_sample).item()) + 1
    pooled_x = torch.zeros(batch_size, *x.shape[1:], dtype=dtype, device=device)
    view_shape = (-1,) + (1,) * (x.dim() - 1)
    pool_to = nd_to_sample.view(*view_shape).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x


def compute_gradient_penalty(
    D, x, x_fake,
    given_y=None, given_w=None, nd_to_sample=None,
    data_parallel=None, ed_to_sample=None,
):
    batch_size = int(torch.max(nd_to_sample).item()) + 1
    device = x.device
    u = torch.empty(x.shape[0], 1, 1, device=device).uniform_(0, 1)
    x_both = x.data * u + x_fake.data * (1 - u)
    x_both = x_both.requires_grad_(True)
    grad_outputs = torch.ones(batch_size, 1, device=device)
    out = D(x_both, given_y, given_w, nd_to_sample)
    grad = torch.autograd.grad(
        outputs=out, inputs=x_both, grad_outputs=grad_outputs,
        retain_graph=True, create_graph=True, only_inputs=True,
    )[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty


def conv_block(in_channels, out_channels, k, s, p,
               act=None, upsample=False, spec_norm=False, batch_norm=True):
    block = []
    if upsample:
        layer = nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=k, stride=s, padding=p, bias=True)
    else:
        layer = nn.Conv2d(in_channels, out_channels,
                          kernel_size=k, stride=s, padding=p, bias=True)
    block.append(spectral_norm(layer) if spec_norm else layer)
    if batch_norm:
        block.append(nn.InstanceNorm2d(out_channels))
    if act and "leaky" in act:
        block.append(nn.LeakyReLU(0.1, inplace=True))
    elif act and "relu" in act:
        block.append(nn.ReLU(inplace=True))
    elif act and "tanh" in act:
        block.append(nn.Tanh())
    return block


class CMP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(3 * in_channels, 3 * in_channels, 3, resample=None),
            ResidualBlock(3 * in_channels, 3 * in_channels, 3, resample=None),
            *conv_block(3 * in_channels, in_channels, 3, 1, 1, act="relu"),
        )

    def forward(self, feats, edges=None):
        dtype, device = feats.dtype, feats.device
        if edges is None or edges.numel() == 0:
            zeros = torch.zeros_like(feats)
            return self.encoder(torch.cat([feats, zeros, zeros], 1))

        edges = edges.view(-1, 3)
        V = feats.size(0)
        C, H, W = feats.shape[-3:]
        pooled_v_pos = torch.zeros(V, C, H, W, dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(V, C, H, W, dtype=dtype, device=device)

        pos_inds = torch.where(edges[:, 1] > 0)[0]
        if pos_inds.numel() > 0:
            pos_v_src = torch.cat([edges[pos_inds, 0], edges[pos_inds, 2]]).long()
            pos_v_dst = torch.cat([edges[pos_inds, 2], edges[pos_inds, 0]]).long()
            pos_vecs_src = feats[pos_v_src]
            pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src)
            pooled_v_pos.scatter_add_(0, pos_v_dst, pos_vecs_src)

        neg_inds = torch.where(edges[:, 1] < 0)[0]
        if neg_inds.numel() > 0:
            neg_v_src = torch.cat([edges[neg_inds, 0], edges[neg_inds, 2]]).long()
            neg_v_dst = torch.cat([edges[neg_inds, 2], edges[neg_inds, 0]]).long()
            neg_vecs_src = feats[neg_v_src]
            neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src)
            pooled_v_neg.scatter_add_(0, neg_v_dst, neg_vecs_src)

        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        return self.encoder(enc_in)


class Generator(nn.Module):
    def __init__(self, noise_dim=128, cond_dim=18, hidden_dim=16):
        super().__init__()
        self.init_size = 8
        self.hidden_dim = hidden_dim

        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim // 2 * self.init_size ** 2)
        )
        self.encoder = nn.Sequential(
            *conv_block(2, hidden_dim // 2, 3, 2, 1, act="relu"),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2, 3, resample='down'),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2, 3, resample='down'),
        )
        self.cmp_1 = CMP(hidden_dim)
        self.upsample_1 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')
        self.cmp_2 = CMP(hidden_dim)
        self.upsample_2 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')
        self.cmp_3 = CMP(hidden_dim)
        self.upsample_3 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='up')
        self.cmp_4 = CMP(hidden_dim)
        self.head = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            *conv_block(hidden_dim, 1, 3, 1, 1, act="tanh", batch_norm=False),
        )

    def forward(self, z, given_m=None, given_y=None, given_w=None, given_v=None):
        z = z.view(-1, z.shape[-1])
        y = given_y.view(-1, given_y.shape[-1])
        z = torch.cat([z, y], 1)

        x = self.l1(z)
        f = x.view(-1, self.hidden_dim // 2, self.init_size, self.init_size)
        p = self.encoder(given_m)
        f = torch.cat([f, p], 1)

        x = self.cmp_1(f, given_w)
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w)
        x = self.upsample_2(x)
        x = self.cmp_3(x, given_w)
        x = self.upsample_3(x)
        x = self.cmp_4(x, given_w)
        x = self.head(x)
        x = x.view(-1, x.shape[2], x.shape[3])
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.cmp_1 = CMP(hidden_dim)
        self.cmp_2 = CMP(hidden_dim)
        self.cmp_3 = CMP(hidden_dim)
        self.cmp_4 = CMP(hidden_dim)
        self.l1 = nn.Sequential(nn.Linear(18, 8 * 64 ** 2))
        self.downsample_1 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_2 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_3 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.downsample_4 = ResidualBlock(hidden_dim, hidden_dim, 3, resample='down')
        self.encoder = nn.Sequential(
            *conv_block(9, hidden_dim, 3, 1, 1, act="relu"),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample=None),
        )
        self.head_global_cnn = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
        )
        self.head_global_l1 = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.head_local_cnn = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
            ResidualBlock(hidden_dim, hidden_dim, 3, resample='down'),
        )
        self.head_local_l1 = nn.Sequential(nn.Linear(hidden_dim, 1))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
        x = x.view(-1, 1, 64, 64)
        y = self.l1(given_y).view(-1, 8, 64, 64)
        x = torch.cat([x, y], 1)

        x = self.encoder(x)
        x = self.cmp_1(x, given_w)
        x = self.downsample_1(x)
        x = self.cmp_2(x, given_w)
        x = self.downsample_2(x)
        x = self.cmp_3(x, given_w)
        x = self.downsample_4(x)
        x_l = self.cmp_4(x, given_w)

        x_g = add_pool(x_l, nd_to_sample)
        x_g = self.head_global_cnn(x_g)
        validity_global = self.head_global_l1(x_g.view(-1, x_g.shape[1]))

        x_l = self.head_local_cnn(x_l)
        x_l = add_pool(x_l, nd_to_sample)
        validity_local = self.head_local_l1(x_l.view(-1, x_l.shape[1]))

        return validity_global + validity_local


__all__ = [
    "MODEL_NAME",
    "ResidualBlock",
    "add_pool",
    "compute_gradient_penalty",
    "conv_block",
    "CMP",
    "Generator",
    "Discriminator",
]
