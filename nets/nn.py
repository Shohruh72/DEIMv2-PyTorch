import copy
import math

import torch

from utils import util


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


# ---------------------------------------------------------------- HGNetV2 backbone (Atto, Femto, Pico, N)
class LAB(torch.nn.Module):
    # Learnable Affine Block
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBN(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, groups=g, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_ch)
        self.act = torch.nn.ReLU() if act else torch.nn.Identity()
        self.lab = LAB() if act else torch.nn.Identity()

    def forward(self, x):
        return self.lab(self.act(self.bn(self.conv(x))))

    def fuse_forward(self, x):
        return self.lab(self.act(self.conv(x)))


class LightConvBN(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv1 = ConvBN(in_ch, out_ch, act=False)
        self.conv2 = ConvBN(out_ch, out_ch, k, g=out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Stem(torch.nn.Module):
    def __init__(self, mid_ch, out_ch):
        super().__init__()
        self.stem1 = ConvBN(3, mid_ch, k=3, s=2)
        self.stem2a = ConvBN(mid_ch, mid_ch // 2, k=2)
        self.stem2b = ConvBN(mid_ch // 2, mid_ch, k=2)
        self.stem3 = ConvBN(mid_ch * 2, mid_ch, k=3, s=2)
        self.stem4 = ConvBN(mid_ch, out_ch)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = torch.nn.functional.pad(self.stem1(x), (0, 1, 0, 1))
        y = self.stem2b(torch.nn.functional.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat(tensors=(self.pool(x), y), dim=1)))


class HGBlock(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, n, k, residual, light):
        super().__init__()
        self.residual = residual
        self.layers = torch.nn.ModuleList(LightConvBN(in_ch if i == 0 else mid_ch, mid_ch, k) if light else
                                          ConvBN(in_ch if i == 0 else mid_ch, mid_ch, k) for i in range(n))
        self.aggregation = torch.nn.Sequential(ConvBN(in_ch + n * mid_ch, out_ch // 2),
                                               ConvBN(out_ch // 2, out_ch))

    def forward(self, x):
        y = [x]
        for layer in self.layers:
            y.append(layer(y[-1]))
        y = self.aggregation(torch.cat(y, dim=1))
        return y + x if self.residual else y


class HGStage(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_blocks, down, light, k, n):
        super().__init__()
        self.downsample = ConvBN(in_ch, in_ch, k=3, s=2, g=in_ch, act=False) if down else torch.nn.Identity()
        self.blocks = torch.nn.Sequential(*(HGBlock(in_ch if i == 0 else out_ch, mid_ch, out_ch, n, k, i > 0, light)
                                            for i in range(num_blocks)))

    def forward(self, x):
        return self.blocks(self.downsample(x))


class HGNetV2(torch.nn.Module):
    url = 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'

    def __init__(self, stages, indices):
        super().__init__()
        self.indices = indices
        self.stem = Stem(16, 16)
        self.stages = torch.nn.ModuleList(HGStage(*args) for args in stages)

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.indices:
                outputs.append(x)
        return outputs

    def load(self, path=None):
        # ImageNet stage-1 weights of PP-HGNetV2-B0, shape-matched for the smaller variants
        src = torch.hub.load_state_dict_from_url(self.url, './weights', map_location='cpu')
        dst = {k: v for k, v in self.state_dict().items() if not k.endswith('num_batches_tracked')}
        ckpt = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}
        self.load_state_dict(ckpt, strict=False)
        print(f'Backbone: loaded {len(ckpt)}/{len(dst)} tensors from {self.url}')


# ---------------------------------------------------------------- ViT backbone (DINOv3 / distilled ViT-Tiny) + STA
class RoPE(torch.nn.Module):
    def __init__(self, dim, rescale=None):
        super().__init__()
        self.rescale = rescale
        self.register_buffer('periods', 100 ** (2 * torch.arange(dim // 4, dtype=torch.float32) / (dim // 2)))

    def forward(self, h, w):
        y = torch.arange(0.5, h, device=self.periods.device, dtype=torch.float32) / h
        x = torch.arange(0.5, w, device=self.periods.device, dtype=torch.float32) / w
        coords = 2.0 * torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1).flatten(0, 1) - 1.0
        if self.training and self.rescale is not None:
            s = math.log(self.rescale)
            coords *= torch.empty(1, device=coords.device).uniform_(-s, s).exp()
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return angles.sin(), angles.cos()


def apply_rope(x, sin, cos):
    # rotate patch tokens only, prefix (cls + register) tokens are left untouched
    n = x.shape[-2] - sin.shape[-2]
    y = x[..., n:, :].float()
    y1, y2 = y.chunk(2, dim=-1)
    y = y * cos + torch.cat(tensors=(-y2, y1), dim=-1) * sin
    return torch.cat(tensors=(x[..., :n, :], y.to(x.dtype)), dim=-2)


class Attention(torch.nn.Module):
    def __init__(self, dim, num_head, mask_k_bias):
        super().__init__()
        self.num_head = num_head
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)
        if mask_k_bias:
            self.register_buffer('bias_mask', torch.cat((torch.ones(dim), torch.zeros(dim), torch.ones(dim))))

    def forward(self, x, rope):
        b, n, c = x.shape
        bias = self.qkv.bias * self.bias_mask if hasattr(self, 'bias_mask') else self.qkv.bias
        qkv = torch.nn.functional.linear(x, self.qkv.weight, bias)
        q, k, v = qkv.view(b, n, 3, self.num_head, c // self.num_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if torch.onnx.is_in_onnx_export() and hasattr(self, 'bias_mask'):
            # SDPA accumulates softmax(q k) in FP32, the exported FP16 MatMul/Softmax breaks DINOv3 in TensorRT
            with torch.autocast('cuda', enabled=False):
                attn = ((q.float() * q.shape[-1] ** -0.5) @ k.float().transpose(-2, -1)).softmax(-1)
            x = attn.to(v.dtype) @ v
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class MLP(torch.nn.Module):
    def __init__(self, dim, ch):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, ch)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(ch, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwiGLU(torch.nn.Module):
    def __init__(self, dim, ch):
        super().__init__()
        ch = int(ch * 2 / 3)
        ch = ch + (-ch % 8)
        self.w1 = torch.nn.Linear(dim, ch)
        self.w2 = torch.nn.Linear(dim, ch)
        self.w3 = torch.nn.Linear(ch, dim)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


class LayerScale(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x):
        return x * self.gamma


class Block(torch.nn.Module):
    def __init__(self, dim, num_head, ffn_ratio, swiglu, dinov3):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, eps=1e-5 if dinov3 else 1e-6)
        self.attn = Attention(dim, num_head, mask_k_bias=dinov3)
        self.ls1 = LayerScale(dim) if dinov3 else torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(dim, eps=1e-5 if dinov3 else 1e-6)
        self.mlp = SwiGLU(dim, dim * ffn_ratio) if swiglu else MLP(dim, dim * ffn_ratio)
        self.ls2 = LayerScale(dim) if dinov3 else torch.nn.Identity()

    def forward(self, x, rope):
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class ViT(torch.nn.Module):
    # dinov3=True: DINOv3 ViT-S/S+ (4 register tokens, LayerScale, masked k-bias, final norm, RoPE rescale)
    # dinov3=False: DEIMv2 ViT-Tiny/Tiny+ distilled from DINOv3
    def __init__(self, dim, num_head, indices, ffn_ratio=4, swiglu=False, dinov3=False):
        super().__init__()
        self.dim = dim
        self.indices = indices
        self.num_prefix = 5 if dinov3 else 1

        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, dim))
        if dinov3:
            self.storage_tokens = torch.nn.Parameter(torch.zeros(1, 4, dim))
            self.mask_token = torch.nn.Parameter(torch.zeros(1, dim))
            self.norm = torch.nn.LayerNorm(dim, eps=1e-5)
        self.rope_embed = RoPE(dim // num_head, rescale=2 if dinov3 else None)
        self.blocks = torch.nn.ModuleList(Block(dim, num_head, ffn_ratio, swiglu, dinov3) for _ in range(12))

    def forward(self, x):
        x = self.patch_embed.proj(x)
        b, _, h, w = x.shape
        tokens = [self.cls_token.expand(b, -1, -1)]
        if hasattr(self, 'storage_tokens'):
            tokens = [(self.cls_token + 0 * self.mask_token).expand(b, -1, -1), self.storage_tokens.expand(b, -1, -1)]
        x = torch.cat(tensors=tokens + [x.flatten(2).transpose(1, 2)], dim=1)

        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope_embed(h, w))
            if i in self.indices:
                y = self.norm(x) if hasattr(self, 'norm') else x
                outputs.append(y[:, self.num_prefix:])
        return outputs


class STA(torch.nn.Module):
    # Spatial Tuning Adapter: light CNN that supplies fine-grained detail at 1/8, 1/16, 1/32
    def __init__(self, ch):
        super().__init__()
        self.stem = torch.nn.Sequential(torch.nn.Conv2d(3, ch, 3, 2, 1, bias=False),
                                        torch.nn.BatchNorm2d(ch), torch.nn.GELU(),
                                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(ch, 2 * ch, 3, 2, 1, bias=False),
                                         torch.nn.BatchNorm2d(2 * ch))
        self.conv3 = torch.nn.Sequential(torch.nn.GELU(),
                                         torch.nn.Conv2d(2 * ch, 4 * ch, 3, 2, 1, bias=False),
                                         torch.nn.BatchNorm2d(4 * ch))
        self.conv4 = torch.nn.Sequential(torch.nn.GELU(),
                                         torch.nn.Conv2d(4 * ch, 4 * ch, 3, 2, 1, bias=False),
                                         torch.nn.BatchNorm2d(4 * ch))

    def forward(self, x):
        p2 = self.conv2(self.stem(x))
        p3 = self.conv3(p2)
        return p2, p3, self.conv4(p3)


class DINOv3STA(torch.nn.Module):
    def __init__(self, vit, out_ch, sta_ch):
        super().__init__()
        self.dinov3 = vit
        self.sta = STA(sta_ch)
        in_ch = [vit.dim + sta_ch * 2, vit.dim + sta_ch * 4, vit.dim + sta_ch * 4]
        self.convs = torch.nn.ModuleList(torch.nn.Conv2d(ch, out_ch, kernel_size=1, bias=False) for ch in in_ch)
        self.norms = torch.nn.ModuleList(torch.nn.BatchNorm2d(out_ch) for _ in in_ch)
        # ImageNet normalization lives inside the model: every variant takes RGB in [0, 1]
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        b, _, h, w = x.shape
        h, w = h // 16, w // 16
        outputs = []
        for i, (y, z) in enumerate(zip(self.dinov3(x), self.sta(x))):
            y = y.transpose(1, 2).reshape(b, -1, h, w)
            y = torch.nn.functional.interpolate(y, size=(int(h * 2 ** (1 - i)), int(w * 2 ** (1 - i))),
                                                mode='bilinear', align_corners=False)
            outputs.append(self.norms[i](self.convs[i](torch.cat(tensors=(y, z), dim=1))))
        return outputs

    def load(self, path):
        # DINOv3 weights: https://github.com/facebookresearch/dinov3, ViT-Tiny(+): see README.md
        src = torch.load(path, map_location='cpu')
        self.dinov3.load_state_dict({k.replace('.qkv.bias_mask', '.bias_mask'): v for k, v in src.items()})
        print(f'Backbone: loaded {path}')


# ---------------------------------------------------------------- Encoder
class RepConv(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())
        self.relu = torch.nn.SiLU()

    def forward(self, x):
        return self.relu(self.conv1(x) + self.conv2(x))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch * 2, torch.nn.SiLU())
        self.bottlenecks = torch.nn.Sequential(*(RepConv(out_ch) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return x + self.bottlenecks(y)


class ELAN(torch.nn.Module):
    # RepNCSPELAN; `post_conv=True` is the D-FINE flavour with an extra 3x3 conv per branch
    def __init__(self, in_ch, out_ch, mid_ch1, mid_ch2, n, post_conv):
        super().__init__()
        extra = [Conv(mid_ch2, mid_ch2, torch.nn.SiLU(), k=3)] if post_conv else []
        self.cv1 = Conv(in_ch, mid_ch1, torch.nn.SiLU())
        self.cv2 = torch.nn.Sequential(CSP(mid_ch1 // 2, mid_ch2, n), *copy.deepcopy(extra))
        self.cv3 = torch.nn.Sequential(CSP(mid_ch2, mid_ch2, n), *copy.deepcopy(extra))
        self.cv4 = Conv(mid_ch1 + 2 * mid_ch2, out_ch, torch.nn.SiLU())

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in (self.cv2, self.cv3))
        return self.cv4(torch.cat(y, dim=1))


class SCDown(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cv1 = Conv(ch, ch, torch.nn.Identity())
        self.cv2 = Conv(ch, ch, torch.nn.Identity(), k=3, s=2, g=ch)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class AIFI(torch.nn.Module):
    # post-norm transformer encoder layer with 2D sin-cos position embedding
    def __init__(self, ch, num_head, ffn_ch):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(ch, num_head, batch_first=True)
        self.linear1 = torch.nn.Linear(ch, ffn_ch)
        self.linear2 = torch.nn.Linear(ffn_ch, ch)
        self.norm1 = torch.nn.LayerNorm(ch)
        self.norm2 = torch.nn.LayerNorm(ch)
        self.act = torch.nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        q = k = x + util.sincos_embedding(w, h, c).to(x.device)
        x = self.norm1(x + self.self_attn(q, k, value=x)[0])
        x = self.norm2(x + self.linear2(self.act(self.linear1(x))))
        return x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()


class HybridEncoder(torch.nn.Module):
    def __init__(self, in_ch, ch, ffn_ch, expansion, depth, post_conv=False):
        super().__init__()
        n = len(in_ch)
        block = ELAN(ch, ch, ch * 2, round(expansion * ch // 2), round(3 * depth), post_conv)
        self.input_proj = torch.nn.ModuleList(Conv(c, ch, torch.nn.Identity()) if c != ch else torch.nn.Identity()
                                              for c in in_ch)
        self.aifi = AIFI(ch, 8, ffn_ch)
        self.lateral_convs = torch.nn.ModuleList(Conv(ch, ch, torch.nn.Identity()) for _ in range(n - 1))
        self.fpn_blocks = torch.nn.ModuleList(copy.deepcopy(block) for _ in range(n - 1))
        self.downsample_convs = torch.nn.ModuleList(SCDown(ch) for _ in range(n - 1))
        self.pan_blocks = torch.nn.ModuleList(copy.deepcopy(block) for _ in range(n - 1))

    def forward(self, x):
        x = [m(i) for m, i in zip(self.input_proj, x)]
        x[-1] = self.aifi(x[-1])

        n = len(x)
        inner = [x[-1]]
        for i in range(n - 1, 0, -1):
            inner[0] = self.lateral_convs[n - 1 - i](inner[0])
            y = torch.nn.functional.interpolate(inner[0], scale_factor=2., mode='nearest')
            inner.insert(0, self.fpn_blocks[n - 1 - i](y + x[i - 1]))

        outputs = [inner[0]]
        for i in range(n - 1):
            outputs.append(self.pan_blocks[i](self.downsample_convs[i](outputs[-1]) + inner[i + 1]))
        return outputs


class LiteEncoder(torch.nn.Module):
    # single input level (1/16) -> two output levels (1/16, 1/32) with global-average fusion
    def __init__(self, in_ch, ch, expansion, depth):
        super().__init__()
        block = ELAN(ch, ch, ch * 2, round(expansion * ch // 2), round(3 * depth), post_conv=True)
        down = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                                   torch.nn.Conv2d(ch, ch, 1, bias=False),
                                   torch.nn.BatchNorm2d(ch), torch.nn.SiLU())
        self.input_proj = torch.nn.ModuleList([Conv(in_ch, ch, torch.nn.Identity())])
        self.down_sample1 = copy.deepcopy(down)
        self.down_sample2 = copy.deepcopy(down)
        self.bi_fusion = torch.nn.Module()
        self.bi_fusion.cv = Conv(ch, ch, torch.nn.SiLU())
        self.fpn_block = copy.deepcopy(block)
        self.pan_block = copy.deepcopy(block)

    def forward(self, x):
        p4 = self.input_proj[0](x[0])
        p5 = self.down_sample1(p4)
        p5 = self.bi_fusion.cv(p5 + torch.nn.functional.adaptive_avg_pool2d(p5, 1))
        p4 = self.fpn_block(p4 + torch.nn.functional.interpolate(p5, scale_factor=2., mode='nearest'))
        p5 = self.pan_block(p5 + self.down_sample2(p4))
        return [p4, p5]


# ---------------------------------------------------------------- Decoder
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.scale


class Head(torch.nn.Module):
    # MLP head
    def __init__(self, in_ch, ch, out_ch, n):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(i, o) for i, o in zip([in_ch] + [ch] * (n - 1),
                                                                                  [ch] * (n - 1) + [out_ch]))
        self.act = torch.nn.SiLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class FFN(torch.nn.Module):
    def __init__(self, ch, mid_ch):
        super().__init__()
        self.w12 = torch.nn.Linear(ch, 2 * mid_ch)
        self.w3 = torch.nn.Linear(mid_ch, ch)
        for m in (self.w12, self.w3):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(torch.nn.functional.silu(x1) * x2)


class Gate(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = torch.nn.Linear(2 * ch, 2 * ch)
        self.norm = RMSNorm(ch)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

    def forward(self, x1, x2):
        g1, g2 = torch.sigmoid(self.gate(torch.cat(tensors=(x1, x2), dim=-1))).chunk(2, dim=-1)
        return self.norm(g1 * x1 + g2 * x2)


class DeformableAttention(torch.nn.Module):
    # multi-scale deformable attention without value/output projections
    def __init__(self, ch, num_head, num_points):
        super().__init__()
        self.num_head = num_head
        self.num_points = num_points
        n = sum(num_points)
        self.sampling_offsets = torch.nn.Linear(ch, num_head * n * 2)
        self.attention_weights = torch.nn.Linear(ch, num_head * n)
        self.register_buffer('num_points_scale', torch.tensor([1 / i for i in num_points for _ in range(i)]))

        theta = torch.arange(num_head, dtype=torch.float32) * (2.0 * math.pi / num_head)
        grid = torch.stack(tensors=(theta.cos(), theta.sin()), dim=-1)
        grid = (grid / grid.abs().max(-1, keepdim=True).values).view(num_head, 1, 2).tile(1, n, 1)
        grid *= torch.cat([torch.arange(1, i + 1) for i in num_points]).view(1, -1, 1)
        torch.nn.init.zeros_(self.sampling_offsets.weight)
        self.sampling_offsets.bias.data[...] = grid.flatten()
        torch.nn.init.zeros_(self.attention_weights.weight)
        torch.nn.init.zeros_(self.attention_weights.bias)

    def forward(self, query, ref, value, shapes):
        b, q = query.shape[:2]
        n = sum(self.num_points)
        offsets = self.sampling_offsets(query).view(b, q, self.num_head, n, 2)
        weights = self.attention_weights(query).view(b, q, self.num_head, n).softmax(-1)

        scale = self.num_points_scale.to(query.dtype).unsqueeze(-1)
        locations = ref[:, :, None, :, :2] + offsets * scale * ref[:, :, None, :, 2:] * 0.5
        grids = (2 * locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(self.num_points, dim=-2)

        c = value[0].shape[2]
        if torch.onnx.is_in_onnx_export():
            # TensorRT GridSample corrupts the decoder (TensorRT 10.13 / 11.4), export a gather-based equivalent
            samples = [util.grid_sample(v.reshape(b * self.num_head, c, h, w), g)
                       for v, g, (h, w) in zip(value, grids, shapes)]
        else:
            samples = [torch.nn.functional.grid_sample(v.reshape(b * self.num_head, c, h, w), g,
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
                       for v, g, (h, w) in zip(value, grids, shapes)]
        weights = weights.permute(0, 2, 1, 3).reshape(b * self.num_head, 1, q, n)
        output = (torch.cat(samples, dim=-1) * weights).sum(-1).reshape(b, self.num_head * c, q)
        return output.permute(0, 2, 1)


class DecoderLayer(torch.nn.Module):
    def __init__(self, ch, num_head, ffn_ch, num_points, gateway):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(ch, num_head, batch_first=True)
        self.norm1 = RMSNorm(ch)
        self.cross_attn = DeformableAttention(ch, num_head, num_points)
        if gateway:
            self.gateway = Gate(ch)
        else:
            self.norm2 = RMSNorm(ch)
        self.swish_ffn = FFN(ch, ffn_ch // 2)
        self.norm3 = RMSNorm(ch)

    def forward(self, x, ref, value, shapes, mask, pos):
        q = k = x + pos
        x = self.norm1(x + self.self_attn(q, k, value=x, attn_mask=mask)[0])
        y = self.cross_attn(x + pos, ref, value, shapes)
        x = self.gateway(x, y) if hasattr(self, 'gateway') else self.norm2(x + y)
        return self.norm3((x + self.swish_ffn(x)).clamp(min=-65504, max=65504))


class LQE(torch.nn.Module):
    # Location Quality Estimator: refines scores with statistics of the box distributions
    def __init__(self, reg_max, k=4):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = Head(4 * (k + 1), 64, 1, 2)
        torch.nn.init.zeros_(self.reg_conf.layers[-1].weight)
        torch.nn.init.zeros_(self.reg_conf.layers[-1].bias)

    def forward(self, scores, corners):
        b, n, _ = corners.shape
        p = corners.view(b, n, 4, self.reg_max + 1).softmax(-1).topk(self.k, dim=-1)[0]
        return scores + self.reg_conf(torch.cat(tensors=(p, p.mean(-1, keepdim=True)), dim=-1).view(b, n, -1))


class Decoder(torch.nn.Module):
    # DEIM decoder with Fine-grained Distribution Refinement (FDR) and contrastive denoising
    def __init__(self, nc, ch, ffn_ch, num_layers, num_points,
                 num_queries=300, gateway=True, share_head=False, reg_max=32, reg_scale=4.0):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.num_queries = num_queries
        self.num_head = 8
        self.num_denoising = 100
        self.register_buffer('project', util.weighting_function(reg_max, 0.5, reg_scale), persistent=False)

        self.layers = torch.nn.ModuleList(DecoderLayer(ch, self.num_head, ffn_ch, num_points, gateway)
                                          for _ in range(num_layers))
        self.lqe_layers = torch.nn.ModuleList(LQE(reg_max) for _ in range(num_layers))

        self.denoising_class_embed = torch.nn.Embedding(nc + 1, ch, padding_idx=nc)
        self.enc_score_head = torch.nn.Linear(ch, nc)
        self.enc_bbox_head = Head(ch, ch, 4, 3)
        self.query_pos_head = Head(4, ch, ch, 3)
        self.pre_bbox_head = Head(ch, ch, 4, 3)
        self.dec_score_head = torch.nn.ModuleList(torch.nn.Linear(ch, nc) for _ in range(num_layers))
        box_head = Head(ch, ch, 4 * (reg_max + 1), 3)
        self.dec_bbox_head = torch.nn.ModuleList(box_head if share_head else copy.deepcopy(box_head)
                                                 for _ in range(num_layers))
        self._anchors = {}
        self.initialize_biases()

    def initialize_biases(self):
        bias = -math.log((1 - 0.01) / 0.01)
        torch.nn.init.normal_(self.denoising_class_embed.weight[:-1])
        for m in [self.enc_score_head, *self.dec_score_head]:
            torch.nn.init.constant_(m.bias, bias)
        for m in [self.enc_bbox_head, self.pre_bbox_head, *self.dec_bbox_head]:
            torch.nn.init.zeros_(m.layers[-1].weight)
            torch.nn.init.zeros_(m.layers[-1].bias)
        for m in self.query_pos_head.layers:
            torch.nn.init.xavier_uniform_(m.weight)

    def make_anchors(self, shapes, device, eps=1e-2):
        key = (tuple(tuple(i) for i in shapes), str(device))
        if key not in self._anchors:
            anchors = []
            for i, (h, w) in enumerate(shapes):
                y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                xy = (torch.stack(tensors=(x, y), dim=-1).unsqueeze(0) + 0.5) / torch.tensor([w, h])
                anchors.append(torch.cat(tensors=(xy, torch.full_like(xy, 0.05 * 2.0 ** i)), dim=-1).view(1, -1, 4))
            anchors = torch.cat(anchors, dim=1).to(device)
            mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
            # border anchors are invalid, 1E4 instead of the original inf: sigmoid / gradients are identical,
            # but TensorRT corrupts the decoder when top-k selects inf anchors (images with few objects)
            anchors = torch.where(mask, torch.log(anchors / (1 - anchors)), 1E4)
            self._anchors[key] = anchors, mask
        return self._anchors[key]

    def forward(self, x, targets=None):
        shapes = [i.shape[2:] for i in x]
        memory = torch.cat([i.flatten(2).permute(0, 2, 1) for i in x], dim=1)
        b = memory.shape[0]

        dn_meta, mask = None, None
        if self.training and targets is not None:
            dn_logits, dn_boxes, mask, dn_meta = util.denoising(targets, self.nc, self.num_queries,
                                                                 self.denoising_class_embed, self.num_denoising)

        # query selection
        anchors, valid = self.make_anchors(shapes, memory.device)
        masked = valid.to(memory.dtype) * memory
        enc_logits = self.enc_score_head(masked)
        index = torch.topk(enc_logits.max(-1).values, self.num_queries, dim=1)[1].unsqueeze(-1)
        topk_memory = masked.gather(1, index.expand(-1, -1, memory.shape[-1]))
        topk_boxes = self.enc_bbox_head(topk_memory) + anchors.expand(b, -1, -1).gather(1, index.expand(-1, -1, 4))

        output, ref = topk_memory.detach(), topk_boxes.detach()
        if dn_meta is not None:
            output = torch.cat(tensors=(dn_logits, output), dim=1)
            ref = torch.cat(tensors=(dn_boxes, ref), dim=1)

        # decoder with fine-grained distribution refinement
        h, c = self.num_head, memory.shape[-1]
        value = memory.view(b, -1, h, c // h).permute(0, 2, 3, 1).split([i * j for i, j in shapes], dim=-1)
        ref = ref.sigmoid()
        pos = self.query_pos_head(ref).clamp(min=-10, max=10)

        output_detach = corners_undetach = 0
        boxes, logits, corners, refs = [], [], [], []
        for i, layer in enumerate(self.layers):
            output = layer(output, ref.unsqueeze(2), value, shapes, mask, pos)
            if i == 0:
                pre_boxes = torch.sigmoid(self.pre_bbox_head(output) + util.inverse_sigmoid(ref))
                pre_logits = self.dec_score_head[0](output)
                ref_initial = pre_boxes.detach()

            pred_corners = self.dec_bbox_head[i](output + output_detach) + corners_undetach
            distance = pred_corners.view(*pred_corners.shape[:-1], 4, -1).softmax(-1) @ self.project
            pred_boxes = util.distance2bbox(ref_initial, distance, self.reg_scale)
            if self.training or i == len(self.layers) - 1:
                logits.append(self.lqe_layers[i](self.dec_score_head[i](output), pred_corners))
                boxes.append(pred_boxes)
                corners.append(pred_corners)
                refs.append(ref_initial)

            corners_undetach = pred_corners
            ref = pred_boxes.detach()
            output_detach = output.detach()

        if not self.training:
            return boxes[-1], logits[-1]

        if dn_meta is not None:
            # split denoising queries from matching queries
            n = dn_meta['dn_num_split']
            dn_pre_logits, pre_logits = pre_logits.split(n, dim=1)
            dn_pre_boxes, pre_boxes = pre_boxes.split(n, dim=1)
            dn = [[j.split(n, dim=1)[0] for j in i] for i in (logits, boxes, corners, refs)]
            logits, boxes, corners, refs = [[j.split(n, dim=1)[1] for j in i] for i in (logits, boxes, corners, refs)]

        index = index.expand(-1, -1, self.nc)
        outputs = {'pred_logits': logits[-1], 'pred_boxes': boxes[-1],
                   'pred_corners': corners[-1], 'ref_points': refs[-1],
                   'aux_outputs': self.pack(logits, boxes, corners, refs)[:-1],
                   'enc_aux_outputs': [{'pred_logits': enc_logits.gather(1, index),
                                        'pred_boxes': topk_boxes.sigmoid()}],
                   'pre_outputs': {'pred_logits': pre_logits, 'pred_boxes': pre_boxes}}
        if dn_meta is not None:
            outputs['dn_outputs'] = self.pack(*dn)
            outputs['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_boxes}
            outputs['dn_meta'] = dn_meta
        return outputs

    @staticmethod
    def pack(logits, boxes, corners, refs):
        # per-layer outputs, the last layer acts as teacher for decoupled distillation (DDF)
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_corners': c, 'ref_points': d,
                 'teacher_corners': corners[-1], 'teacher_logits': logits[-1]}
                for a, b, c, d in zip(logits, boxes, corners, refs)]


class DEIMv2(torch.nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        h, w = x.shape[2:]
        x = self.decoder(self.encoder(self.backbone(x)), targets)
        if self.training:
            return x
        # (batch, queries, 4 + nc): boxes in input pixels (x1, y1, x2, y2) and class scores
        box, cls = x
        box = torch.cat(tensors=(box[..., :2] - box[..., 2:] / 2, box[..., :2] + box[..., 2:] / 2), dim=-1)
        return torch.cat(tensors=(box * torch.tensor([w, h, w, h], device=box.device), cls.sigmoid()), dim=-1)

    def fuse(self):
        for m in self.modules():
            if type(m) in (Conv, ConvBN) and not hasattr(m, 'fused'):
                m.conv = fuse_conv(m.conv, m.norm if type(m) is Conv else m.bn)
                m.forward = m.fuse_forward
                m.fused = True
                delattr(m, 'norm' if type(m) is Conv else 'bn')
        for m in self.modules():
            if type(m) is RepConv and hasattr(m, 'conv1'):
                m.conv = m.conv1.conv
                m.conv.weight.data += torch.nn.functional.pad(m.conv2.conv.weight, [1, 1, 1, 1])
                m.conv.bias.data += m.conv2.conv.bias
                m.forward = m.fuse_forward
                delattr(m, 'conv1')
                delattr(m, 'conv2')
        return self


def hgnet_v2(num_stages):
    # in_ch, mid_ch, out_ch, num_blocks, downsample, light_block, kernel_size, num_layers
    stages = [[16, 16, 64, 1, False, False, 3, 3],
              [64, 32, 256, 1, True, False, 3, 3],
              [256, 64, 512, 2, True, True, 5, 3],
              [512, 128, 1024, 1, True, True, 5, 3]]
    return stages[:num_stages]


def deim_v2_atto(num_classes: int = 80):
    stages = hgnet_v2(3)
    stages[2] = [256, 64, 256, 1, True, True, 3, 3]
    backbone = HGNetV2(stages, indices=[2])
    encoder = LiteEncoder(256, 64, expansion=0.34, depth=0.5)
    decoder = Decoder(num_classes, 64, 160, 3, [4, 2], num_queries=100, gateway=False, share_head=True)
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_femto(num_classes: int = 80):
    stages = hgnet_v2(3)
    stages[2] = [256, 64, 512, 1, True, True, 5, 3]
    backbone = HGNetV2(stages, indices=[2])
    encoder = LiteEncoder(512, 96, expansion=0.34, depth=0.5)
    decoder = Decoder(num_classes, 96, 256, 3, [4, 2], num_queries=150, gateway=False, share_head=True)
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_pico(num_classes: int = 80):
    backbone = HGNetV2(hgnet_v2(3), indices=[2])
    encoder = LiteEncoder(512, 112, expansion=0.34, depth=0.5)
    decoder = Decoder(num_classes, 112, 320, 3, [4, 2], num_queries=200, gateway=False, share_head=True)
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_n(num_classes: int = 80):
    backbone = HGNetV2(hgnet_v2(4), indices=[2, 3])
    encoder = HybridEncoder([512, 1024], 128, 512, expansion=0.34, depth=0.5, post_conv=True)
    decoder = Decoder(num_classes, 128, 512, 3, [6, 6])
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_s(num_classes: int = 80):
    backbone = DINOv3STA(ViT(192, 3, indices=[3, 7, 11]), out_ch=192, sta_ch=16)
    encoder = HybridEncoder([192, 192, 192], 192, 512, expansion=0.34, depth=0.67)
    decoder = Decoder(num_classes, 192, 512, 4, [3, 6, 3])
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_m(num_classes: int = 80):
    backbone = DINOv3STA(ViT(256, 4, indices=[3, 7, 11]), out_ch=256, sta_ch=16)
    encoder = HybridEncoder([256, 256, 256], 256, 512, expansion=0.67, depth=1.0)
    decoder = Decoder(num_classes, 256, 512, 4, [3, 6, 3])
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_l(num_classes: int = 80):
    backbone = DINOv3STA(ViT(384, 6, indices=[5, 8, 11], dinov3=True), out_ch=224, sta_ch=32)
    encoder = HybridEncoder([224, 224, 224], 224, 896, expansion=1.0, depth=1.0)
    decoder = Decoder(num_classes, 224, 1792, 4, [3, 6, 3])
    return DEIMv2(backbone, encoder, decoder)


def deim_v2_x(num_classes: int = 80):
    backbone = DINOv3STA(ViT(384, 6, indices=[5, 8, 11], ffn_ratio=6, swiglu=True, dinov3=True), 256, 64)
    encoder = HybridEncoder([256, 256, 256], 256, 1024, expansion=1.25, depth=1.37)
    decoder = Decoder(num_classes, 256, 2048, 6, [3, 6, 3])
    return DEIMv2(backbone, encoder, decoder)
