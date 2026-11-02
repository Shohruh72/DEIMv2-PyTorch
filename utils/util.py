import copy
import functools
import math
import random
import re

import numpy
import torch
from scipy.optimize import linear_sum_assignment


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


class End2End(torch.nn.Module):
    """
    Model + top-k: (N, 3, H, W) RGB in [0, 1] -> (N, max_det, 6) as x1, y1, x2, y2, score, class in input pixels
    """

    def __init__(self, model, max_det=300):
        super().__init__()
        self.model = model
        self.max_det = max_det

    def forward(self, x):
        return post_process(self.model(x).float(), self.max_det, batched=True)


def export_onnx(model, params, f='./weights/best.onnx', half=False):
    """
    Exports End2End with a dynamic batch.
    half=False: FP32 graph for onnxruntime / CPU.
    half=True: traced under CUDA autocast, so the graph carries the FP16/FP32 casts of AMP
               (LayerNorm, softmax, box decoding stay FP32), as needed by strongly typed TensorRT.
    """
    import onnx

    device = torch.device('cuda' if half else 'cpu')
    model = End2End(model.float().fuse().eval(), params['max_det']).to(device)
    x = torch.zeros((1, 3, params['input_size'], params['input_size']), device=device)

    with torch.no_grad(), torch.autocast('cuda', enabled=half):
        torch.onnx.export(model, x,
                          f=f,
                          verbose=False,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['outputs'],
                          dynamic_axes={'images': {0: 'batch'}, 'outputs': {0: 'batch'}},
                          dynamo=False)

    # Checks
    onnx.checker.check_model(onnx.load(f))
    print(f'Exported {f}')
    return f


def export_engine(f='./weights/best.onnx', batch=1, workspace=4):
    """
    Builds a TensorRT engine (*.engine next to the ONNX) for batch sizes 1 ... `batch`, tuned for batch 1.
    TensorRT >= 10.6, the network is strongly typed: the precision is the one of the ONNX (see `export_onnx`).
    The engine only runs on the GPU model and TensorRT version that built it.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(f):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)

    shape = tuple(network.get_input(0).shape)[1:]
    profile = builder.create_optimization_profile()
    profile.set_shape('images', (1, *shape), (1, *shape), (batch, *shape))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f'TensorRT failed to build {f}')

    f = f.replace('.onnx', '.engine')
    with open(f, 'wb') as file:
        file.write(engine)
    print(f'Exported {f}')
    return f


class TensorRT(torch.nn.Module):
    """
    Runs an engine of `export_engine` on CUDA tensors, same input / output as End2End
    """

    def __init__(self, f):
        import tensorrt as trt

        super().__init__()
        with open(f, 'rb') as file:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(file.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()  # TensorRT syncs extra on the default stream

    def forward(self, x):
        x = x.float().contiguous()
        self.context.set_input_shape('images', tuple(x.shape))
        y = torch.empty(tuple(self.context.get_tensor_shape('outputs')), device=x.device)
        self.context.set_tensor_address('images', x.data_ptr())
        self.context.set_tensor_address('outputs', y.data_ptr())
        current = torch.cuda.current_stream(x.device)
        self.stream.wait_stream(current)
        self.context.execute_async_v3(self.stream.cuda_stream)
        current.wait_stream(self.stream)
        return y


# ---------------------------------------------------------------- boxes
def wh2xy(x):
    cx, cy, w, h = x.unbind(-1)
    w, h = w.clamp(min=0.0), h.clamp(min=0.0)
    return torch.stack(tensors=(cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def xy2wh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    return torch.stack(tensors=((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(box1, box2):
    iou, union = box_iou(box1, box2)
    lt = torch.min(box1[:, None, :2], box2[:, :2])
    rb = torch.max(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def grid_sample(x, grid):
    """
    grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False) with gathers, for ONNX / TensorRT
    """
    n, c, h, w = x.shape
    gh, gw = grid.shape[1:3]
    px = ((grid[..., 0] + 1) * w - 1) / 2
    py = ((grid[..., 1] + 1) * h - 1) / 2
    x0, y0 = px.floor(), py.floor()
    x = x.flatten(2)
    output = 0
    for xi, yi, weight in ((x0, y0, (x0 + 1 - px) * (y0 + 1 - py)),
                           (x0 + 1, y0, (px - x0) * (y0 + 1 - py)),
                           (x0, y0 + 1, (x0 + 1 - px) * (py - y0)),
                           (x0 + 1, y0 + 1, (px - x0) * (py - y0))):
        valid = (xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)
        index = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).long().flatten(1)
        sample = x.gather(2, index[:, None].expand(-1, c, -1)).view(n, c, gh, gw)
        output = output + sample * (weight * valid)[:, None]
    return output


def inverse_sigmoid(x, eps=1e-5):
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


@functools.lru_cache(maxsize=8)
def sincos_embedding(w, h, dim, temperature=10000.):
    grid_w, grid_h = torch.meshgrid(torch.arange(int(w), dtype=torch.float32),
                                    torch.arange(int(h), dtype=torch.float32), indexing='ij')
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1. / (temperature ** omega)
    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]
    return torch.cat(tensors=(out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()), dim=1)[None, :, :]


# ---------------------------------------------------------------- fine-grained distribution refinement (FDR)
def weighting_function(reg_max, up, reg_scale):
    """
    Non-uniform bin values W(n) of the edge-distance distribution
    """
    up, reg_scale = torch.tensor([up]), torch.tensor([reg_scale])
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-step ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [step ** i - 1 for i in range(1, reg_max // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    return torch.cat(values, 0)


def distance2bbox(points, distance, reg_scale):
    # (cx, cy, w, h) reference + (left, top, right, bottom) distances -> (cx, cy, w, h)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    return xy2wh(torch.stack(tensors=(x1, y1, x2, y2), dim=-1))


def bbox2distance(points, box, reg_max, reg_scale, up, eps=0.1):
    # inverse of distance2bbox, returns soft-label targets (left bin, right weight, left weight)
    left = (points[:, 0] - box[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (points[:, 1] - box[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (box[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (box[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    gt = torch.stack(tensors=(left, top, right, bottom), dim=-1).reshape(-1)

    values = weighting_function(reg_max, up, reg_scale).to(gt.device)
    indices = ((values.unsqueeze(0) - gt.unsqueeze(1)) <= 0).sum(1).float() - 1

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)
    valid = (indices >= 0) & (indices < reg_max)
    i = indices[valid].long()
    left_diffs = torch.abs(gt[valid] - values[i])
    right_diffs = torch.abs(values[i + 1] - gt[valid])
    weight_right[valid] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid] = 1.0 - weight_right[valid]

    weight_left[indices < 0] = 1.0
    indices[indices < 0] = 0.0
    weight_right[indices >= reg_max] = 1.0
    weight_left[indices >= reg_max] = 0.0
    indices[indices >= reg_max] = reg_max - 0.1
    return indices.clamp(min=0, max=reg_max - eps), weight_right, weight_left


# ---------------------------------------------------------------- contrastive denoising
def denoising(targets, nc, num_queries, embed, num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0):
    num_gts = [len(t['cls']) for t in targets]
    device = targets[0]['cls'].device
    max_gt = max(num_gts)
    if max_gt == 0:
        return None, None, None, None

    num_group = max(num_denoising // max_gt, 1)
    bs = len(num_gts)
    query_cls = torch.full([bs, max_gt], nc, dtype=torch.int32, device=device)
    query_box = torch.zeros([bs, max_gt, 4], device=device)
    pad_mask = torch.zeros([bs, max_gt], dtype=torch.bool, device=device)
    for i, n in enumerate(num_gts):
        if n > 0:
            query_cls[i, :n] = targets[i]['cls']
            query_box[i, :n] = targets[i]['box']
            pad_mask[i, :n] = 1

    # each group has positive and negative queries
    query_cls = query_cls.tile([1, 2 * num_group])
    query_box = query_box.tile([1, 2 * num_group, 1])
    pad_mask = pad_mask.tile([1, 2 * num_group])
    negative = torch.zeros([bs, max_gt * 2, 1], device=device)
    negative[:, max_gt:] = 1
    negative = negative.tile([1, num_group, 1])
    positive = (1 - negative).squeeze(-1) * pad_mask
    positive_idx = torch.split(torch.nonzero(positive)[:, 1], [n * num_group for n in num_gts])
    num_denoising = int(max_gt * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(query_cls, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_cls = torch.randint_like(mask, 0, nc, dtype=query_cls.dtype)
        query_cls = torch.where(mask & pad_mask, new_cls, query_cls)

    if box_noise_scale > 0:
        known_box = wh2xy(query_box)
        diff = torch.tile(query_box[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(query_box, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(query_box)
        rand_part = (rand_part + 1.0) * negative + rand_part * (1 - negative)
        known_box += (rand_sign * rand_part * diff)
        query_box = xy2wh(torch.clip(known_box, min=0.0, max=1.0))
        query_box[query_box < 0] *= -1

    # matching queries cannot see denoising queries, denoising groups cannot see each other
    size = num_denoising + num_queries
    attn_mask = torch.full([size, size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        a, b = max_gt * 2 * i, max_gt * 2 * (i + 1)
        if i == 0:
            attn_mask[a:b, b:num_denoising] = True
        if i == num_group - 1:
            attn_mask[a:b, :a] = True
        else:
            attn_mask[a:b, b:num_denoising] = True
            attn_mask[a:b, :a] = True

    dn_meta = {'dn_positive_idx': positive_idx, 'dn_num_group': num_group, 'dn_num_split': [num_denoising, num_queries]}
    return embed(query_cls), inverse_sigmoid(query_box), attn_mask, dn_meta


# ---------------------------------------------------------------- inference & metrics
def post_process(outputs, max_det=300, batched=False):
    # DETR-style: no NMS, top-k over (query, class) pairs -> [x1, y1, x2, y2, score, class]
    nc = outputs.shape[-1] - 4
    scores, index = torch.topk(outputs[..., 4:].flatten(1), max_det, dim=-1)
    boxes = outputs[..., :4].gather(1, (index // nc).unsqueeze(-1).expand(-1, -1, 4))
    outputs = torch.cat(tensors=(boxes, scores.unsqueeze(-1), (index % nc).unsqueeze(-1).to(boxes.dtype)), dim=-1)
    return outputs if batched else list(outputs)


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, output, target, eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px = numpy.linspace(start=0, stop=1, num=1000)
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapezoid(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


# ---------------------------------------------------------------- training helpers
def strip_optimizer(filename):
    x = torch.load(filename, map_location='cpu', weights_only=False)
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f=filename)


def load_weight(model, path):
    """
    Loads weights from this repo (*.pt), official DEIMv2 checkpoints (*.pth) or Hugging Face (*.safetensors).
    Tensors with a different shape (e.g. class heads of another dataset) are skipped.
    """
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        src = load_file(path)
    else:
        src = torch.load(path, map_location='cpu', weights_only=False)
        if 'ema' in src:
            src = src['ema']['module']
        elif isinstance(src['model'], torch.nn.Module):
            src = src['model'].float().state_dict()
        else:
            src = src['model']

    # official key layout -> this repo
    rules = (('backbone.dinov3._model.', 'backbone.dinov3.'),
             ('decoder.decoder.', 'decoder.'),
             ('encoder.encoder.0.layers.0.', 'encoder.aifi.'),
             (r'downsample_convs\.(\d+)\.0\.', r'downsample_convs.\1.'),
             (r'\.qkv\.bias_mask', '.bias_mask'))
    for a, b in rules:
        src = {re.sub(a, b, k): v for k, v in src.items()}

    dst = model.state_dict(keep_vars=True)
    ckpt = {k: v for k, v in src.items() if k in dst and v.shape == dst[k].shape}
    model.load_state_dict(state_dict=ckpt, strict=False)
    # shared heads (atto/femto/pico) appear under several names, count unique tensors
    loaded = {id(dst[k]) for k in ckpt}
    missing = [k for k in dst if id(dst[k]) not in loaded]
    print(f'Loaded {len(ckpt)} tensors from {path}, missing {len(missing)}: {missing[:5]}')
    return model


def set_params(model, params):
    # pretrained backbone weights get `lr_backbone`, 1-D tensors (norm, bias, LAB) get no weight decay
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        backbone = name.startswith('backbone.') and not name.startswith(('backbone.sta.', 'backbone.convs.',
                                                                          'backbone.norms.'))
        lr = params['lr_backbone'] if backbone else params['lr']
        decay = 0.0 if p.ndim <= 1 else params['weight_decay']
        groups.setdefault((lr, decay), []).append(p)
    return [{'params': v, 'lr': lr, 'weight_decay': decay} for (lr, decay), v in groups.items()]


class FlatCosineLR:
    """
    Quadratic warm-up -> flat -> cosine decay to `lr * lr_gamma` -> flat during no-augmentation epochs
    """

    def __init__(self, params, num_steps, accumulate):
        self.warmup = params['warmup_iter'] * accumulate
        self.flat = params['flat_epoch'] * num_steps
        self.total = params['epochs'] * num_steps
        self.no_aug = params['no_aug_epoch'] * num_steps
        self.gamma = params['lr_gamma']
        self.base = None

    def step(self, step, optimizer):
        if self.base is None:
            self.base = [group['lr'] for group in optimizer.param_groups]
        for base, group in zip(self.base, optimizer.param_groups):
            if step <= self.warmup:
                lr = base * (step / self.warmup) ** 2
            elif step <= self.flat:
                lr = base
            elif step >= self.total - self.no_aug:
                lr = base * self.gamma
            else:
                alpha = 0.5 * (1 + math.cos(math.pi * (step - self.flat) / (self.total - self.flat - self.no_aug)))
                lr = base * self.gamma + (base - base * self.gamma) * alpha
            group['lr'] = lr


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=1000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


# ---------------------------------------------------------------- matching & loss
class Assigner:
    """
    One-to-one Hungarian matching. After `change_epoch` the cost becomes -score * IoU^alpha (DEIMv2)
    """

    def __init__(self, params):
        self.cost_cls, self.cost_box, self.cost_giou = 2.0, 5.0, 2.0
        self.alpha, self.gamma = 0.25, 2.0
        self.iou_alpha = 4.0
        self.change_epoch = params['matcher_change_epoch']

    @torch.no_grad()
    def __call__(self, outputs, targets, epoch):
        bs, nq = outputs['pred_logits'].shape[:2]
        prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
        box = outputs['pred_boxes'].flatten(0, 1)
        tgt_cls = torch.cat([t['cls'] for t in targets])
        tgt_box = torch.cat([t['box'] for t in targets])

        prob = prob[:, tgt_cls]
        if epoch >= self.change_epoch:
            cost = -(prob * box_iou(wh2xy(box), wh2xy(tgt_box))[0] ** self.iou_alpha)
        else:
            neg = (1 - self.alpha) * (prob ** self.gamma) * (-(1 - prob + 1e-8).log())
            pos = self.alpha * ((1 - prob) ** self.gamma) * (-(prob + 1e-8).log())
            cost = self.cost_box * torch.cdist(box, tgt_box, p=1) + self.cost_cls * (pos - neg) \
                   - self.cost_giou * generalized_box_iou(wh2xy(box), wh2xy(tgt_box))

        cost = torch.nan_to_num(cost.view(bs, nq, -1).cpu(), nan=1.0)
        sizes = [len(t['box']) for t in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class ComputeLoss:
    """
    DEIM criterion: Matchability-Aware Loss (MAL), L1 + GIoU, Fine-Grained Localization (FGL)
    and Decoupled Distillation Focal (DDF) losses over decoder layers, encoder and denoising queries
    """

    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module

        self.nc = model.decoder.nc
        self.reg_max = model.decoder.reg_max
        self.reg_scale = model.decoder.reg_scale
        self.params = params
        self.gamma = 1.5
        self.local = params['local']  # FGL + DDF
        self.uni_set = params['uni_set']  # box losses on the union of matches over all layers
        self.assigner = Assigner(params)

    @staticmethod
    def permute(indices):
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        return batch, torch.cat([src for (src, _) in indices])

    @staticmethod
    def num_boxes(n, device):
        n = torch.as_tensor([n], dtype=torch.float, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(n)
            n /= torch.distributed.get_world_size()
        return torch.clamp(n, min=1).item()

    @staticmethod
    def union(indices, aux_indices):
        # union of matched (query, target) pairs over all layers, most frequent pair wins
        for aux in aux_indices:
            indices = [(torch.cat([i[0], j[0]]), torch.cat([i[1], j[1]])) for i, j in zip(indices, aux)]
        results = []
        for ind in [torch.cat([i[:, None], j[:, None]], 1) for i, j in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            pairs = {}
            for i, j in unique[torch.argsort(counts, descending=True)].tolist():
                pairs.setdefault(i, j)
            results.append((torch.tensor(list(pairs.keys())).long(), torch.tensor(list(pairs.values())).long()))
        return results

    def loss_mal(self, outputs, targets, indices, n):
        idx = self.permute(indices)
        logits = outputs['pred_logits']
        src_box = outputs['pred_boxes'][idx]
        tgt_box = torch.cat([t['box'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou = torch.diag(box_iou(wh2xy(src_box), wh2xy(tgt_box))[0]).detach()

        tgt_cls = torch.full(logits.shape[:2], self.nc, dtype=torch.int64, device=logits.device)
        tgt_cls[idx] = torch.cat([t['cls'][j] for t, (_, j) in zip(targets, indices)])
        target = torch.nn.functional.one_hot(tgt_cls, num_classes=self.nc + 1)[..., :-1]

        score = torch.zeros_like(tgt_cls, dtype=logits.dtype)
        score[idx] = iou.to(score.dtype)
        score = (score.unsqueeze(-1) * target).pow(self.gamma)
        weight = logits.sigmoid().detach().pow(self.gamma) * (1 - target) + target
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, score, weight=weight, reduction='none')
        return loss.mean(1).sum() * logits.shape[1] / n

    @staticmethod
    def loss_box(outputs, targets, indices, n):
        src_box = outputs['pred_boxes'][ComputeLoss.permute(indices)]
        tgt_box = torch.cat([t['box'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_l1 = torch.nn.functional.l1_loss(src_box, tgt_box, reduction='none').sum() / n
        loss_giou = (1 - torch.diag(generalized_box_iou(wh2xy(src_box), wh2xy(tgt_box)))).sum() / n
        return loss_l1, loss_giou

    def loss_local(self, outputs, targets, indices, n, dn, T=5):
        idx = self.permute(indices)
        tgt_box = torch.cat([t['box'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        corners = outputs['pred_corners'][idx].reshape(-1, self.reg_max + 1)
        key = 'dn' if dn else 'main'
        if key not in self.fgl_targets:  # the reference points are shared by all layers
            with torch.no_grad():
                self.fgl_targets[key] = bbox2distance(outputs['ref_points'][idx].detach(), wh2xy(tgt_box),
                                                      self.reg_max, self.reg_scale, 0.5)
        target, weight_right, weight_left = self.fgl_targets[key]

        iou = torch.diag(box_iou(wh2xy(outputs['pred_boxes'][idx]), wh2xy(tgt_box))[0])
        weight = iou.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
        left = target.long()
        loss = torch.nn.functional.cross_entropy(corners, left, reduction='none') * weight_left.reshape(-1) \
               + torch.nn.functional.cross_entropy(corners, left + 1, reduction='none') * weight_right.reshape(-1)
        loss_fgl = (loss * weight.float()).sum() / n

        loss_ddf = None
        if outputs['pred_corners'] is not outputs.get('teacher_corners', outputs['pred_corners']):
            pred = outputs['pred_corners'].reshape(-1, self.reg_max + 1)
            teacher = outputs['teacher_corners'].reshape(-1, self.reg_max + 1)
            weight = outputs['teacher_logits'].sigmoid().max(dim=-1)[0]
            mask = torch.zeros_like(weight, dtype=torch.bool)
            mask[idx] = True
            mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)
            weight[idx] = iou.reshape_as(weight[idx]).to(weight.dtype)
            weight = weight.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(pred / T, dim=1),
                                              torch.nn.functional.softmax(teacher.detach() / T, dim=1),
                                              reduction='none').sum(-1) * weight * (T ** 2)
            if not dn:
                scale = 8 / outputs['pred_boxes'].shape[0]  # avoid the influence of batch size per GPU
                self.num_pos, self.num_neg = (mask.sum() * scale) ** 0.5, ((~mask).sum() * scale) ** 0.5
            loss_pos = loss[mask].mean() if mask.any() else 0
            loss_neg = loss[~mask].mean() if (~mask).any() else 0
            loss_ddf = (loss_pos * self.num_pos + loss_neg * self.num_neg) / (self.num_pos + self.num_neg)
        return loss_fgl, loss_ddf

    def __call__(self, outputs, targets, epoch=0):
        device = outputs['pred_logits'].device
        self.fgl_targets, self.num_pos, self.num_neg = {}, None, None

        indices = self.assigner(outputs, targets, epoch)
        aux_indices = [self.assigner(x, targets, epoch) for x in outputs['aux_outputs'] + [outputs['pre_outputs']]]
        enc_indices = [self.assigner(x, targets, epoch) for x in outputs['enc_aux_outputs']]
        indices_go = self.union(indices, aux_indices + enc_indices)
        n = self.num_boxes(sum(len(t['cls']) for t in targets), device)
        n_go = self.num_boxes(sum(len(i) for i, _ in indices_go), device)

        losses = {'cls': [], 'l1': [], 'giou': [], 'fgl': [], 'ddf': []}

        def compute(x, ind, num, dn=False):
            # classification uses its own matches, box & local losses use the union set if enabled
            losses['cls'].append(self.loss_mal(x, targets, ind, num))
            if self.uni_set and not dn:
                ind, num = indices_go, n_go
            loss_l1, loss_giou = self.loss_box(x, targets, ind, num)
            losses['l1'].append(loss_l1)
            losses['giou'].append(loss_giou)
            if self.local and 'pred_corners' in x:
                loss_fgl, loss_ddf = self.loss_local(x, targets, ind, num, dn)
                losses['fgl'].append(loss_fgl)
                if loss_ddf is not None:
                    losses['ddf'].append(loss_ddf)

        compute(outputs, indices, n)
        for x, i in zip(outputs['aux_outputs'] + [outputs['pre_outputs']] + outputs['enc_aux_outputs'],
                        aux_indices + enc_indices):
            compute(x, i, n)

        if 'dn_outputs' in outputs:
            meta = outputs['dn_meta']
            dn_indices = []
            for t, positive in zip(targets, meta['dn_positive_idx']):
                j = torch.arange(len(t['cls']), dtype=torch.int64, device=device).tile(meta['dn_num_group'])
                dn_indices.append((positive, j))
            for x in outputs['dn_outputs'] + [outputs['dn_pre_outputs']]:
                compute(x, dn_indices, n * meta['dn_num_group'], dn=True)

        p = self.params
        loss_cls = sum(losses['cls']) * p['mal']
        loss_box = sum(losses['l1']) * p['l1'] + sum(losses['giou']) * p['giou']
        loss_fgl = sum(losses['fgl']) * p['fgl'] if losses['fgl'] else torch.zeros((), device=device)
        loss_ddf = sum(losses['ddf']) * p['ddf'] if losses['ddf'] else torch.zeros((), device=device)
        return tuple(torch.nan_to_num(x, nan=0.0) for x in (loss_cls, loss_box, loss_fgl, loss_ddf))
