import os
import random

import numpy
import torch
from PIL import Image
from torch.utils import data
from torchvision import tv_tensors
from torchvision.transforms import v2

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.augment = augment
        self.input_size = input_size
        self.epoch = 0  # set by the training loop, drives the augmentation schedule

        # Read labels
        labels = self.load_label(filenames)
        self.labels = list(labels.values())
        self.filenames = list(labels.keys())  # update
        self.n = len(self.filenames)  # number of samples

        # Augmentations
        self.mosaic_cache = []
        self.mosaic_resize = v2.Resize(input_size // 2)
        self.mosaic_affine = v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=0)
        self.photometric = v2.RandomPhotometricDistort(p=0.5)
        self.zoom_out = v2.RandomZoomOut(fill=0)
        self.iou_crop = v2.RandomIoUCrop()
        self.sanitize = v2.SanitizeBoundingBoxes(min_size=params['min_size'])
        self.flip = v2.RandomHorizontalFlip()
        self.resize = v2.Resize((input_size, input_size))
        self.scales = scales(input_size, params['multi_scale']) if params['multi_scale'] else None

    def __getitem__(self, index):
        image, target = self.load_image(index)

        if self.augment:
            # 4 stages: no-aug -> mosaic or zoom-out/IoU-crop -> zoom-out/IoU-crop -> no-aug
            start, flat, stop = self.params['aug_epochs']
            if start <= self.epoch < stop:
                mosaic = self.epoch < flat and random.random() <= self.params['mosaic']
                if mosaic:
                    image, target = self.load_mosaic(image, target)
                image, target = self.photometric(image, target)
                if not mosaic:
                    image, target = self.zoom_out(image, target)
                    if random.random() < 0.8:
                        image, target = self.iou_crop(image, target)
            image, target = self.flip(*self.sanitize(image, target))
            image, target = self.sanitize(*self.resize(image, target))
        else:
            image, target = self.resize(image, target)

        # PIL to normalized CHW float, boxes to normalized (cx, cy, w, h)
        sample = v2.functional.pil_to_tensor(image).float() / 255.
        box = target['boxes'].as_subclass(torch.Tensor).float()
        box = torch.stack(tensors=((box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2,
                                   box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]), dim=-1)
        box = box / self.input_size
        return sample, {'cls': target['labels'], 'box': box}

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        image = Image.open(self.filenames[i]).convert('RGB')
        w, h = image.size
        label = torch.from_numpy(self.labels[i])
        box = label[:, 1:5] * torch.tensor([w, h, w, h])
        box = torch.cat(tensors=(box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2), dim=1)
        keep = (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
        box = tv_tensors.BoundingBoxes(box[keep], format='XYXY', canvas_size=(h, w))
        return image, {'boxes': box, 'labels': label[keep, 0].long()}

    def load_mosaic(self, image, target):
        # 2x2 mosaic from a per-worker cache of recent samples, followed by a random affine
        image, target = self.mosaic_resize(image, target)
        self.mosaic_cache.append((image, target))
        if len(self.mosaic_cache) > 50:
            self.mosaic_cache.pop(random.randint(0, len(self.mosaic_cache) - 2))  # keep the newest
        samples = [(image, target)] + random.choices(self.mosaic_cache, k=3)

        h = max(i.size[1] for i, _ in samples)
        w = max(i.size[0] for i, _ in samples)
        image4 = Image.new(mode=image.mode, size=(w * 2, h * 2), color=0)
        boxes, labels = [], []
        for (image, target), (x, y) in zip(samples, ((0, 0), (w, 0), (0, h), (w, h))):
            image4.paste(image, (x, y))
            boxes.append(target['boxes'].as_subclass(torch.Tensor) + torch.tensor([x, y, x, y]))
            labels.append(target['labels'])

        box = tv_tensors.BoundingBoxes(torch.cat(boxes), format='XYXY', canvas_size=(h * 2, w * 2))
        return self.mosaic_affine(image4, {'boxes': box, 'labels': torch.cat(labels)})

    def collate_fn(self, batch):
        samples = torch.stack([b[0] for b in batch], dim=0)
        targets = [b[1] for b in batch]
        if not self.augment:
            return samples, targets

        params = self.params
        beta = round(random.uniform(0.45, 0.55), 6)
        if random.random() < params['mix_up'] and params['mix_up_epochs'][0] <= self.epoch < params['mix_up_epochs'][1]:
            # MixUp with the previous sample of the batch
            samples = samples.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(samples.mul(beta))
            targets = [{k: torch.cat((t[k], s[k])) for k in t} for t, s in zip(targets, targets[-1:] + targets[:-1])]
        elif params['copy_blend_epochs'][0] <= self.epoch < params['copy_blend_epochs'][1] \
                and random.random() < params['copy_blend']:
            samples, targets = copy_blend(samples, targets, beta, params)

        # Multi-scale training
        if self.scales is not None and self.epoch < params['epochs'] - params['no_aug_epoch']:
            samples = torch.nn.functional.interpolate(samples, size=random.choice(self.scales))
        return samples, targets

    @staticmethod
    def load_label(filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt') as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert (label >= 0).all()
                        assert label.shape[1] == 5
                        assert (label[:, 1:] <= 1).all()
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
            except FileNotFoundError:
                label = numpy.zeros((0, 5), dtype=numpy.float32)
            except AssertionError:
                continue
            x[filename] = label
        torch.save(x, path)
        return x


def scales(base_size, repeat):
    # multi-scale sizes around `base_size`, the base size is repeated to be sampled more often
    n = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    sizes = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(n)]
    sizes += [base_size] * repeat
    sizes += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(n)]
    return sizes


def copy_blend(samples, targets, beta, params):
    # blends random objects of the batch (with some context) into random places of each image
    b, _, h, w = samples.shape
    pool = [(i, box, cls) for i, t in enumerate(targets) for box, cls in zip(t['box'], t['cls'])
            if box[2] * w * box[3] * h >= params['area_threshold']]
    if not pool:
        return samples, targets

    outputs = samples.clone()
    for i in range(b):
        boxes, labels = [], []
        for j, box, cls in random.sample(pool, min(params['num_objects'], len(pool))):
            cx, cy, bw, bh = box.tolist()
            x1_src, y1_src = max(int((cx - bw / 2) * w), 0), max(int((cy - bh / 2) * h), 0)
            x2_src, y2_src = min(int((cx + bw / 2) * w), w), min(int((cy + bh / 2) * h), h)
            pw, ph = x2_src - x1_src, y2_src - y1_src
            if pw <= 0 or ph <= 0:
                continue

            x1 = random.randint(0, w - pw) if pw < w else 0
            y1 = random.randint(0, h - ph) if ph < h else 0
            x2, y2 = x1 + pw, y1 + ph
            boxes.append(torch.tensor([(x1 + pw / 2) / w, (y1 + ph / 2) / h, pw / w, ph / h]))
            labels.append(cls)

            # expand the copied region by a random ratio, bounded by both images
            r = round(random.uniform(*params['expand_ratios']), 6)
            ex1, ey1 = min(x1_src, int(pw * r)), min(y1_src, int(ph * r))
            ex2, ey2 = min(x2_src + int(pw * r), w) - x2_src, min(y2_src + int(ph * r), h) - y2_src
            ex1, ey1 = min(x1, ex1), min(y1, ey1)
            ex2, ey2 = min(x2 + ex2, w) - x2, min(y2 + ey2, h) - y2
            x1_src, y1_src, x2_src, y2_src = x1_src - ex1, y1_src - ey1, x2_src + ex2, y2_src + ey2
            x1, y1, x2, y2 = x1 - ex1, y1 - ey1, x2 + ex2, y2 + ey2

            patch = samples[j, :, y1_src:y2_src, x1_src:x2_src]
            outputs[i, :, y1:y2, x1:x2] = outputs[i, :, y1:y2, x1:x2] * beta + patch * (1 - beta)

        if boxes:
            targets[i] = {'cls': torch.cat((targets[i]['cls'], torch.stack(labels))),
                          'box': torch.cat((targets[i]['box'], torch.stack(boxes)))}
    return outputs, targets
