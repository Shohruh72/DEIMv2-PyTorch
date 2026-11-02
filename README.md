[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) re-implementation using PyTorch

Real-time DETR detector: 8 models from 0.5M to 50M parameters, no NMS, verified against the official COCO results.

### Installation

```
conda create -n DEIM python=3.11
conda activate DEIM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python PyYAML scipy tqdm
pip install safetensors     # official Hugging Face weights
pip install onnxruntime-gpu # *.onnx inference (or onnxruntime for CPU)
pip install onnx tensorrt   # --export and *.engine inference
```

### Quick start

* Download weights from the [Download](#download) table into `./weights`
* Run `python main.py --model s --demo --weights weights/DEIMv2_DINOv3_S_COCO.safetensors` (webcam, `q` to quit)

### Download

All weights are in [Releases](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/tag/v1.0), click a size to download:

| Version | Input | Params | COCO AP | PyTorch | ONNX FP32 | ONNX FP16 | TensorRT FP16 |
|:-------:|:-----:|-------:|--------:|--------:|----------:|----------:|--------------:|
| atto | 320 | 0.5M | 23.7 | [2.1 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_ATTO_COCO.safetensors) | [2.5 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_ATTO_COCO.onnx) | [1.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_ATTO_COCO_fp16.onnx) | [4.9 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_ATTO_COCO_fp16.engine) |
| femto | 416 | 1.0M | 31.0 | [4.0 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_FEMTO_COCO.safetensors) | [4.4 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_FEMTO_COCO.onnx) | [2.5 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_FEMTO_COCO_fp16.onnx) | [6.3 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_FEMTO_COCO_fp16.engine) |
| pico | 640 | 1.5M | 38.5 | [6.3 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_PICO_COCO.safetensors) | [6.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_PICO_COCO.onnx) | [3.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_PICO_COCO_fp16.onnx) | [8.4 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_PICO_COCO_fp16.engine) |
| n | 640 | 3.6M | 43.0 | [14.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_N_COCO.safetensors) | [14.9 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_N_COCO.onnx) | [7.8 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_N_COCO_fp16.onnx) | [11.4 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_HGNetv2_N_COCO_fp16.engine) |
| s | 640 | 9.8M | 50.9 | [39.4 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_S_COCO.safetensors) | [40.2 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_S_COCO.onnx) | [20.8 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_S_COCO_fp16.onnx) | [29.5 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_S_COCO_fp16.engine) |
| m | 640 | 18.2M | 53.0 | [73.8 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_M_COCO.safetensors) | [73.7 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_M_COCO.onnx) | [37.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_M_COCO_fp16.onnx) | [46.8 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_M_COCO_fp16.engine) |
| l | 640 | 32.2M | 56.0 | [130.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_L_COCO.safetensors) | [130.0 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_L_COCO.onnx) | [65.8 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_L_COCO_fp16.onnx) | [74.6 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_L_COCO_fp16.engine) |
| x | 640 | 50.4M | 57.8 | [205.3 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_X_COCO.safetensors) | [202.7 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_X_COCO.onnx) | [102.4 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_X_COCO_fp16.onnx) | [108.9 MB](https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_X_COCO_fp16.engine) |

* ONNX FP32: onnxruntime on CPU or GPU. ONNX FP16: source of a TensorRT engine for your GPU
* TensorRT FP16: prebuilt on RTX 4070 Laptop + TensorRT 11.4 and runs only there, otherwise build it from ONNX FP16 (below)

### Inference

Examples use `s` (`DEIMv2_DINOv3_S_COCO`), replace it with any version of the table.

ONNX Runtime:

```
wget -P weights https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_S_COCO.onnx
python main.py --model s --demo --weights weights/DEIMv2_DINOv3_S_COCO.onnx
```

TensorRT (engines are not portable, build once on your GPU):

```
wget -P weights https://github.com/Shohruh72/DEIMv2-PyTorch/releases/download/v1.0/DEIMv2_DINOv3_S_COCO_fp16.onnx
python main.py --model s --export engine --weights weights/DEIMv2_DINOv3_S_COCO_fp16.onnx
python main.py --model s --demo --weights weights/DEIMv2_DINOv3_S_COCO_fp16.engine
```

`--model` must match the file. Without this repo, only `numpy`, `opencv-python` and `onnxruntime` are needed:

```python
import cv2, numpy, onnxruntime

size = 640  # atto: 320, femto: 416, others: 640
session = onnxruntime.InferenceSession('DEIMv2_DINOv3_S_COCO.onnx')
image = cv2.imread('image.jpg')
h, w = image.shape[:2]
x = cv2.resize(image, (size, size))[..., ::-1].transpose(2, 0, 1)[None].astype(numpy.float32) / 255
outputs = session.run(None, {'images': x})[0][0]  # (300, 6): x1, y1, x2, y2, score, class
outputs = outputs[outputs[:, 4] > 0.5]
outputs[:, :4] *= [w / size, h / size, w / size, h / size]  # back to image pixels
```

### Train

* Configure your dataset path in `main.py` (`data_dir`), see [Dataset structure](#dataset-structure)
* Put the pretrained backbone into `./weights` (`atto` ... `n` download it automatically)
    * `s`, `m`: [ViT-Tiny](https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing),
      [ViT-Tiny+](https://drive.google.com/file/d/1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt/view?usp=sharing)
    * `l`, `x`: ViT-S/16, ViT-S+/16 from [DINOv3](https://github.com/facebookresearch/dinov3)
* Run `bash main.sh $ --model s --train` for training, `$` is number of GPUs
* Add `--weights weights/DEIMv2_DINOv3_S_COCO.safetensors` to fine-tune from COCO, class heads of another class count are re-initialized

`--batch-size` is per GPU, gradients are accumulated up to the official total batch size,
so every recipe also trains on a single small GPU (`s`: batch 8 uses 6.5 GB).
Hyper-parameters and class names: `utils/args.yaml`.

### Test

* Run `python main.py --model s --test` to evaluate `./weights/best.pt`

### Demo

* Run `python main.py --model s --demo` to run `./weights/best.pt` on the webcam
* `--weights` takes `*.pt`, official `*.pth` / `*.safetensors`, `*.onnx` or a TensorRT `*.engine`
* `--source 1`, `--source video.mp4`, `--source rtsp://...` select another camera, a file or a stream
* `--conf 0.5` sets the score threshold

### Export

* Run `python main.py --model s --export` for ONNX (FP32, onnxruntime / CPU)
* Run `python main.py --model s --export engine --batch-size 4` for TensorRT FP16 (max batch 4)
* `--export engine --weights *_fp16.onnx` builds the engine directly from an exported ONNX
* Input: `images` (N, 3, H, W) RGB in [0, 1], normalization is inside the model
* Output: `outputs` (N, 300, 6) as `x1, y1, x2, y2, score, class` in input pixels, no NMS
* TensorRT >= 10.6. An engine runs only on the GPU and TensorRT version that built it

### Results

| Version | Input | Params | GFLOPs | COCO AP | TensorRT AP | TensorRT ms |                                                               Download |
|:-------:|:-----:|-------:|-------:|--------:|------------:|------------:|-----------------------------------------------------------------------:|
|  atto   |  320  |   0.5M |    0.8 |    23.7 |        23.8 |         0.7 |  [Model](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_ATTO_COCO) |
|  femto  |  416  |   1.0M |    1.7 |    31.0 |           - |           - | [Model](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO) |
|  pico   |  640  |   1.5M |    5.2 |    38.5 |           - |           - |  [Model](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_PICO_COCO) |
|    n    |  640  |   3.6M |    7.0 |    43.0 |        43.0 |         1.4 |     [Model](https://huggingface.co/Intellindust/DEIMv2_HGNetv2_N_COCO) |
|    s    |  640  |   9.8M |   49.6 |    50.9 |        50.9 |         3.2 |      [Model](https://huggingface.co/Intellindust/DEIMv2_DINOv3_S_COCO) |
|    m    |  640  |  18.2M |   84.1 |    53.0 |           - |           - |      [Model](https://huggingface.co/Intellindust/DEIMv2_DINOv3_M_COCO) |
|    l    |  640  |  32.2M |  144.5 |    56.0 |           - |           - |      [Model](https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO) |
|    x    |  640  |  50.4M |  199.6 |    57.8 |        57.8 |        20.3 |      [Model](https://huggingface.co/Intellindust/DEIMv2_DINOv3_X_COCO) |

* COCO AP: official weights evaluated with this code on COCO val2017, identical to the official results
* TensorRT: FP16 engine, RTX 4070 Laptop, TensorRT 11.4, batch 1, `-` is not evaluated
* `x` keeps DINOv3 attention in FP32 (FP16 overflows), PyTorch AMP is faster for it (16 ms)

### Dataset structure

    ├── COCO
        ├── images
            ├── train2017
                ├── 1111.jpg
            ├── val2017
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
            ├── val2017
                ├── 2222.txt
        ├── train2017.txt
        ├── val2017.txt

Labels are YOLO format: `class cx cy w h`, normalized to [0, 1], one object per line.

### Project structure

    ├── main.py          train / test / demo / export
    ├── main.sh          multi-GPU launcher
    ├── nets/nn.py       HGNetV2, DINOv3 ViT + STA, hybrid / lite encoder, DEIM decoder
    ├── utils/util.py    matcher, losses, denoising, LR, EMA, metrics, weight loading, ONNX / TensorRT
    ├── utils/dataset.py dataset, augmentations (mosaic, zoom-out, IoU crop, photometric, flip, copy-blend, multi-scale)
    └── utils/args.yaml  hyper-parameters per model

### Changes from the official code

* Plain PyTorch modules and one `args.yaml` instead of the registry / YAML-injection framework
* Official `*.pth` and Hugging Face `*.safetensors` load directly (`util.load_weight`)
* Fixed: a missing ViT checkpoint silently trained the backbone from scratch, the first iteration skipped warm-up,
  settings leaked between models built in one process
* TensorRT: deformable attention is exported with gathers, TensorRT `GridSample` corrupts the decoder
* The Hugging Face `config.json` of `s` lists ViT layers `[5, 8, 11]`, the weights need `[3, 7, 11]`

#### Reference

* https://github.com/Intellindust-AI-Lab/DEIMv2

#### License

Derived from DEIMv2, see [LICENSE.md](./LICENSE.md): **non-commercial use only**.
Commercial use requires a license from INTELLINDUST.
