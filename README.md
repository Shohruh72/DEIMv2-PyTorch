[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) re-implementation using PyTorch

Real-time DETR detector: 8 models from 0.5M to 50M parameters, no NMS, verified against the official COCO results.

### Installation

```
conda create -n DEIM python=3.11
conda activate DEIM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python PyYAML scipy tqdm
pip install safetensors     # official Hugging Face weights
pip install onnx tensorrt   # only for --export
```

### Quick start

* Download official COCO weights from the [Results](#results) table into `./weights`
* Run `python main.py --model s --demo --weights weights/DEIMv2_DINOv3_S_COCO.safetensors` (webcam, `q` to quit)

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
* `--weights` takes `*.pt`, official `*.pth` / `*.safetensors` or a TensorRT `*.engine`
* `--source 1`, `--source video.mp4`, `--source rtsp://...` select another camera, a file or a stream
* `--conf 0.5` sets the score threshold

### Export

* Run `python main.py --model s --export` for ONNX (FP32, onnxruntime / CPU)
* Run `python main.py --model s --export engine --batch-size 4` for TensorRT FP16 (max batch 4)
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
* https://github.com/jahongir7174/YOLOv11-pt

#### License

Derived from DEIMv2, see [LICENSE.md](./LICENSE.md): **non-commercial use only**.
Commercial use requires a license from INTELLINDUST.
