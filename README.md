# image-corruption-detector

A PyTorch project that detects and classifies image corruption — things like blur, noise, JPEG artifacts, and brightness issues. Built as a way to get hands-on with transfer learning and see how well a pretrained ResNet-18 can adapt to a completely different task.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Live demo:** [HF Spaces](https://huggingface.co/spaces/mannh12/image-corruption-detector)

---

## What it does

The model looks at an image and outputs one of 7 labels:

- `clean` — no corruption
- `gaussian_blur` — out-of-focus or motion blur
- `gaussian_noise` — sensor-style white noise
- `salt_pepper` — random stuck pixels
- `jpeg_artifacts` — low-quality compression blocking
- `brightness` — over/under-exposed
- `contrast` — washed-out, low contrast

The training data is generated from CIFAR-10 — I take clean images and apply each corruption at 3 severity levels to create a controlled, balanced dataset. That way there's no dependency on a hand-labelled dataset.

---

## How the model works

ResNet-18 pretrained on ImageNet is used as the backbone. The first half (conv1 through layer2) stays frozen — those weights already know how to detect edges and textures, which transfers fine. Only layer3, layer4, and a new classification head get trained.

```
Input (224×224×3)
    ↓
ResNet-18 backbone
  conv1, layer1, layer2  ← frozen (ImageNet weights)
  layer3, layer4         ← fine-tuned
  AvgPool → 512-dim
    ↓
Custom head
  Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→7)
    ↓
7-class output
```

This setup converges fast — past 90% validation accuracy in the first few epochs.

---

## Getting started

```bash
pip install -r requirements.txt

# generate the dataset (~3 min, downloads CIFAR-10)
python src/data/generate_dataset.py

# train
python scripts/train.py

# run on a single image
python scripts/predict.py --model-path checkpoints/best_model.pt --image-path photo.jpg
```

---

## Project layout

```
├── app.py                        # Gradio demo (HF Spaces entry point)
├── config/default_config.yaml    # all hyperparameters in one place
├── src/
│   ├── data/
│   │   ├── corruption_pipeline.py  # the corruption functions (OpenCV)
│   │   ├── generate_dataset.py     # builds the dataset from CIFAR-10
│   │   └── dataset.py              # PyTorch Dataset + DataLoader
│   ├── models/resnet_classifier.py # model definition
│   ├── training/
│   │   ├── trainer.py              # training loop with early stopping etc.
│   │   └── metrics.py              # accuracy, F1, confusion matrix
│   ├── inference/predict.py        # single image + batch inference
│   └── utils/
│       ├── logger.py
│       └── visualization.py        # training curves, confusion matrix plots
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── notebooks/exploration.ipynb     # EDA + results walkthrough
```

---

## Results

Numbers from a 25-epoch run on the generated dataset:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| clean | ~0.97 | ~0.98 | ~0.97 |
| gaussian_blur | ~0.94 | ~0.95 | ~0.94 |
| gaussian_noise | ~0.96 | ~0.95 | ~0.96 |
| salt_pepper | ~0.97 | ~0.96 | ~0.97 |
| jpeg_artifacts | ~0.93 | ~0.92 | ~0.93 |
| brightness | ~0.98 | ~0.98 | ~0.98 |
| contrast | ~0.97 | ~0.97 | ~0.97 |

Overall accuracy ~96%, macro F1 ~0.96. Results will vary a bit with different seeds.

---

## Corruption severity levels

Each function takes a severity 1–3:

| Type | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| Gaussian blur | kernel 5 | kernel 9 | kernel 15 |
| Gaussian noise | σ=10 | σ=25 | σ=50 |
| Salt & pepper | 2% pixels | 5% | 10% |
| JPEG | quality 20 | quality 10 | quality 5 |
| Brightness | +40 | −80 | +120 |
| Contrast | 0.7× | 0.4× | 0.15× |

---

## Training setup

- Adam optimizer, lr=0.001, weight decay 1e-4
- ReduceLROnPlateau (halves LR after 3 epochs of no improvement)
- Early stopping with patience 5
- Gradient clipping at norm 1.0
- Light augmentation during training: random horizontal flip + mild color jitter

Everything is in `config/default_config.yaml` so you can tweak without touching source files. All the CLI scripts also accept `--epochs`, `--lr`, `--batch-size` etc. as overrides.

---

## Stuff I'd add given more time

- Test on higher-res images (CIFAR-10 is 32×32, which is pretty small)
- Compare against EfficientNet or a small ViT
- Wrap inference in a FastAPI endpoint so it can run as a microservice
- AMD ROCm support — the code uses `torch.device` throughout so it should just work once ROCm is set up, but I haven't tested it

---

## License

MIT — see [LICENSE](LICENSE)
