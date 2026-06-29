# MCM YOLO Object Detection Model

This repository documents a first **YOLO26s object-detection model** for synthetic underwater **Mine Countermeasure (MCM)** perception.

The model was trained on a synthetic underwater dataset generated from a Gazebo maritime scene. The dataset includes variation in ambient lighting, vehicle light intensity, water color, fog density, and image blur to approximate different underwater visibility conditions.

![YOLO26 inference result](https://raw.githubusercontent.com/FM-Robotics/MCM_yolo26_model/main/inference_result.jpg)

---

## Model

- **Architecture:** YOLO26s
- **Task:** Object Detection
- **Image size:** 640 × 400
- **Epochs:** 150
- **Dataset size:** 288 synthetic images
- **Train / Validation / Test split:** 70 / 20 / 10

### Classes

- Mine
- Mine Anchor
- Mine Chain
- Barrel
- Tire
- Stone
- Crate

---

## Validation Results

| Metric | Value |
|:-------|------:|
| Precision | **0.898** |
| Recall | **0.904** |
| mAP@50 | **0.955** |
| mAP@50-95 | **0.791** |

The current model represents an initial proof of concept. The **Mine** and **Mine Anchor** classes are well represented, while **Barrel**, **Stone**, **Mine Chain**, **Crate**, and **Tire** require additional training samples in future dataset iterations.

---

## Environment Variability

The training dataset includes controlled environmental variation:

- Ambient light intensity
- Vehicle illumination
- Water color
- Fog density
- Image blur

These variations were introduced to improve robustness under different underwater visibility conditions.

---

## Requirements

Tested with:

- Python 3.10
- PyTorch (CUDA)
- Ultralytics

Install:

```bash
python3 -m venv ultralytics_training
source ultralytics_training/bin/activate

pip install --upgrade pip
pip install ultralytics
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset Structure

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Example `data.yaml`

```yaml
train: train/images
val: valid/images
test: test/images

nc: 7

names:
  - barrel
  - crate
  - mine
  - mine_anchor
  - mine_chain
  - stone
  - tire
```

---

## Training

Example `train.py`

```python
from ultralytics import YOLO
import torch


def main():

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo26s.pt")

    model.train(
        data="dataset/data.yaml",
        imgsz=640,
        epochs=150,
        batch=24,
        device=0,
        patience=0,
        project="runs",
        name="mines_training_yolo26"
    )


if __name__ == "__main__":
    main()
```

Run training

```bash
python train.py
```

The trained model is stored in

```
best.pt
```

---

## Validation

```bash
yolo detect val \
    model=best.pt \
    data=dataset/data.yaml \
    split=test \
    imgsz=640
```

---

## Inference

Single image

```bash
yolo detect predict \
    model=best.pt \
    source=dataset/test/images/image.jpg \
    imgsz=640 \
    conf=0.25 \
    save=True
```

Entire folder

```bash
yolo detect predict \
    model=best.pt \
    source=dataset/test/images \
    imgsz=640 \
    conf=0.25 \
    save=True \
    project=inference \
    name=test_results
```

Predictions are saved to

```
inference/test_results/
```

---

## Repository Contents

```
.
├── README.md
├── train.py
├── best.pt
├── requirements.txt
└── examples/
    ├── prediction_01.jpg
    ├── prediction_02.jpg
    └── prediction_03.jpg
```

---

## Future Work

Planned improvements include:

- Larger synthetic dataset
- More debris samples
- Additional environmental randomization
- Instance segmentation for large structures (pipeline, quay wall)
- Integration into a semantic world model for autonomous underwater navigation

## Additional information

https://www.fm-sw.com/project-mission_autonomy.html

## License

This repository is released under the GNU Affero General Public License v3.0 (AGPL-3.0).

The repository contains the training script, trained model, and documentation for a synthetic underwater object detection experiment based on the Ultralytics YOLO framework.

The synthetic training dataset is not included.
