# Automatic License Plate Recognition for Iranian LIcense Plates
## Institut Polytechnique de Paris
## Course: XINF-573 Image Processing and Computer Vision
## Soheil Lotfi, Ken Browder

This repository is a PyTorch implementation of the method proposed in the paper:

**S. Hatami, M. Sadedel, and F. Jamali, “Iranian license plate recognition using a reliable deep learning approach,” arXiv preprint arXiv:2305.02292, 2023.**

## Overview

The repository provides tools for Iranian license plate detection and character recognition using deep learning. It includes training and testing pipelines, model weights, and inference notebooks for both images and videos.

---

## Repository Structure and Key Files

### 1. **Training**
- **`train.ipynb`**:
  - Trains the character recognition model.
  - The best and final weights are saved in the `saved_models2` directory.

- **`yolo_train.ipynb`**:
  - Trains the YOLOv11 model for license plate detection.
  - The best and last weights are saved in `runs/detect/train7/weights/`.

- **`train_transformer.ipynb`**:
  - An experiment with replacing the LSTM network proposed in the original paper with a transformer-based architecture.
  - The last trained weights for the transformer model are saved in `model_weights_epoch_300_transformer.pth`.

---

### 2. **Testing**
- **`test_pipeline.ipynb`**:
  - Tests the trained network end-to-end on unseen datasets.
  - Allows the user to specify the path to the test dataset (stored locally or on cloud storage).
  - A sample test dataset can be downloaded [here](https://drive.google.com/drive/folders/1ZCwF2mJuQN1LRow7NGCS_27zBy_g2-wJ?usp=sharing). Download and update the dataset path in the notebook manually.

- **`test_pipeline_transformer.ipynb`**:
  - Tests the transformer-based model end-to-end.

---

### 3. **Inference**
- **`single_image_inference.ipynb`**:
  - Performs inference on individual images using the complete end-to-end model.

- **`video_inference.ipynb`**:
  - Performs inference on videos, simulating real-time license plate detection and character recognition.

---

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/pandanautinspace/IR-ALPR.git
   cd IR-ALPR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Download the dataset from the [provided Google Drive link](https://drive.google.com/drive/folders/1ZCwF2mJuQN1LRow7NGCS_27zBy_g2-wJ?usp=sharing).
   - Update the dataset path in the respective notebook to match your local setup.

4. **Run the desired notebook**:
   - Use `train.ipynb` or `yolo_train.ipynb` for training.
   - Use `test_pipeline.ipynb` or `test_pipeline_transformer.ipynb` for testing.
   - Use `single_image_inference.ipynb` or `video_inference.ipynb` for inference.

---

## Results and Metrics

- **Training Losses**: `training_losses.png`
- **Validation Losses**: `validation_losses.png`
- **Learning Rates**: `learning_rates.png`

---

## Citation

If you use this repository, please cite the original paper:

```
@article{hatami2023iranian,
  title={Iranian license plate recognition using a reliable deep learning approach},
  author={Hatami, S. and Sadedel, M. and Jamali, F.},
  journal={arXiv preprint arXiv:2305.02292},
  year={2023}
}
```

---

## Contact

If you have any questions please contact either of the authors of this repo:
soheil.lotfi@ip-paris.fr
kenneth.browder@ip-paris.fr

