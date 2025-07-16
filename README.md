
# Diabetic Retinopathy Classification using FCN-CLIP with CBAM

This project implements a hybrid deep learning model that classifies diabetic retinopathy (DR) from fundus images. It combines visual features from a pre-trained CLIP model with a fully convolutional network (FCN) enhanced by CBAM (Convolutional Block Attention Module) for better lesion localization. This is part of the final project for ELEC 4840: AI for Medical Image Analysis, contributed by Gyumin Lee.

> This repository only contains the code and model implementation related to the **FCN-CLIP with CBAM** approach.

---

## Overview

### FCN-CLIP Hybrid Model

- The model uses the **CLIP image encoder (ResNet-based)** as a fixed feature extractor.
- Extracted feature maps are passed through a **deep FCN decoder** to generate predictions.
- A **CBAM module** is inserted to guide the network to focus on DR-relevant regions such as microaneurysms and hemorrhages.
- The network is trained using **class-weighted cross-entropy** to mitigate class imbalance.

### CBAM Attention

- **Channel Attention**: Highlights important feature channels using global pooling and bottleneck layers.
- **Spatial Attention**: Focuses on relevant spatial regions using a 7×7 convolution and sigmoid gating.

---

## Dataset Setup

### 1. APTOS 2019 Dataset
- Download the dataset from Kaggle:  
  https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- Place all training images in a folder named `train_images` in the same directory as the notebook.

```
/Q1
 ├─ train_images/
 │   └─ *.png
```

### 2. Messidor-2 Dataset
- Place the Messidor images inside the `IMAGES` folder:

```
/Q1
 ├─ IMAGES/
 │   └─ *.tif
```

---

## How to Run

1. Open the notebook: `elec4840finalproject.ipynb` in Google Colab.
2. Set the working directory:
   ```python
   import os
   os.chdir('/content/drive/MyDrive/elec4840/Code_Template_new/Q1')
   ```
3. Download and import all utility modules from the notebook’s first code cells.
4. Execute all code cells **sequentially from top to bottom**.

---

## Performance Summary

| Model              | Accuracy | AUROC | Notes                                |
|--------------------|----------|--------|--------------------------------------|
| FCN-CLIP + CBAM    | 92.49%   | 0.99   | Strong at lesion localization        |
| Zero-shot CLIP     | 55.17%   | 0.75   | No fine-tuning, low generalization   |
| Ranking-aware CLIP | 93.42%   | 0.9845 | Higher recall, excels on severe DR   |

- The hybrid model is especially effective in distinguishing **mild and moderate DR** stages.
- Achieves **74% recall** for Proliferative DR.

---

## Training Details

| Hyperparameter   | Value        |
|------------------|--------------|
| Optimizer        | AdamW        |
| Learning Rate    | 2e-4         |
| Epochs           | 20           |
| Batch Size       | 32           |
| Early Stopping   | Patience = 7 |
| Loss Function    | Focal Loss (γ = 2.0) |
| CBAM Reduction   | 16           |

---

## References

1. Radford et al., *Learning Transferable Visual Models from Natural Language Supervision*, ICML 2021.
2. Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018.
3. APTOS 2019 Blindness Detection Dataset – [Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
4. Messidor-2 Dataset

---

## Notes

- This repository includes only the code and components relevant to the **FCN-CLIP with CBAM** model developed by Gyumin Lee.
- This is not the full version of the final project. 
