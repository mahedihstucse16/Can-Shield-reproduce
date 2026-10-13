# CAN-SHIELD: reproduction code

Code for the paper "A Novelty Shortcut in CAN Intrusion Detection: Fuzzing Attacks in Training Destroy Cross-Vehicle Transfer, and Protocol-Level Features Are Robust to It" (M. M. Hassan, R. U. Azad, M. R. Alam, M. F. Elahe, A. H. M. S. Sadi, M. S. Sayeed).

## Contents
- `CAN_SHIELD_Reproduction.ipynb`: the full pipeline, all five experiments (E1 to E5) and every figure and table in the paper.

## Data
The datasets are not included. Download them from the official pages:
- Car-Hacking (Hyundai YF Sonata): https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
- OTIDS (KIA Soul): https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset

## Environment
Python 3.13.16, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, imbalanced-learn 0.14.2, SciPy 1.16.3, shap 0.52.0.
Seed 42 for single-draw experiments; seeds 0 to 4 for repeated ones.

## How to run
1. Download both datasets and place the files as described at the top of the notebook.
2. Open the notebook (Jupyter or Google Colab) and run all cells in order.

## Licence
MIT (see LICENSE).
