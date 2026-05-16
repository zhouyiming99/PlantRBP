# PlantRBP
## RBP
# RBP Prediction System

A PPI-enhanced protein RNA-binding protein (RBP) prediction system based on sequence features, ESM embeddings, and graph neural networks.

## Overview

This project provides a single prediction workflow for identifying RBPs from protein FASTA sequences.

### Supported workflow
- **PPI-based RBP prediction only**

### Basic idea
The model combines:
- **Protein sequence features**
- **ESM-2 embeddings**
- **Protein-protein interaction (PPI) network information**

The final predictor is a **5-fold ensemble model** trained in advance.

---

## Workflow

### Step 1: Training

Run `train.py` to train the model.

After training, five fold models will be generated and saved in the `models/` directory, for example:

```bash
models/balanced_fold_1.pt
models/balanced_fold_2.pt
models/balanced_fold_3.pt
models/balanced_fold_4.pt
models/balanced_fold_5.pt

Step 2: Prediction
Only one prediction mode is kept: PPI-based prediction.

In practical deployment, the server should store built-in species-specific resources, including:

PPI network file
Background proteome FASTA file
Optional precomputed ESM cache file
This means that end users only need to provide a protein FASTA file and select the species.
The system will then output a prediction result table for the target proteins.
Example Prediction Command
For Arabidopsis thaliana, an example command is:
python predict.py --mode ppi \
    --input ara_rbp_10.fasta \
    --model_dir ./models \
    --ppi 3702.protein.v12.0.Arabidopsis_thaliana.TAIR10.pep.all.txt \
    --background Arabidopsis_thaliana.TAIR10.pep.all.fasta \
    --output results.csv \
    --esm_cache ./balanced_cache_v2_650M/esm_features_650M.pkl
