# PlantRBP

## RBP Prediction System

A PPI-enhanced RNA-binding protein (RBP) prediction system based on sequence features, ESM-2 embeddings, and graph neural networks.

---

## Overview

This project provides a single prediction workflow for identifying RBPs from protein FASTA sequences.

### Supported Workflow

- **PPI-based RBP prediction only**

### Basic Idea

The model integrates:

- **Protein sequence features**
- **ESM-2 embeddings**
- **Protein–protein interaction (PPI) network information**

The final predictor is a **5-fold ensemble model** trained in advance.

---

## Workflow

### Step 1: Training

Run `train.py` to train the model.

```bash
python train.py \
    --fasta all_proteins.fasta \
    --rbp_list rbp_ids.txt \
    --ppi ppi_network.txt \
    --model_dir ./models
```

After training, five fold models will be generated and saved in the `models/` directory, for example:

```bash
models/balanced_fold_1.pt
models/balanced_fold_2.pt
models/balanced_fold_3.pt
models/balanced_fold_4.pt
models/balanced_fold_5.pt
```

These five models are all required during prediction.

---

### Step 2: Prediction

Only one prediction mode is kept: **PPI-based prediction**.

In practical deployment, the server should store built-in species-specific resources, including:

- **PPI network file**
- **Background proteome FASTA file**
- **Optional precomputed ESM cache file**

This means that **end users only need to provide a protein FASTA file and select the species**.  
The system will then output a prediction result table for the target proteins.

---

## Example Prediction Command

For **Arabidopsis thaliana**, an example command is:

```bash
python predict.py --mode ppi \
    --input ara_rbp_10.fasta \
    --model_dir ./models \
    --ppi 3702.protein.v12.0.Arabidopsis_thaliana.TAIR10.pep.all.txt \
    --background Arabidopsis_thaliana.TAIR10.pep.all.fasta \
    --output results.csv \
    --esm_cache ./balanced_cache_v2_650M/esm_features_650M.pkl
```

---

## Deployment Design

For practical deployment, it is recommended to store species-specific reference data on the server side.

### Server-side Built-in Resources

For each supported species, the following files should be prepared:

- PPI network file
- Background proteome FASTA file
- Optional ESM feature cache

For example, for *Arabidopsis thaliana*:

- `3702.protein.v12.0.Arabidopsis_thaliana.TAIR10.pep.all.txt`
- `Arabidopsis_thaliana.TAIR10.pep.all.fasta`

Additional species can be supported later by adding their corresponding PPI and background proteome files.

### End-user Input

Users only need to provide:

- A FASTA file of target protein sequences
- The selected species

The backend can automatically load the correct species-specific reference files and produce the final prediction results.

---

## Input

### User Input

- Protein FASTA file containing target sequences
- Selected species

### Reference Data

- PPI network file
- Background proteome FASTA file
- Optional ESM cache file

---

## Output

The output is a CSV table containing prediction results for the target proteins.

Typical columns include:

- `protein_id`
- `rbp_score`
- `prediction`
- `confidence`
- `ppi_degree`

Example output file:

```bash
results.csv
```

---

## Requirements

Main dependencies include:

- Python 3.8+
- PyTorch
- PyTorch Geometric
- fair-esm
- scikit-learn
- pandas
- numpy
- scipy
- tqdm

Install ESM with:

```bash
pip install fair-esm
```

---


