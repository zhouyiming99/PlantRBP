# RNA-Protein Interaction Prediction Suite (MHGAT)

This repository provides a comprehensive pipeline for predicting RNA-protein interactions, specifically designed to identify RNA binding sites using a Heterogeneous Graph Attention Network (MHGAT). The workflow covers everything from raw structural and genomic data processing to bias-corrected deep learning inference and downstream bioinformatics analysis.

---

## 1. Data Preprocessing

The preprocessing module converts raw experimental and structural data into a unified format suitable for deep learning.

### Q-BioLiP Structure Data Processor
**Script:** `scripts/process_qbiolip_gpu.py`
*   **Functionality:** Parses PDB structure files for both proteins and RNA using GPU-accelerated coordinate calculations to determine spatial distances.
*   **Usage Command:**
    ```bash
    python scripts/process_qbiolip_gpu.py \
        --receptor_dir data/qbiolip/receptors \
        --ligand_dir data/qbiolip/ligands \
        --annotation data/qbiolip/annotation.txt \
        --fasta_dir data/qbiolip/fastas \
        --output_dir data/qbiolip/processed \
        --device gpu \
        --distance_cutoff 5.0
    ```

### Arabidopsis CLIP-seq Data Processor
**Script:** `scripts/process_arabidopsis.py`
*   **Functionality:** Integrates Arabidopsis thaliana CLIP-seq peak data with genomic sequences and generates synthetic negative samples.
*   **Usage Command:**
    ```bash
    python scripts/process_arabidopsis.py \
        --genome data/genome/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
        --proteins data/genome/Arabidopsis_thaliana.TAIR10.pep.all.fa \
        --gtf data/genome/Arabidopsis_thaliana.TAIR10.57.gtf \
        --bed data/clip/peaks.bed \
        --output_dir data/arabidopsis/processed \
        --window_size 101 \
        --neg_ratio 1.0
    ```

---

## 2. Model Architecture and Training

The core of this suite is the Multi-scale Heterogeneous Graph Attention Network (MHGAT).

### MHGAT Model and Trainer
**Script:** `train_mhgat.py`
*   **Functionality:** Trains the MHGAT model using multi-scale RNA graphs, ESM-2 protein embeddings, and heterogeneous cross-attention.
*   **Usage Command:**
    ```bash
    # Standard Training (Pre-training + Fine-tuning)
    python train_mhgat.py \
        --mode train \
        --qbiolip_data data/qbiolip/processed/qbiolip_unified.json \
        --output_dir outputs/mhgat_model \
        --batch_size 16 \
        --epochs 100

    # Transfer Learning to Target Species
    python train_mhgat.py \
        --mode transfer \
        --checkpoint outputs/mhgat_model/best_finetune.pt \
        --arabidopsis_data data/arabidopsis/processed/arabidopsis_unified.json \
        --output_dir outputs/mhgat_transfer \
        --transfer_epochs 50
    ```

---

## 3. Prediction and Post-processing

This module handles inference on new sequences and ensures results reliability through systematic bias correction.

### Prediction and Bias Correction
**Script:** `predict_and_fix_mhgat.py`
*   **Functionality:** Performs inference and applies corrections such as boundary effect elimination and temperature scaling.
*   **Usage Command:**
    ```bash
    # Batch Prediction from FASTA with Auto-Preprocessing and Combined Correction
    python predict_and_fix_mhgat.py \
        --mode repredict \
        --checkpoint outputs/mhgat_transfer/best_transfer.pt \
        --rna_fasta data/fasta/target_rna.fasta \
        --protein_fasta data/fasta/target_protein.fasta \
        --output_json predictions_fixed.json \
        --method combined \
        --pair_mode one_to_all

    # Fix existing prediction results without re-running inference
    python predict_and_fix_mhgat.py \
        --mode fix_json \
        --input raw_predictions.json \
        --output fixed_predictions.json \
        --method boundary
    ```

---

## 4. Downstream Analysis

Converts raw model outputs into standard formats used in bioinformatics pipelines.

### Binding Site Extraction
**Script:** `extract_binding_sites.py`
*   **Functionality:** Transforms prediction probabilities into actionable BED, TSV, and statistical reports.
*   **Usage Command:**
    ```bash
    python extract_binding_sites.py \
        --input predictions_fixed.json \
        --rna_fasta data/fasta/target_rna.fasta \
        --bed_output results/binding_sites.bed \
        --tsv_output results/binding_probabilities.tsv \
        --summary_output results/stats_summary.txt \
        --min_prob 0.5 \
        --add_nucleotide
    ```
