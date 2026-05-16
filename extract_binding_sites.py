import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_fasta(fasta_file: str) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    cur_id, cur_seq = None, []
    try:
        with open(fasta_file, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if cur_id is not None:
                        seqs[cur_id] = ''.join(cur_seq)
                    cur_id = line[1:].split()[0]
                    cur_seq = []
                else:
                    cur_seq.append(line.upper())
        if cur_id is not None:
            seqs[cur_id] = ''.join(cur_seq)
        logger.info(f"Loaded {len(seqs)} sequences from {fasta_file}")
        return seqs
    except Exception as e:
        logger.error(f"Failed to read FASTA file: {e}")
        return {}

def generate_bed(
    predictions: List[Dict],
    output_bed: str,
    min_prob: float = 0.5,
    only_binding: bool = True,
    rna_filter: Optional[Set[str]] = None,
    protein_filter: Optional[Set[str]] = None,
    rna_sequences: Optional[Dict[str, str]] = None,
    add_nucleotide: bool = False,
) -> Tuple[int, int]:
    logger.info("=" * 70)
    logger.info("Generating BED file (base-wise)")
    logger.info(f"  Output:            {output_bed}")
    logger.info(f"  Min Probability:   {min_prob}")
    logger.info(f"  Only Binding:      {only_binding}")
    logger.info(f"  RNA Filter:        {len(rna_filter) if rna_filter else 'None'}")
    logger.info(f"  Protein Filter:    {len(protein_filter) if protein_filter else 'None'}")
    logger.info(f"  Add Nucleotide:    {add_nucleotide}")
    logger.info("=" * 70)

    bed_records = []
    total_bases = 0
    filtered_bases = 0

    for pred in predictions:
        if 'error' in pred:
            continue
        rna_id = pred.get('rna_id', pred.get('sample_id', 'unknown'))
        protein_id = pred.get('protein_id', 'unknown')
        if rna_filter and rna_id not in rna_filter:
            continue
        if protein_filter and protein_id not in protein_filter:
            continue
        probs = pred.get('binding_probs', [])
        labels = pred.get('binding_labels', [])
        rna_seq = rna_sequences.get(rna_id, '') if rna_sequences else ''
        for i, (prob, label) in enumerate(zip(probs, labels)):
            total_bases += 1
            if only_binding and label != 1:
                filtered_bases += 1
                continue
            if prob < min_prob:
                filtered_bases += 1
                continue
            chrom = rna_id
            chrom_start = i
            chrom_end = i + 1
            name = rna_id
            score = int(prob * 1000)
            strand = '+'
            fields = [chrom, str(chrom_start), str(chrom_end), name, str(score), strand, protein_id, f"{prob:.6f}"]
            if add_nucleotide:
                nucleotide = rna_seq[i] if rna_seq and i < len(rna_seq) else 'N'
                fields.append(nucleotide)
            bed_records.append('\t'.join(fields))

    Path(output_bed).parent.mkdir(parents=True, exist_ok=True)
    with open(output_bed, 'w') as fh:
        header_fields = ['#chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'protein_id', 'binding_prob']
        if add_nucleotide:
            header_fields.append('nucleotide')
        fh.write('\t'.join(header_fields) + '\n')
        for record in bed_records:
            fh.write(record + '\n')
    logger.info(f"BED generation complete. Total bases: {total_bases}, Filtered: {filtered_bases}, Output: {len(bed_records)}")
    return total_bases, len(bed_records)

def generate_tsv(
    predictions: List[Dict],
    output_tsv: str,
    rna_filter: Optional[Set[str]] = None,
    protein_filter: Optional[Set[str]] = None,
    rna_sequences: Optional[Dict[str, str]] = None,
    include_zero_prob: bool = True,
) -> Tuple[int, int]:
    logger.info("=" * 70)
    logger.info("Generating TSV file")
    logger.info(f"  Output:           {output_tsv}")
    logger.info(f"  RNA Filter:       {len(rna_filter) if rna_filter else 'None'}")
    logger.info(f"  Protein Filter:   {len(protein_filter) if protein_filter else 'None'}")
    logger.info(f"  Include Zero:     {include_zero_prob}")
    logger.info("=" * 70)

    tsv_records = []
    total_samples = 0
    total_bases = 0
    for pred in predictions:
        if 'error' in pred:
            continue
        rna_id = pred.get('rna_id', pred.get('sample_id', 'unknown'))
        protein_id = pred.get('protein_id', 'unknown')
        if rna_filter and rna_id not in rna_filter:
            continue
        if protein_filter and protein_id not in protein_filter:
            continue
        total_samples += 1
        probs = pred.get('binding_probs', [])
        labels = pred.get('binding_labels', [])
        rna_seq = rna_sequences.get(rna_id, '') if rna_sequences else ''
        for i, (prob, label) in enumerate(zip(probs, labels), 1):
            if not include_zero_prob and prob == 0:
                continue
            total_bases += 1
            nucleotide = rna_seq[i - 1] if rna_seq and i <= len(rna_seq) else 'N'
            fields = [rna_id, protein_id, str(i), nucleotide, f"{prob:.6f}", str(label)]
            tsv_records.append('\t'.join(fields))

    Path(output_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_tsv, 'w') as fh:
        header = ['transcript_id', 'protein_id', 'position', 'nucleotide', 'binding_prob', 'binding_label']
        fh.write('\t'.join(header) + '\n')
        for record in tsv_records:
            fh.write(record + '\n')
    logger.info(f"TSV generation complete. Samples: {total_samples}, Bases: {total_bases}, Rows: {len(tsv_records)}")
    return total_samples, total_bases

def generate_summary(predictions: List[Dict], output_txt: str) -> Dict:
    logger.info("=" * 70)
    logger.info("Generating statistical summary")
    logger.info(f"  Output: {output_txt}")
    logger.info("=" * 70)

    valid_preds = [p for p in predictions if 'error' not in p]
    error_preds = [p for p in predictions if 'error' in p]
    rna_counts = defaultdict(int)
    protein_counts = defaultdict(int)
    binding_ratios, total_sites, max_probs, mean_probs = [], [], [], []

    for pred in valid_preds:
        rna_id = pred.get('rna_id', pred.get('sample_id', 'unknown'))
        protein_id = pred.get('protein_id', 'unknown')
        stats = pred.get('statistics', {})
        rna_counts[rna_id] += 1
        protein_counts[protein_id] += 1
        binding_ratios.append(stats.get('binding_ratio', 0))
        total_sites.append(stats.get('num_binding_regions', 0))
        max_probs.append(stats.get('max_binding_prob', 0))
        mean_probs.append(stats.get('mean_binding_prob', 0))

    stats_dict = {
        'total_samples': len(predictions),
        'valid_samples': len(valid_preds),
        'error_samples': len(error_preds),
        'unique_rnas': len(rna_counts),
        'unique_proteins': len(protein_counts),
    }

    if binding_ratios:
        stats_dict.update({'binding_ratio_mean': np.mean(binding_ratios), 'binding_ratio_median': np.median(binding_ratios), 'binding_ratio_std': np.std(binding_ratios), 'binding_ratio_min': np.min(binding_ratios), 'binding_ratio_max': np.max(binding_ratios)})
    if total_sites:
        stats_dict.update({'sites_mean': np.mean(total_sites), 'sites_median': np.median(total_sites), 'sites_total': sum(total_sites)})

    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, 'w') as fh:
        fh.write("=" * 70 + "\nMHGAT Prediction Statistical Summary\n" + "=" * 70 + "\n\n")
        fh.write(f"Total Samples:      {stats_dict['total_samples']:,}\n  Successful:       {stats_dict['valid_samples']:,}\n  Errors:           {stats_dict['error_samples']:,}\n\n")
        fh.write(f"Unique RNAs:        {stats_dict['unique_rnas']:,}\nUnique Proteins:    {stats_dict['unique_proteins']:,}\n\n")
        if binding_ratios:
            fh.write(f"Binding Ratio Stats:\n  Mean:             {np.mean(binding_ratios):.4f}\n  Median:           {np.median(binding_ratios):.4f}\n  Std Dev:          {np.std(binding_ratios):.4f}\n  Range:            [{np.min(binding_ratios):.4f}, {np.max(binding_ratios):.4f}]\n\n")
        if total_sites:
            fh.write(f"Binding Regions Stats:\n  Mean:             {np.mean(total_sites):.2f}\n  Median:           {np.median(total_sites):.0f}\n  Total:            {sum(total_sites):,}\n\n")
        fh.write("=" * 70 + "\nTOP 10 RNA (by prediction count)\n" + "=" * 70 + "\n")
        for rna_id, cnt in sorted(rna_counts.items(), key=lambda x: -x[1])[:10]:
            fh.write(f"  {rna_id:40s}  {cnt:5d}\n")
        fh.write("\n" + "=" * 70 + "\nTOP 10 Protein (by prediction count)\n" + "=" * 70 + "\n")
        for prot_id, cnt in sorted(protein_counts.items(), key=lambda x: -x[1])[:10]:
            fh.write(f"  {prot_id:40s}  {cnt:5d}\n")
        if error_preds:
            fh.write("\n" + "=" * 70 + "\nError Samples List\n" + "=" * 70 + "\n")
            for i, pred in enumerate(error_preds[:20], 1):
                sid = pred.get('sample_id', pred.get('rna_id', 'unknown'))
                fh.write(f"{i:3d}. {sid}: {pred.get('error', 'unknown error')}\n")
    logger.info(f"Statistical summary saved: {output_txt}")
    return stats_dict

def generate_prob_distribution(predictions: List[Dict], output_file: str):
    logger.info(f"Generating probability distribution stats: {output_file}")
    all_probs, binding_probs = [], []
    for pred in predictions:
        if 'error' in pred: continue
        p, l = pred.get('binding_probs', []), pred.get('binding_labels', [])
        all_probs.extend(p)
        binding_probs.extend([val for val, lbl in zip(p, l) if lbl == 1])
    if not all_probs: return
    all_probs = np.array(all_probs)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as fh:
        fh.write("=" * 70 + "\nBinding Probability Distribution\n" + "=" * 70 + "\n\n")
        fh.write(f"All Bases Stats:\n  Total Bases:      {len(all_probs):,}\n  Mean:             {all_probs.mean():.6f}\n  Median:           {np.median(all_probs):.6f}\n  Max:              {all_probs.max():.6f}\n\n")
        fh.write("Probability Bin Distribution:\n")
        bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for low, high in bins:
            count = np.sum((all_probs >= low) & (all_probs < high))
            pct = count / len(all_probs) * 100
            fh.write(f"  [{low:.1f}, {high:.1f}):  {count:8,}  ({pct:5.2f}%)  {'█' * int(pct / 2)}\n")
    logger.info(f"Probability distribution saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract base-wise binding sites from MHGAT JSON results')
    parser.add_argument('--input', type=str, required=True, help='Prediction result JSON file')
    parser.add_argument('--bed_output', type=str, default='binding_sites.bed')
    parser.add_argument('--tsv_output', type=str, default='binding_probs.tsv')
    parser.add_argument('--summary_output', type=str, default='summary.txt')
    parser.add_argument('--prob_dist_output', type=str, default='prob_distribution.txt')
    parser.add_argument('--rna_fasta', type=str, default=None, help='RNA FASTA for nucleotide info')
    parser.add_argument('--min_prob', type=float, default=0.5, help='Min probability threshold [0-1]')
    parser.add_argument('--no_only_binding', action='store_true', help='Output all bases (default only binding labels)')
    parser.add_argument('--rna_filter', type=str, nargs='*', default=None)
    parser.add_argument('--protein_filter', type=str, nargs='*', default=None)
    parser.add_argument('--add_nucleotide', action='store_true', help='Add nucleotide column to BED (requires --rna_fasta)')
    parser.add_argument('--skip_bed', action='store_true'); parser.add_argument('--skip_tsv', action='store_true')
    parser.add_argument('--skip_summary', action='store_true'); parser.add_argument('--skip_prob_dist', action='store_true')
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    with open(args.input, 'r') as fh:
        data = json.load(fh)
    predictions = data['predictions'] if 'predictions' in data else (data if isinstance(data, list) else [data])
    rna_seqs = load_fasta(args.rna_fasta) if args.rna_fasta else None
    rf, pf = set(args.rna_filter) if args.rna_filter else None, set(args.protein_filter) if args.protein_filter else None

    if not args.skip_bed: generate_bed(predictions, args.bed_output, args.min_prob, not args.no_only_binding, rf, pf, rna_seqs, args.add_nucleotide)
    if not args.skip_tsv: generate_tsv(predictions, args.tsv_output, rf, pf, rna_seqs)
    if not args.skip_summary: generate_summary(predictions, args.summary_output)
    if not args.skip_prob_dist: generate_prob_distribution(predictions, args.prob_dist_output)
    logger.info("Processing complete.")

if __name__ == '__main__':
    main()
