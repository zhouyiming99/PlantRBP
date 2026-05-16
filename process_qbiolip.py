import os
import json
import numpy as np
import logging
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RNA_BASES = {
    'A': 'A', 'ADE': 'A', 'U': 'U', 'URA': 'U', 'G': 'G', 'GUA': 'G', 'C': 'C', 'CYT': 'C',
    'I': 'I', 'PSU': 'U', '5MU': 'U', '1MA': 'A', '7MG': 'G', '5MC': 'C'
}
AMINO_ACIDS = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'MSE': 'M'
}

class FastPDBParser:
    @staticmethod
    def parse_coords_and_seq(pdb_file: Path, is_rna: bool) -> Tuple[np.ndarray, List[str], np.ndarray, List]:
        coords = []
        atom_res_indices = []
        residues_seen = {} 
        residue_order = [] 
        
        valid_res_names = RNA_BASES if is_rna else AMINO_ACIDS
        
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        res_name = line[17:20].strip()
                        if res_name not in valid_res_names:
                            continue
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            chain_id = line[21].strip() or 'A'
                            res_seq = int(line[22:26])
                        except ValueError:
                            continue

                        key = (chain_id, res_seq)
                        coords.append([x, y, z])
                        atom_res_indices.append(key)
                        
                        if key not in residues_seen:
                            residues_seen[key] = res_name
                            residue_order.append(key)
        except Exception:
            return np.array([]), [], np.array([]), []

        if not coords:
            return np.array([]), [], np.array([]), []

        sequence = []
        for key in residue_order:
            res_name = residues_seen[key]
            if is_rna:
                sequence.append(RNA_BASES.get(res_name, 'N'))
            else:
                sequence.append(AMINO_ACIDS.get(res_name, 'X'))
        
        key_to_idx = {k: i for i, k in enumerate(residue_order)}
        atom_res_map = [key_to_idx[k] for k in atom_res_indices]
        
        return np.array(coords, dtype=np.float32), sequence, np.array(atom_res_map, dtype=np.int32), residue_order

class DataManager:
    def __init__(self, annotation_file, fasta_dir):
        self.annotation_map = self._load_annotations(annotation_file)
        self.fasta_dir = Path(fasta_dir)
        self.fasta_index = self._index_fasta(self.fasta_dir)
        
    def _load_annotations(self, path):
        anno = {}
        with open(path, 'r') as f:
            f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2].strip() == 'RNA':
                    assembly_id = parts[0].strip()
                    anno[assembly_id] = {
                        'ligand_file': parts[1].strip(),
                        'binding_str': parts[3].strip() if len(parts) > 3 else ""
                    }
        return anno

    def _index_fasta(self, directory):
        idx = {}
        for p in directory.glob("*.fasta"):
            idx[p.stem] = p
        for p in directory.glob("*.fa"):
            if p.stem not in idx:
                idx[p.stem] = p
        return idx

def process_single_entry(args):
    assembly_id, anno_data, rec_dir, lig_dir, fasta_path, cutoff, use_gpu, num_gpus = args
    
    try:
        rec_path = None
        for name in [f"{assembly_id}.pdb", f"{assembly_id.lower()}.pdb", f"{assembly_id.upper()}.pdb"]:
            if (rec_dir / name).exists():
                rec_path = rec_dir / name
                break
        if not rec_path: return None

        lig_file_base = anno_data['ligand_file']
        lig_path = None
        cand = lig_dir / f"{lig_file_base}.pdb"
        if cand.exists():
            lig_path = cand
        else:
            base = lig_file_base.split('_RNA')[0]
            try:
                found = list(lig_dir.glob(f"{base}*RNA*.pdb"))
                if found: lig_path = found[0]
            except: pass
        if not lig_path: return None

        prot_coords, prot_seq_pdb, _, _ = FastPDBParser.parse_coords_and_seq(rec_path, is_rna=False)
        rna_coords, rna_seq_list, rna_atom_map, rna_res_keys = FastPDBParser.parse_coords_and_seq(lig_path, is_rna=True)
        
        if len(prot_coords) == 0 or len(rna_coords) == 0:
            return None

        rna_sequence = "".join(rna_seq_list)
        protein_sequence = None
        if fasta_path and fasta_path.exists():
            try:
                with open(fasta_path, 'r') as f:
                    lines = f.readlines()
                    seq_lines = [l.strip() for l in lines if not l.startswith('>')]
                    protein_sequence = "".join(seq_lines)
            except: pass
        
        if not protein_sequence:
            protein_sequence = "".join(prot_seq_pdb)

        if not protein_sequence or not rna_sequence:
            return None

        n_residues = len(rna_sequence)
        binding_labels = [0] * n_residues
        atom_contact = None

        if use_gpu and HAS_TORCH and num_gpus > 0:
            gpu_id = random.randint(0, num_gpus - 1)
            device = torch.device(f"cuda:{gpu_id}")
            try:
                t_rna = torch.from_numpy(rna_coords).to(device)
                t_prot = torch.from_numpy(prot_coords).to(device)
                dists = torch.cdist(t_rna, t_prot)
                min_dists = dists.min(dim=1).values
                atom_contact = (min_dists <= cutoff).cpu().numpy()
            except RuntimeError:
                from scipy.spatial.distance import cdist
                dists = cdist(rna_coords, prot_coords, metric='euclidean')
                atom_contact = dists.min(axis=1) <= cutoff
        else:
            from scipy.spatial.distance import cdist
            dists = cdist(rna_coords, prot_coords, metric='euclidean')
            atom_contact = dists.min(axis=1) <= cutoff

        for idx, is_contact in enumerate(atom_contact):
            if is_contact:
                res_idx = rna_atom_map[idx]
                if res_idx < n_residues:
                    binding_labels[res_idx] = 1
                    
        if sum(binding_labels) == 0:
            return None

        rna_chains = "_".join(sorted(set(k[0] for k in rna_res_keys)))
        
        return {
            "rna_sequence": rna_sequence,
            "protein_sequence": protein_sequence,
            "binding_labels": binding_labels,
            "sample_id": f"qbiolip_{assembly_id}",
            "source": "qbiolip",
            "pdb_id": assembly_id.split('_')[0],
            "rna_chains": rna_chains
        }
    except Exception:
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fast Q-BioLiP Processor (GPU/CPU)')
    parser.add_argument('--receptor_dir', required=True, help='Directory containing receptor PDB files')
    parser.add_argument('--ligand_dir', required=True, help='Directory containing ligand PDB files')
    parser.add_argument('--annotation', required=True, help='Path to annotation.txt')
    parser.add_argument('--fasta_dir', required=True, help='Directory containing protein FASTA files')
    parser.add_argument('--output_dir', default='data/qbiolip/processed', help='Output directory')
    parser.add_argument('--distance_cutoff', type=float, default=5.0)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Computing device')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel processes (0 for auto)')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rec_dir = Path(args.receptor_dir)
    lig_dir = Path(args.ligand_dir)
    
    num_gpus = 0
    if args.device == 'gpu':
        if HAS_TORCH and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Detected {num_gpus} GPU device(s)")
        else:
            logger.warning("GPU or PyTorch not detected, falling back to CPU")
            args.device = 'cpu'

    if args.workers == 0:
        if args.device == 'gpu':
            args.workers = min(32, num_gpus * 8) if num_gpus > 0 else 8
        else:
            args.workers = os.cpu_count()

    logger.info("Indexing data...")
    manager = DataManager(args.annotation, args.fasta_dir)
    assembly_ids = list(manager.annotation_map.keys())
    
    logger.info(f"Total tasks: {len(assembly_ids)}")
    logger.info(f"Mode: {args.device.upper()}, Workers: {args.workers}")

    tasks = []
    use_gpu_flag = (args.device == 'gpu')
    for aid in assembly_ids:
        f_path = manager.fasta_index.get(aid) or manager.fasta_index.get(aid.lower()) or manager.fasta_index.get(aid.upper())
        tasks.append((aid, manager.annotation_map[aid], rec_dir, lig_dir, f_path, args.distance_cutoff, use_gpu_flag, num_gpus))

    chunk_size = max(1, len(tasks) // (args.workers * 4))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        processed_iter = list(tqdm(executor.map(process_single_entry, tasks, chunksize=chunk_size), total=len(tasks), desc="Processing", unit="pdb"))
        
    valid_samples = [r for r in processed_iter if r is not None]
    output_file = output_dir / "qbiolip_unified.json"
    
    logger.info(f"Processing complete. Success: {len(valid_samples)} / Total: {len(assembly_ids)}")
    
    if valid_samples:
        avg_len = np.mean([len(s['rna_sequence']) for s in valid_samples])
        logger.info(f"Average RNA length: {avg_len:.2f}")
    
    final_data = {
        "metadata": {"source": "Q-BioLiP-Fast-GPU", "total": len(valid_samples), "cutoff": args.distance_cutoff},
        "samples": valid_samples
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
