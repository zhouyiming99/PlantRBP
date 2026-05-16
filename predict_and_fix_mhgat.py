import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
import esm
from scipy.signal import savgol_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_mhgat import MHGAT, ModelConfig, RNATokenizer, MHGATPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_rna_header(header_line: str) -> Dict:
    header = header_line.lstrip('>')
    parts  = header.split()
    info = {
        'transcript_id':      parts[0] if parts else 'unknown',
        'seq_type':           parts[1] if len(parts) > 1 else '',
        'gene_id':            '',
        'gene_symbol':        '',
        'transcript_biotype': '',
        'gene_biotype':       '',
        'description':        '',
        'raw_header':         header,
    }
    rest = ' '.join(parts[1:])
    patterns = {
        'gene_id':            r'gene:(\S+)',
        'gene_symbol':        r'gene_symbol:(\S+)',
        'transcript_biotype': r'transcript_biotype:(\S+)',
        'gene_biotype':       r'gene_biotype:(\S+)',
        'description':        r'description:(.+?)(?:\s*\[Source|\Z)',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, rest)
        if m:
            info[key] = m.group(1).strip()
    return info

def parse_protein_header(header_line: str) -> Dict:
    header  = header_line.lstrip('>')
    parts   = header.split()
    raw_id  = parts[0] if parts else 'unknown'
    info = {
        'protein_id':         raw_id,
        'gene_id':            '',
        'gene_symbol':        '',
        'gene_biotype':       '',
        'transcript_biotype': '',
        'description':        '',
        'raw_header':         header,
    }
    if '|' in raw_id:
        pipe = raw_id.split('|')
        if pipe[0] in ('sp', 'tr'):
            info['protein_id']  = pipe[1]
            info['gene_symbol'] = pipe[2] if len(pipe) > 2 else ''
        else:
            info['protein_id'] = pipe[-1]
    rest = ' '.join(parts[1:])
    patterns = {
        'gene_id':            r'gene:(\S+)',
        'gene_symbol':        r'gene_symbol:(\S+)',
        'gene_biotype':       r'gene_biotype:(\S+)',
        'transcript_biotype': r'transcript_biotype:(\S+)',
        'description':        r'description:(.+?)(?:\s*\[Source|\Z)',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, rest)
        if m:
            info[key] = m.group(1).strip()
    if not info['gene_symbol'] and info['description']:
        info['gene_symbol'] = info['description'].split()[0]
    return info

def preprocess_rna_fasta(
    input_fasta:    str,
    output_fasta:   str,
    biotype_filter: Optional[List[str]] = None,
    max_len:        int  = 510,
    min_len:        int  = 20,
    truncate_mode:  str  = 'end',
    save_stats:     bool = True,
) -> Tuple[int, Dict]:
    logger.info("=" * 55)
    logger.info("RNA FASTA Preprocessing")
    logger.info(f"  Input:    {input_fasta}")
    logger.info(f"  Output:   {output_fasta}")
    logger.info(f"  Filter:   {biotype_filter or 'None'}")
    logger.info(f"  Range:    {min_len} ~ {max_len} nt")
    logger.info("=" * 55)

    VALID_CHARS = set('AUGCTNRYSWKMBDHV')
    raw_seqs, biotype_cnts = [], defaultdict(int)
    cur_info, cur_seq = None, []

    with open(input_fasta, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if cur_info is not None:
                    raw_seqs.append((cur_info, ''.join(cur_seq)))
                cur_info = parse_rna_header(line)
                cur_seq  = []
                biotype_cnts[cur_info['transcript_biotype']] += 1
            else:
                cur_seq.append(line.upper())
    if cur_info is not None:
        raw_seqs.append((cur_info, ''.join(cur_seq)))

    stats, records = defaultdict(int), []
    for info, raw in raw_seqs:
        if biotype_filter and info['transcript_biotype'] not in biotype_filter:
            stats['filtered_biotype'] += 1
            continue
        seq = raw.replace('T', 'U')
        seq = ''.join(c if c in VALID_CHARS else 'N' for c in seq)
        orig_len = len(seq)
        if orig_len < min_len:
            stats['filtered_short'] += 1
            continue
        if orig_len > max_len:
            stats['truncated'] += 1
            if truncate_mode == 'end': seq = seq[:max_len]
            elif truncate_mode == 'start': seq = seq[-max_len:]
            elif truncate_mode == 'middle':
                s = (orig_len - max_len) // 2
                seq = seq[s: s + max_len]
        records.append((info, seq, orig_len))
        stats['output'] += 1

    Path(output_fasta).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fasta, 'w') as fh:
        for info, seq, orig_len in records:
            tid, gene, sym, bt = info['transcript_id'], info['gene_id'] or info['transcript_id'], info['gene_symbol'] or '', info['transcript_biotype'] or 'unknown'
            fh.write(f">{tid} gene:{gene} symbol:{sym} biotype:{bt} orig_len:{orig_len} used_len:{len(seq)}\n")
            for i in range(0, len(seq), 60): fh.write(seq[i:i+60] + '\n')

    stats_dict = {'total_input': len(raw_seqs), 'filtered_biotype': stats['filtered_biotype'], 'filtered_short': stats['filtered_short'], 'truncated': stats['truncated'], 'total_output': stats['output']}
    logger.info(f"RNA Preprocessing Complete. Input: {len(raw_seqs)}, Output: {stats['output']}")
    return stats['output'], stats_dict

def preprocess_protein_fasta(
    input_fasta:  str,
    output_fasta: str,
    max_len:      int  = 1024,
    min_len:      int  = 10,
    save_stats:   bool = True,
) -> Tuple[int, Dict]:
    logger.info("=" * 55)
    logger.info("Protein FASTA Preprocessing")
    logger.info(f"  Input:    {input_fasta}")
    logger.info(f"  Output:   {output_fasta}")
    logger.info("=" * 55)

    VALID_AA = set('ACDEFGHIKLMNPQRSTVWYBJOUXZ')
    raw_records, cur_info, cur_seq = [], None, []

    with open(input_fasta, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if cur_info is not None: raw_records.append((cur_info, ''.join(cur_seq)))
                cur_info, cur_seq = parse_protein_header(line), []
            else: cur_seq.append(line.upper())
    if cur_info is not None: raw_records.append((cur_info, ''.join(cur_seq)))

    stats, records = defaultdict(int), []
    for info, raw in raw_records:
        seq = ''.join(aa if aa in VALID_AA else 'X' for aa in raw).rstrip('*')
        orig_len = len(seq)
        if orig_len < min_len:
            stats['filtered_short'] += 1
            continue
        if orig_len > max_len:
            seq = seq[:max_len]
            stats['truncated'] += 1
        records.append((info, seq, orig_len))
        stats['output'] += 1

    Path(output_fasta).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fasta, 'w') as fh:
        for info, seq, orig_len in records:
            pid, gene, sym, bt = info['protein_id'], info['gene_id'] or info['protein_id'], info['gene_symbol'] or '', info['gene_biotype'] or 'unknown'
            fh.write(f">{pid} gene:{gene} symbol:{sym} biotype:{bt} orig_len:{orig_len} used_len:{len(seq)}\n")
            for i in range(0, len(seq), 60): fh.write(seq[i:i+60] + '\n')

    logger.info(f"Protein Preprocessing Complete. Input: {len(raw_records)}, Output: {stats['output']}")
    return stats['output'], {'total_input': len(raw_records), 'total_output': stats['output']}

def load_fasta(fasta_file: str) -> Dict[str, str]:
    seqs, cur_id, cur_seq = {}, None, []
    with open(fasta_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if cur_id is not None: seqs[cur_id] = ''.join(cur_seq)
                cur_id, cur_seq = line[1:].split()[0], []
            else: cur_seq.append(line)
    if cur_id is not None: seqs[cur_id] = ''.join(cur_seq)
    logger.info(f"Loaded {len(seqs)} sequences from {fasta_file}")
    return seqs

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple:
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {dev}")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    config = ckpt.get('model_config', ModelConfig())
    model = MHGAT(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(dev).eval()
    return model, config, dev

def _auto_preprocess_path(original: str, suffix: str) -> str:
    p = Path(original)
    return str(p.parent / (p.stem + suffix))

def diagnose_offset(predictions: List[Dict]) -> Dict:
    logger.info("=" * 55)
    logger.info("Offset Diagnosis")
    logger.info("=" * 55)
    peaks, boundary, early, late, prob_arrays = [], [], [], [], []
    for pred in predictions:
        if 'error' in pred: continue
        probs = np.array(pred.get('binding_probs', []))
        if len(probs) < 60: continue
        prob_arrays.append(probs)
        peaks.append(int(np.argmax(probs)) + 1)
        boundary.append(probs[47:55].tolist())
        early.append(probs[:20].mean())
        if len(probs) > 100: late.append(probs[100:].mean())
    if not peaks: return {}
    top1_pos, top1_cnt = Counter(peaks).most_common(1)[0]
    conc = top1_cnt / len(peaks)
    offset_type = "conv_boundary_effect" if conc > 0.3 and top1_pos in range(48, 56) else "none"
    logger.info(f"Offset Type: {offset_type}, Concentration: {conc:.1%}, Top Peak: {top1_pos}")
    return {'offset_type': offset_type, 'top_peak_pos': top1_pos, 'concentration': conc, 'num_samples': len(peaks)}

def correct_cls_offset(probs: np.ndarray, rna_len: int) -> np.ndarray:
    pl = len(probs)
    if pl == rna_len: return probs.copy()
    if pl == rna_len + 1: return probs[1:].copy()
    if pl == rna_len + 2: return probs[1:-1].copy()
    return probs[:rna_len].copy()

def correct_boundary_effect(probs: np.ndarray, region_size: int = 10, stride: int = 5, smooth_window: int = 15, smooth_poly: int = 3) -> np.ndarray:
    if len(probs) < smooth_window: return probs.copy()
    corrected = probs.copy()
    win = min(smooth_window, len(probs) - 1)
    if win % 2 == 0: win -= 1
    if win >= smooth_poly + 1:
        try: corrected = savgol_filter(corrected, win, smooth_poly)
        except: pass
    br_s, br_e = max(0, stride * region_size - stride * 2), min(len(corrected), stride * region_size + stride * 2)
    if br_e > br_s:
        lbl = corrected[max(0, br_s - 20):br_s].mean() if br_s > 0 else 0.0
        rbl = corrected[br_e:min(len(corrected), br_e + 20)].mean() if br_e < len(corrected) else 0.0
        excess = corrected[br_s:br_e] - ((lbl + rbl) / 2)
        corrected[br_s:br_e] -= np.maximum(0, excess * 0.6)
    return np.clip(corrected, 0, 1)

def apply_temperature_scaling(probs: np.ndarray, temp: float = 1.5) -> np.ndarray:
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)) / temp
    return 1.0 / (1.0 + np.exp(-logits))

def normalize_by_local_background(probs: np.ndarray, window: int = 51, pct: float = 20) -> np.ndarray:
    if len(probs) < window: return probs.copy()
    hw, bg = window // 2, np.zeros_like(probs)
    for i in range(len(probs)):
        bg[i] = np.percentile(probs[max(0, i-hw):min(len(probs), i+hw+1)], pct)
    norm = (probs / np.maximum(bg, 1e-7))
    return 1.0 / (1.0 + np.exp(-(norm - norm.mean())))

def full_correction_pipeline(probs: np.ndarray, rna_len: int, method: str = 'combined', region_size: int = 10, stride: int = 5, temperature: float = 1.2, smooth_window: int = 11, threshold: float = 0.5) -> Tuple[np.ndarray, List[int]]:
    c = correct_cls_offset(probs, rna_len)
    if len(c) > rna_len: c = c[:rna_len]
    elif len(c) < rna_len: c = np.pad(c, (0, rna_len - len(c)))
    if method in ('boundary', 'combined'): c = correct_boundary_effect(c, region_size, stride, smooth_window)
    if method in ('normalize', 'combined'): c = normalize_by_local_background(c)
    if method in ('temperature', 'combined'): c = apply_temperature_scaling(c, temperature)
    return c, [1 if p > threshold else 0 for p in c]

def extract_binding_sites(probs: List[float], labels: List[int], min_len: int = 1, merge_gap: int = 3) -> List[Dict]:
    merged = labels.copy()
    i = 0
    while i < len(merged):
        if merged[i] == 0:
            gap_e = i
            while gap_e < len(merged) and merged[gap_e] == 0: gap_e += 1
            if (gap_e - i) <= merge_gap and i > 0 and gap_e < len(merged) and merged[i-1] == 1 and merged[gap_e] == 1:
                for j in range(i, gap_e): merged[j] = 1
            i = gap_e
        else: i += 1
    sites, in_s, s_start = [], False, 0
    for i, lbl in enumerate(merged):
        if lbl == 1 and not in_s: in_s, s_start = True, i
        elif lbl == 0 and in_s:
            in_s = False
            if (i - s_start) >= min_len:
                sp = np.array(probs[s_start:i])
                sites.append({'start': s_start + 1, 'end': i, 'length': i - s_start, 'max_prob': float(sp.max()), 'mean_prob': float(sp.mean())})
    if in_s and (len(merged) - s_start) >= min_len:
        sp = np.array(probs[s_start:])
        sites.append({'start': s_start + 1, 'end': len(merged), 'length': len(merged) - s_start, 'max_prob': float(sp.max()), 'mean_prob': float(sp.mean())})
    return sites

def format_result(raw: Dict, sample_id: Optional[str] = None, threshold: float = 0.5) -> Dict:
    probs = raw['binding_probs']
    lbls = [1 if p > threshold else 0 for p in probs]
    sites = extract_binding_sites(probs, lbls)
    return {
        'sample_id': sample_id or raw.get('sample_id', ''), 'rna_length': raw['rna_length'], 'protein_length': raw['protein_length'],
        'statistics': {'num_binding_bases': sum(lbls), 'total_rna_length': len(lbls), 'binding_ratio': round(sum(lbls)/len(lbls), 4) if lbls else 0, 'mean_binding_prob': round(float(np.mean(probs)), 4), 'max_binding_prob': round(float(np.max(probs)), 4), 'num_binding_regions': len(sites)},
        'binding_sites': sites, 'binding_probs': [round(p, 4) for p in probs], 'binding_labels': lbls, 'region_scores': [round(s, 4) for s in raw.get('region_scores', [])]
    }

def format_corrected_result(orig: Dict, c_probs: np.ndarray, c_lbls: List[int], method: str) -> Dict:
    sites = extract_binding_sites(c_probs.tolist(), c_lbls)
    tl = len(c_lbls)
    return {
        'sample_id': orig.get('sample_id', ''), 'rna_id': orig.get('rna_id', ''), 'protein_id': orig.get('protein_id', ''), 'rna_length': orig.get('rna_length', tl), 'protein_length': orig.get('protein_length', 0), 'correction_method': method,
        'statistics': {'num_binding_bases': sum(c_lbls), 'total_rna_length': tl, 'binding_ratio': round(sum(c_lbls)/tl, 4) if tl else 0, 'mean_binding_prob': round(float(c_probs.mean()), 4), 'max_binding_prob': round(float(c_probs.max()), 4), 'num_binding_regions': len(sites)},
        'binding_sites': sites, 'binding_probs': [round(float(p), 4) for p in c_probs], 'binding_labels': c_lbls
    }

def print_result_summary(res: Dict):
    stats, sites = res['statistics'], res['binding_sites']
    print("\n" + "=" * 60 + f"\nSample ID: {res.get('sample_id') or 'N/A'}\nRNA Length: {res['rna_length']} nt\nProtein Length: {res['protein_length']} aa\n" + "-" * 60)
    print(f"Binding: {stats['num_binding_bases']}/{stats['total_rna_length']} ({stats['binding_ratio']:.2%})\nMean Prob: {stats['mean_binding_prob']:.4f}\nMax Prob: {stats['max_binding_prob']:.4f}\nRegions: {stats['num_binding_regions']}")
    if sites:
        for i, s in enumerate(sites[:10], 1): print(f"  Region {i}: {s['start']}-{s['end']} (Len={s['length']}, MaxProb={s['max_prob']:.4f})")
    print("=" * 60)

def visualize_binding(rna: str, lbls: List[int], prbs: List[float], width: int = 80):
    print(f"\nVisualization (■=Binding, □=None) Length: {len(rna)}")
    for s in range(0, len(rna), width):
        e = min(s + width, len(rna))
        chars = "".join('█▓▒░ '[min(4, int((1-p)*5))] for p in prbs[s:e])
        print(f"\nPos {s+1}-{e}:\nSeq: {rna[s:e]}\nBin: {''.join('■' if l else '□' for l in lbls[s:e])}\nInt: {chars}")

def _save_json(data: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh: json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {path}")

def predict_single(predictor: MHGATPredictor, rna: str, prot: str, sid: Optional[str] = None, threshold: float = 0.5, visualize: bool = True, output: Optional[str] = None) -> Dict:
    raw = predictor.predict(rna, prot)
    res = format_result(raw, sid, threshold)
    print_result_summary(res)
    if visualize and len(rna) <= 200: visualize_binding(rna[:res['rna_length']], res['binding_labels'], res['binding_probs'])
    if output: _save_json(res, output)
    return res

def predict_from_json(predictor: MHGATPredictor, input_j: str, output_j: str, threshold: float = 0.5) -> List[Dict]:
    with open(input_j, 'r') as fh: data = json.load(fh)
    samples = data if isinstance(data, list) else data.get('samples', [])
    results = []
    for s in tqdm(samples, desc="Predicting"):
        try:
            raw = predictor.predict(s['rna_sequence'], s['protein_sequence'])
            results.append(format_result(raw, s.get('sample_id'), threshold))
        except Exception as e:
            results.append({'sample_id': s.get('sample_id', '?'), 'error': str(e)})
    _save_json({'total_samples': len(results), 'threshold': threshold, 'predictions': results}, output_j)
    return results

def predict_from_fasta(predictor, rna_f, prot_f, out_j, pair_mode='one_to_one', threshold=0.5, auto_rna=False, b_filter=None, max_r=510, min_r=20, trunc='end', out_rna=None, auto_prot=False, max_p=1024, min_p=10, out_prot=None) -> List[Dict]:
    a_rna, a_prot = rna_f, prot_f
    if auto_rna:
        a_rna = out_rna or _auto_preprocess_path(rna_f, '_rna_preprocessed.fasta')
        preprocess_rna_fasta(rna_f, a_rna, b_filter, max_r, min_r, trunc)
    if auto_prot:
        a_prot = out_prot or _auto_preprocess_path(prot_f, '_protein_preprocessed.fasta')
        preprocess_protein_fasta(prot_f, a_prot, max_p, min_p)
    rs, ps = load_fasta(a_rna), load_fasta(a_prot)
    r_ids, p_ids = list(rs.keys()), list(ps.keys())
    if pair_mode == 'one_to_one': pairs = list(zip(r_ids, p_ids))
    else: pairs = [(r, p) for r in r_ids for p in p_ids]
    results = []
    for rid, pid in tqdm(pairs, desc="Predicting"):
        try:
            res = format_result(predictor.predict(rs[rid], ps[pid]), f"{rid}__vs__{pid}", threshold)
            res.update({'rna_id': rid, 'protein_id': pid})
            results.append(res)
        except Exception as e: results.append({'sample_id': f"{rid}__vs__{pid}", 'error': str(e)})
    _save_json({'total_samples': len(results), 'predictions': results}, out_j)
    return results

def fix_json_predictions(input_j, output_j, method='combined', threshold=0.5, rs=10, stride=5, temp=1.2) -> List[Dict]:
    with open(input_j, 'r') as fh: data = json.load(fh)
    preds = data['predictions'] if 'predictions' in data else (data if isinstance(data, list) else [data])
    diag = diagnose_offset([p for p in preds if 'error' not in p])
    results = []
    for pred in tqdm(preds, desc="Fixing"):
        if 'error' in pred: results.append(pred); continue
        prbs = np.array(pred.get('binding_probs', []))
        if not len(prbs): results.append(pred); continue
        c_prb, c_lbl = full_correction_pipeline(prbs, pred.get('rna_length', len(prbs)), method, rs, stride, temp, threshold=threshold)
        results.append(format_corrected_result(pred, c_prb, c_lbl, method))
    _save_json({'correction_info': {'method': method, 'threshold': threshold, 'diagnosis': diag}, 'predictions': results}, output_j)
    return results

class FixedPredictor:
    def __init__(self, ckpt, device='cuda', method='combined', threshold=0.5, rs=10, stride=5, temp=1.2):
        self.method, self.threshold, self.rs, self.stride, self.temp = method, threshold, rs, stride, temp
        self.model, self.config, self.device = load_model(ckpt, device)
        self.tok = RNATokenizer(self.config.max_rna_len)
        _, self.alpha = esm.pretrained.load_model_and_alphabet(self.config.esm_model_name)
        self.bc = self.alpha.get_batch_converter()

    @torch.no_grad()
    def predict(self, rna, prot, sid=''):
        rt, rm = self.tok.encode(rna)
        ptunc = prot[:self.config.max_protein_len]
        _, _, pt = self.bc([("protein", ptunc)])
        pt, pm = pt.squeeze(0), (pt.squeeze(0) != self.alpha.padding_idx).long()
        out = self.model({'rna_tokens': rt.unsqueeze(0).to(self.device), 'rna_mask': rm.unsqueeze(0).to(self.device), 'protein_tokens': pt.unsqueeze(0).to(self.device), 'protein_mask': pm.unsqueeze(0).to(self.device)})
        raw_p = F.softmax(out['base_logits'][0], dim=-1)[:, 1].cpu().numpy()[1:min(len(rna), self.config.max_rna_len-2)+1]
        c_prb, c_lbl = full_correction_pipeline(raw_p, len(raw_p), self.method, self.rs, self.stride, self.temp, threshold=self.threshold)
        res = format_corrected_result({'sample_id': sid, 'rna_length': len(raw_p), 'protein_length': len(ptunc)}, c_prb, c_lbl, self.method)
        res['raw_peak_position'] = int(np.argmax(raw_p)) + 1
        return res

    def predict_fasta(self, rna_f, prot_f, out_j, pair_mode='one_to_all'):
        rs, ps = load_fasta(rna_f), load_fasta(prot_f)
        pairs = list(zip(rs.keys(), ps.keys())) if pair_mode == 'one_to_one' else [(r, p) for r in rs for p in ps]
        results = []
        for rid, pid in tqdm(pairs, desc="Predict+Fix"):
            try: results.append(self.predict(rs[rid], ps[pid], f"{rid}__vs__{pid}"))
            except Exception as e: results.append({'sample_id': f"{rid}__vs__{pid}", 'error': str(e)})
        _save_json({'predictions': results}, out_j)
        return results

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['predict_single', 'predict_json', 'predict_fasta', 'preprocess_rna', 'preprocess_protein', 'diagnose', 'fix_json', 'repredict'], required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--device', default='cuda')
    p.add_argument('--rna_seq', default=None); p.add_argument('--protein_seq', default=None)
    p.add_argument('--input_json', default=None); p.add_argument('--rna_fasta', default=None); p.add_argument('--protein_fasta', default=None); p.add_argument('--input', default=None)
    p.add_argument('--output_json', default='predictions.json'); p.add_argument('--output', default='predictions_fixed.json')
    p.add_argument('--preprocessed_rna_fasta', default=None); p.add_argument('--preprocessed_protein_fasta', default=None)
    p.add_argument('--threshold', type=float, default=0.5); p.add_argument('--pair_mode', default='one_to_one', choices=['one_to_one', 'one_to_all'])
    p.add_argument('--auto_preprocess', action='store_true'); p.add_argument('--biotype_filter', nargs='*', default=None)
    p.add_argument('--max_rna_len', type=int, default=510); p.add_argument('--auto_preprocess_protein', action='store_true')
    p.add_argument('--method', default='combined', choices=['boundary', 'normalize', 'temperature', 'combined', 'raw'])
    p.add_argument('--temp', type=float, default=1.2, dest='temperature')
    args = p.parse_args()

    if args.mode == 'preprocess_rna': preprocess_rna_fasta(args.rna_fasta, args.preprocessed_rna_fasta or _auto_preprocess_path(args.rna_fasta, '_rna_preprocessed.fasta'), args.biotype_filter, args.max_rna_len)
    elif args.mode == 'preprocess_protein': preprocess_protein_fasta(args.protein_fasta, args.preprocessed_protein_fasta or _auto_preprocess_path(args.protein_fasta, '_protein_preprocessed.fasta'))
    elif args.mode == 'diagnose':
        with open(args.input, 'r') as fh: data = json.load(fh)
        print(json.dumps(diagnose_offset(data.get('predictions', data if isinstance(data, list) else [data])), indent=2))
    elif args.mode == 'fix_json': fix_json_predictions(args.input, args.output, args.method, args.threshold, temperature=args.temperature)
    elif args.mode == 'repredict': FixedPredictor(args.checkpoint, args.device, args.method, args.threshold, temp=args.temperature).predict_fasta(args.rna_fasta, args.protein_fasta, args.output_json, args.pair_mode)
    else:
        m, c, d = load_model(args.checkpoint, args.device)
        pr = MHGATPredictor(m, c, str(d))
        if args.mode == 'predict_single': predict_single(pr, args.rna_seq, args.protein_seq, threshold=args.threshold, output=args.output_json)
        elif args.mode == 'predict_json': predict_from_json(pr, args.input_json, args.output_json, args.threshold)
        elif args.mode == 'predict_fasta': predict_from_fasta(pr, args.rna_fasta, args.protein_fasta, args.output_json, args.pair_mode, args.threshold, args.auto_preprocess, args.biotype_filter, args.max_rna_len, auto_prot=args.auto_preprocess_protein)

if __name__ == '__main__': main()
