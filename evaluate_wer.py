#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Marvin4000 - Real-time Audio Transcription & Translation
# © 2025 XOREngine (WallyByte)
# https://github.com/XOREngine/marvin4000

import jiwer
import re
from typing import List, Dict


librispeech_references = [
    "...",
    "...",
    "...",
]

marvin_hypothesis = [
    "...",
    "...",
    "...",
]
# === JIWER OFFICIAL ===

def calculate_wer_jiwer(hypothesis: List[str], references: List[str]) -> Dict:
    print("🔬 ANÁLISIS WER CON JIWER")
    print("=" * 50)
    
    print("\n📊 ANÁLISIS POR SEGMENTOS:")
    wer_scores = []
    
    for i, (hyp, ref) in enumerate(zip(hypothesis, references)):
        # Sin transformaciones - jiwer básico
        wer_score = jiwer.wer(ref, hyp)
        wer_scores.append(wer_score)
        print(f"Segmento {i+1}: WER = {wer_score*100:.1f}%")
        
        if i < 3:
            print(f"  Ref: '{ref}'")
            print(f"  Hip: '{hyp}'")
    
    # Corpus level - concatenar todo
    ref_concat = " ".join(references)
    hyp_concat = " ".join(hypothesis)
    
    corpus_wer = jiwer.wer(ref_concat, hyp_concat)
    measures = jiwer.compute_measures(ref_concat, hyp_concat)
    
    print(f"\n🏆 RESULTADOS FINALES JIWER:")
    print(f"Corpus WER: {corpus_wer*100:.1f}%")
    print(f"Substitutions: {measures['substitutions']}")
    print(f"Deletions: {measures['deletions']}")
    print(f"Insertions: {measures['insertions']}")
    print(f"Hits: {measures['hits']}")
    print(f"Total palabras ref: {measures['substitutions'] + measures['deletions'] + measures['hits']}")
    
    return {
        'corpus_wer': corpus_wer,
        'segment_wers': wer_scores,
        'measures': measures
    }

# === MANUAL IMPLEMENTATION ===

def tokenize(text: str) -> List[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

def levenshtein_wer(hyp_tokens: List[str], ref_tokens: List[str], debug: bool = False) -> Dict:
    m, n = len(ref_tokens), len(hyp_tokens)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    i, j = m, n
    substitutions = deletions = insertions = 0
    operations = []
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            operations.append(f"MATCH: '{ref_tokens[i-1]}'")
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            operations.append(f"SUB: '{ref_tokens[i-1]}' -> '{hyp_tokens[j-1]}'")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            operations.append(f"DEL: '{ref_tokens[i-1]}'")
            i -= 1
        else:
            insertions += 1
            operations.append(f"INS: '{hyp_tokens[j-1]}'")
            j -= 1
    
    total_errors = substitutions + deletions + insertions
    wer = total_errors / m if m > 0 else 0
    
    if debug:
        print(f"\nTokens Ref: {ref_tokens}")
        print(f"Tokens Hip: {hyp_tokens}")
        print(f"SUB: {substitutions}, DEL: {deletions}, INS: {insertions}")
        print(f"WER: {total_errors}/{m} = {wer:.4f} ({wer*100:.1f}%)")
    
    return {
        'wer': wer,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'total_errors': total_errors,
        'ref_length': m,
        'hyp_length': n,
        'operations': operations[::-1]
    }

def calculate_corpus_wer_manual(hypothesis: List[str], references: List[str]) -> Dict:
    print("🔬 ANÁLISIS WER MANUAL")
    print("=" * 50)
    
    total_sub = total_del = total_ins = 0
    total_ref_tokens = total_hyp_tokens = 0
    
    print("\n📊 ANÁLISIS POR SEGMENTOS:")
    
    for i, (hyp, ref) in enumerate(zip(hypothesis, references)):
        hyp_tokens = tokenize(hyp)
        ref_tokens = tokenize(ref)
        
        result = levenshtein_wer(hyp_tokens, ref_tokens, debug=(i < 3))
        
        total_sub += result['substitutions']
        total_del += result['deletions']
        total_ins += result['insertions']
        total_ref_tokens += result['ref_length']
        total_hyp_tokens += result['hyp_length']
        
        print(f"Segmento {i+1}: WER={result['wer']*100:.1f}% (S:{result['substitutions']}, D:{result['deletions']}, I:{result['insertions']})")
    
    total_errors = total_sub + total_del + total_ins
    corpus_wer = total_errors / total_ref_tokens if total_ref_tokens > 0 else 0
    
    print(f"\n🏆 RESULTADOS FINALES MANUAL:")
    print(f"Total tokens Ref: {total_ref_tokens}")
    print(f"Total tokens Hip: {total_hyp_tokens}")
    print(f"Total substitutions: {total_sub}")
    print(f"Total deletions: {total_del}")
    print(f"Total insertions: {total_ins}")
    print(f"Total errors: {total_errors}")
    print(f"Corpus WER: {total_errors}/{total_ref_tokens} = {corpus_wer:.4f} ({corpus_wer*100:.1f}%)")
    
    return {
        'corpus_wer': corpus_wer,
        'total_substitutions': total_sub,
        'total_deletions': total_del,
        'total_insertions': total_ins,
        'total_errors': total_errors,
        'total_ref_tokens': total_ref_tokens,
        'total_hyp_tokens': total_hyp_tokens
    }

# === EXECUTION ===

if __name__ == "__main__":
    
    try:
        result_jiwer = calculate_wer_jiwer(marvin_hypothesis, librispeech_references)
    except ImportError:
        print("⚠️  jiwer no instalado. Usa: pip install jiwer")
    
    print("\n" + "="*70 + "\n")
    
    result_manual = calculate_corpus_wer_manual(marvin_hypothesis, librispeech_references)