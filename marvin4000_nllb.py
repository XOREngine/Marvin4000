#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Marvin4000 - Real-time Audio Transcription & Translation
# © 2025 XOREngine (WallyByte)
# https://github.com/XOREngine/marvin4000

from __future__ import annotations
import argparse
import queue
import signal
import subprocess as sp
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import re

import math
import numpy as np
from scipy.signal import resample_poly
import torch
import webrtcvad

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from transformers import AutoProcessor, SeamlessM4Tv2Model

CACHE_DIR          = Path("./models_cache")
ASR_MODEL          = "openai/whisper-large-v3-turbo" 
NMT_MODEL          = "facebook/nllb-200-3.3B"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF           = DEVICE == "cuda"

CHUNK_SEC          = 1.5
NATIVE_SR          = 48000
TARGET_SR          = 16000
CHANNELS           = 2
QUEUE_MAX          = 32
TIMEOUT_SEC        = 12.0
SILENCE_SEC        = 0.8
MIN_SEGMENT_SEC    = 0.5
MIN_WORDS_FOR_SENT = 4
MIN_PARTIAL_WORDS  = 5

# Audio processing thresholds
AUDIO_RMS_THRESHOLD = 0.0025  # Minimum RMS level to process audio

# VAD silence detection parameters
VAD_FRAME_MS       = 30    # VAD frame duration in milliseconds
VAD_SILENCE_DURATION_SEC = 1.2  # Required silence duration for detection
MIN_CUT_DURATION_SEC = 2.5      # Minimum audio duration before cut

# Translation reuse configuration
REUSE_THRESHOLD    = 0.95 
MAX_CACHE_SIZE     = 50   

# Langs
# Whisper: en, es, de, fr, it, pt, ru, ja, ko, zh, ar, hi, ...
# NLLB: eng_Latn, spa_Latn, deu_Latn, fra_Latn, ita_Latn, por_Latn, rus_Cyrl, jpn_Jpan, kor_Hang, zho_Hans, arb_Arab, hin_Deva, ...
ASR_SOURCE_LANG = "en"          
NMT_SOURCE_LANG = "eng_Latn"    
NMT_TARGET_LANG = "spa_Latn" 

GREEN              = "\033[32m"
RESET              = "\033[0m"

# Lock to serialize GPU usage
gpu_lock = threading.Lock()

def log(msg: str, color: str = "") -> None:
    ts = datetime.now().isoformat(timespec="milliseconds")
    print(f"[{ts}] {color}{msg}{RESET}", flush=True)

def to_mono_16k(x: np.ndarray, sr_orig: int) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr_orig == TARGET_SR:
        return x.astype("float32")
    g = math.gcd(sr_orig, TARGET_SR)
    return resample_poly(x, TARGET_SR // g, sr_orig // g).astype("float32")

def text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between texts (0.0 to 1.0)"""
    if not text1 or not text2:
        return 0.0
    
    # Simplified calculation for performance
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    common = words1.intersection(words2)
    return len(common) / max(len(words1), len(words2))

class AudioProducer(threading.Thread):
    def __init__(self, q: queue.Queue[np.ndarray], audio_device: str):
        super().__init__(daemon=True)
        self.q = q
        self.audio_device = audio_device
        self._stop = threading.Event()
    def stop(self):
        self._stop.set()
    def run(self):
        frames = int(CHUNK_SEC * NATIVE_SR)
        chunk_bytes = frames * CHANNELS * 4
        cmd = [
            "parec", f"--device={self.audio_device}", "--format=float32le",
            f"--rate={NATIVE_SR}", f"--channels={CHANNELS}"
        ]
        log(f"Producer start: {' '.join(cmd)}")
        proc = sp.Popen(cmd, stdout=sp.PIPE, bufsize=chunk_bytes*4)
        try:
            while not self._stop.is_set():
                buf = proc.stdout.read(chunk_bytes)
                if not buf:
                    break
                audio = np.frombuffer(buf, dtype="<f4").reshape(-1, CHANNELS)
                mono  = to_mono_16k(audio, NATIVE_SR)
                try:
                    self.q.put(mono, timeout=1)
                except queue.Full:
                    log("Producer queue full, drop chunk")
        finally:
            proc.terminate()
            log("Producer stopped")


class Transcriber(threading.Thread):
    def __init__(self, q: queue.Queue[np.ndarray]):
        super().__init__(daemon=True)
        self.q            = q
        self.audio_buffer : List[np.ndarray] = []
        self.last_flush   = time.time()
        self.vad          = webrtcvad.Vad(1)

        self.last_partial_text = ""
        self.last_partial_translation = ""
        self.translation_cache = {} 

        # WHISPER ASR
        log(f"Loading ASR model: {ASR_MODEL}")
        self.asr_processor = WhisperProcessor.from_pretrained(ASR_MODEL, cache_dir=CACHE_DIR)

        # === 16bit (default) ===
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(DEVICE)
        log("ASR model loaded: 16bit")

        # === 8bit quantization (uncomment to use) ===
        # quant_asr = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        # self.asr_model = WhisperForConditionalGeneration.from_pretrained(
        #     ASR_MODEL,
        #     cache_dir=CACHE_DIR,
        #     quantization_config=quant_asr,
        #     device_map="auto"
        # )
        # log("ASR model loaded: 8bit quantized")


        # NLLB NMT
        log(f"Loading NMT model: {NMT_MODEL}")
        self.nmt_tokenizer = AutoTokenizer.from_pretrained(NMT_MODEL, cache_dir=CACHE_DIR)
        
        self.nmt_tokenizer.src_lang = NMT_SOURCE_LANG
        self.nmt_tokenizer.pad_token = self.nmt_tokenizer.unk_token
        self.nmt_tokenizer.pad_token_id = self.nmt_tokenizer.unk_token_id
        
        # === 16bit (default) ===
        self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
            NMT_MODEL,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16, 
            load_in_8bit=False,
            device_map="auto"
        ).to(DEVICE)
        log("NMT model loaded: 16bit")
        
        # === 8bit quantization (uncomment to use) ===
        # quant_nmt = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        # self.nmt_model = AutoModelForSeq2SeqLM.from_pretrained(
        #     NMT_MODEL,
        #     cache_dir=CACHE_DIR,
        #     quantization_config=quant_nmt,
        #     device_map="auto"
        # )
        # log("NMT model loaded: 8bit quantized")


        self.nmt_model.resize_token_embeddings(len(self.nmt_tokenizer))
        self.nmt_model.eval()  

        # Fixed language config
        self.src_lang = NMT_SOURCE_LANG
        self.tgt_lang = NMT_TARGET_LANG


    def _has_sentence_end(self, txt: str) -> bool:
        if not txt:
            return False
        ends         = txt.rstrip().endswith((".", "?", "!"))
        starts_upper = txt[0].isupper()
        return ends and starts_upper and len(txt.split()) >= MIN_WORDS_FOR_SENT

    
    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        # # Calculate RMS - objective volume measure
        # rms = np.sqrt(np.mean(np.square(audio)))
        # if rms < AUDIO_RMS_THRESHOLD:
        #     log(f"Audio discarded due to low level: {rms:.6f}")
        #     return None


        dur = len(audio) / TARGET_SR
        # Use MIN_SEGMENT_SEC (now 0.5s) to filter
        if dur < MIN_SEGMENT_SEC:
            return None

        norm = audio / (np.max(np.abs(audio)) + 1e-8)
        feats = self.asr_processor(norm, sampling_rate=TARGET_SR,
                               return_tensors="pt", padding=True).input_features.to(DEVICE)
        if USE_HALF:
            feats = feats.half()
        attn   = torch.ones(feats.shape[:-1], device=DEVICE, dtype=torch.long)
        forced = self.asr_processor.get_decoder_prompt_ids(language=ASR_SOURCE_LANG, task="transcribe")
        with gpu_lock, torch.inference_mode():
            gen = self.asr_model.generate(
                feats,
                attention_mask=attn,
                forced_decoder_ids=forced,
                max_length=448,
                num_beams=3,            
                early_stopping=True,
                temperature=0.0,        
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                return_timestamps=False,
                use_cache=True,
            )

        return self.asr_processor.batch_decode(gen, skip_special_tokens=True)[0].strip()



    def _translate(self, txt: str, kind: str) -> Optional[str]:
        if not txt or len(txt) < 2:
            return None

        if txt in self.translation_cache:
            cached_translation = self.translation_cache[txt]
            if kind == "F":
                log(f"{GREEN}[FINAL-TR] {cached_translation}{RESET}")
            return cached_translation

        for previous_transcription, cached_translation in self.translation_cache.items():
            if text_similarity(txt, previous_transcription) > REUSE_THRESHOLD:
                if kind == "F":
                    log(f"{GREEN}[FINAL-TR] {cached_translation}{RESET}")
                return cached_translation

        try:
            src_texts = [txt.strip()]
            
            inputs = self.nmt_tokenizer(
                src_texts, 
                return_tensors="pt", 
                padding=True,              
                max_length=64,             
                truncation=True
            ).to(DEVICE)
            
           
            with gpu_lock, torch.inference_mode():
                forced_bos_token_id = self.nmt_tokenizer.convert_tokens_to_ids(self.tgt_lang)
              
                generated_tokens = self.nmt_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=120,              
                    min_length=8,                
                    num_beams=4,                 
                    do_sample=False,             
                    repetition_penalty=1.1,      
                    no_repeat_ngram_size=2,      
                    early_stopping=True,         
                )
                
            translation = self.nmt_tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]

            translation = re.sub(r'\s+([.,!?])', r'\1', translation)
            translation = re.sub(r'\.{2,}', '.', translation)
            translation = translation.strip()

            prefix = "[PARTIAL-TR]" if kind == "P" else f"{GREEN}[FINAL-TR]"
            log(f"{prefix} {translation}{RESET}" if kind == "F" else f"{prefix} {translation}")

            self.translation_cache[txt] = translation
            if len(self.translation_cache) > MAX_CACHE_SIZE:
                self.translation_cache.pop(next(iter(self.translation_cache)))

            return translation

        except Exception as e:
            log(f"Error in NLLB translation: {str(e)}")
            return None


    def _find_silence_split(self, audio: np.ndarray) -> Optional[int]:
        audio_i16 = (audio * 32767).astype(np.int16)
        frame_sz = int(TARGET_SR * VAD_FRAME_MS / 1000)
        req = int(VAD_SILENCE_DURATION_SEC * 1000 / VAD_FRAME_MS)
        
        cnt = 0
        last_speech_idx = 0
        
        for i in range(0, len(audio_i16) - frame_sz, frame_sz):
            frame = audio_i16[i:i+frame_sz]
            if len(frame) < frame_sz:
                continue
                
            frame_bytes = frame.tobytes()
            
            if not self.vad.is_speech(frame_bytes, TARGET_SR):
                cnt += 1
            else:
                cnt = 0
                last_speech_idx = i
                
            if cnt >= req:
                silence_duration = (i - last_speech_idx) / TARGET_SR
                if silence_duration > SILENCE_SEC:
                    cut = last_speech_idx + frame_sz * 2
                    if cut / TARGET_SR >= MIN_CUT_DURATION_SEC:
                        return cut
                        
        return None

    def _post_process_asr(self, txt: str) -> str:
        # txt = txt.replace("?", "'").replace("?","\"").replace("?","\"")
        # txt = re.sub(r"[^\x20-\x7EáéíóúüÁÉÍÓÚÜñÑ¿¡]", "", txt)   # remove weird chars
        return txt
    
    def run(self):
        log("Transcriber ready")
        while True:
            try:
                chunk = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Calculate RMS - objective volume measure
            chunk_rms = np.sqrt(np.mean(np.square(chunk)))
            if chunk_rms < AUDIO_RMS_THRESHOLD:
                log(f"Audio discarded due to low level: {chunk_rms:.6f}")
                
                # Process accumulated audio as final segment
                if self.audio_buffer:
                    audio_cat = np.concatenate(self.audio_buffer)
                    if len(audio_cat) / TARGET_SR >= MIN_SEGMENT_SEC:
                        txt_raw = self._transcribe(audio_cat) or ""
                        txt = self._post_process_asr(txt_raw)
                        if txt.strip():
                            log(f"[FINAL-SILENCE] {txt.upper()}", GREEN)
                            self._translate(txt, "F")
                    
                    self.audio_buffer.clear()
                    self.translation_cache.clear()  
                
                self.last_flush = time.time()  # Reset timeout
                continue   # Skip current chunk - low volume
            
            self.audio_buffer.append(chunk)

            audio_cat = np.concatenate(self.audio_buffer)
            txt_raw   = self._transcribe(audio_cat) or ""
            txt       = self._post_process_asr(txt_raw)

            if len(txt.split()) >= MIN_PARTIAL_WORDS:
                partial_tr = self._translate(txt, "P")
                if partial_tr:
                    self.last_partial_text = txt
                    self.last_partial_translation = partial_tr

            now       = time.time()
            split_idx = self._find_silence_split(audio_cat)
            end_sent  = self._has_sentence_end(txt)
            timed_out = (now - self.last_flush) >= TIMEOUT_SEC

            if (split_idx is not None and end_sent) or timed_out:
                cut = split_idx if split_idx is not None else len(audio_cat)
                out, rem = audio_cat[:cut], audio_cat[cut:]

                # Use MIN_SEGMENT_SEC to decide
                if len(out) / TARGET_SR >= MIN_SEGMENT_SEC:
                    # Process final block
                    log(f"[FINAL] {txt.upper()}", GREEN)
                    sim = text_similarity(txt, self.last_partial_text)
                    if sim > REUSE_THRESHOLD and self.last_partial_translation:
                        log(f"[FINAL-TR] {self.last_partial_translation}", GREEN)
                    else:
                        self._translate(txt, "F")
                    
                    # Reset buffers, preserve remainder, clear translation cache
                    self.audio_buffer = [rem] if rem.size else []
                    self.translation_cache.clear()  
                    self.last_flush = now
                else:
                    # Fragment too short: DON'T discard it, leave buffers intact
                    log(f"Fragment of {len(out)/TARGET_SR:.2f}s (< {MIN_SEGMENT_SEC}s), waiting for more before flush")


# Main
def main():
    parser = argparse.ArgumentParser(description="Marvin4000 - Real-time speech to text transcription and translation")
    parser.add_argument("--audio-device", required=True, help="Audio device name (e.g., 'alsa_output.device.monitor')")
    parser.add_argument("--asr-lang", default="en", help="ASR source language (Whisper. default: en)")
    parser.add_argument("--nmt-source", default="eng_Latn", help="NMT source language (NLLB. default: eng_Latn)")
    parser.add_argument("--nmt-target", default="spa_Latn", help="NMT target language (NLLB. default: spa_Latn)")
    args = parser.parse_args()

    # Override global variables
    global ASR_SOURCE_LANG, NMT_SOURCE_LANG, NMT_TARGET_LANG
    ASR_SOURCE_LANG = args.asr_lang
    NMT_SOURCE_LANG = args.nmt_source
    NMT_TARGET_LANG = args.nmt_target

    q_audio = queue.Queue(maxsize=QUEUE_MAX)
    
    # Initialize transcriber blocking until models load
    transcriber = Transcriber(q_audio)
    log("Models loaded, starting audio capture")
    prod = AudioProducer(q_audio, args.audio_device)

    transcriber.start()
    prod.start()

    def stop_all(*_):
        prod.stop(); prod.join(timeout=1)
        log("Shutting down")
        raise SystemExit

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)
    signal.pause()

if __name__ == "__main__":
    main()