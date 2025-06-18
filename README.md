<!-- Marvin4000 - Real-time Audio Transcription & Translation -->
<!-- © 2025 XOREngine (WallyByte) -->
<!-- https://github.com/XOREngine/marvin4000 -->

# Marvin4000

> Real-time audio transcription and translation using Whisper and multilingual models (SeamlessM4T / NLLB‑200)

[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/GPU-Accelerated-green)](https://developer.nvidia.com/cuda-toolkit)

**🌐 Languages:** [English](README.md) | [Español](README.es.md)

<br>

**Marvin4000** captures, transcribes, and translates system audio in real-time using local hardware.

<br>

> ⚠️ **IMPORTANT:**
>
> * If you're on **Windows**, audio capture must be manually implemented using an alternative to `parec` that provides system audio data in `float32` format.

<br>

## 📊 Proven Performance

| GPU & Models Used                                                | Latency (s) | WER       | BLEU-1/4/Corpus | VRAM        |
| ---------------------------------------------------------------- | ----------- | --------- | --------------- | ----------- |
| **RTX 4060 Ti 16GB<br>whisper-large-v3 (8b), seamless-m4t-v2-large** | **2-3**     | **5 %** | **74/39/52**    | **12.2 GB** |

#### Test Corpus

* **Audio**: 25 random audiobook fragments from [LibriSpeech](https://www.openslr.org/12) (avg: 5 min/fragment)
* **Reference Transcription**: Official LibriSpeech transcriptions
* **Reference Translation**: Generated with Claude & GPT and manually reviewed
* **Total Evaluated**: ~120 minutes of audio

#### Metrics Calculation

* **WER**: Calculated with [jiwer](https://github.com/jitsi/jiwer), normalized for punctuation
* **BLEU**: Corpus-level implementation with lowercase tokenization, n-gram clipping and brevity penalty
* **BLEU-1/4/Corpus**: 1-gram / 4-gram precision / full corpus score
* **Latency**: Measured under real conditions with RTX 4060 Ti 16GB and RTX 2060 6GB

#### Limitations

While reference translations are high quality, we acknowledge they are not equivalent to professional human translations. However, they provide a consistent standard for comparing system performance, following methodologies similar to those employed in evaluations like [FLEURS](https://arxiv.org/abs/2205.12446) and [CoVoST 2](https://arxiv.org/abs/2007.10310).

<br>

## 🚀 Installation and Usage

### Requirements

```bash
sudo apt install python3-pip pulseaudio-utils ffmpeg
git clone https://github.com/XOREngine/marvin4000.git
cd marvin4000
pip install -r requirements.txt
```

### Basic Execution

```bash
# 1. Play some audio content on your system
vlc example_video.mp4
# ffmpeg.ffplay -nodisp -autoexit -ss 1 example.mp3
# or play audio from browser, etc.

# 2. Detect valid audio devices
python detect_audio_devices.py
# Example output:
# $ python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

# 3. Start transcription/translation with appropriate monitor device
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
```

### Language Configuration

Marvin4000 uses SeamlessM4T and NLLB‑200 for transcription and translation between 100+ languages. Supports real-time multilingual applications.

<br>

## 🔬 Technical Architecture

* **Threading Separation**: Audio capture | ASR | NMT. 68% latency reduction
* **Int8 Quantization**: bits-and-bytes implementation for models
* **Intelligent VAD**: WebRTC + conservative segmentation (1.2s minimum silence) + linguistic validation
* **Memory Efficient**: Circular buffer + translation cache (0.95 similarity)
* **Hybrid Latency**: Progressive partials (2-3s perceived) with explicit `attention_mask` for enhanced ASR control
* **Adaptive Segmentation**: Avoids <0.5s fragments, 2.5s minimum cuts
* **Forced Decoding**: Use of `forced_decoder_ids` to indicate language and task to Whisper, improving transcription accuracy

<br>

### Adjustable Configuration Parameters

> **Note:** If you experience too much latency, you can reduce `num_beams` or shorten `max_new_tokens`. This will make inferences faster at the cost of slight quality loss.

**Segmentation and Flow:**

```python
TIMEOUT_SEC = 12.0           # Maximum time without flush
MIN_SEGMENT_SEC = 0.5        # Minimum accepted segment duration
MIN_PARTIAL_WORDS = 5        # Minimum words to show partial
REUSE_THRESHOLD = 0.95       # Similarity threshold for cache
SILENCE_SEC = 0.8            # Silence required for segmentation
VAD_SILENCE_DURATION_SEC = 1.2
MIN_CUT_DURATION_SEC = 2.5
AUDIO_RMS_THRESHOLD = 0.0025 # Minimum accepted volume level
```

**ASR Inference (Whisper):**

```python
gen = self.asr.generate(
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
```

**NMT Inference (SeamlessM4T):**

```python
generated_tokens = self.nmt_model.generate(
    **inputs,
    tgt_lang=self.tgt_lang,
    generate_speech=False,
    max_new_tokens=140,
    num_beams=5,
    do_sample=False,
    repetition_penalty=1.02,
    length_penalty=1.25,
    early_stopping=False,
    no_repeat_ngram_size=4,
    use_cache=True
)
```

### Optimizations for High-End Hardware

For GPUs with >20GB VRAM (RTX 4090, A40, A100), **CUDA streams** can be implemented for ASR/NMT parallelization:

```python
# Suggested modifications for high-end hardware:
asr_lock = threading.Lock()     # Instead of shared gpu_lock
nmt_lock = threading.Lock()     # Independent locks

stream_asr = torch.cuda.Stream()
stream_nmt = torch.cuda.Stream()
# Estimated potential improvement: +15-25% throughput
```

<br>

## 📜 Models and Licenses

* Marvin4000 Code: [MIT](LICENSE)
* Whisper: [MIT](https://github.com/openai/whisper/blob/main/LICENSE) (OpenAI)
* SeamlessM4T: [CC-BY-NC 4.0](https://github.com/facebookresearch/seamless_communication/blob/main/LICENSE) (Meta AI)
* NLLB-200: [CC-BY-NC 4.0](https://huggingface.co/facebook/nllb-200-3.3B) (Meta AI)

<br>

## 🙏 Acknowledgments and References

### Models and Libraries Used

* [OpenAI Whisper](https://github.com/openai/whisper)
* [Meta SeamlessM4T](https://github.com/facebookresearch/seamless_communication)
* [Meta NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)
* [WebRTC VAD](https://webrtc.org/)

### Technical Inspiration and Papers

* [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) – real-time execution
* [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) – quantization
* [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) – efficient buffering
* [snakers4/silero-vad](https://github.com/snakers4/silero-vad) – optimized VAD
* [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
* [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)
* [NLLB-200: No Language Left Behind](https://arxiv.org/abs/2207.04672)
* [Efficient Low-Bit Quantization of Transformer-Based Language Models](https://arxiv.org/abs/2305.12889)

---

<br>

This project is designed as a flexible foundation. If you want to modify it, use it creatively, improve it, or simply adapt it to your needs...

> 💪 **Go for it.**

If you also share improvements or mention us as a reference, it will always be welcome 🙌😜.

<br>

© [XOREngine](https://xorengine.com) · Open source commitment

<br>

<!-- keywords: whisper, seamlessM4T, realtime transcription, translation, streaming audio, cuda, multilingual, vad, low latency, NLLB, ASR, NMT -->