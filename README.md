<!-- Marvin4000 - Real-time Audio Transcription & Translation -->
<!-- © 2025 XOREngine (WallyByte) -->
<!-- https://github.com/XOREngine/marvin4000 -->

# Marvin4000

> Transcripción y traducción de audio en tiempo real con Whisper y modelos multilingües (SeamlessM4T / NLLB‑200)

[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/GPU-Accelerated-green)](https://developer.nvidia.com/cuda-toolkit)

<br>

**Marvin4000** captura, transcribe y traduce audio del sistema en tiempo real usando hardware local.

<br>

> ⚠️ **IMPORTANTE:**
>
> * Si estás en **Windows**, la captura de audio debe ser implementada manualmente mediante una alternativa a `parec` que proporcione datos de audio del sistema en formato `float32`.

<br>

## 📊 Rendimiento probado

| GPU & Modelos usados                                                | Latencia (s) | WER       | BLEU-1/4/Corpus | VRAM        |
| ------------------------------------------------------------------- | ------------ | --------- | --------------- | ----------- |
| **RTX 4060 Ti 16GB<br>whisper-large-v3 (8b), seamless-m4t-v2-large** | **2-3**      | **5 %** | **74/39/52**    | **12.2 GB** |

#### Corpus de prueba

* **Audio**: 25 fragmentos aleatorios de audiolibros de [LibriSpeech](https://www.openslr.org/12) (media: 5 min/fragmento)
* **Transcripción de referencia**: Transcripciones oficiales de LibriSpeech
* **Traducción de referencia**: Generada con Claude & GPT y revisada manualmente
* **Total evaluado**: \~120 minutos de audio

#### Cálculo de métricas

* **WER**: Calculado con [jiwer](https://github.com/jitsi/jiwer), normalizado para puntuación
* **BLEU**: Implementación corpus-level con tokenización lowercase, clipping de n-gramas y brevity penalty
* **BLEU-1/4/Corpus**: Precisión 1-grama / 4-grama / score corpus completo
* **Latencia**: Medida en condiciones reales con RTX 4060 Ti 16GB y RTX 2060 6GB

#### Limitaciones

Aunque las traducciones de referencia son de alta calidad, reconocemos que no son equivalentes a traducciones humanas profesionales. Sin embargo, proveen un estándar consistente para comparar el rendimiento del sistema, siguiendo metodologías similares a las empleadas en evaluaciones como [FLEURS](https://arxiv.org/abs/2205.12446) y [CoVoST 2](https://arxiv.org/abs/2007.10310).

<br>

## 🚀 Instalación y uso

### Requisitos

```bash
sudo apt install python3-pip pulseaudio-utils ffmpeg
git clone https://github.com/XOREngine/marvin4000.git
cd marvin4000
pip install -r requirements.txt
```

### Ejecución básica

```bash
# 1. Reproducir algún contenido con audio en tu sistema
vlc video_ejemplo.mp4
# ffmpeg.ffplay -nodisp -autoexit -ss 1 example.mp3
# o reproducir audio desde el navegador, etc.

# 2. Detectar dispositivos de audio válidos
python detect_audio_devices.py
# Ejemplo salida:
# $ python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

# 3. Iniciar transcripción/traducción con el dispositivo monitor adecuado
python marvin4000_seam.py --audio-device "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
```

### Configuración de idiomas

Marvin4000 utiliza SeamlessM4T y NLLB‑200 para transcripción y traducción entre más de 100 idiomas. Soporta aplicaciones multilingües en tiempo real.

<br>

## 🔬 Arquitectura técnica

* **Separación de hilos (Threading)**: Captura de audio | ASR | NMT. Reducción 68% latencia
* **Cuantización Int8**: Implementación bits-and-bytes para los modelos
* **VAD inteligente**: WebRTC + segmentación conservadora (1.2s silencio mínimo) + validación lingüística
* **Memoria eficiente**: Buffer circular + caché de traducciones (similitud 0.95)
* **Latencia híbrida**: Parciales progresivos (2-3s percibida) con `attention_mask` explícito para mayor control en ASR
* **Segmentación adaptativa**: Evita fragmentos <0.5s, cortes mínimos 2.5s
* **Decodificación forzada**: Uso de `forced_decoder_ids` para indicar idioma y tarea a Whisper, mejorando precisión de transcripción

<br>

### Parámetros de configuración ajustables

> **Nota:** Si experimentas demasiada latencia, puedes reducir `num_beams` a 2 o 3 y acortar `max_new_tokens`. Esto hará las inferencias más rápidas a costa de una leve pérdida de calidad.

**Segmentación y flujo:**

```python
TIMEOUT_SEC = 12.0           # Tiempo máximo sin flush
MIN_SEGMENT_SEC = 0.5        # Mínima duración aceptada de segmento
MIN_PARTIAL_WORDS = 5        # Palabras mínimas para mostrar parcial
REUSE_THRESHOLD = 0.95       # Umbral de similitud para cache
SILENCE_SEC = 0.8            # Silencio requerido para segmentar
VAD_SILENCE_DURATION_SEC = 1.2
MIN_CUT_DURATION_SEC = 2.5
AUDIO_RMS_THRESHOLD = 0.0025 # Nivel mínimo de volumen aceptado
```

**Inferencia ASR (Whisper):**

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

**Inferencia NMT (SeamlessM4T):**

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

### Optimizaciones para hardware potente

Para GPUs con >20GB VRAM (RTX 4090, A40, A100), se pueden implementar **CUDA streams** para paralelización ASR/NMT:

```python
# Modificaciones sugeridas para hardware potente:
asr_lock = threading.Lock()     # En lugar de gpu_lock compartido
nmt_lock = threading.Lock()     # Locks independientes

stream_asr = torch.cuda.Stream()
stream_nmt = torch.cuda.Stream()
# Potencial mejora estimada: +15-25% throughput
```

<br>

## 📜 Modelos y licencias

* Código Marvin4000: [MIT](LICENSE)
* Whisper: [MIT](https://github.com/openai/whisper/blob/main/LICENSE) (OpenAI)
* SeamlessM4T: [CC-BY-NC 4.0](https://github.com/facebookresearch/seamless_communication/blob/main/LICENSE) (Meta AI)
* NLLB-200: [CC-BY-NC 4.0](https://huggingface.co/facebook/nllb-200-3.3B) (Meta AI)

<br>

## 🙏 Agradecimientos y referencias

### Modelos y librerías usadas

* [OpenAI Whisper](https://github.com/openai/whisper)
* [Meta SeamlessM4T](https://github.com/facebookresearch/seamless_communication)
* [Meta NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)
* [WebRTC VAD](https://webrtc.org/)

### Inspiración técnica y papers

* [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) – ejecución tiempo real
* [TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) – cuantización
* [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) – buffering eficiente
* [snakers4/silero-vad](https://github.com/snakers4/silero-vad) – VAD optimizado
* [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
* [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)
* [NLLB-200: No Language Left Behind](https://arxiv.org/abs/2207.04672)
* [Efficient Low-Bit Quantization of Transformer-Based Language Models](https://arxiv.org/abs/2305.12889)

---

<br>

Este proyecto está pensado como una base flexible. Si quieres modificarlo, usarlo de forma creativa, mejorarlo o simplemente adaptarlo a tus necesidades...

> 💪 **Hazlo.**

Si además compartes mejoras o nos mencionas como referencia, será siempre bien recibido 🙌😜.

<br>

© [XOREngine](https://xorengine.com) · Compromiso open source

<br>

<!-- keywords: whisper, seamlessM4T, realtime transcription, translation, streaming audio, cuda, multilingual, vad, low latency, NLLB, ASR, NMT -->