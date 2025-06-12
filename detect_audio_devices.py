#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Marvin4000 - Real-time Audio Transcription & Translation
# © 2025 XOREngine (WallyByte)
# https://github.com/XOREngine/marvin4000

import sys
import time
import subprocess
import os
import numpy as np
import sounddevice as sd


def detect_audio_in_device(device_id, test_duration=3, sample_rate=44100, channels=2):
    max_level = 0
    max_energy = 0
    audio_detected = False
    
    def callback(indata, frames, time, status):
        nonlocal max_level, max_energy, audio_detected
        
        volume_norm = np.linalg.norm(indata) / np.sqrt(frames)
        energy = np.sqrt(np.mean(indata**2))
        
        max_level = max(max_level, volume_norm)
        max_energy = max(max_energy, energy)
        
        if volume_norm > 0.001 or energy > 0.001:
            audio_detected = True
    
    try:
        with sd.InputStream(device=device_id, channels=channels, 
                           callback=callback, 
                           blocksize=int(sample_rate * 0.1),
                           samplerate=sample_rate):
            time.sleep(test_duration)
            
        return {
            'audio_detected': audio_detected,
            'max_level': max_level,
            'max_energy': max_energy
        }
            
    except Exception as e:
        return {
            'audio_detected': False,
            'max_level': 0,
            'max_energy': 0,
            'error': str(e)
        }


def test_pulseaudio_monitor_sources():
    working_devices = []
    
    try:
        result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        
        sources = result.stdout.strip().split('\n')
        monitor_sources = []
        
        for source in sources:
            parts = source.split()
            if len(parts) < 2:
                continue
                
            source_name = parts[1]
            
            # Solo dispositivos monitor/virtuales/loopback
            if any(keyword in source_name.lower() for keyword in 
                   ['monitor', 'virtual', 'loopback', 'sink']):
                monitor_sources.append((parts[0], source_name))
        
        print(f"Probando {len(monitor_sources)} fuentes monitor/virtuales...")
        
        for source_id, source_name in monitor_sources:
            print(f"\nProbando: {source_name}")
            
            try:
                temp_file = "/tmp/test_audio.raw"
                cmd = f"timeout 3 parec --device={source_name} --format=float32le --rate=16000 --channels=1 > {temp_file} 2>/dev/null"
                
                os.system(cmd)
                
                if os.path.exists(temp_file):
                    data = np.fromfile(temp_file, dtype=np.float32)
                    if len(data) > 0:
                        max_level = np.max(np.abs(data))
                        energy = np.sqrt(np.mean(data**2))
                        
                        if max_level > 0.001:
                            working_devices.append({
                                'device_name': source_name,
                                'max_level': max_level,
                                'energy': energy
                            })
                            print(f"✅ AUDIO DETECTADO - Nivel: {max_level:.4f}")
                        else:
                            print(f"❌ Sin audio - Nivel: {max_level:.4f}")
                    
                    os.remove(temp_file)
                else:
                    print("❌ Error en grabación")
            
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"Error al listar fuentes: {e}")
    
    return working_devices


def test_sounddevice_monitors():
    """Prueba dispositivos monitor con sounddevice"""
    working_devices = []
    devices = sd.query_devices()
    
    monitor_devices = []
    for i, device in enumerate(devices):
        if (device['max_input_channels'] > 0 and 
            any(keyword in device['name'].lower() for keyword in 
                ['monitor', 'virtual', 'loopback', 'pulse', 'pipewire'])):
            monitor_devices.append((i, device))
    
    print(f"Probando {len(monitor_devices)} dispositivos monitor con sounddevice...")
    
    for device_id, device_info in monitor_devices:
        print(f"\nProbando ID {device_id}: {device_info['name']}")
        
        result = detect_audio_in_device(device_id)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        elif result['audio_detected']:
            working_devices.append({
                'device_id': device_id,
                'device_name': device_info['name'],
                'max_level': result['max_level'],
                'energy': result['max_energy']
            })
            print(f"✅ AUDIO DETECTADO - Nivel: {result['max_level']:.4f}")
        else:
            print(f"❌ Sin audio - Nivel: {result['max_level']:.4f}")
    
    return working_devices


def main():
    
    print("\n📡 PROBANDO FUENTES PULSEAUDIO...")
    pulse_devices = test_pulseaudio_monitor_sources()

    print("\n🎵 PROBANDO CON SOUNDDEVICE...")
    sd_devices = None 
    # sd_devices = test_sounddevice_monitors()
    
    print("\n" + "=" * 60)
    print("🏆 DISPOSITIVOS CON AUDIO DETECTADO:")
    
    if pulse_devices:
        for device in pulse_devices:
            print(f"    $ python marvin4000_seam.py --audio-device \"{device['device_name']}\"")
    
    # if sd_devices:
    #     for device in sd_devices:
    #         print(f"    $ python marvin4000.py --audio-device-id {device['device_id']}")
    
    if not pulse_devices and not sd_devices:
        print("❌ No se detectó audio en ningún dispositivo monitor/virtual")
        print("💡 Asegúrate de que hay audio reproduciéndose en el sistema")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()