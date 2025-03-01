#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 容器适配版：2025-03-01 更新

import os
import re
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from speechbrain.pretrained import EncoderClassifier
import matplotlib.pyplot as plt

# ██████████ 容器关键配置 ██████████
PROJECT_DIR = Path("/workspace/src")  # ← 容器内统一路径
AUDIO_DIR = PROJECT_DIR / "audio_output"
FILTERED_DIR = PROJECT_DIR / "filtered_audio"
LOG_DIR = PROJECT_DIR / "logs"
REPORT_DIR = PROJECT_DIR / "reports"
# ████████████████████████████████

class TurboAudioProcessor:
    def __init__(self):
        # CUDA加速配置（适配RTX 3060）
        torch.backends.cudnn.benchmark = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🟢 初始化硬件加速 | 设备: {self.device.upper()} | CUDA版本: {torch.version.cuda}")
        
        # 预训练模型加载（自动缓存到容器内）
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(PROJECT_DIR / "pretrained_models"),  # 容器内专用缓存目录
            run_opts={"device": self.device}
        ).eval()
        
        self._prepare_directories()
    
    def _prepare_directories(self):
        """容器路径初始化"""
        for d in [AUDIO_DIR, FILTERED_DIR, LOG_DIR, REPORT_DIR]:
            d.mkdir(exist_ok=True, parents=True)
    
    def _dynamic_batch_size(self):
        """RTX 3060专用批处理优化"""
        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return 8 if total_mem > 10*1024**3 else 4  # 12GB显存用8，8GB用4
    
    def process_files(self):
        """核心处理流程（保留多线程）"""
        file_pattern = re.compile(r'.*?_part\d{3}_\d{3}\.wav$')
        audio_files = [f for f in AUDIO_DIR.glob('*.wav') if file_pattern.match(f.name)]
        
        if not audio_files:
            print(f"❌ 在目录 {AUDIO_DIR} 中未找到音频文件")
            return
        
        with tqdm(total=len(audio_files), desc="🚀 容器处理进度", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余: {remaining}]") as pbar:
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = {executor.submit(self.process_single, f): f for f in audio_files}
                valid, invalid = 0, 0
                
                for future in concurrent.futures.as_completed(futures):
                    file = futures[future]
                    try:
                        result = future.result()
                        valid += result['valid']
                        invalid += result['invalid']
                        pbar.update()
                        pbar.set_postfix_str(f"✅合格:{valid} ❌丢弃:{invalid}")
                    except Exception as e:
                        print(f"\n🔥 {file.name} 处理失败: {str(e)}")
            
            self.generate_report(valid, invalid)
    
    # 其余方法保持不变（process_single/_load_audio/generate_report等）

if __name__ == "__main__":
    TurboAudioProcessor().process_files()
