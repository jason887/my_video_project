#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 容器适配版：2025-03-01 更新（增强频谱分析版）

import os
import re
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import matplotlib.pyplot as plt

# ██████████ 容器关键配置 ██████████
PROJECT_DIR = Path("/workspace/src") 
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
        print(f"\U0001F7E2 初始化硬件加速 | 设备: {self.device.upper()} | CUDA版本: {torch.version.cuda}")
        
        # 预训练模型加载
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(PROJECT_DIR / "pretrained_models"),
            run_opts={"device": self.device}
        ).eval()
        
        # 初始化梅尔频谱转换器（新增配置）
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,      # 根据实际音频采样率调整
            n_fft=512,
            hop_length=256,
            n_mels=64
        ).to(self.device)
        
        self._prepare_directories()
    
    def _prepare_directories(self):
        """容器路径初始化"""
        for d in [AUDIO_DIR, FILTERED_DIR, LOG_DIR, REPORT_DIR]:
            d.mkdir(exist_ok=True, parents=True)
    
    def _dynamic_batch_size(self):
        """RTX 3060专用批处理优化"""
        if self.device == "cuda":
            total_mem = torch.cuda.get_device_properties(0).total_memory
            return 8 if total_mem > 10*1024**3 else 4
    
    def process_single(self, file_path):
        """单文件处理方法（已集成频谱分析）"""
        try:
            waveform = self._load_audio(file_path)
            embeddings = self.model.encode_batch(waveform)
            
            # 频谱特征分析（新增代码段）
            spec = self.mel_spec(waveform)
            spec_std = spec.std().item()  # 计算频谱标准差
            
            # 双重过滤条件（调整阈值）
            if embeddings.norm() > 0.4 and spec_std < 1.5:  # 联合判断条件
                target_path = FILTERED_DIR / file_path.name
                os.rename(file_path, target_path)
                return {'valid':1, 'invalid':0}
            else:
                os.remove(file_path)
                return {'valid':0, 'invalid':1}
        except Exception as e:
            print(f"\U0001F525 处理 {file_path.name} 时发生异常: {str(e)}")
            return {'valid':0, 'invalid':1}

    def _load_audio(self, path):
        """音频加载实现"""
        waveform, _ = torchaudio.load(path)
        return waveform.to(self.device)
    
    def process_files(self):
        """核心处理流程"""
        print(f"\U0001F50D 扫描目录: {AUDIO_DIR}")
        file_pattern = re.compile(r'.*?_part\d{3}_\d{3}\.wav$')
        audio_files = [f for f in AUDIO_DIR.glob('*.wav') if file_pattern.match(f.name)]
        
        if not audio_files:
            print(f"❌ 在目录 {AUDIO_DIR} 中未找到音频文件")
            return
        
        with tqdm(total=len(audio_files), desc="\U0001F680 容器处理进度", 
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
                        print(f"\n\U0001F525 {file.name} 处理失败: {str(e)}")
            
            self.generate_report(valid, invalid)
    
    def generate_report(self, valid, invalid):
        """生成处理报告"""
        report_content = f"""\U0001F4CA 音频处理报告
        处理时间：{np.datetime64('now')}
        ----------------------------
        总处理文件：{valid + invalid}
        有效保留数：{valid} ({(valid/(valid+invalid))*100:.1f}%)
        无效过滤数：{invalid} ({(invalid/(valid+invalid))*100:.1f}%)
        """
        report_path = REPORT_DIR / f"report_{np.datetime64('now','D')}.txt"
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"\U0001F4CA 报告已生成：{report_path}")

if __name__ == "__main__":
    TurboAudioProcessor().process_files()
