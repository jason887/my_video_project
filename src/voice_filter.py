#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 全自动主播语音过滤器（含自检测试功能）

import os
import shutil
import datetime
import argparse
import random
import wave
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from pathlib import Path
import statistics

# 声纹识别库检测
VOICE_PRINT_AVAILABLE = False
try:
    import torch
    import torchaudio
    from speechbrain.inference import EncoderClassifier
    VOICE_PRINT_AVAILABLE = True
except ImportError:
    print("提示：安装声纹库可使用高级功能 → pip install torch torchaudio speechbrain")

# ======== 全局配置 ========
PROJECT_DIR = Path("D:/my_video_project")
AUDIO_DIR = PROJECT_DIR / "audio"
FILTERED_DIR = PROJECT_DIR / "filtered_audio"
LOG_FILE = PROJECT_DIR / "audio_processing.log"
TEST_NOISE_FILE = PROJECT_DIR / "test_noise.wav"

def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

class AutoVoiceFilter:
    def __init__(self, use_cpu=False):
        self.model = None
        self.reference_voice = None
        self.similarity_threshold = 0.7
        self.embedding_cache = {}
        
        if not VOICE_PRINT_AVAILABLE:
            return

        # 硬件配置
        self.device = "cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"初始化语音引擎 → 使用设备: {self.device.upper()}")
        
        try:
            model_path = PROJECT_DIR / "voice_models"
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_path),
                run_opts={"device": self.device}
            )
            self.model.eval()
        except Exception as e:
            log_message(f"模型加载失败: {str(e)}")

    def _get_voiceprint(self, audio_path):
        """获取声纹特征（带缓存）"""
        if not self.model:
            return None
            
        cache_key = str(audio_path)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            # 加载并预处理音频
            waveform, orig_freq = torchaudio.load(audio_path)
            if orig_freq != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq, 16000)(waveform)
            if waveform.shape[1] < 16000:
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                
            # 提取特征
            embedding = self.model.encode_batch(waveform.to(self.device))
            result = embedding.squeeze().cpu().detach().numpy()
            
            self.embedding_cache[cache_key] = result
            return result
        except Exception as e:
            log_message(f"音频处理失败 [{audio_path.name}]: {str(e)}")
            return None

    def auto_setup(self, audio_files):
        """智能配置参考声纹（增加质量检测）"""
        if not self.model or len(audio_files) < 5:
            return False

        # 随机采样计算相似度矩阵
        samples = random.sample(audio_files, min(15, len(audio_files)))
        voiceprints = {}
        for f in samples:
            if vp := self._get_voiceprint(f):
                voiceprints[f] = vp

        # 计算样本间平均相似度
        similarities = []
        for f1, v1 in voiceprints.items():
            for f2, v2 in voiceprints.items():
                if f1 != f2:
                    sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
                    similarities.append(sim)
        
        if len(similarities) == 0:
            log_message("警告：样本间相似度过低，可能包含混杂音频")
            return False

        # 寻找最具代表性的样本（相似度高于平均值+标准差）
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        best_sample = max(
            voiceprints.items(),
            key=lambda x: sum(
                np.dot(x[1], v)/(np.linalg.norm(x[1])*np.linalg.norm(v))
                for k, v in voiceprints.items() if k != x[0]
            )
        )[0]

        # 自动设置阈值（更严格）
        self.similarity_threshold = max(0.65, avg_sim - 1.5*std_sim)
        
        self.reference_voice = voiceprints[best_sample]
        log_message(f"样本平均相似度: {avg_sim:.2f} | 自动阈值: {self.similarity_threshold:.2f}")
        return True

    def is_clean_audio(self, audio_path):
        """判断是否为纯净音频（带相似度记录）"""
        if not self.model or not self.reference_voice:
            return True
            
        current_vp = self._get_voiceprint(audio_path)
        if current_vp is None:
            return False

        similarity = np.dot(self.reference_voice, current_vp) / (
            np.linalg.norm(self.reference_voice) * np.linalg.norm(current_vp)
        )
        log_message(f"相似度检测 [{audio_path.name}]: {similarity:.2f}")  # 新增日志
        
        return similarity >= self.similarity_threshold

def generate_test_noise():
    """生成测试用噪音文件"""
    sample_rate = 16000
    duration = 5  # 5秒
    t = np.linspace(0, duration, int(sample_rate * duration))
    noise = np.random.normal(0, 0.5, len(t)).astype(np.float32)
    
    with wave.open(str(TEST_NOISE_FILE), 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((noise * 32767).astype(np.int16).tobytes())

def inject_test_files(host_dir, num=3):
    """为主播目录注入测试文件"""
    if not TEST_NOISE_FILE.exists():
        generate_test_noise()
    
    host_dir = Path(host_dir)
    for i in range(num):
        test_file = host_dir / f"TEST_NOISE_{i}.wav"
        shutil.copy(TEST_NOISE_FILE, test_file)
        log_message(f"已注入测试文件: {test_file.name}")

def process_host(host_prefix, src_dir, dst_dir, use_ai=True):
    """处理单个主播的音频（增加测试功能）"""
    audio_files = [f for f in Path(src_dir).glob(f"{host_prefix}*") if f.suffix.lower() in ('.wav','.mp3','.flac')]
    if not audio_files:
        log_message(f"主播 {host_prefix} 没有找到音频文件")
        return

    # 注入测试文件
    inject_test_files(src_dir)
    audio_files += [f for f in Path(src_dir).glob("TEST_NOISE_*.wav")]
    
    log_message(f"开始处理主播: {host_prefix} (共 {len(audio_files)} 个文件)")
    
    # 初始化过滤器
    filter = AutoVoiceFilter()
    if use_ai and VOICE_PRINT_AVAILABLE:
        if not filter.auto_setup(audio_files):
            log_message("智能模式失败，转为简单复制")
            use_ai = False

    # 创建输出目录
    output_dir = Path(dst_dir) / host_prefix
    output_dir.mkdir(exist_ok=True)

    # 处理文件
    test_results = {"pass": 0, "block": 0}
    for audio_file in audio_files:
        is_test = "TEST_NOISE" in audio_file.name
        
        if not use_ai or filter.is_clean_audio(audio_file):
            dest = output_dir / audio_file.name
            shutil.copy2(audio_file, dest)
            log_msg = f"保留: {audio_file.name}"
            if is_test: 
                test_results["pass"] += 1
                log_msg += " ✘ 测试未通过"
        else:
            log_msg = f"过滤: {audio_file.name}"
            if is_test: 
                test_results["block"] += 1
                log_msg += " ✔ 测试通过"
        
        log_message(log_msg)

    # 输出测试结果
    log_message(f"测试结果: 拦截 {test_results['block']}/{test_results['pass']+test_results['block']} 个噪音文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="主播语音自动过滤器")
    parser.add_argument("--src", default=str(AUDIO_DIR), help="输入目录路径")
    parser.add_argument("--dst", default=str(FILTERED_DIR), help="输出目录路径")
    parser.add_argument("--prefix", help="主播名前缀（多个用逗号分隔）")
    parser.add_argument("--auto", action="store_true", help="启用智能过滤")
    args = parser.parse_args()

    log_message("="*40)
    log_message("主播语音过滤开始".center(40))
    
    # 自动检测所有主播
    if args.prefix:
        prefixes = args.prefix.split(",")
    else:
        all_files = list(Path(args.src).glob("*"))
        prefixes = sorted({f.name.split("_")[0] for f in all_files if f.is_file()})
    
    for prefix in prefixes:
        process_host(prefix.strip(), args.src, args.dst, args.auto)

    log_message("处理完成".center(40))
    log_message("="*40)
