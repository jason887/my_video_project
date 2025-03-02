import os
import logging
import soundfile as sf
import numpy as np
import librosa
import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition
import noisereduce as nr
import threading
import time
import shutil
from datetime import datetime
from tqdm import tqdm

# 配置日志记录
log_file = 'audio_processing.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8')
logging.info('----- 脚本开始执行 -----')

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
    logging.info(f'CUDA 可用，使用设备: {torch.cuda.get_device_name(0)}')
    logging.info(f'CUDA 可用内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    device = torch.device('cpu')
    logging.info('CUDA 不可用，使用 CPU 设备')

# 确保所有操作在同一设备上进行
torch.set_default_tensor_type('torch.FloatTensor')

# 定义文件夹路径
INPUT_FOLDER = "input_audio"
OUTPUT_FOLDER = "filtered_output"
VOICE_PRINTS_FOLDER = "voice_prints"
STREAMERS_FOLDER = "streamers"

# 创建必要的文件夹
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VOICE_PRINTS_FOLDER, exist_ok=True)
os.makedirs(STREAMERS_FOLDER, exist_ok=True)

# 加载声纹识别模型
VOICE_PRINT_AVAILABLE = True
try:
    verification_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    verification_model.to(device)
    for param in verification_model.parameters():
        if param.device != device:
            param.data = param.data.to(device)

    logging.info(f'声纹识别模型加载成功，设备: {next(verification_model.parameters()).device}')
except Exception as e:
    VOICE_PRINT_AVAILABLE = False
    logging.error(f'声纹识别模型加载失败: {e}')
    logging.exception(e)

# 定义白名单用户
WHITELIST_USER_VOICE_PRINTS = {
    "user1": "voice_prints/user1/user1_voice_print.npy",
    "user2": "voice_prints/user2/user2_voice_print.npy",
}

# 设置功能开关
NOISE_REDUCTION_ENABLED = True
VOICE_FILTER_ENABLED = True if VOICE_PRINT_AVAILABLE else False
logging.info(f'降噪功能启用: {NOISE_REDUCTION_ENABLED}')
logging.info(f'声纹过滤功能启用: {VOICE_FILTER_ENABLED}')

def extract_voice_print(audio_path):
    """提取音频文件的声纹特征"""
    if not VOICE_PRINT_AVAILABLE:
        logging.warning('声纹识别功能不可用，无法提取声纹。')
        return None

    try:
        signal, sampling_rate = torchaudio.load(audio_path)
        logging.debug(f'加载音频文件 {audio_path} 成功，采样率: {sampling_rate}')

        signal = signal.to(device)
        logging.debug(f'信号设备: {signal.device}, 模型设备: {next(verification_model.parameters()).device}')

        with torch.no_grad():
            embeddings = verification_model.encode_batch(signal).squeeze(0)

        logging.debug(f'提取声纹特征成功，embeddings shape: {embeddings.shape}')
        return embeddings.cpu().detach().numpy()
    except Exception as e:
        logging.error(f'提取声纹特征失败: {e}')
        logging.exception(e)
        return None

def compare_voice_prints(voice_print1, voice_print2):
    """比较两个声纹的相似度"""
    if voice_print1 is None or voice_print2 is None:
        logging.warning('无法比较声纹，因为其中一个或两个声纹为空。')
        return 0.0

    try:
        vp1 = torch.from_numpy(voice_print1).unsqueeze(0).to(device)
        vp2 = torch.from_numpy(voice_print2).unsqueeze(0).to(device)

        logging.debug(f'声纹1设备: {vp1.device}, 声纹2设备: {vp2.device}')

        with torch.no_grad():
            score, prediction = verification_model.verify_batch(vp1, vp2)

        similarity_score = score.item()
        logging.debug(f'声纹比较成功，相似度得分: {similarity_score}')
        return similarity_score
    except Exception as e:
        logging.error(f'声纹比较失败: {e}')
        logging.exception(e)
        return 0.0

def denoise_audio(audio_path, output_path):
    """使用 noisereduce 对音频进行降噪处理"""
    try:
        signal, sampling_rate = librosa.load(audio_path, sr=None)
        logging.debug(f'使用 librosa 加载音频文件 {audio_path} 成功，采样率: {sampling_rate}')

        reduced_noise = nr.reduce_noise(y=signal, sr=sampling_rate)

        sf.write(output_path, reduced_noise, sampling_rate)
        logging.debug(f'保存音频文件 {output_path} 成功，采样率: {sampling_rate}')
        return True
    except Exception as e:
        logging.error(f'音频降噪处理失败 (使用 noisereduce): {e}')
        logging.exception(e)
        return False

def create_voice_print(audio_path, streamer_name):
    """从纯净音频创建主播声纹"""
    if not VOICE_PRINT_AVAILABLE:
        print("错误: 声纹识别功能不可用，无法创建声纹。")
        logging.error("声纹识别功能不可用，无法创建声纹。")
        return False

    try:
        # 为每个主播创建专门的目录
        streamer_dir = os.path.join(VOICE_PRINTS_FOLDER, streamer_name)
        os.makedirs(streamer_dir, exist_ok=True)

        # 提取声纹
        voice_print = extract_voice_print(audio_path)
        if voice_print is None:
            print(f"错误: 无法从音频 {audio_path} 提取声纹。")
            logging.error(f"无法从音频 {audio_path} 提取声纹。")
            return False

        # 生成带时间戳的文件名，避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        voice_print_filename = f"{streamer_name}_{timestamp}.npy"
        voice_print_path = os.path.join(streamer_dir, voice_print_filename)

        # 保存声纹文件
        np.save(voice_print_path, voice_print)

        # 同时更新白名单配置
        WHITELIST_USER_VOICE_PRINTS[streamer_name] = voice_print_path

        # 复制一份音频到主播文件夹作为参考
        audio_filename = os.path.basename(audio_path)
        streamer_audio_dir = os.path.join(STREAMERS_FOLDER, streamer_name)
        os.makedirs(streamer_audio_dir, exist_ok=True)
        reference_audio_path = os.path.join(streamer_audio_dir, f"reference_{timestamp}_{audio_filename}")
        shutil.copy2(audio_path, reference_audio_path)

        print(f"成功: 已为主播 {streamer_name} 创建声纹文件: {voice_print_path}")
        print(f"参考音频已保存至: {reference_audio_path}")
        logging.info(f"创建声纹成功，主播: {streamer_name}，路径: {voice_print_path}")
        return True
    except Exception as e:
        logging.error(f'创建声纹失败: {e}')
        logging.exception(e)
        return False

# 新增主程序流程
def main_menu():
    while True:
        print("\n=== 音频处理系统 ===")
        print("1. 创建主播声纹")
        print("2. 处理音频文件")
        print("3. 退出系统")
        
        choice = input("请选择操作 (1-3): ")
        
        if choice == "1":
            audio_path = input("请输入纯净音频文件路径: ")
            streamer_name = input("请输入主播名称: ")
            create_voice_print(audio_path, streamer_name)
            
        elif choice == "2":
            print("音频处理功能需要结合您已实现的模块运行")
            
        elif choice == "3":
            print("系统退出")
            break
            
        else:
            print("无效的输入，请重新选择")

if __name__ == "__main__":
    main_menu()
