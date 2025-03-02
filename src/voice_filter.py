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
    "user1": "voice_prints/user1_voice_print.npy",
    "user2": "voice_prints/user2_voice_print.npy",
}

# 确保voice_prints目录存在
os.makedirs("voice_prints", exist_ok=True)

INPUT_FOLDER = "input_audio"
OUTPUT_FOLDER = "filtered_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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


def process_audio_file(filename, pbar=None):
    """处理单个音频文件"""
    input_path = os.path.join(INPUT_FOLDER, filename)
    base_filename, ext = os.path.splitext(filename)
    denoised_filename = f"{base_filename}_denoised{ext}"
    denoised_output_path = os.path.join(OUTPUT_FOLDER, denoised_filename)
    filtered_filename = f"{base_filename}_filtered{ext}"
    filtered_output_path = os.path.join(OUTPUT_FOLDER, filtered_filename)

    logging.info(f'开始处理音频文件: {filename}')
    
    if pbar:
        pbar.set_description(f"处理: {filename}")

    # 降噪处理
    if NOISE_REDUCTION_ENABLED:
        logging.info(f'开始降噪处理: {filename}')
        if pbar:
            pbar.set_description(f"降噪处理: {filename}")
        
        success = denoise_audio(input_path, denoised_output_path)
        if success:
            audio_to_process_path = denoised_output_path
            logging.info(f'音频降噪处理完成，输入: {input_path}, 输出: {denoised_output_path}')
        else:
            audio_to_process_path = input_path
            logging.warning(f'音频降噪处理失败，将使用原始音频继续处理: {input_path}')
    else:
        audio_to_process_path = input_path
        denoised_output_path = input_path

    # 声纹过滤
    if VOICE_FILTER_ENABLED:
        logging.info(f'开始声纹过滤: {filename}')
        if pbar:
            pbar.set_description(f"声纹过滤: {filename}")
            
        input_voice_print = extract_voice_print(audio_to_process_path)

        is_whitelist_user = False
        for username, voice_print_file in WHITELIST_USER_VOICE_PRINTS.items():
            try:
                if not os.path.exists(voice_print_file):
                    logging.warning(f'白名单声纹文件不存在: {voice_print_file}')
                    continue
                    
                whitelist_voice_print = np.load(voice_print_file)
                similarity = compare_voice_prints(input_voice_print, whitelist_voice_print)
                logging.debug(f'与白名单用户 {username} 声纹相似度: {similarity}')
                if similarity > 0.7:
                    is_whitelist_user = True
                    logging.info(f'音频文件来自白名单用户 {username}, 相似度: {similarity}')
                    break
            except Exception as e:
                logging.warning(f'加载或比较白名单用户 {username} 声纹时出错: {e}')
                logging.exception(e)

        if is_whitelist_user:
            try:
                sf.copy(audio_to_process_path, filtered_output_path)
                logging.info(f'白名单用户音频，已复制到输出目录: {filtered_output_path}')
            except Exception as e:
                logging.error(f'复制白名单用户音频到输出目录失败: {e}')
                logging.exception(e)
        else:
            logging.info(f'非白名单用户音频，已丢弃: {filename}')
    else:
        try:
            sf.copy(audio_to_process_path, filtered_output_path)
            logging.info(f'声纹过滤未启用，音频已复制到输出目录: {filtered_output_path}')
        except Exception as e:
            logging.error(f'复制音频到输出目录失败: {e}')
            logging.exception(e)

    if pbar:
        pbar.update(1)
        pbar.set_description(f"已完成: {filename}")
        
    logging.info(f'音频文件 {filename} 处理完成')
    logging.info('----- 本次音频文件处理结束 -----\n')


def process_with_semaphore(filename, semaphore, pbar):
    """使用信号量控制并发处理"""
    with semaphore:
        process_audio_file(filename, pbar)


if __name__ == "__main__":
    logging.info('----- 音频处理流程开始 -----')
    print("="*50)
    print("音频处理系统")
    print("="*50)
    print(f"降噪功能: {'启用' if NOISE_REDUCTION_ENABLED else '禁用'}")
    print(f"声纹过滤功能: {'启用' if VOICE_FILTER_ENABLED else '禁用'}")
    print(f"声纹识别功能: {'可用' if VOICE_PRINT_AVAILABLE else '不可用'}")
    print("-"*50)
    
    logging.info(f'降噪功能: {"启用" if NOISE_REDUCTION_ENABLED else "禁用"}')
    logging.info(f'声纹过滤功能: {"启用" if VOICE_FILTER_ENABLED else "禁用"}')
    logging.info(f'声纹识别功能可用: {VOICE_PRINT_AVAILABLE}')
    
    # 如果使用CUDA，清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info(f'已清理CUDA缓存')

    if not VOICE_PRINT_AVAILABLE and VOICE_FILTER_ENABLED:
        logging.warning('声纹识别功能不可用，声纹过滤将不会启用。')
        print("警告: 声纹识别功能不可用，声纹过滤已自动禁用。")
        VOICE_FILTER_ENABLED = False

    if not os.path.exists(INPUT_FOLDER):
        logging.error(f'输入文件夹 {INPUT_FOLDER} 不存在')
        os.makedirs(INPUT_FOLDER)
        logging.info(f'已尝试创建输入文件夹 {INPUT_FOLDER}')
        print(f"错误: 输入文件夹 '{INPUT_FOLDER}' 不存在，已尝试创建，请将音频文件放入其中并重新运行脚本。")
    else:
        audio_files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f)) and f.lower().endswith(('.wav', '.mp3', '.flac'))]
        logging.info(f'找到音频文件: {audio_files}')

        if not audio_files:
            logging.warning(f'输入文件夹 {INPUT_FOLDER} 中没有找到任何音频文件。')
            print(f"警告: 输入文件夹 '{INPUT_FOLDER}' 中没有找到任何音频文件。请将音频文件放入该文件夹中。")
        else:
            total_files = len(audio_files)
            print(f"找到 {total_files} 个音频文件，开始处理...")
            
            # 创建进度条
            with tqdm(total=total_files, desc="总体进度", unit="文件") as pbar:
                # 设置最大线程数
                max_threads = min(4, os.cpu_count() or 2)
                semaphore = threading.Semaphore(max_threads)
                threads = []
                
                # 创建并启动线程
                for audio_file in audio_files:
                    thread = threading.Thread(
                        target=process_with_semaphore,
                        args=(audio_file, semaphore, pbar)
                    )
                    threads.append(thread)
                    thread.start()
                
                # 等待所有线程完成
                for thread in threads:
                    thread.join()
            
            print(f"处理完成! 共处理 {total_files} 个音频文件。")
            print(f"处理后的文件保存在 '{OUTPUT_FOLDER}' 文件夹中。")
            print(f"详细日志已保存到 '{log_file}'。")
            
    logging.info('----- 音频处理流程结束 -----')
