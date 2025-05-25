import sounddevice as sd
import numpy as np
import queue
import sys
import threading
from scipy.io import wavfile
import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import time

# 设置音频参数
SAMPLE_RATE = 16000  # 采样率
CHANNELS = 1         # 单声道
DTYPE = np.int16     # 音频数据类型
CHUNK_DURATION = 3   # 每个音频块的持续时间（秒）
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 每个音频块的大小

# 初始化音频队列
audio_queue = queue.Queue()
# 用于存储已处理的文本，避免重复
processed_texts = set()

def init_models():
    """初始化语音识别和翻译模型"""
    print("正在加载模型...")
    
    # 初始化 Whisper 模型，使用 CUDA
    whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
    
    # 初始化翻译模型
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translation_model = MarianMTModel.from_pretrained(model_name).to("cuda")
    
    return whisper_model, translation_model, tokenizer

def audio_callback(indata, frames, time, status):
    """音频回调函数，用于接收音频数据"""
    if status:
        print(f"音频流状态: {status}")
    audio_queue.put(indata.copy())

def process_audio(whisper_model, translation_model, tokenizer):
    """处理音频数据的主函数"""
    while True:
        # 从队列中获取音频数据
        audio_data = audio_queue.get()
        if audio_data is None:
            break

        # 将音频数据转换为适合 Whisper 处理的格式
        audio_data = audio_data.flatten().astype(np.float32) / 32768.0

        # 使用 Whisper 进行语音识别
        result = whisper_model.transcribe(audio_data, language="en")
        
        for segment in result[0]:
            text = segment.text.strip()
            
            # 检查文本是否已经处理过
            if text and text not in processed_texts:
                processed_texts.add(text)
                
                # 翻译文本
                inputs = tokenizer(text, return_tensors="pt").to("cuda")
                translated = translation_model.generate(**inputs)
                translation = tokenizer.decode(translated[0], skip_special_tokens=True)
                
                # 打印结果
                print("\n" + "="*50)
                print(f"英文: {text}")
                print(f"中文: {translation}")
                print("="*50)

                # 限制已处理文本集合的大小，防止内存占用过大
                if len(processed_texts) > 100:
                    processed_texts.clear()

def main():
    """主函数"""
    try:
        # 初始化模型
        whisper_model, translation_model, tokenizer = init_models()
        print("模型加载完成！开始录音...")

        # 创建音频处理线程
        processing_thread = threading.Thread(target=process_audio, 
                                          args=(whisper_model, translation_model, tokenizer))
        processing_thread.start()

        # 开始录音
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                          callback=audio_callback, blocksize=CHUNK_SIZE):
            print("正在录音...按 Ctrl+C 停止")
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n录音已停止")
        # 向队列发送终止信号
        audio_queue.put(None)
        processing_thread.join()
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 