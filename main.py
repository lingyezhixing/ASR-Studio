from pathlib import Path
import argparse
from inference import GLMASR

def main():
    parser = argparse.ArgumentParser(description='将音频文件转录为文本。')
    parser.add_argument('path', type=str, help='音频文件或目录路径')
    args = parser.parse_args()

    # 移除Windows路径周围的引号
    input_path_str = args.path.strip('"').strip("'")
    input_path = Path(input_path_str)
    
    if input_path.is_file():
        process_file(input_path)
    elif input_path.is_dir():
        for audio_file in input_path.glob('*.wav'):
            process_file(audio_file)
    else:
        print(f"错误: {input_path_str} 不是有效的文件或目录。")

def process_file(audio_path):
    asr = GLMASR(model_path="models/GLM-ASR-Nano-2512")
    asr.transcribe(str(audio_path), operation_mode="save")

if __name__ == "__main__":
    main()