import torch
import os
import json
import subprocess
import tempfile
from pydub import AudioSegment
from silero_vad import read_audio, get_speech_timestamps
from multiprocessing import Pool
from tqdm import tqdm

class AudioSplitter:
    """
    一个用于智能或强制分割长音频的类。

    采用懒加载策略：VAD模型仅在需要时才会被加载一次。
    新增功能：分割后可选择使用ffmpeg对所有音频块进行并发标准化。
    """
    def __init__(self, use_onnx: bool = True):
        """
        初始化 AudioSplitter。
        """
        print("AudioSplitter 已初始化。VAD模型将在首次进行智能分割时按需加载。")
        self._vad_model = None
        self.use_onnx = use_onnx

    @property
    def vad_model(self):
        """
        VAD模型的懒加载属性。
        """
        if self._vad_model is None:
            print("检测到首次使用VAD，正在加载 Silero VAD 模型（此操作仅执行一次）...")
            torch.set_num_threads(1)
            # Silero VAD is loaded from torch.hub
            self._vad_model, _ = torch.hub.load(repo_or_dir='snakers4_silero-vad_master',
                                                model='silero_vad',
                                                force_reload=False,
                                                onnx=self.use_onnx,
                                                source='local')
            print("VAD 模型加载完成。")
        return self._vad_model

    @staticmethod
    def _normalize_worker(filepath: str):
        """
        [新增] 用于多进程的单个文件标准化工作函数。

        Args:
            filepath (str): 需要标准化的音频文件路径。
        
        Returns:
            tuple: (文件路径, 是否成功, 错误信息或成功消息)
        """
        if not os.path.exists(filepath):
            return (filepath, False, "File not found.")

        temp_dir = os.path.dirname(filepath)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as temp_f:
                temp_filepath = temp_f.name
            
            command = [
                'ffmpeg', '-y', '-i', filepath,
                '-ar', '16000', '-ac', '1', '-acodec', 'pcm_s16le', '-f', 'wav',
                temp_filepath
            ]
            subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            os.replace(temp_filepath, filepath)
            return (filepath, True, "Success")

        except subprocess.CalledProcessError as e:
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath): os.remove(temp_filepath)
            error_msg = f"FFmpeg error for {os.path.basename(filepath)}:\n{e.stderr}"
            return (filepath, False, error_msg)
        except Exception as e:
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath): os.remove(temp_filepath)
            return (filepath, False, str(e))

    def _get_full_timeline(self, audio_path: str) -> list:
        wav = read_audio(audio_path)
        speech_timestamps = get_speech_timestamps(wav, self.vad_model, sampling_rate=16000, return_seconds=True)
        audio_duration_seconds = len(wav) / 16000
        timeline = []
        current_time = 0.0
        for segment in speech_timestamps:
            if segment['start'] > current_time: timeline.append({'start': current_time, 'end': segment['start'], 'type': 'silence'})
            timeline.append({'start': segment['start'], 'end': segment['end'], 'type': 'speech'})
            current_time = segment['end']
        if current_time < audio_duration_seconds: timeline.append({'start': current_time, 'end': audio_duration_seconds, 'type': 'silence'})
        return timeline

    @staticmethod
    def _force_split_segment(segment: dict, max_length: float) -> list:
        sub_segments = []
        start_time, end_time, seg_type = segment['start'], segment['end'], segment['type']
        while start_time < end_time:
            sub_end_time = min(start_time + max_length, end_time)
            sub_segments.append({'start': start_time, 'end': sub_end_time, 'type': seg_type})
            start_time = sub_end_time
        return sub_segments


    def split(
        self,
        audio_path: str,
        output_dir: str,
        target_length: float = 13.0,
        max_length: float = 15.0,
        overlap_length: float = 1.0,
        skip_vad: bool = False,
        normalize_audio: bool = True,
        norm_processes: int = 4
    ):
        if not os.path.exists(audio_path): raise FileNotFoundError(f"错误：音频文件不存在于 '{audio_path}'")
        if not skip_vad and target_length >= max_length: raise ValueError("错误：在VAD模式下, target_length 必须小于 max_length。")
        if skip_vad and target_length <= overlap_length: raise ValueError("错误：在强制分割模式下, target_length 必须大于 overlap_length。")

        os.makedirs(output_dir, exist_ok=True)
        chunks, force_split_count = [], 0

        if skip_vad:
            print("模式: 跳过VAD，使用固定长度分割... (无需加载VAD模型)")
            audio = AudioSegment.from_file(audio_path)
            total_duration = audio.duration_seconds
            start_time = 0.0
            while start_time < total_duration:
                end_time = min(start_time + target_length, total_duration)
                chunks.append({'start': start_time, 'end': end_time})
                next_start_time = end_time - overlap_length
                if end_time >= total_duration or next_start_time <= start_time: break
                start_time = next_start_time
        else:
            print("模式: 使用VAD进行智能分割...")
            timeline = self._get_full_timeline(audio_path)
            processed_timeline = []
            for segment in timeline:
                if (segment['end'] - segment['start']) > max_length:
                    processed_timeline.extend(self._force_split_segment(segment, max_length))
                    force_split_count += 1
                else: processed_timeline.append(segment)
            current_chunk_segments = []
            for segment in processed_timeline:
                current_duration = (current_chunk_segments[-1]['end'] - current_chunk_segments[0]['start']) if current_chunk_segments else 0
                if current_chunk_segments and current_duration + (segment['end'] - segment['start']) > max_length:
                    chunks.append({'start': current_chunk_segments[0]['start'], 'end': current_chunk_segments[-1]['end']})
                    overlap_point = max(0, current_chunk_segments[-1]['end'] - overlap_length)
                    new_start_idx = next((i for i, s in reversed(list(enumerate(current_chunk_segments))) if s['start'] < overlap_point), -1)
                    current_chunk_segments = current_chunk_segments[new_start_idx:] + [segment] if new_start_idx != -1 else [segment]
                else: current_chunk_segments.append(segment)
            if current_chunk_segments: chunks.append({'start': current_chunk_segments[0]['start'], 'end': current_chunk_segments[-1]['end']})

        print(f"分割完成，共生成 {len(chunks)} 个片段。正在导出文件...")
        original_audio = AudioSegment.from_file(audio_path)
        info_data, filepaths_to_normalize = [], []
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        for i, chunk_info in enumerate(tqdm(chunks, desc=f"导出 {base_filename} 切片")):
            start_ms, end_ms = int(chunk_info['start'] * 1000), int(chunk_info['end'] * 1000)
            chunk_audio = original_audio[start_ms:end_ms]
            output_filename = f"{base_filename}_chunk_{i+1:04d}.wav"
            output_filepath = os.path.join(output_dir, output_filename)
            chunk_audio.export(output_filepath, format="wav")
            info_data.append({"file_path": output_filepath, "original_start_time": chunk_info['start'], "original_end_time": chunk_info['end'], "duration": chunk_info['end'] - chunk_info['start']})
            filepaths_to_normalize.append(output_filepath)
        
        if normalize_audio:
            print(f"\n正在对 {len(filepaths_to_normalize)} 个音频分段进行标准化 (使用 {norm_processes} 个进程)...")
            with Pool(processes=norm_processes) as pool:
                results = list(tqdm(pool.imap_unordered(self._normalize_worker, filepaths_to_normalize), total=len(filepaths_to_normalize), desc="标准化"))
            
            failures = [res for res in results if not res[1]]
            if failures:
                print(f"\n警告: {len(failures)} 个文件在标准化过程中失败。错误信息如下:")
                for res in failures: print(f"- {res[0]}: {res[2]}")
            else:
                print("所有音频分段已成功标准化。")
        
        info_filepath = os.path.join(output_dir, "_vad_complete.json")
        with open(info_filepath, 'w', encoding='utf-8') as f: json.dump(info_data, f, indent=4, ensure_ascii=False)

        print("\n--- 分割统计 ---")
        print(f"总计: 成功分割为 {len(chunks)} 个音频文件。")
        if not skip_vad: print(f"总计: 对 {force_split_count} 个过长的原始片段执行了强制切分。")
        print("------------------")
        print(f"\n处理完成！分段音频保存在: '{output_dir}'")
        print(f"分段信息文件保存在: '{info_filepath}'")


# --- 程序主入口：演示新功能 ---
if __name__ == "__main__":
    
    # 1. 实例化分割器
    splitter = AudioSplitter()

    # --- 运行VAD智能分割，并启用后续的标准化处理 ---
    print("\n" + "="*50)
    print("场景: VAD智能分割 + 音频标准化")
    print("="*50)

    # 确保演示文件存在，如果不存在则创建一个假的
    audio_file_to_process = r"audio.wav"
    if not os.path.exists(audio_file_to_process):
        print(f"警告: 演示音频 '{audio_file_to_process}' 不存在。将创建一个1分钟的静音文件用于演示。")
        AudioSegment.silent(duration=60000).export(audio_file_to_process, format="wav")

    try:
        splitter.split(
            audio_path=audio_file_to_process,
            output_dir=r"audio_split_output_vad",
            skip_vad=False,
            target_length=13,
            max_length=15,
            overlap_length=1,
            normalize_audio=True,      # <-- 启用标准化
            norm_processes=8           # <-- 设置并发进程数 (请根据您的CPU核心数调整)
        )
    except Exception as e:
        print(f"\n程序运行出错: {e}")