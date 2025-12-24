import torch
import os
import json
import subprocess
import tempfile
from pathlib import Path
from pydub import AudioSegment
from silero_vad import read_audio, get_speech_timestamps
from multiprocessing import Pool
from tqdm import tqdm

class AudioSplitter:
    """
    音频分割器：支持智能VAD分割和固定长度分割，可配置输出模式。

    采用懒加载策略：VAD模型仅在首次使用时加载一次。
    支持三种操作模式：仅返回对象、仅保存文件、同时返回并保存。
    输出文件命名规范：{原始文件名}_chunk_{编号:04d}.wav，确保顺序可追溯。
    """
    def __init__(self, use_onnx: bool = True):
        """
        初始化音频分割器。

        Args:
            use_onnx (bool): 是否使用 ONNX 版本的 Silero VAD 模型，默认为 True
        """
        print("AudioSplitter 已初始化。VAD模型将在首次进行智能分割时按需加载。")
        self._vad_model = None
        self.use_onnx = use_onnx

    @property
    def vad_model(self):
        """
        懒加载 Silero VAD 模型，仅在首次调用时加载。

        Returns:
            torch.nn.Module: 加载的 VAD 模型
        """
        if self._vad_model is None:
            print("检测到首次使用VAD，正在加载 Silero VAD 模型（此操作仅执行一次）...")
            torch.set_num_threads(1)
            # Silero VAD is loaded from torch.hub
            self._vad_model, _ = torch.hub.load(
                repo_or_dir='models/snakers4_silero-vad_master',
                model='silero_vad',
                force_reload=False,
                onnx=self.use_onnx,
                source='local'
            )
            print("VAD 模型加载完成。")
        return self._vad_model

    @staticmethod
    def _normalize_worker(filepath: str) -> tuple[str, bool, str]:
        """
        多进程标准化音频文件（16kHz, mono, PCM_S16LE）。

        Args:
            filepath (str): 待标准化的音频文件路径

        Returns:
            tuple[str, bool, str]: (文件路径, 是否成功, 错误信息或成功消息)
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
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            error_msg = f"FFmpeg error for {os.path.basename(filepath)}:\n{e.stderr}"
            return (filepath, False, error_msg)
        except Exception as e:
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return (filepath, False, str(e))

    def _get_full_timeline(self, audio_path: str) -> list[dict]:
        """
        使用 VAD 模型分析音频，生成包含语音/静音段的时间轴。

        Args:
            audio_path (str): 音频文件路径

        Returns:
        list[dict]: 时间轴列表，每个元素包含 'start', 'end', 'type'（speech/silence）
        """
        wav = read_audio(audio_path)
        speech_timestamps = get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=16000,
            return_seconds=True
        )
        audio_duration_seconds = len(wav) / 16000
        timeline = []
        current_time = 0.0
        for segment in speech_timestamps:
            if segment['start'] > current_time:
                timeline.append({'start': current_time, 'end': segment['start'], 'type': 'silence'})
            timeline.append({'start': segment['start'], 'end': segment['end'], 'type': 'speech'})
            current_time = segment['end']
        if current_time < audio_duration_seconds:
            timeline.append({'start': current_time, 'end': audio_duration_seconds, 'type': 'silence'})
        return timeline

    @staticmethod
    def _force_split_segment(segment: dict, max_length: float) -> list[dict]:
        """
        将超长音频段强制分割为不超过 max_length 的子段。

        Args:
            segment (dict): 原始段，包含 'start', 'end', 'type'
            max_length (float): 最大段长（秒）

        Returns:
            list[dict]: 子段列表
        """
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
        output_dir: str | None = None,
        target_length: float = 20.0,
        max_length: float = 24.0,
        overlap_length: float = 0.0,
        skip_vad: bool = False,
        normalize_audio: bool = True,
        norm_processes: int = 4,
        operation_mode: str = "save"
    ) -> list[dict] | None:
        """
        智能分割音频文件，支持三种操作模式。

        Args:
            audio_path (str): 输入音频文件路径
            output_dir (str | None): 输出目录路径。当 operation_mode 为 "return" 时可为 None
            target_length (float): 目标片段长度（秒），默认 13.0
            max_length (float): 最大片段长度（秒），默认 15.0
            overlap_length (float): 片段重叠长度（秒），默认 1.0
            skip_vad (bool): 是否跳过 VAD 智能分割，使用固定长度分割
            normalize_audio (bool): 是否对输出音频进行标准化（16kHz, mono, PCM_S16LE）
            norm_processes (int): 标准化并发进程数
            operation_mode (str): 操作模式
                - "return": 仅返回 AudioSegment 对象列表，不保存文件
                - "save": 仅保存文件到 output_dir，不返回对象
                - "return_and_save": 同时返回对象并保存文件

        Returns:
            list[dict] | None:
                - "return" 或 "return_and_save": 返回列表，每个元素为 {
                    "filename": str,           # 文件名
                    "audio_segment": AudioSegment,  # PyDub 音频对象
                    "start": float,            # 起始时间（秒）
                    "end": float               # 结束时间（秒）
                  }
                - "save": 返回 None

        Raises:
            FileNotFoundError: 输入文件不存在
            ValueError: 参数组合无效
        """
        # 输入验证
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if not skip_vad and target_length >= max_length:
            raise ValueError("VAD模式下，target_length 必须小于 max_length")
        
        if skip_vad and target_length <= overlap_length:
            raise ValueError("强制分割模式下，target_length 必须大于 overlap_length")
        
        if operation_mode not in {"return", "save", "return_and_save"}:
            raise ValueError("operation_mode 必须为 'return', 'save', 或 'return_and_save'")
        
        if operation_mode in {"save", "return_and_save"} and output_dir is None:
            raise ValueError("当 operation_mode 为 'save' 或 'return_and_save' 时，output_dir 必须提供")
        
        # 初始化
        chunks: list[dict] = []
        force_split_count = 0
        
        # 创建输出目录（仅在需要时）
        if operation_mode in {"save", "return_and_save"}:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 分割逻辑
        if skip_vad:
            print("模式: 跳过VAD，使用固定长度分割... (无需加载VAD模型)")
            audio = AudioSegment.from_file(audio_path)
            total_duration = audio.duration_seconds
            start_time = 0.0
            while start_time < total_duration:
                end_time = min(start_time + target_length, total_duration)
                chunks.append({'start': start_time, 'end': end_time})
                next_start_time = end_time - overlap_length
                if end_time >= total_duration or next_start_time <= start_time:
                    break
                start_time = next_start_time
        else:
            print("模式: 使用VAD进行智能分割...")
            timeline = self._get_full_timeline(audio_path)
            processed_timeline = []
            for segment in timeline:
                duration = segment['end'] - segment['start']
                if duration > max_length:
                    processed_timeline.extend(self._force_split_segment(segment, max_length))
                    force_split_count += 1
                else:
                    processed_timeline.append(segment)
                
            current_chunk_segments: list[dict] = []
            for segment in processed_timeline:
                current_duration = (current_chunk_segments[-1]['end'] - current_chunk_segments[0]['start']) if current_chunk_segments else 0
                if current_chunk_segments and current_duration + (segment['end'] - segment['start']) > max_length:
                    chunks.append({
                        'start': current_chunk_segments[0]['start'],
                        'end': current_chunk_segments[-1]['end']
                    })
                    overlap_point = max(0, current_chunk_segments[-1]['end'] - overlap_length)
                    new_start_idx = next(
                        (i for i, s in reversed(list(enumerate(current_chunk_segments))) if s['start'] < overlap_point),
                        -1
                    )
                    current_chunk_segments = (
                        current_chunk_segments[new_start_idx:] + [segment]
                        if new_start_idx != -1 else [segment]
                    )
                else:
                    current_chunk_segments.append(segment)
            
            if current_chunk_segments:
                chunks.append({
                    'start': current_chunk_segments[0]['start'],
                    'end': current_chunk_segments[-1]['end']
                })
        
        # 导出音频
        print(f"分割完成，共生成 {len(chunks)} 个片段。正在处理...")
        original_audio = AudioSegment.from_file(audio_path)
        info_data: list[dict] = []
        filepaths_to_normalize: list[str] = []
        base_filename = Path(audio_path).stem
        returned_segments: list[dict] = []
        
        for i, chunk_info in enumerate(tqdm(chunks, desc=f"处理 {base_filename} 切片")):
            start_ms, end_ms = int(chunk_info['start'] * 1000), int(chunk_info['end'] * 1000)
            chunk_audio = original_audio[start_ms:end_ms]
            output_filename = f"{base_filename}_chunk_{i+1:04d}.wav"
            
            # 保存文件（save 或 return_and_save 模式）
            if operation_mode in {"save", "return_and_save"}:
                output_filepath = str(Path(output_dir) / output_filename)
                chunk_audio.export(output_filepath, format="wav")
                info_data.append({
                    "file_path": output_filepath,
                    "original_start_time": chunk_info['start'],
                    "original_end_time": chunk_info['end'],
                    "duration": chunk_info['end'] - chunk_info['start']
                })
                filepaths_to_normalize.append(output_filepath)
            
            # 返回对象（return 或 return_and_save 模式）
            if operation_mode in {"return", "return_and_save"}:
                returned_segments.append({
                    "filename": output_filename,
                    "audio_segment": chunk_audio,
                    "start": chunk_info['start'],
                    "end": chunk_info['end']
                })
        
        # 标准化音频
        if normalize_audio and filepaths_to_normalize:
            print(f"\n正在对 {len(filepaths_to_normalize)} 个音频分段进行标准化 (使用 {norm_processes} 个进程)...")
            with Pool(processes=norm_processes) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self._normalize_worker, filepaths_to_normalize),
                    total=len(filepaths_to_normalize),
                    desc="标准化"
                ))
            
            failures = [res for res in results if not res[1]]
            if failures:
                print(f"\n警告: {len(failures)} 个文件在标准化过程中失败。错误信息如下:")
                for res in failures:
                    print(f"- {res[0]}: {res[2]}")
            else:
                print("所有音频分段已成功标准化.")
        
        # 生成信息文件
        if operation_mode in {"save", "return_and_save"}:
            info_filepath = str(Path(output_dir) / "_vad_complete.json")
            with open(info_filepath, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, indent=4, ensure_ascii=False)
        
        # 输出统计信息
        print("\n--- 分割统计 ---")
        print(f"总计: 成功分割为 {len(chunks)} 个片段。")
        if not skip_vad:
            print(f"总计: 对 {force_split_count} 个过长的原始片段执行了强制切分。")
        print("------------------")
        
        if operation_mode in {"save", "return_and_save"}:
            print(f"\n处理完成！分段音频保存在: '{output_dir}'")
            print(f"分段信息文件保存在: '{info_filepath}'")
        
        # 返回结果
        if operation_mode == "return":
            return returned_segments
        elif operation_mode == "save":
            return None
        else:  # return_and_save
            return returned_segments

# --- 程序主入口：演示新功能 ---
if __name__ == "__main__":
    
    # 1. 实例化分割器
    splitter = AudioSplitter()
    
    # --- 演示：保存模式 ---
    print("\n" + "="*50)
    print("场景: 保存模式（仅保存文件）")
    print("="*50)
    audio_file_to_process = r"test_audio.wav"
    if not os.path.exists(audio_file_to_process):
        print(f"警告: 演示音频 '{audio_file_to_process}' 不存在。将创建一个1分钟的静音文件用于演示。")
        AudioSegment.silent(duration=60000).export(audio_file_to_process, format="wav")
    try:
        splitter.split(
            audio_path=audio_file_to_process,
            output_dir=r"audio_split_output_save",
            skip_vad=False,
            target_length=20,
            max_length=24,
            overlap_length=0,
            normalize_audio=True,
            norm_processes=8,
            operation_mode="save"
        )
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    # --- 演示：返回模式 ---
    print("\n" + "="*50)
    print("场景: 返回模式（仅返回对象，不保存文件）")
    print("="*50)
    try:
        segments = splitter.split(
            audio_path=audio_file_to_process,
            output_dir=None,
            skip_vad=False,
            target_length=20,
            max_length=24,
            overlap_length=0,
            normalize_audio=True,
            norm_processes=8,
            operation_mode="return"
        )
        print(f"成功返回 {len(segments)} 个音频片段对象。")
        print(f"第一个片段: {segments[0]['filename']}, 长度: {segments[0]['audio_segment'].duration_seconds:.2f}s")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    # --- 演示：返回并保存模式 ---
    print("\n" + "="*50)
    print("场景: 返回并保存模式（同时返回对象并保存文件）")
    print("="*50)
    try:
        segments = splitter.split(
            audio_path=audio_file_to_process,
            output_dir=r"audio_split_output_return_and_save",
            skip_vad=False,
            target_length=20,
            max_length=24,
            overlap_length=0,
            normalize_audio=True,
            norm_processes=8,
            operation_mode="return_and_save"
        )
        print(f"成功返回 {len(segments)} 个音频片段对象，并已保存至文件。")
        print(f"第一个片段: {segments[0]['filename']}, 长度: {segments[0]['audio_segment'].duration_seconds:.2f}s")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
