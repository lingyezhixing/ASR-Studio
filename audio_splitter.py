import torch
import os
import json
import tempfile
from pathlib import Path
from pydub import AudioSegment
# 假设 silero_vad 相关依赖文件已在本地正确配置
from silero_vad import read_audio, get_speech_timestamps
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
            # 保持原始设定：从本地加载
            self._vad_model, _ = torch.hub.load(
                repo_or_dir='models/snakers4_silero-vad_master',
                model='silero_vad',
                force_reload=False,
                onnx=self.use_onnx,
                source='local'
            )
            print("VAD 模型加载完成。")
        return self._vad_model

    def _get_full_timeline(self, audio_path: str) -> list[dict]:
        """
        使用 VAD 模型分析音频，生成包含语音/静音段的时间轴。
        已修复：强制将音频转换为 16kHz 单声道供 VAD 分析。

        Args:
            audio_path (str): 音频文件路径

        Returns:
        list[dict]: 时间轴列表，每个元素包含 'start', 'end', 'type'（speech/silence）
        """
        # 1. 预处理：生成 16k 单声道临时文件供 VAD 读取
        # 必须使用 tempfile 确保不污染原目录，且确保 VAD 输入合规
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_vad_f:
            temp_vad_path = temp_vad_f.name

        try:
            # 使用 pydub 进行转换
            audio_for_vad = AudioSegment.from_file(audio_path)
            audio_for_vad = audio_for_vad.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_for_vad.export(temp_vad_path, format="wav")

            # 2. 读取转换后的音频进行 VAD 分析
            wav = read_audio(temp_vad_path)
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                sampling_rate=16000,
                return_seconds=True
            )
            # 使用原音频的长度（秒），pydub读取的duration比较准确
            audio_duration_seconds = audio_for_vad.duration_seconds

        finally:
            # 清理临时文件
            if os.path.exists(temp_vad_path):
                os.remove(temp_vad_path)

        # 3. 构建时间轴
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
        target_length: float = 12.0,
        max_length: float = 15.0,
        overlap_length: float = 0.0,
        skip_vad: bool = False,
        normalize_audio: bool = True,
        operation_mode: str = "save"
    ) -> list[dict] | None:
        """
        智能分割音频文件，支持三种操作模式。
        已修复：重叠逻辑错误、内存对象归一化、去除FFmpeg依赖。

        Args:
            audio_path (str): 输入音频文件路径
            output_dir (str | None): 输出目录路径
            target_length (float): 目标片段长度（秒）
            max_length (float): 最大片段长度（秒）
            overlap_length (float): 片段重叠长度（秒）
            skip_vad (bool): 是否跳过 VAD
            normalize_audio (bool): 是否对输出音频进行标准化（16kHz, mono, PCM_S16LE）
            norm_processes (int): (已弃用，保留兼容性) 标准化现在并在主线程通过pydub执行
            operation_mode (str): "return", "save", "return_and_save"

        Returns:
            list[dict] | None
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
                # 计算如果加入当前段，总时长是否超标
                current_duration = (current_chunk_segments[-1]['end'] - current_chunk_segments[0]['start']) if current_chunk_segments else 0
                if current_chunk_segments and current_duration + (segment['end'] - segment['start']) > max_length:
                    # 1. 记录当前切片
                    chunks.append({
                        'start': current_chunk_segments[0]['start'],
                        'end': current_chunk_segments[-1]['end']
                    })
                    
                    # 2. 计算重叠点
                    overlap_point = max(0, current_chunk_segments[-1]['end'] - overlap_length)
                    
                    # 3. 【修正逻辑】查找结束时间晚于重叠点的第一个片段
                    # 原逻辑中的 start < overlap_point 是错误的
                    new_start_idx = -1
                    for i, s in enumerate(current_chunk_segments):
                        if s['end'] > overlap_point:
                            new_start_idx = i
                            break
                    
                    # 4. 防止死循环逻辑：如果重叠点导致要保留所有片段（idx=0），且列表长度>1，则强制推进
                    if new_start_idx == 0 and len(current_chunk_segments) > 1:
                        new_start_idx = 1
                    
                    # 5. 重构 buffer
                    if new_start_idx != -1:
                        current_chunk_segments = current_chunk_segments[new_start_idx:] + [segment]
                    else:
                        # 没有片段满足重叠条件（中间可能是长静音），直接重新开始
                        current_chunk_segments = [segment]
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
        base_filename = Path(audio_path).stem
        returned_segments: list[dict] = []
        
        # 单线程处理导出（包含内存中标准化），取代原有的多进程 FFmpeg
        # 使用 pydub 的处理速度通常足够快，且能保证 save 和 return 的数据一致性
        for i, chunk_info in enumerate(tqdm(chunks, desc=f"处理 {base_filename} 切片")):
            start_ms, end_ms = int(chunk_info['start'] * 1000), int(chunk_info['end'] * 1000)
            chunk_audio = original_audio[start_ms:end_ms]
            
            # 【核心修正】在此处统一进行标准化
            # 无论 operation_mode 是什么，只要 normalize_audio=True，
            # 这里的 AudioSegment 对象就会被转换。这解决了 "return" 模式未归一化的问题。
            if normalize_audio:
                chunk_audio = chunk_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

            output_filename = f"{base_filename}_chunk_{i+1:04d}.wav"
            
            # 保存文件（save 或 return_and_save 模式）
            if operation_mode in {"save", "return_and_save"}:
                output_filepath = str(Path(output_dir) / output_filename)
                # 直接导出已归一化的 AudioSegment，不再需要 ffmpeg 再次处理
                chunk_audio.export(output_filepath, format="wav")
                info_data.append({
                    "file_path": output_filepath,
                    "original_start_time": chunk_info['start'],
                    "original_end_time": chunk_info['end'],
                    "duration": chunk_info['end'] - chunk_info['start']
                })
            
            # 返回对象（return 或 return_and_save 模式）
            if operation_mode in {"return", "return_and_save"}:
                returned_segments.append({
                    "filename": output_filename,
                    "audio_segment": chunk_audio, # 返回的是已归一化的对象
                    "start": chunk_info['start'],
                    "end": chunk_info['end']
                })
        
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
        # 创建一个 44.1kHz Stereo 的文件来测试归一化效果
        AudioSegment.silent(duration=60000).set_frame_rate(44100).set_channels(2).export(audio_file_to_process, format="wav")
    try:
        splitter.split(
            audio_path=audio_file_to_process,
            output_dir=r"audio_split_output_save",
            skip_vad=False,
            target_length=12.0,
            max_length=15.0,
            overlap_length=0.0, # 测试重叠
            normalize_audio=True, # 测试归一化
            operation_mode="save"
        )
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    # --- 演示：返回模式 ---
    print("\n" + "="*50)
    print("场景: 返回模式（仅返回对象，不保存文件）")
    print("="*50)
    try:
        segments = splitter.split(
            audio_path=audio_file_to_process,
            output_dir=None,
            skip_vad=False,
            target_length=12.0,
            max_length=15.0,
            overlap_length=0.0,
            normalize_audio=True,
            operation_mode="return"
        )
        print(f"成功返回 {len(segments)} 个音频片段对象。")
        if segments:
            seg0 = segments[0]['audio_segment']
            print(f"第一个片段: {segments[0]['filename']}")
            print(f"检查归一化结果: 采样率={seg0.frame_rate}Hz, 通道数={seg0.channels}")
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
            target_length=12.0,
            max_length=15.0,
            overlap_length=0.0,
            normalize_audio=True,
            operation_mode="return_and_save"
        )
        print(f"成功返回 {len(segments)} 个音频片段对象，并已保存至文件。")
        if segments:
            print(f"第一个片段长度: {segments[0]['audio_segment'].duration_seconds:.2f}s")
    except Exception as e:
        print(f"\n程序运行出错: {e}")