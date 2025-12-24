import torch
import torchaudio
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor
from pathlib import Path
import os
import tempfile
from tqdm import tqdm

class GLMASR:
    WHISPER_FEAT_CFG = {
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 128,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }

    def __init__(self, model_path: str, device: str = None, max_new_tokens: int = 128):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.feature_extractor = WhisperFeatureExtractor(**self.WHISPER_FEAT_CFG)
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def get_audio_token_length(self, seconds, merge_factor=2):
        def get_T_after_cnn(L_in, dilation=1):
            for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            return L_out

        mel_len = int(seconds * 100)
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1
        audio_token_num = min(audio_token_num, 1500 // merge_factor)

        return audio_token_num

    def build_prompt(self, audio_path: Path) -> dict:
        audio_path = Path(audio_path)
        wav, sr = torchaudio.load(str(audio_path))
        wav = wav[:1, :]
        if sr != self.feature_extractor.sampling_rate:
            wav = torchaudio.transforms.Resample(sr, self.feature_extractor.sampling_rate)(wav)

        tokens = []
        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\n")

        audios = []
        audio_offsets = []
        audio_length = []
        chunk_size = self.WHISPER_FEAT_CFG['chunk_length'] * self.feature_extractor.sampling_rate
        for start in range(0, wav.shape[1], chunk_size):
            chunk = wav[:, start : start + chunk_size]
            mel = self.feature_extractor(
                chunk.numpy(),
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding="max_length",
            )["input_features"]
            audios.append(mel)
            seconds = chunk.shape[1] / self.feature_extractor.sampling_rate
            num_tokens = self.get_audio_token_length(seconds, self.config.merge_factor)
            tokens += self.tokenizer.encode("<|begin_of_audio|>")
            audio_offsets.append(len(tokens))
            tokens += [0] * num_tokens
            tokens += self.tokenizer.encode("<|end_of_audio|>")
            audio_length.append(num_tokens)

        if not audios:
            raise ValueError("音频内容为空或加载失败。")

        tokens += self.tokenizer.encode("<|user|>")
        tokens += self.tokenizer.encode("\nPlease transcribe this audio into text")

        tokens += self.tokenizer.encode("<|assistant|>")
        tokens += self.tokenizer.encode("\n")

        batch = {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "audios": torch.cat(audios, dim=0),
            "audio_offsets": [audio_offsets],
            "audio_length": [audio_length],
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        }
        return batch

    def prepare_inputs(self, batch: dict) -> tuple[dict, int]:
        tokens = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        audios = batch["audios"].to(self.device)
        model_inputs = {
            "inputs": tokens,
            "attention_mask": attention_mask,
            "audios": audios.to(torch.bfloat16),
            "audio_offsets": batch["audio_offsets"],
            "audio_length": batch["audio_length"],
        }
        return model_inputs, tokens.size(1)

    def transcribe(self, audio_path: str, operation_mode: str = "return") -> str | None:
        audio_path = Path(audio_path)
        wav, sr = torchaudio.load(str(audio_path))
        duration = wav.shape[1] / sr

        if duration > 25.0:
            from audio_splitter import AudioSplitter
            splitter = AudioSplitter()
            segments = splitter.split(
                audio_path=str(audio_path),
                output_dir=None,
                skip_vad=False,
                target_length=20,
                max_length=24,
                overlap_length=0,
                normalize_audio=True,
                norm_processes=4,
                operation_mode="return"
            )

            combined_transcription = []
            for seg in tqdm(segments, desc="Processing audio segments"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    seg["audio_segment"].export(temp_file.name, format="wav")
                    temp_path = temp_file.name

                seg_transcription = self.transcribe(temp_path, operation_mode="return")
                combined_transcription.append(seg_transcription)
                os.unlink(temp_path)

            # Smart merge with universal punctuation check
            if combined_transcription:
                transcription = combined_transcription[0]
                for i in range(1, len(combined_transcription)):
                    # Check if previous segment ends with any punctuation
                    if combined_transcription[i-1].strip() and combined_transcription[i-1].strip()[-1] in [".", "!", "?", "。", "！", "？", "；", "：", "\"", "“", "”", "‘", "’", "，"]:
                        transcription += combined_transcription[i]
                    else:
                        transcription += ',' + combined_transcription[i]
            else:
                transcription = ""
        else:
            batch = self.build_prompt(audio_path)
            model_inputs, prompt_len = self.prepare_inputs(batch)
            with torch.inference_mode():
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            transcript_ids = generated[0, prompt_len:].cpu().tolist()
            transcription = self.tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
        
        if operation_mode in ["save", "return_and_save"]:
            output_path = Path(audio_path).with_suffix('.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
        
        if operation_mode == "return":
            return transcription
        elif operation_mode == "save":
            return None
        else:  # return_and_save
            return transcription

if __name__ == "__main__":
    asr = GLMASR(model_path="models/GLM-ASR-Nano-2512")
    transcription = asr.transcribe("test_audio.wav")
    print("----------")
    print(transcription or "[Empty transcription]")