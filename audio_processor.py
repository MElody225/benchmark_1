#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频预处理模块
将音频文件转换为Mel频谱图，用于后续训练
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频预处理器"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 10.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = "hann",
        fmin: float = 0,
        fmax: Optional[float] = None,
        normalize: bool = True
    ):
        """
        参数:
            sample_rate: 目标采样率 (Hz)
            duration: 音频时长（秒），不足则填充，超过则截断
            n_mels: Mel频谱通道数
            n_fft: FFT窗口大小
            hop_length: 帧移
            win_length: 窗口长度
            window: 窗函数类型
            fmin: 最小频率
            fmax: 最大频率（None则为sample_rate/2）
            normalize: 是否归一化
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.window = window
        self.fmin = fmin
        self.fmax = fmax if fmax else sample_rate / 2.0
        self.normalize = normalize
        
        # 计算期望的样本数
        self.expected_samples = int(sample_rate * duration)
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        加载音频文件
        
        返回:
            audio: (n_samples,) 单声道音频，采样率为self.sample_rate
        """
        audio_path = Path(audio_path)
        
        # 检查文件是否存在
        if not audio_path.exists():
            logger.error(f"音频文件不存在: {audio_path}")
            return None
        
        if not audio_path.is_file():
            logger.error(f"路径不是文件: {audio_path}")
            return None
        
        try:
            # 使用绝对路径确保librosa能正确读取
            # 加载音频并重采样
            audio, sr = librosa.load(str(audio_path.resolve()), sr=self.sample_rate, mono=True)
            
            if audio is None or len(audio) == 0:
                logger.error(f"音频为空: {audio_path}")
                return None
            
            # 记录音频信息
            duration = len(audio) / self.sample_rate
            logger.debug(f"音频信息: {audio_path.name} - {duration:.2f}s, {len(audio)} samples")
            
            return audio
        
        except Exception as e:
            logger.error(f"加载音频失败 {audio_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """
        填充或截断音频到固定长度
        
        参数:
            audio: (n_samples,)
        
        返回:
            audio: (expected_samples,)
        """
        if len(audio) > self.expected_samples:
            # 截断（从中间截取，保留最重要的部分）
            start = (len(audio) - self.expected_samples) // 2
            audio = audio[start:start + self.expected_samples]
            logger.debug(f"音频截断: {len(audio) + start} -> {len(audio)} samples")
        
        elif len(audio) < self.expected_samples:
            # 填充（零填充到末尾）
            padding = self.expected_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            logger.debug(f"音频填充: {len(audio) - padding} -> {len(audio)} samples")
        
        return audio
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        计算Mel频谱图
        
        参数:
            audio: (n_samples,) 音频信号
        
        返回:
            mel_spec: (n_mels, n_frames) Mel频谱图
        """
        # 计算短时傅里叶变换
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )
        
        # 计算幅度谱
        magnitude = np.abs(stft)
        
        # 转换为Mel频谱
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude ** 2,  # 功率谱
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # 转换为dB尺度（log-Mel）
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def normalize_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        归一化频谱图到[0, 1]
        
        参数:
            mel_spec: (n_mels, n_frames) dB尺度
        
        返回:
            normalized: (n_mels, n_frames) [0, 1]
        """
        # dB范围通常是[-80, 0]
        # 归一化到[0, 1]
        min_val = mel_spec.min()
        max_val = mel_spec.max()
        
        # 处理全零或常数值的情况
        if max_val - min_val < 1e-8:
            # 如果所有值相同，返回零矩阵或保持原值
            if abs(max_val) < 1e-8:
                return np.zeros_like(mel_spec)
            else:
                # 常数值归一化到0.5
                return np.ones_like(mel_spec) * 0.5
        
        mel_spec_norm = (mel_spec - min_val) / (max_val - min_val)
        return mel_spec_norm
    
    def resize_spectrogram(self, mel_spec: np.ndarray, target_width: int = 128) -> np.ndarray:
        """
        调整频谱图宽度（时间维度）
        
        参数:
            mel_spec: (n_mels, n_frames)
            target_width: 目标宽度
        
        返回:
            resized: (n_mels, target_width)
        """
        from scipy.ndimage import zoom
        
        current_width = mel_spec.shape[1]
        
        if current_width == target_width:
            return mel_spec
        
        # 计算缩放因子
        zoom_factor = target_width / current_width
        
        # 调整大小（仅时间轴）
        resized = zoom(mel_spec, (1.0, zoom_factor), order=1)
        
        return resized
    
    def process_single_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        target_shape: Tuple[int, int] = (128, 128)
    ) -> Optional[np.ndarray]:
        """
        处理单个音频文件
        
        参数:
            audio_path: 音频文件路径
            output_path: 输出.npy文件路径（可选）
            target_shape: 目标频谱图形状 (height, width)
        
        返回:
            mel_spec: (n_mels, target_width) 或 None（失败时）
        """
        # 加载音频
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        
        # 填充或截断
        audio = self.pad_or_truncate(audio)
        
        # 计算Mel频谱图
        mel_spec = self.compute_mel_spectrogram(audio)
        
        # 归一化
        if self.normalize:
            mel_spec = self.normalize_spectrogram(mel_spec)
        
        # 调整到目标形状
        if target_shape:
            target_height, target_width = target_shape
            
            # 调整高度（频率维度）
            if mel_spec.shape[0] != target_height:
                from scipy.ndimage import zoom
                zoom_factor = target_height / mel_spec.shape[0]
                mel_spec = zoom(mel_spec, (zoom_factor, 1.0), order=1)
            
            # 调整宽度（时间维度）
            mel_spec = self.resize_spectrogram(mel_spec, target_width)
        
        # 保存（如果指定了输出路径）
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, mel_spec.astype(np.float32))
            logger.debug(f"已保存: {output_path}")
        
        return mel_spec
    
    def process_audio_batch(
        self,
        audio_dir: str,
        output_dir: str,
        num_workers: int = 4,
        file_extension: str = "*.wav",
        target_shape: Tuple[int, int] = (128, 128),
        recursive: bool = True
    ):
        """
        批量处理音频目录
        
        参数:
            audio_dir: 音频文件夹路径
            output_dir: 输出文件夹路径
            num_workers: 并行进程数
            file_extension: 音频文件扩展名模式
            target_shape: 目标频谱图形状
            recursive: 是否递归搜索子目录（默认: True）
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有音频文件（支持递归搜索）
        if recursive:
            # 递归搜索所有子目录中的音频文件
            pattern = f"**/{file_extension}" if not file_extension.startswith("**/") else file_extension
            audio_files = list(audio_dir.glob(pattern))
        else:
            # 只搜索当前目录
            audio_files = list(audio_dir.glob(file_extension))
        
        if len(audio_files) == 0:
            logger.error(f"在 {audio_dir} 中未找到音频文件 ({file_extension}, recursive={recursive})")
            return
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        logger.info(f"配置: {self.sample_rate}Hz, {self.duration}s, {self.n_mels} Mel通道")
        logger.info(f"输出形状: {target_shape}")
        
        # 构建输入输出路径对
        tasks = []
        for audio_path in audio_files:
            # 计算相对路径，保持目录结构或使用唯一文件名
            try:
                relative_path = audio_path.relative_to(audio_dir)
                # 将路径中的目录分隔符替换为下划线，创建唯一文件名
                # 例如: S1_C3_E144_V0060_ID1_T1/S1_C3_E144_V0060_ID1_T1_c0.wav 
                # -> S1_C3_E144_V0060_ID1_T1_S1_C3_E144_V0060_ID1_T1_c0.npy
                safe_name = str(relative_path).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
                safe_name = safe_name.replace(".wav", ".npy").replace(".mp3", ".npy").replace(".flac", ".npy")
                output_path = output_dir / safe_name
            except ValueError:
                # 如果无法计算相对路径，使用文件名
                output_path = output_dir / f"{audio_path.stem}.npy"
            
            # 跳过已处理的文件
            if output_path.exists():
                logger.debug(f"已存在，跳过: {output_path}")
                continue
            
            tasks.append((str(audio_path), str(output_path)))
        
        if len(tasks) == 0:
            logger.info("所有音频已处理完成！")
            return
        
        logger.info(f"需要处理 {len(tasks)} 个音频")
        
        # 多进程处理
        success_count = 0
        failed_count = 0
        
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._process_wrapper, audio_path, output_path, target_shape): (audio_path, output_path)
                    for audio_path, output_path in tasks
                }
                
                with tqdm(total=len(tasks), desc="处理音频") as pbar:
                    for future in as_completed(futures):
                        audio_path, output_path = futures[future]
                        try:
                            result = future.result()
                            if result:
                                success_count += 1
                            else:
                                failed_count += 1
                                logger.error(f"处理失败: {audio_path}")
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"处理异常 {audio_path}: {e}")
                        pbar.update(1)
        else:
            # 单进程
            for audio_path, output_path in tqdm(tasks, desc="处理音频"):
                result = self.process_single_audio(audio_path, output_path, target_shape)
                if result is not None:
                    success_count += 1
                else:
                    failed_count += 1
        
        # 输出统计
        logger.info("=" * 70)
        logger.info("处理完成！")
        logger.info(f"✅ 成功: {success_count}")
        logger.info(f"❌ 失败: {failed_count}")
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info("=" * 70)
    
    def _process_wrapper(self, audio_path: str, output_path: str, target_shape: Tuple[int, int]) -> bool:
        """多进程包装函数"""
        try:
            result = self.process_single_audio(audio_path, output_path, target_shape)
            return result is not None
        except Exception as e:
            logger.error(f"处理失败 {audio_path}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="音频预处理工具")
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="音频文件夹路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="目标采样率 (默认: 16000)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="音频时长（秒） (默认: 10.0)"
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=128,
        help="Mel频谱通道数 (默认: 128)"
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=128,
        help="目标频谱图宽度 (默认: 128)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行进程数 (默认: 4)"
    )
    parser.add_argument(
        "--file_extension",
        type=str,
        default="*.wav",
        help="音频文件扩展名 (默认: *.wav)"
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="不递归搜索子目录（默认会递归搜索）"
    )
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels
    )
    
    # 批量处理
    processor.process_audio_batch(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        file_extension=args.file_extension,
        target_shape=(args.n_mels, args.target_width),
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()