#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频预处理模块
提取视频帧，转换为张量格式，用于后续训练
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
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


class VideoProcessor:
    """视频预处理器"""
    
    def __init__(
        self,
        num_frames: int = 16,
        resolution: int = 224,
        sampling_strategy: str = "uniform",
        normalize: bool = True
    ):
        """
        参数:
            num_frames: 每个视频采样的帧数
            resolution: 输出分辨率 (resolution x resolution)
            sampling_strategy: 采样策略 ('uniform', 'random', 'dense')
            normalize: 是否归一化到[0,1]
        """
        self.num_frames = num_frames
        self.resolution = resolution
        self.sampling_strategy = sampling_strategy
        self.normalize = normalize
        
        # ImageNet标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    
    def read_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        读取视频文件
        
        返回:
            frames: (T, H, W, C) numpy数组，BGR格式
        """
        video_path = Path(video_path)
        
        # 检查文件是否存在
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return None
        
        if not video_path.is_file():
            logger.error(f"路径不是文件: {video_path}")
            return None
        
        try:
            # 使用绝对路径确保OpenCV能正确读取
            cap = cv2.VideoCapture(str(video_path.resolve()))
            
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                # 尝试获取更多错误信息
                logger.error(f"文件大小: {video_path.stat().st_size if video_path.exists() else 'N/A'} bytes")
                return None
            
            # 获取视频属性用于验证
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.debug(f"视频信息: {video_path.name} - {frame_count}帧, {fps:.2f}fps, {width}x{height}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                logger.error(f"视频为空或无法读取帧: {video_path}")
                return None
            
            return np.stack(frames, axis=0)  # (T, H, W, C)
        
        except Exception as e:
            logger.error(f"读取视频失败 {video_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        采样固定数量的帧
        
        参数:
            frames: (T, H, W, C) 原始帧
        
        返回:
            sampled_frames: (num_frames, H, W, C)
        """
        total_frames = len(frames)
        
        if total_frames < self.num_frames:
            # 帧数不足，重复最后一帧
            logger.warning(f"视频帧数不足 ({total_frames} < {self.num_frames})，填充最后一帧")
            padding = [frames[-1]] * (self.num_frames - total_frames)
            frames = np.concatenate([frames, np.stack(padding, axis=0)], axis=0)
            total_frames = len(frames)
        
        if self.sampling_strategy == "uniform":
            # 均匀采样
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        elif self.sampling_strategy == "random":
            # 随机采样
            indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))
        
        elif self.sampling_strategy == "dense":
            # 从开头密集采样
            indices = np.arange(min(self.num_frames, total_frames))
            if len(indices) < self.num_frames:
                indices = np.pad(indices, (0, self.num_frames - len(indices)), 
                                constant_values=indices[-1])
        else:
            raise ValueError(f"未知的采样策略: {self.sampling_strategy}")
        
        return frames[indices]
    
    def resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        调整帧大小
        
        参数:
            frames: (T, H, W, C)
        
        返回:
            resized_frames: (T, resolution, resolution, C)
        """
        resized = []
        for frame in frames:
            # OpenCV resize (H, W)
            frame_resized = cv2.resize(
                frame, 
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_LINEAR
            )
            resized.append(frame_resized)
        
        return np.stack(resized, axis=0)
    
    def to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """
        转换为PyTorch张量
        
        参数:
            frames: (T, H, W, C) BGR格式, uint8, [0, 255]
        
        返回:
            tensor: (C, T, H, W) RGB格式, float32, [0, 1] 或 归一化后
        """
        # BGR转RGB
        frames = frames[..., ::-1].copy()  # (T, H, W, C)
        
        # 转换为float并归一化到[0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # 转换为tensor并调整维度: (T, H, W, C) -> (C, T, H, W)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        # ImageNet标准化
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        
        return tensor
    
    def process_single_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """
        处理单个视频
        
        参数:
            video_path: 视频文件路径
            output_path: 输出.pt文件路径（可选）
        
        返回:
            tensor: (C, T, H, W) 或 None（失败时）
        """
        # 读取视频
        frames = self.read_video(video_path)
        if frames is None:
            return None
        
        # 采样帧
        frames = self.sample_frames(frames)
        
        # 调整大小
        frames = self.resize_frames(frames)
        
        # 转换为tensor
        tensor = self.to_tensor(frames)
        
        # 保存（如果指定了输出路径）
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor, output_path)
            logger.debug(f"已保存: {output_path}")
        
        return tensor
    
    def process_video_batch(
        self,
        video_dir: str,
        output_dir: str,
        num_workers: int = 4,
        file_extension: str = "*.mp4",
        recursive: bool = True
    ):
        """
        批量处理视频目录
        
        参数:
            video_dir: 视频文件夹路径
            output_dir: 输出文件夹路径
            num_workers: 并行进程数
            file_extension: 视频文件扩展名模式
            recursive: 是否递归搜索子目录（默认: True）
        """
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有视频文件（支持递归搜索）
        if recursive:
            # 递归搜索所有子目录中的视频文件
            pattern = f"**/{file_extension}" if not file_extension.startswith("**/") else file_extension
            video_files = list(video_dir.glob(pattern))
        else:
            # 只搜索当前目录
            video_files = list(video_dir.glob(file_extension))
        
        if len(video_files) == 0:
            logger.error(f"在 {video_dir} 中未找到视频文件 ({file_extension}, recursive={recursive})")
            return
        
        logger.info(f"找到 {len(video_files)} 个视频文件")
        logger.info(f"配置: {self.num_frames}帧, {self.resolution}x{self.resolution}, {self.sampling_strategy}采样")
        
        # 构建输入输出路径对
        tasks = []
        for video_path in video_files:
            # 计算相对路径，保持目录结构或使用唯一文件名
            try:
                relative_path = video_path.relative_to(video_dir)
                # 将路径中的目录分隔符替换为下划线，创建唯一文件名
                # 例如: S1_C3_E144_V0060_ID1_T1/S1_C3_E144_V0060_ID1_T1_c0.mp4 
                # -> S1_C3_E144_V0060_ID1_T1_S1_C3_E144_V0060_ID1_T1_c0.pt
                safe_name = str(relative_path).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
                safe_name = safe_name.replace(".mp4", ".pt").replace(".avi", ".pt").replace(".mov", ".pt")
                output_path = output_dir / safe_name
            except ValueError:
                # 如果无法计算相对路径，使用文件名
                output_path = output_dir / f"{video_path.stem}.pt"
            
            # 跳过已处理的文件
            if output_path.exists():
                logger.debug(f"已存在，跳过: {output_path}")
                continue
            
            tasks.append((str(video_path), str(output_path)))
        
        if len(tasks) == 0:
            logger.info("所有视频已处理完成！")
            return
        
        logger.info(f"需要处理 {len(tasks)} 个视频")
        
        # 多进程处理
        success_count = 0
        failed_count = 0
        
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(self._process_wrapper, video_path, output_path): (video_path, output_path)
                    for video_path, output_path in tasks
                }
                
                with tqdm(total=len(tasks), desc="处理视频") as pbar:
                    for future in as_completed(futures):
                        video_path, output_path = futures[future]
                        try:
                            result = future.result()
                            if result:
                                success_count += 1
                            else:
                                failed_count += 1
                                logger.error(f"处理失败: {video_path}")
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"处理异常 {video_path}: {e}")
                        pbar.update(1)
        else:
            # 单进程
            for video_path, output_path in tqdm(tasks, desc="处理视频"):
                result = self.process_single_video(video_path, output_path)
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
    
    def _process_wrapper(self, video_path: str, output_path: str) -> bool:
        """多进程包装函数"""
        try:
            result = self.process_single_video(video_path, output_path)
            return result is not None
        except Exception as e:
            logger.error(f"处理失败 {video_path}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="视频预处理工具")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="视频文件夹路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="每个视频采样帧数 (默认: 16)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="输出分辨率 (默认: 224)"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "random", "dense"],
        help="帧采样策略 (默认: uniform)"
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
        default="*.mp4",
        help="视频文件扩展名 (默认: *.mp4)"
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="不递归搜索子目录（默认会递归搜索）"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="不进行ImageNet标准化"
    )
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = VideoProcessor(
        num_frames=args.num_frames,
        resolution=args.resolution,
        sampling_strategy=args.sampling_strategy,
        normalize=not args.no_normalize
    )
    
    # 批量处理
    processor.process_video_batch(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        file_extension=args.file_extension,
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()