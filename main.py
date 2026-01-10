"""
视频/图片处理器
监控下载文件夹，自动提取视频最后一帧，自动切割分镜图片
"""
import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Gemini 水印去除模块
try:
    from gemini_watermark import remove_watermark, is_gemini_image
    HAS_WATERMARK_REMOVER = True
except ImportError:
    HAS_WATERMARK_REMOVER = False

CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "watch_folder": str(Path.home() / "Downloads"),
    "output_folder": str(Path.home() / "Downloads" / "LastFrames"),
    "video_extensions": [".mp4", ".mov", ".webm", ".avi", ".mkv"],
    "image_extensions": [".jpg", ".jpeg", ".png", ".webp"],
    "video_counter": 0,
    "image_counter": 0,
    "enable_grid_split": True,      # 启用分镜切割
    "enable_watermark_remove": True  # 启用去水印
}

# 深色科技风格配色
COLORS = {
    "bg": "#1a1a2e",
    "bg_light": "#16213e",
    "accent": "#0f3460",
    "highlight": "#00d9ff",
    "text": "#e0e0e0",
    "text_dim": "#888888",
    "success": "#00ff88",
    "error": "#ff4757"
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class GridDetector:
    """
    智能网格分镜检测和切割
    完整移植自 ComfyUI-AutoSplitGridImage
    使用 HSV+LAB 色彩空间 + Canny 边缘检测 + 精确边界调整
    """
    
    @staticmethod
    def _read_image(image_path):
        """读取图片，支持中文路径"""
        # cv2.imread 不支持中文路径，用 numpy + PIL 读取
        img_pil = Image.open(image_path)
        img_rgb = np.array(img_pil.convert('RGB'))
        # 转成 BGR 给 cv2 用
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    
    @staticmethod
    def detect_grid(image_path, logger=None):
        """
        检测图片的网格布局
        返回 (rows, cols, h_splits, v_splits) 或 None
        h_splits/v_splits 是分割线区域的 (start, end) 列表
        """
        if not HAS_CV2:
            return GridDetector._detect_simple(image_path)
        
        try:
            img = GridDetector._read_image(image_path)
        except:
            return None
        
        if img is None:
            return None
        
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
        
        # 方法：检测高亮度行/列（分割线是白色的）
        row_mean = np.mean(gray, axis=1)
        col_mean = np.mean(gray, axis=0)
        
        # 找水平分割线（亮度>200的连续行）
        h_splits = GridDetector._find_bright_regions(row_mean, height, threshold=200)
        
        # 找垂直分割线
        v_splits = GridDetector._find_bright_regions(col_mean, width, threshold=200)
        
        rows = len(h_splits) + 1
        cols = len(v_splits) + 1
        
        if rows == 1 and cols == 1:
            # 没找到分割线，尝试用 Canny 边缘检测
            h_splits_pos = GridDetector._find_split_positions(img_rgb, height, is_vertical=False)
            v_splits_pos = GridDetector._find_split_positions(img_rgb, width, is_vertical=True)
            
            # 转换为区域格式 (pos-10, pos+10)
            h_splits = [(max(0, p-10), min(height, p+10)) for p in h_splits_pos]
            v_splits = [(max(0, p-10), min(width, p+10)) for p in v_splits_pos]
            
            rows = len(h_splits) + 1
            cols = len(v_splits) + 1
        
        if rows == 1 and cols == 1:
            return None
        
        if rows > 4 or cols > 4:
            return None
        
        return (rows, cols, h_splits, v_splits)
    
    @staticmethod
    def _find_bright_regions(line_mean, total_size, threshold=200, min_width=8):
        """
        找高亮度区域（分割线）
        返回 [(start, end), ...] 列表
        """
        regions = []
        min_gap = total_size * 0.15  # 最小间距
        
        i = int(total_size * 0.1)
        end_search = int(total_size * 0.9)
        
        while i < end_search:
            if line_mean[i] > threshold:
                # 找到亮区域的起点
                start = i
                while i < end_search and line_mean[i] > threshold * 0.8:
                    i += 1
                end = i
                
                # 检查宽度和间距
                if (end - start) >= min_width:
                    if not regions or (start - regions[-1][1]) > min_gap:
                        regions.append((start, end))
            else:
                i += 1
        
        return regions
    
    @staticmethod
    def _find_split_positions(img_rgb, size, is_vertical):
        """
        使用 Canny 边缘检测找分割线位置
        找边缘密度最低的位置（分割线处边缘少）
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        if is_vertical:
            edge_density = np.sum(edges, axis=0)  # 每列的边缘密度
        else:
            edge_density = np.sum(edges, axis=1)  # 每行的边缘密度
        
        # 平滑处理
        window_size = max(20, len(edge_density) // 20)
        smoothed = np.convolve(edge_density, np.ones(window_size)/window_size, mode='same')
        
        # 找局部最小值
        splits = []
        min_gap = size * 0.2  # 最小间距20%
        threshold = np.mean(smoothed) * 0.5  # 阈值
        
        # 在中间区域搜索
        start_search = int(size * 0.15)
        end_search = int(size * 0.85)
        
        i = start_search
        while i < end_search:
            if smoothed[i] < threshold:
                # 找这个低谷区域的最小点
                min_val = smoothed[i]
                min_pos = i
                while i < end_search and smoothed[i] < threshold * 2:
                    if smoothed[i] < min_val:
                        min_val = smoothed[i]
                        min_pos = i
                    i += 1
                
                # 检查间距
                if not splits or (min_pos - splits[-1]) > min_gap:
                    splits.append(min_pos)
            else:
                i += 1
        
        # 最多3条分割线（4x4网格）
        if len(splits) > 3:
            # 保留边缘密度最低的3条
            split_scores = [(s, smoothed[s]) for s in splits]
            split_scores.sort(key=lambda x: x[1])
            splits = sorted([s[0] for s in split_scores[:3]])
        
        return splits
    
    @staticmethod
    def _detect_simple(image_path):
        """无 OpenCV 时的简单检测"""
        try:
            img = Image.open(image_path)
            gray = np.array(img.convert('L'))
            height, width = gray.shape
        except:
            return None
        
        row_means = np.mean(gray, axis=1)
        col_means = np.mean(gray, axis=0)
        
        h_splits = []
        v_splits = []
        
        for i in range(int(height * 0.1), int(height * 0.9)):
            if row_means[i] > 220 or row_means[i] < 40:
                if not h_splits or (i - h_splits[-1]) > height * 0.1:
                    h_splits.append(i)
        
        for i in range(int(width * 0.1), int(width * 0.9)):
            if col_means[i] > 220 or col_means[i] < 40:
                if not v_splits or (i - v_splits[-1]) > width * 0.1:
                    v_splits.append(i)
        
        if not h_splits and not v_splits:
            return None
        
        return (len(h_splits) + 1, len(v_splits) + 1, h_splits, v_splits)
    
    @staticmethod
    def _is_black_region(region):
        """检查区域是否为黑色区域"""
        if len(region.shape) == 3:
            region_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            region_gray = region
        dark_ratio = np.mean(region_gray < 40)
        return dark_ratio > 0.6

    @staticmethod
    def _is_white_region(region):
        """检查区域是否为白色区域"""
        if len(region.shape) == 3:
            region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            sat_mean = np.mean(region_hsv[:,:,1])
            val_mean = np.mean(region_hsv[:,:,2])
            return sat_mean < 30 and val_mean > 225
        return False

    @staticmethod
    def _detect_split_borders(img_strip, is_vertical=True):
        """
        检测分割线区域的边框（HSV+LAB色彩空间）
        """
        hsv = cv2.cvtColor(img_strip, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_strip, cv2.COLOR_RGB2LAB)
        
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        l_channel = lab[:, :, 0]
        
        # 检测白色和黑色区域: (sat < 10 & val > 248) | (l_channel < 30)
        is_border = ((sat < 10) & (val > 248)) | (l_channel < 30)
        
        if is_vertical:
            border_ratios = np.mean(is_border, axis=1)
            indices = np.where(border_ratios < 0.95)[0]
        else:
            border_ratios = np.mean(is_border, axis=0)
            indices = np.where(border_ratios < 0.95)[0]
            
        if len(indices) == 0:
            return 0, img_strip.shape[1] if is_vertical else img_strip.shape[0]
            
        return indices[0], indices[-1]

    @staticmethod
    def _adjust_split_line(img_np, split_pos, is_vertical=True, margin=15):
        """
        调整分割线附近的边界
        返回 (left_bound, right_bound) 或 (top_bound, bottom_bound)
        """
        height, width = img_np.shape[:2]
        
        if is_vertical:
            left_bound = max(0, split_pos - margin)
            right_bound = min(width, split_pos + margin)
            strip = img_np[:, left_bound:right_bound]
            start, end = GridDetector._detect_split_borders(strip, False)
            
            start = max(0, start - 5)
            end = min(strip.shape[1], end + 5)
            
            return left_bound + start, left_bound + end
        else:
            top_bound = max(0, split_pos - margin)
            bottom_bound = min(height, split_pos + margin)
            strip = img_np[top_bound:bottom_bound, :]
            start, end = GridDetector._detect_split_borders(strip, True)
            
            start = max(0, start - 5)
            end = min(strip.shape[0], end + 5)
            
            return top_bound + start, top_bound + end

    @staticmethod
    def _remove_external_borders(img_np):
        """
        去除图片四周的白边/黑边
        完整移植自 ComfyUI-AutoSplitGridImage
        使用 chunk-based 检测（纯numpy实现）
        """
        if img_np.size == 0 or img_np is None:
            return img_np

        # 转灰度
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        height, width = img_np.shape[:2]
        
        min_trim = 18
        check_width = 30
        chunk_size = 5

        def find_border(is_vertical=True, from_start=True):
            """查找边界"""
            total_size = width if is_vertical else height

            if from_start:
                range_iter = range(0, total_size - chunk_size, chunk_size)
            else:
                range_iter = range(total_size - chunk_size, 0, -chunk_size)

            for i in range_iter:
                if is_vertical:
                    chunk = img_np[:, i:i+chunk_size] if from_start else img_np[:, i-chunk_size:i]
                else:
                    chunk = img_np[i:i+chunk_size, :] if from_start else img_np[i-chunk_size:i, :]
                
                # 分别检查黑边和白边
                if not (GridDetector._is_black_region(chunk) or GridDetector._is_white_region(chunk)):
                    return i if from_start else i
                    
            return min_trim if from_start else total_size - min_trim

        # 检测四个边界
        left = find_border(is_vertical=True, from_start=True)
        right = find_border(is_vertical=True, from_start=False)
        top = find_border(is_vertical=False, from_start=True)
        bottom = find_border(is_vertical=False, from_start=False)

        # 强制应用最小裁剪（只检测黑边，ComfyUI原版逻辑）
        left_region = gray[:, :check_width]
        if np.mean(left_region < 40) > 0.3:
            left = max(left, min_trim)

        right_region = gray[:, -check_width:]
        if np.mean(right_region < 40) > 0.3:
            right = min(right, width - min_trim)

        top_region = gray[:check_width, :]
        if np.mean(top_region < 40) > 0.3:
            top = max(top, min_trim)

        bottom_region = gray[-check_width:, :]
        if np.mean(bottom_region < 40) > 0.3:
            bottom = min(bottom, height - min_trim)

        # 确保裁剪合理
        if (right - left) < width * 0.5 or (bottom - top) < height * 0.5:
            return img_np

        cropped = img_np[top:bottom, left:right]
        
        # 二次检查右边缘（只检测黑边）
        if cropped.shape[1] > 2 * min_trim:
            right_edge = cv2.cvtColor(cropped[:, -min_trim:], cv2.COLOR_RGB2GRAY)
            if np.mean(right_edge < 40) > 0.3:
                cropped = cropped[:, :-min_trim]
        
        # 二次检查左边缘（只检测黑边）
        if cropped.shape[1] > 2 * min_trim:
            left_edge = cv2.cvtColor(cropped[:, :min_trim], cv2.COLOR_RGB2GRAY)
            if np.mean(left_edge < 40) > 0.3:
                cropped = cropped[:, min_trim:]
                
        return cropped

    @staticmethod
    def split_image(image_path, output_dir, base_name, grid_info, logger=None):
        """
        使用检测到的分割线位置切割图片
        """
        def log(msg):
            if logger:
                logger(msg)
            print(msg)
        
        log(f"[切割] 开始: h_splits={grid_info[2]}, v_splits={grid_info[3]}")
        
        if not HAS_CV2:
            log("[切割] 错误: 没有cv2")
            return []
        
        rows, cols, h_splits, v_splits = grid_info
        
        try:
            img = GridDetector._read_image(image_path)
            log(f"[切割] 读取图片成功")
        except Exception as e:
            log(f"[切割] 读取图片失败: {e}")
            return []
        
        if img is None:
            log("[切割] 图片为空")
            return []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img_rgb.shape[:2]
        log(f"[切割] 图片尺寸: {width}x{height}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建内容区域边界
        # h_splits 是 [(start, end), ...] 格式，表示分割线区域
        # 内容区域是分割线之间的部分
        h_bounds = [0]
        for item in h_splits:
            if isinstance(item, tuple):
                start, end = item
            else:
                # 兼容单数字格式
                start, end = item, item + 20
            h_bounds.append(start)
            h_bounds.append(end)
        h_bounds.append(height)
        
        v_bounds = [0]
        for item in v_splits:
            if isinstance(item, tuple):
                start, end = item
            else:
                start, end = item, item + 20
            v_bounds.append(start)
            v_bounds.append(end)
        v_bounds.append(width)
        
        log(f"[切割] h_bounds={h_bounds}")
        log(f"[切割] v_bounds={v_bounds}")
        
        saved_files = []
        idx = 1
        
        # 遍历内容区域（跳过分割线区域）
        for i in range(0, len(h_bounds) - 1, 2):
            for j in range(0, len(v_bounds) - 1, 2):
                top = h_bounds[i]
                bottom = h_bounds[i + 1]
                left = v_bounds[j]
                right = v_bounds[j + 1]
                
                log(f"[切割] 区域{idx}: top={top}, bottom={bottom}, left={left}, right={right}")
                
                # 切割
                cell = img_rgb[top:bottom, left:right]
                
                if cell.size == 0:
                    log(f"[切割] 区域{idx}: 空，跳过")
                    continue
                
                log(f"[切割] 区域{idx}: 切割后尺寸 {cell.shape}")
                
                # 去除外部边框
                cell = GridDetector._remove_external_borders(cell)
                
                if cell.size == 0:
                    log(f"[切割] 区域{idx}: 去边框后为空，跳过")
                    continue
                
                log(f"[切割] 区域{idx}: 去边框后尺寸 {cell.shape}")
                
                # 保存
                cell_pil = Image.fromarray(cell)
                output_path = output_dir / f"{base_name}_grid_{idx}.jpg"
                cell_pil.save(output_path, "JPEG", quality=95)
                saved_files.append(output_path)
                log(f"[切割] 区域{idx}: 已保存 {output_path.name}")
                idx += 1
        
        log(f"[切割] 完成，共 {len(saved_files)} 张")
        return saved_files


class VideoHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app
        self._processed = set()
        self._lock = threading.Lock()
    
    def _try_process(self, filepath):
        """尝试处理文件"""
        if isinstance(filepath, bytes):
            filepath = filepath.decode('utf-8')
        
        # 跳过临时文件
        lower_path = filepath.lower()
        if any(lower_path.endswith(x) for x in ['.tmp', '.crdownload', '.part', '.partial']):
            return
        
        ext = os.path.splitext(filepath)[1].lower()
        is_video = ext in self.app.config["video_extensions"]
        is_image = ext in self.app.config.get("image_extensions", [])
        
        if not is_video and not is_image:
            return
        
        # 防重复
        with self._lock:
            if filepath in self._processed:
                return
            self._processed.add(filepath)
        
        self.app.log(f"[检测到] {os.path.basename(filepath)}")
        
        if is_video:
            threading.Thread(target=self.process_video, args=(filepath,), daemon=True).start()
        else:
            threading.Thread(target=self.process_image, args=(filepath,), daemon=True).start()
    
    def on_created(self, event):
        if not event.is_directory:
            self._try_process(event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory:
            self._try_process(event.dest_path)
    
    def on_modified(self, event):
        if not event.is_directory:
            self._try_process(event.src_path)
    
    def process_video(self, filepath):
        filename = os.path.basename(filepath)
        self.app.log(f"[视频] {filename}")
        
        if not self.wait_for_file(filepath):
            self.app.log(f"[错误] 文件不可用", error=True)
            return
        
        self.app.log(f"[处理] 提取中...")
        output_path = self.extract_frames(filepath)
        
        if output_path:
            self.app.log(f"[完成] {os.path.basename(output_path)}", success=True)
        else:
            self.app.log(f"[失败] 提取出错", error=True)
    
    def process_image(self, filepath):
        """处理图片：去除 Gemini 水印，然后检测网格并切割"""
        filename = os.path.basename(filepath)
        self.app.log(f"[图片] {filename}")
        
        if not self.wait_for_file(filepath):
            self.app.log(f"[错误] 文件不可用", error=True)
            return
        
        try:
            output_dir = Path(self.app.config["output_folder"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取序号
            counter = self.app.config.get("image_counter", 0) + 1
            self.app.config["image_counter"] = counter
            save_config(self.app.config)
            base_name = f"{counter:03d}"
            
            img = Image.open(filepath)
            processed_img = img
            ext = os.path.splitext(filepath)[1]
            
            # 1. 去水印
            if self.app.config.get("enable_watermark_remove", True) and HAS_WATERMARK_REMOVER:
                if is_gemini_image(img):
                    self.app.log(f"[水印] 正在去除...")
                    processed_img = remove_watermark(img)
            
            # 2. 检测分镜并切割
            if self.app.config.get("enable_grid_split", True):
                # 先保存去水印后的图片用于检测
                temp_path = output_dir / f"_temp_{base_name}{ext}"
                processed_img.save(temp_path, quality=95)
                
                grid_info = GridDetector.detect_grid(str(temp_path))
                
                if grid_info:
                    rows, cols, h_splits, v_splits = grid_info
                    self.app.log(f"[分镜] {rows}行x{cols}列 (水平线:{h_splits} 垂直线:{v_splits})")
                    
                    # 创建子目录
                    grid_dir = output_dir / f"{base_name}_grid"
                    grid_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 切割图片
                    saved = GridDetector.split_image(str(temp_path), str(grid_dir), base_name, grid_info)
                    
                    # 保存原图（去水印后的）
                    import shutil
                    shutil.move(str(temp_path), grid_dir / f"{base_name}_source{ext}")
                    
                    self.app.log(f"[完成] 切割为 {len(saved)} 张", success=True)
                else:
                    # 不是分镜图，直接保存去水印后的图片
                    final_path = output_dir / f"{base_name}_nowm{ext}"
                    temp_path.rename(final_path)
                    self.app.log(f"[完成] 已保存: {final_path.name}", success=True)
            else:
                # 分镜关闭，只保存去水印后的图片
                output_path = output_dir / f"{base_name}_nowm{ext}"
                processed_img.save(output_path, quality=95)
                self.app.log(f"[完成] 已保存: {output_path.name}", success=True)
                
        except Exception as e:
            self.app.log(f"[错误] {e}", error=True)
    
    def wait_for_file(self, filepath, timeout=60):
        start_time = time.time()
        last_size = -1
        stable_count = 0
        
        while time.time() - start_time < timeout:
            try:
                if not os.path.exists(filepath):
                    return False
                current_size = os.path.getsize(filepath)
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 3:
                        return True
                else:
                    stable_count = 0
                last_size = current_size
                time.sleep(1)
            except:
                time.sleep(1)
        return False
    
    def get_video_info(self, video_path):
        """获取视频时长和帧率"""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration,r_frame_rate",
            "-of", "json", video_path
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            import json as js
            data = js.loads(result.stdout)
            stream = data["streams"][0]
            duration = float(stream.get("duration", 0))
            fps_str = stream.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30
            return duration, fps
        except:
            return 0, 30
    
    def get_next_number(self):
        """获取下一个序号"""
        counter = self.app.config.get("video_counter", 0) + 1
        self.app.config["video_counter"] = counter
        save_config(self.app.config)
        return counter
    
    def extract_frames(self, video_path):
        """提取最后一帧 + 最后2秒所有帧 + 复制原视频"""
        try:
            import shutil
            
            output_dir = Path(self.app.config["output_folder"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_ext = os.path.splitext(video_path)[1]  # 保留原扩展名
            
            # 获取序号，数字在前面
            num = self.get_next_number()
            base_name = f"{num:03d}"  # 001, 002, 003...
            
            # 1. 提取最后一帧（单独放根目录）
            last_frame_path = output_dir / f"{base_name}_last.jpg"
            cmd_last = [
                "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
                "-vsync", "0", "-q:v", "2", "-update", "1", str(last_frame_path)
            ]
            subprocess.run(
                cmd_last, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            # 2. 获取视频信息
            duration, fps = self.get_video_info(video_path)
            if duration <= 0:
                self.app.log(f"[警告] 无法获取视频时长", error=True)
                return str(last_frame_path) if last_frame_path.exists() else None
            
            # 3. 创建子文件夹存放最后2秒帧
            tail_dir = output_dir / f"{base_name}_tail"
            tail_dir.mkdir(parents=True, exist_ok=True)
            
            # 4. 复制原视频到子文件夹（原封不动）
            video_copy_path = tail_dir / f"{base_name}_source{video_ext}"
            shutil.copy2(video_path, video_copy_path)
            self.app.log(f"[复制] 原视频已复制")
            
            # 5. 提取最后2秒所有帧到临时位置
            tail_seconds = min(2, duration)  # 如果视频不足2秒就取全部
            start_time = max(0, duration - tail_seconds)
            
            # 先提取到临时文件 frame_0001.jpg, frame_0002.jpg ...
            temp_pattern = str(tail_dir / "temp_%04d.jpg")
            cmd_tail = [
                "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
                "-q:v", "2", temp_pattern
            ]
            subprocess.run(
                cmd_tail, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            
            # 6. 重命名为倒数格式 -Xs_YY.jpg
            temp_files = sorted(tail_dir.glob("temp_*.jpg"))
            total_frames = len(temp_files)
            frames_per_sec = int(fps)
            
            for i, temp_file in enumerate(temp_files):
                # 计算这是倒数第几秒第几帧
                reverse_idx = total_frames - 1 - i  # 从最后往前数
                sec_from_end = reverse_idx // frames_per_sec + 1  # 倒数第几秒
                frame_in_sec = (reverse_idx % frames_per_sec) + 1  # 该秒内第几帧
                
                new_name = f"-{sec_from_end}s_{frame_in_sec:02d}.jpg"
                new_path = tail_dir / new_name
                temp_file.rename(new_path)
            
            frame_count = len(list(tail_dir.glob("*.jpg")))
            self.app.log(f"[提取] 尾帧 + {frame_count}帧(最后{tail_seconds:.1f}秒)")
            
            return str(last_frame_path) if last_frame_path.exists() else None
        except Exception as e:
            self.app.log(f"[错误] {e}", error=True)
            return None


class App:
    def __init__(self):
        self.config = load_config()
        self.observer = None
        self.running = False
        
        self.root = tk.Tk()
        self.root.title("媒体处理器")
        self.root.geometry("420x380")
        self.root.resizable(False, False)
        self.root.configure(bg=COLORS["bg"])
        self.root.attributes("-topmost", True)
        
        self.setup_ui()
        
    def setup_ui(self):
        # 标题
        title = tk.Label(
            self.root, text="媒体处理器", 
            font=("Microsoft YaHei UI", 14, "bold"),
            fg=COLORS["highlight"], bg=COLORS["bg"]
        )
        title.pack(pady=(12, 8))
        
        # 设置区域
        settings_frame = tk.Frame(self.root, bg=COLORS["bg_light"], padx=15, pady=10)
        settings_frame.pack(fill=tk.X, padx=15)
        
        # 监控文件夹
        tk.Label(
            settings_frame, text="监控目录", 
            font=("Microsoft YaHei UI", 9), fg=COLORS["text_dim"], bg=COLORS["bg_light"]
        ).grid(row=0, column=0, sticky=tk.W)
        
        self.watch_var = tk.StringVar(value=self.config["watch_folder"])
        watch_entry = tk.Entry(
            settings_frame, textvariable=self.watch_var, width=32,
            font=("Consolas", 9), bg=COLORS["accent"], fg=COLORS["text"],
            insertbackground=COLORS["text"], relief=tk.FLAT, bd=5
        )
        watch_entry.grid(row=0, column=1, padx=(10, 5))
        
        watch_btn = tk.Button(
            settings_frame, text="...", width=3,
            font=("Consolas", 9), bg=COLORS["accent"], fg=COLORS["highlight"],
            relief=tk.FLAT, cursor="hand2",
            command=lambda: self.browse_folder("watch")
        )
        watch_btn.grid(row=0, column=2)
        
        # 输出文件夹
        tk.Label(
            settings_frame, text="输出目录", 
            font=("Microsoft YaHei UI", 9), fg=COLORS["text_dim"], bg=COLORS["bg_light"]
        ).grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        
        self.output_var = tk.StringVar(value=self.config["output_folder"])
        output_entry = tk.Entry(
            settings_frame, textvariable=self.output_var, width=32,
            font=("Consolas", 9), bg=COLORS["accent"], fg=COLORS["text"],
            insertbackground=COLORS["text"], relief=tk.FLAT, bd=5
        )
        output_entry.grid(row=1, column=1, padx=(10, 5), pady=(8, 0))
        
        output_btn = tk.Button(
            settings_frame, text="...", width=3,
            font=("Consolas", 9), bg=COLORS["accent"], fg=COLORS["highlight"],
            relief=tk.FLAT, cursor="hand2",
            command=lambda: self.browse_folder("output")
        )
        output_btn.grid(row=1, column=2, pady=(8, 0))
        
        # 功能开关区域
        toggle_frame = tk.Frame(self.root, bg=COLORS["bg"])
        toggle_frame.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        # 去水印开关
        self.watermark_var = tk.BooleanVar(value=self.config.get("enable_watermark_remove", True))
        watermark_cb = tk.Checkbutton(
            toggle_frame, text="去水印", variable=self.watermark_var,
            font=("Microsoft YaHei UI", 9), fg=COLORS["text"], bg=COLORS["bg"],
            selectcolor=COLORS["accent"], activebackground=COLORS["bg"],
            activeforeground=COLORS["highlight"], cursor="hand2",
            command=self.save_settings
        )
        watermark_cb.pack(side=tk.LEFT)
        
        # 分镜切割开关
        self.grid_var = tk.BooleanVar(value=self.config.get("enable_grid_split", True))
        grid_cb = tk.Checkbutton(
            toggle_frame, text="分镜切割", variable=self.grid_var,
            font=("Microsoft YaHei UI", 9), fg=COLORS["text"], bg=COLORS["bg"],
            selectcolor=COLORS["accent"], activebackground=COLORS["bg"],
            activeforeground=COLORS["highlight"], cursor="hand2",
            command=self.save_settings
        )
        grid_cb.pack(side=tk.LEFT, padx=(15, 0))
        
        # 控制按钮
        btn_frame = tk.Frame(self.root, bg=COLORS["bg"])
        btn_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.start_btn = tk.Button(
            btn_frame, text="开始监控", width=10,
            font=("Microsoft YaHei UI", 10, "bold"), bg=COLORS["accent"], fg=COLORS["highlight"],
            relief=tk.FLAT, cursor="hand2", command=self.toggle_monitor
        )
        self.start_btn.pack(side=tk.LEFT)
        
        open_btn = tk.Button(
            btn_frame, text="打开输出", width=10,
            font=("Microsoft YaHei UI", 10), bg=COLORS["accent"], fg=COLORS["text"],
            relief=tk.FLAT, cursor="hand2", command=self.open_output
        )
        open_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(
            btn_frame, text="未运行",
            font=("Microsoft YaHei UI", 10), fg=COLORS["text_dim"], bg=COLORS["bg"]
        )
        self.status_label.pack(side=tk.RIGHT)
        
        # 日志区域
        log_frame = tk.Frame(self.root, bg=COLORS["bg_light"])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.log_text = tk.Text(
            log_frame, height=8, width=50,
            font=("Consolas", 9), bg=COLORS["bg_light"], fg=COLORS["text"],
            relief=tk.FLAT, bd=10, insertbackground=COLORS["text"]
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def browse_folder(self, folder_type):
        # 根据类型选择当前设置的目录的父目录，这样对话框会显示当前目录可供选择
        if folder_type == "watch":
            current = self.watch_var.get()
        else:
            current = self.output_var.get()
        
        # 用父目录作为初始位置，这样当前目录会显示在列表中
        parent = str(Path(current).parent)
        if not os.path.exists(parent):
            parent = str(Path.home())
        
        folder = filedialog.askdirectory(initialdir=parent)
        if folder:
            if folder_type == "watch":
                self.watch_var.set(folder)
            else:
                self.output_var.set(folder)
            self.save_settings()
    
    def save_settings(self):
        self.config["watch_folder"] = self.watch_var.get()
        self.config["output_folder"] = self.output_var.get()
        self.config["enable_watermark_remove"] = self.watermark_var.get()
        self.config["enable_grid_split"] = self.grid_var.get()
        save_config(self.config)
    
    def toggle_monitor(self):
        if self.running:
            self.stop_monitor()
        else:
            self.start_monitor()
    
    def start_monitor(self):
        self.save_settings()
        watch_folder = self.config["watch_folder"]
        
        if not os.path.exists(watch_folder):
            messagebox.showerror("错误", f"监控目录不存在:\n{watch_folder}")
            return
        
        os.makedirs(self.config["output_folder"], exist_ok=True)
        
        self.observer = Observer()
        handler = VideoHandler(self)
        self.observer.schedule(handler, watch_folder, recursive=False)
        self.observer.start()
        
        self.running = True
        self.start_btn.config(text="停止监控", fg=COLORS["error"])
        self.status_label.config(text="监控中", fg=COLORS["success"])
        self.log(f"[启动] 监控: {watch_folder}")
    
    def stop_monitor(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.running = False
        self.start_btn.config(text="开始监控", fg=COLORS["highlight"])
        self.status_label.config(text="未运行", fg=COLORS["text_dim"])
        self.log("[停止] 已停止监控")
    
    def open_output(self):
        output_folder = self.config["output_folder"]
        os.makedirs(output_folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(output_folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", output_folder])
        else:
            subprocess.run(["xdg-open", output_folder])
    
    def log(self, message, success=False, error=False):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        
        line_start = self.log_text.index("end-2l linestart")
        line_end = self.log_text.index("end-1l")
        
        if success:
            self.log_text.tag_add("success", line_start, line_end)
            self.log_text.tag_config("success", foreground=COLORS["success"])
        elif error:
            self.log_text.tag_add("error", line_start, line_end)
            self.log_text.tag_config("error", foreground=COLORS["error"])
            
        self.log_text.see(tk.END)
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        self.stop_monitor()
        self.root.destroy()


def main():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
    except:
        messagebox.showerror("错误", "未找到 FFmpeg!\n请安装 FFmpeg 并添加到系统 PATH")
        return
    
    app = App()
    app.run()


if __name__ == "__main__":
    main()
