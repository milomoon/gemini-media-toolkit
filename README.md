# Gemini Media Toolkit

[![Author](https://img.shields.io/badge/Author-Xasia-blue)](https://www.xasia.cc)
[![Website](https://img.shields.io/badge/🌐-www.xasia.cc-green)](https://www.xasia.cc)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A Windows desktop tool for processing Gemini AI generated images and videos.

专为 **AI动漫/AI影视剧** 创作者设计的媒体处理工具。

---

<a href="https://www.xasia.cc">
<img src="docs/author.png" width="100%">
</a>

---

## Features / 功能

- 🎨 **Gemini Watermark Remover / 去水印** - Automatically remove watermarks from Gemini AI generated images using reverse alpha blending algorithm (100% local, no AI needed)
- ✂️ **Grid Image Splitter / 分镜切割** - Auto-detect and split grid/storyboard images into individual frames
- 🎬 **Video Frame Extractor / 视频帧提取** - Extract last frame and tail frames from videos
- 📁 **Folder Monitor / 文件夹监控** - Watch download folder and process new files automatically

## 中文介绍

### 这是什么？

一个专门为 **AI动漫制作** 和 **AI影视剧创作** 设计的效率工具。

当你用 Google Gemini 生成分镜图时，会遇到这些问题：
- 图片带有 Gemini 水印
- 多张分镜合并在一张图里（2x2、3x3 网格）
- 需要手动裁剪每一张

**这个工具帮你一键解决：**

1. **自动去水印** - 使用逆向 Alpha 混合算法，数学计算去除水印，100% 本地运行
2. **自动分镜切割** - 智能检测白色分割线，自动切成单张图片
3. **视频帧提取** - 从 AI 生成的视频中提取关键帧，方便做图生视频的衔接

### 使用场景

- 用 Gemini 生成动漫分镜 → 自动去水印 + 切割
- 用 Veo/Sora/可灵 生成视频 → 提取尾帧做下一段的起始帧
- 批量处理 AI 生成的素材

### 工作流程

```
下载 Gemini 图片 → 工具自动检测 → 去水印 → 检测网格 → 切割保存
```

全程自动，你只需要把图片下载到监控文件夹。

## Keywords

`gemini` `watermark-remover` `watermark-removal` `grid-splitter` `image-splitter` `storyboard` `video-frame-extractor` `google-gemini` `ai-image` `batch-processing` `ai-anime` `ai-movie` `ai-video` `veo` `sora` `kling` `ai动漫` `ai影视` `分镜` `去水印`

## Requirements / 环境要求

- Windows 10/11
- Python 3.8+
- FFmpeg (for video processing)

## Quick Start / 快速开始

### First Time Setup / 首次安装
```bash
# Run install script (creates venv and installs dependencies)
# 双击运行安装脚本
install.bat
```

### Daily Use / 日常使用
```bash
# Double-click to start
# 双击启动
start.vbs
```

## Manual Installation / 手动安装

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

## Install FFmpeg / 安装 FFmpeg

```bash
# Using winget
winget install FFmpeg

# Or download from: https://ffmpeg.org/download.html
```

## Supported Formats / 支持格式

**Images / 图片:** `.jpg` `.jpeg` `.png` `.webp`

**Videos / 视频:** `.mp4` `.mov` `.webm` `.avi` `.mkv`

## How It Works / 原理

### Watermark Removal / 去水印
Uses reverse alpha blending to mathematically remove the semi-transparent Gemini watermark. No AI or cloud service required - runs 100% locally.

使用逆向 Alpha 混合算法，通过数学计算去除半透明水印。无需 AI，无需联网，100% 本地运行。

### Grid Detection / 网格检测
Automatically detects white separator lines in grid/storyboard images and splits them into individual cells.

自动检测分镜图中的白色分割线，智能切割成单张图片。

### Video Processing / 视频处理
Extracts the last frame as a standalone image, plus all frames from the last 2 seconds for review.

提取视频最后一帧作为独立图片，同时提取最后 2 秒的所有帧供选择。

## License

MIT License
