# Gemini Media Toolkit

[![Author](https://img.shields.io/badge/Author-Xasia-blue)](https://www.xasia.cc)
[![Website](https://img.shields.io/badge/🌐-www.xasia.cc-green)](https://www.xasia.cc)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

专为 **AI动漫/AI影视剧** 创作者设计的媒体处理工具

A Windows tool for Gemini AI image/video processing

<a href="https://www.xasia.cc">
<img src="docs/author.png" width="100%">
</a>

---

## ✨ 功能介绍

| 功能 | 说明 |
|------|------|
| 🎨 **去水印** | 自动去除 Gemini 图片水印，100% 本地运行 |
| ✂️ **分镜切割** | 自动检测网格图，切成单张图片 |
| 🎬 **视频帧提取** | 提取视频最后一帧 + 最后2秒所有帧 |
| 📁 **自动监控** | 监控下载文件夹，新文件自动处理 |

---

## 🚀 一键安装（小白看这里）

### 第一步：安装 Python

1. 打开 https://www.python.org/downloads/
2. 点击黄色按钮 **Download Python 3.x.x**
3. 运行安装程序，**勾选 "Add Python to PATH"**（很重要！）
4. 点击 Install Now

### 第二步：安装 FFmpeg（处理视频用）

**方法一：命令安装（推荐）**
```
按 Win+R，输入 cmd，回车，然后输入：
winget install FFmpeg
```

**方法二：手动下载**
1. 打开 https://www.gyan.dev/ffmpeg/builds/
2. 下载 `ffmpeg-release-essentials.zip`
3. 解压到 `C:\ffmpeg`
4. 把 `C:\ffmpeg\bin` 添加到系统环境变量 PATH

### 第三步：下载本工具

```
点击本页面绿色按钮 Code → Download ZIP
解压到任意位置
```

### 第四步：运行

```
双击 install.bat  （首次运行，自动安装依赖）
以后双击 start.vbs 启动
```

---

## 🐢 国内下载慢？用镜像源

如果 `install.bat` 下载很慢或失败，手动执行：

```bash
# 打开 cmd，进入工具目录，执行：
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**常用国内镜像：**
- 清华：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 阿里：`https://mirrors.aliyun.com/pypi/simple`
- 豆瓣：`https://pypi.douban.com/simple`

---

## 📖 使用方法

1. 双击 `start.vbs` 启动程序
2. 设置**监控目录**（你下载图片/视频的文件夹）
3. 设置**输出目录**（处理后文件保存位置）
4. 点击**开始监控**
5. 下载 Gemini 图片或视频，工具自动处理

### 功能开关

- ☑️ **去水印** - 自动去除 Gemini 水印
- ☑️ **分镜切割** - 自动切割网格图

---

## 🎬 使用场景

- **Gemini 生成分镜** → 自动去水印 + 切割成单张
- **Veo/Sora/可灵生成视频** → 提取尾帧做下一段起始帧
- **批量处理 AI 素材** → 监控文件夹自动处理

---

## 📁 支持格式

| 类型 | 格式 |
|------|------|
| 图片 | `.jpg` `.jpeg` `.png` `.webp` |
| 视频 | `.mp4` `.mov` `.webm` `.avi` `.mkv` |

---

## ❓ 常见问题

**Q: 双击 install.bat 闪退？**
A: Python 没装好。重新安装 Python，记得勾选 "Add Python to PATH"

**Q: 提示找不到 ffmpeg？**
A: FFmpeg 没装。按上面方法安装 FFmpeg

**Q: 下载依赖很慢？**
A: 用国内镜像，见上面"国内下载慢"部分

**Q: 去水印后图片有问题？**
A: 只支持 Gemini 生成的图片，其他来源的图片可能不兼容

---

## 🔧 原理说明

### 去水印
使用**逆向 Alpha 混合算法**，通过数学计算还原被水印覆盖的像素。不是 AI，不联网，100% 本地运行。

### 分镜切割
检测图片中的**白色分割线**，自动识别网格布局（2x2、3x3等），切割成单张图片。

### 视频帧提取
用 FFmpeg 提取视频最后一帧，同时提取最后 2 秒的所有帧供选择。

---

## License

MIT License
