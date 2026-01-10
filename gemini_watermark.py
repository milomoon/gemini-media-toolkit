"""
Gemini 水印去除模块
基于 Reverse Alpha Blending 算法
"""
import numpy as np
from PIL import Image
from pathlib import Path

# 水印模板路径
ASSETS_DIR = Path(__file__).parent / "assets"

# 常量
ALPHA_THRESHOLD = 0.002  # 忽略极小的 alpha 值（噪声）
MAX_ALPHA = 0.99         # 避免除以接近零的值
LOGO_VALUE = 255         # 白色水印的颜色值


def detect_watermark_config(width: int, height: int) -> dict:
    """
    根据图片尺寸检测水印配置
    Gemini 规则：宽高都 > 1024 用 96x96，否则用 48x48
    """
    if width > 1024 and height > 1024:
        return {"logo_size": 96, "margin_right": 64, "margin_bottom": 64}
    else:
        return {"logo_size": 48, "margin_right": 32, "margin_bottom": 32}


def calculate_watermark_position(width: int, height: int, config: dict) -> dict:
    """计算水印在图片中的位置（右下角）"""
    logo_size = config["logo_size"]
    margin_right = config["margin_right"]
    margin_bottom = config["margin_bottom"]
    
    return {
        "x": width - margin_right - logo_size,
        "y": height - margin_bottom - logo_size,
        "width": logo_size,
        "height": logo_size
    }


def load_alpha_map(size: int) -> np.ndarray:
    """
    从预捕获的水印模板加载 Alpha 映射
    """
    bg_path = ASSETS_DIR / f"bg_{size}.png"
    if not bg_path.exists():
        raise FileNotFoundError(f"水印模板不存在: {bg_path}")
    
    bg_img = Image.open(bg_path).convert("RGB")
    bg_array = np.array(bg_img, dtype=np.float32)
    
    # 取 RGB 三通道最大值，归一化到 [0, 1]
    alpha_map = np.max(bg_array, axis=2) / 255.0
    return alpha_map


def remove_watermark(image: Image.Image) -> Image.Image:
    """
    去除 Gemini 水印
    
    原理：
    Gemini 添加水印: watermarked = α × logo + (1 - α) × original
    反向求解: original = (watermarked - α × logo) / (1 - α)
    
    Args:
        image: PIL Image 对象
    
    Returns:
        去除水印后的 PIL Image
    """
    width, height = image.size
    
    # 检测水印配置
    config = detect_watermark_config(width, height)
    position = calculate_watermark_position(width, height, config)
    
    # 加载对应尺寸的 alpha 映射
    alpha_map = load_alpha_map(config["logo_size"])
    
    # 转换为 numpy 数组
    img_array = np.array(image, dtype=np.float32)
    
    # 提取水印区域
    x, y = position["x"], position["y"]
    w, h = position["width"], position["height"]
    
    # 处理水印区域的每个像素
    for row in range(h):
        for col in range(w):
            alpha = alpha_map[row, col]
            
            # 跳过极小的 alpha 值
            if alpha < ALPHA_THRESHOLD:
                continue
            
            # 限制 alpha 避免除零
            alpha = min(alpha, MAX_ALPHA)
            one_minus_alpha = 1.0 - alpha
            
            # 对 RGB 三通道应用反向 alpha 混合
            for c in range(3):
                watermarked = img_array[y + row, x + col, c]
                original = (watermarked - alpha * LOGO_VALUE) / one_minus_alpha
                img_array[y + row, x + col, c] = np.clip(original, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))


def is_gemini_image(image: Image.Image) -> bool:
    """
    检测图片是否可能是 Gemini 生成的（有水印）
    放宽检测条件，只要图片尺寸合理就返回 True
    """
    width, height = image.size
    # 只要图片大于 100x100 就尝试去水印
    return width > 100 and height > 100


# 命令行测试
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python gemini_watermark.py <图片路径>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"文件不存在: {input_path}")
        sys.exit(1)
    
    img = Image.open(input_path)
    print(f"处理图片: {input_path} ({img.size[0]}x{img.size[1]})")
    
    if is_gemini_image(img):
        print("检测到可能的 Gemini 水印，正在去除...")
        result = remove_watermark(img)
        
        output_path = input_path.parent / f"{input_path.stem}_nowm{input_path.suffix}"
        result.save(output_path, quality=95)
        print(f"已保存: {output_path}")
    else:
        print("未检测到 Gemini 水印")
