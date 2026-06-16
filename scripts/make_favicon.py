"""產生大尺寸 PWA favicon — 1024x1024 為主視覺，無邊距
黑底 + 綠色 K 線 + 走勢線 + 紅綠實體，元素放大到填滿整個畫面
"""
from PIL import Image, ImageDraw
import os
import shutil

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FAVICON_DIR = os.path.join(OUT_DIR, "..", "dashboard", "static")
os.makedirs(FAVICON_DIR, exist_ok=True)


def make_icon(size: int) -> Image.Image:
    """大尺寸 icon：黑底圓角 + 大 K 線 + 大走勢線，無內邊距"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 黑底圓角矩形 — 滿版（無外邊距）
    radius = int(size * 0.24)
    draw.rounded_rectangle(
        [(0, 0), (size, size)],
        radius=radius,
        fill=(15, 23, 42, 255),  # 深色 #0f172a
    )

    # 綠色外框（很細的，裝飾用）
    draw.rounded_rectangle(
        [(0, 0), (size, size)],
        radius=radius,
        outline=(34, 197, 94, 200),
        width=max(2, int(size * 0.012)),
    )

    # 大 K 線（佔據整個畫面 ~70% 寬度）
    bar_w = size * 0.10  # 影線 + 實體寬度（比之前粗一倍）
    bar_positions = [
        (0.22, 0.65, 0.78, (34, 197, 94)),   # 綠 K
        (0.36, 0.55, 0.68, (239, 68, 68)),   # 紅 K
        (0.50, 0.45, 0.60, (34, 197, 94)),   # 綠 K
        (0.64, 0.32, 0.48, (239, 68, 68)),   # 紅 K
    ]
    for x_frac, top_frac, bot_frac, color in bar_positions:
        x = size * x_frac
        top = size * top_frac
        bot = size * bot_frac
        # 影線
        draw.line(
            [(x, top - size * 0.06), (x, bot + size * 0.06)],
            fill=color, width=max(3, int(size * 0.018))
        )
        # 實體（粗）
        draw.rectangle(
            [(x - bar_w / 2, top), (x + bar_w / 2, bot)],
            fill=color,
        )

    # 大走勢線（從左下到右上，加粗 + 圓點）
    line_pts = [
        (size * 0.15, size * 0.80),
        (size * 0.28, size * 0.70),
        (size * 0.42, size * 0.58),
        (size * 0.58, size * 0.42),
        (size * 0.78, size * 0.25),
    ]
    for i in range(len(line_pts) - 1):
        draw.line(
            [line_pts[i], line_pts[i + 1]],
            fill=(74, 222, 128, 255),
            width=max(4, int(size * 0.05)),
        )
    # 端點圓點（更大）
    for pt in line_pts:
        r = size * 0.045
        draw.ellipse(
            [(pt[0] - r, pt[1] - r), (pt[0] + r, pt[1] + r)],
            fill=(74, 222, 128, 255),
        )

    return img


# 大尺寸 PWA 主檔（1024x1024）
sizes = {
    "favicon-1024.png": 1024,
    "favicon-512.png": 512,
    "favicon-192.png": 192,
    "favicon-180.png": 180,
    "favicon-48.png": 48,
    "favicon-32.png": 32,
    "favicon-16.png": 16,
}

for name, sz in sizes.items():
    icon = make_icon(sz)
    out_path = os.path.join(FAVICON_DIR, name)
    icon.save(out_path)
    print(f"✅ {name} ({sz}x{sz})")

# favicon.ico
ico_16 = make_icon(16)
ico_32 = make_icon(32)
ico_48 = make_icon(48)
ico_48.save(
    os.path.join(FAVICON_DIR, "favicon.ico"),
    format="ICO",
    sizes=[(16, 16), (32, 32), (48, 48)],
    append_images=[ico_16, ico_32],
)
print("✅ favicon.ico (16+32+48)")

# 默認 favicon.png 用 180x180 (Apple 標準)
shutil.copy(
    os.path.join(FAVICON_DIR, "favicon-180.png"),
    os.path.join(FAVICON_DIR, "favicon.png"),
)
print("✅ favicon.png (180x180)")

print(f"\n所有檔案在: {FAVICON_DIR}")
for f in sorted(os.listdir(FAVICON_DIR)):
    sz = os.path.getsize(os.path.join(FAVICON_DIR, f))
    print(f"  {f}  ({sz} bytes)")
