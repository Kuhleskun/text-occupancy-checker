import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image

# === 定数 ===
GRID_SIZE = 10
CELL_SIZE = 80
IMAGE_SIZE = 800
OCCUPANCY_THRESHOLD = 0.05

st.set_page_config(layout="wide")
st.title("📏 テキスト占有率チェッカー")

# === OCR 初期化 ===
@st.cache_resource
def load_reader():
    return easyocr.Reader(['ja'], gpu=False, recog_network='japanese_g2')
reader = load_reader()

# === セル取得 ===
def get_cells_from_box(x1, y1, x2, y2, threshold=OCCUPANCY_THRESHOLD):
    cells = set()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cx1, cy1 = col * CELL_SIZE, row * CELL_SIZE
            cx2, cy2 = cx1 + CELL_SIZE, cy1 + CELL_SIZE
            ix1 = max(x1, cx1)
            iy1 = max(y1, cy1)
            ix2 = min(x2, cx2)
            iy2 = min(y2, cy2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 0 and (inter_area / box_area) >= threshold:
                cells.add(f"{row+1}-{col+1}")
    return cells

# === 画像処理 ===
def process_image(image_file):
    image = Image.open(image_file).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(image)
    results = reader.readtext(img_np)

    occupied_cells = set()
    boxes = []
    for (bbox, text, conf) in results:
        if not text.strip():
            continue
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        boxes.append((text, x1, y1, x2, y2))
        cells = get_cells_from_box(x1, y1, x2, y2)
        occupied_cells |= cells

    return image, boxes, sorted(list(occupied_cells))

# === オーバーレイ描画 ===
def draw_overlay(image, occupied_cells, excluded_cells):
    vis = np.array(image).copy()
    overlay = vis.copy()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cell_id = f"{row+1}-{col+1}"
            if cell_id in occupied_cells:
                color = (255, 100, 100) if cell_id in excluded_cells else (100, 180, 255)
                cv2.rectangle(overlay, (x, y), (x+CELL_SIZE, y+CELL_SIZE), color, -1)
            cv2.rectangle(vis, (x, y), (x+CELL_SIZE, y+CELL_SIZE), (0, 255, 0), 1)
            cv2.putText(vis, cell_id, (x+4, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    return cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

# === 行ベースでチェックボックスに並べる（左詰め）===
def group_cells_by_row(cells):
    row_dict = {str(r): [] for r in range(1, GRID_SIZE + 1)}
    for cell in sorted(cells, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
        row, _ = cell.split('-')
        row_dict[row].append(cell)
    return list(row_dict.values())

# === メイン処理 ===
uploaded = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])
if uploaded:
    image, boxes, detected_cells = process_image(uploaded)
    excluded_cells = []

    col1, col2 = st.columns([1.3, 1])
    with col2:
        st.markdown("### 🛠️ 除外マスを選択")
        with st.form("exclude_form"):
            selected = {}
            grouped = group_cells_by_row(detected_cells)
            for row_cells in grouped:
                if row_cells:
                    cols = st.columns(len(row_cells))
                    for idx, cell in enumerate(row_cells):
                        with cols[idx]:
                            selected[cell] = st.checkbox(cell, key=cell)

            submitted = st.form_submit_button("反映")

    with col1:
        if submitted:
            excluded_cells = [cell for cell, checked in selected.items() if checked]
            final_cells = set(detected_cells) - set(excluded_cells)
            ratio = round(len(final_cells) / (GRID_SIZE * GRID_SIZE) * 100)
            status = "⭕️ 合格" if ratio <= 20 else ("▲ 注意" if ratio <= 30 else "❌ 不合格")

            st.markdown(f"### 📊 テキスト占有率： {ratio}%")
            st.markdown(f"### 📝 最終判定結果： {status}")
            overlay_img = draw_overlay(image, detected_cells, excluded_cells)
            st.image(overlay_img, caption="OCR + セルマップ", use_container_width=True)

            st.markdown(f"**カウント対象マス一覧**： {sorted(list(final_cells))}")
