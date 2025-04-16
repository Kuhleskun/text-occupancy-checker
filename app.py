import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
import random
import io

# === rembg (ONNX 7 MB, 外部 DL 不要) ===
from rembg import remove

# === 定数 ===
GRID_SIZE = 10
CELL_SIZE = 80
IMAGE_SIZE = 800
OCCUPANCY_THRESHOLD = 0.05
PACKAGE_TEXT_THRESHOLD = 0.80  # マスク内率 80%以上 → パッケージ文字とみなす

st.set_page_config(layout="wide")
st.title("📏 テキスト占有率チェッカー（var.250415 + mask_onnx_fix2）")

# --- ファイルアップローダー用キーの初期化 ---
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# --- OCR 初期化 ---
@st.cache_resource
def load_reader():
    return easyocr.Reader(["ja"], gpu=False, recog_network="japanese_g2")

reader = load_reader()

# === 商品マスク生成 (rembg 利用) ===

def get_product_mask(pil_img: Image.Image) -> np.ndarray:
    """rembg で前景マスク (0/1 ndarray) を取得"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    mask_bytes = remove(buf.getvalue(), only_mask=True)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_img = mask_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    mask = (np.array(mask_img) > 128).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask

# --- OCR バウンディングボックスとセルの重なり判定 ---

def get_cells_from_box(x1, y1, x2, y2, threshold=OCCUPANCY_THRESHOLD):
    cells = set()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cx1, cy1 = col * CELL_SIZE, row * CELL_SIZE
            cx2, cy2 = cx1 + CELL_SIZE, cy1 + CELL_SIZE
            ix1, iy1 = max(x1, cx1), max(y1, cy1)
            ix2, iy2 = min(x2, cx2), min(y2, cy2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area = (x2 - x1) * (y2 - y1)
            if area > 0 and inter / area >= threshold:
                cells.add(f"{row + 1}-{col + 1}")
    return cells

# --- グリッド全体のセル集合 ---

def get_all_cells():
    return {f"{row}-{col}" for row in range(1, GRID_SIZE + 1) for col in range(1, GRID_SIZE + 1)}

# --- 行ごとにセルをグループ化（表示用） ---

def group_cells_by_row(cells):
    d = {str(r): [] for r in range(1, GRID_SIZE + 1)}
    for c in sorted(cells, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
        r, _ = c.split('-')
        d[r].append(c)
    return list(d.values())

# --- オーバーレイ描画 ---

def draw_overlay(img, occupied, target, excluded, mask=None):
    vis = np.array(img).copy()
    overlay = vis.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cid = f"{row + 1}-{col + 1}"
            if cid in excluded:
                color = (255, 100, 100)
                cv2.rectangle(overlay, (x, y), (x + CELL_SIZE, y + CELL_SIZE), color, -1)
            elif cid in target:
                color = (0, 255, 0)
                cv2.rectangle(overlay, (x, y), (x + CELL_SIZE, y + CELL_SIZE), color, -1)
            elif cid in (set(occupied) - set(excluded)):
                color = (100, 180, 255)
                cv2.rectangle(overlay, (x, y), (x + CELL_SIZE, y + CELL_SIZE), color, -1)
            cv2.rectangle(vis, (x, y), (x + CELL_SIZE, y + CELL_SIZE), (0, 255, 0), 1)
            cv2.putText(vis, cid, (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

# --- コールバック ---

def apply_excluded():
    st.session_state["excluded_cells"] = st.session_state.get("temp_excluded", [])

def apply_target():
    st.session_state["target_cells"] = st.session_state.get("temp_target", [])

# --- リセット処理 ---

def reset_image():
    for key in [
        "uploaded",
        "image_data",
        "product_mask",
        "occupied_cells",
        "excluded_cells",
        "temp_excluded",
        "target_cells",
        "temp_target",
    ]:
        st.session_state.pop(key, None)
    st.session_state["uploader_key"] += 1
    st.set_query_params(dummy=str(random.randint(0, 100000)))

# --- ファイルアップローダー ---
uploaded = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "png", "jpeg"],
    key=f"uploader_{st.session_state['uploader_key']}"
)
if uploaded:
    st.session_state["uploaded"] = uploaded

# --- 画像アップロード＆OCR処理（初回のみ） ---
if st.session_state.get("uploaded") and st.session_state.get("image_data") is None:
    img = Image.open(st.session_state["uploaded"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    product_mask = get_product_mask(img)
    arr = np.array(img)
    results = reader.readtext(arr)

    occ = set()
    for bbox, text, conf in results:
        if not text.strip():
            continue
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])

        region = product_mask[y1:y2, x1:x2]
        if region.size > 0 and region.mean() >= PACKAGE_TEXT_THRESHOLD:
            continue  # パッケージ文字

        occ |= get_cells_from_box(x1, y1, x2, y2)

    st.session_state["image_data"] = img
    st.session_state["product_mask"] = product_mask
    st.session_state["occupied_cells"] = sorted(occ)
    st.session_state["excluded_cells"] = []
    st.session_state["target_cells"] = []

# --- UI 表示 ---
img_data = st.session_state.get("image_data")
if img_data is not None:
    col1, col2 = st.columns([1.1, 1.2])
    with col1:
        occupied_set = set(st.session_state.get("occupied_cells", []))
        excluded_set = set(st.session_state.get("excluded_cells", []))
        target_set = set(st.session_state.get("target_cells", []))
        final_cells = (occupied_set - excluded_set) | target_set
        ratio = round(len(final_cells) / (GRID_SIZE * GRID_SIZE) * 100)
        status = "⭕️ 合格" if ratio <= 20 else ("▲ 注意" if ratio <= 30 else "❌ 不合格")
        st.markdown(f"📊 **テキスト占有率: {ratio}%**")
        c_status, c_reset = st.columns([4, 1])
        with c_status:
            st.markdown(f"📝 **最終判定結果: {status}**")
        with c_reset:
            if st.button("🔄 リセット"):
                reset_image()
        overlay_img = draw_overlay(
            img_data,
            st.session_state.get("occupied_cells", []),
            st.session_state.get("target_cells", []),
            st.session_state.get("excluded_cells", []),
            mask=st.session_state.get("product_mask"),
        )
        st.image(overlay_img, caption="OCR + セルマップ", width=int(IMAGE_SIZE * 0.8))

    with col2:
        # ===== 除外マスフォーム =====
        st.markdown("### 🛠️ 除外マスを選択")
        with st.form("form_exclusion"):
            if "temp_excluded" not in st.session_state:
                st.session_state["temp_excluded"] = list(st.session_state.get("excluded_cells
