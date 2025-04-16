import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
import random
from pathlib import Path
import torch
from torchvision import transforms


"""
Torch‑based U²‑NetP で商品領域マスクを取得し、
パッケージ文字を除外してテキスト占有率を計算するアプリ。
依存追加なし（torch は easyocr が導入）
外部ダウンロードなし：`models/u2netp.pth` を同梱しておく
UI／レイアウトは従来通り
"""

# -------------------- 定数 --------------------
GRID_SIZE = 10
CELL_SIZE = 80
IMAGE_SIZE = 800
OCCUPANCY_THRESHOLD = 0.05      # 文字 bbox がセルに 5% 以上重なれば占有
PACKAGE_TEXT_THRESHOLD = 0.80   # bbox の 80%以上が商品マスク内ならパッケージ文字
MODEL_PATH = Path(__file__).parent / "models" / "u2netp.pth"  # 4.7 MB 重み

st.set_page_config(layout="wide")
st.title("📏 テキスト占有率チェッカー（Torch U²‑NetP 完全版）")

# -------------------- OCR リーダ --------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(["ja"], gpu=False, recog_network="japanese_g2")

reader = load_reader()

# -------------------- U²‑NetP ロード --------------------
@st.cache_resource
def load_u2netp():
    if not MODEL_PATH.exists():
        st.error("u2netp.pth が models/ フォルダにありません。")
        st.stop()
    model = torch.hub.load("NathanUA/U-2-Net", "u2netp", pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# -------------------- 商品マスク --------------------

def get_product_mask(pil_img: Image.Image) -> np.ndarray:
    model = load_u2netp()
    tr = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    inp = tr(pil_img).unsqueeze(0)
    with torch.no_grad():
        pred = model(inp)[0][0]
    mask = (pred.sigmoid().cpu().numpy() > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask

# -------------------- ユーティリティ --------------------

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


def get_all_cells():
    return {f"{r}-{c}" for r in range(1, GRID_SIZE + 1) for c in range(1, GRID_SIZE + 1)}


def group_cells_by_row(cells):
    rows = {str(r): [] for r in range(1, GRID_SIZE + 1)}
    for cid in sorted(cells, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
        r, _ = cid.split('-')
        rows[r].append(cid)
    return list(rows.values())


def draw_overlay(img, occupied, target, excluded, mask=None):
    vis = np.array(img).copy()
    overlay = vis.copy()

    if mask is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cid = f"{row + 1}-{col + 1}"
            if cid in excluded:
                color = (255, 100, 100)
            elif cid in target:
                color = (0, 255, 0)
            elif cid in (set(occupied) - set(excluded)):
                color = (100, 180, 255)
            else:
                color = None
            if color is not None:
                cv2.rectangle(overlay, (x, y), (x + CELL_SIZE, y + CELL_SIZE), color, -1)
            cv2.rectangle(vis, (x, y), (x + CELL_SIZE, y + CELL_SIZE), (0, 255, 0), 1)
            cv2.putText(vis, cid, (x + 4, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

# -------------------- セッションヘルパ --------------------

def reset_image():
    for k in [
        "uploaded", "image_data", "product_mask", "occupied_cells",
        "excluded_cells", "temp_excluded", "target_cells", "temp_target"
    ]:
        st.session_state.pop(k, None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.set_query_params(dummy=str(random.randint(0, 100000)))

# -------------------- UI: ファイルアップロード --------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "png", "jpeg"],
    key=f"uploader_{st.session_state['uploader_key']}"
)
if uploaded:
    st.session_state["uploaded"] = uploaded

# -------------------- 画像処理（初回のみ） --------------------
if st.session_state.get("uploaded") and st.session_state.get("image_data") is None:
    img = Image.open(st.session_state["uploaded"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    product_mask = get_product_mask(img)

    occ = set()
    for bbox, text, conf in reader.readtext(np.array(img)):
        if not text.strip():
            continue
        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])
        region = product_mask[y1:y2, x1:x2]
        if region.size > 0 and region.mean() >= PACKAGE_TEXT_THRESHOLD:
            continue  # パッケージ文字
        occ |= get_cells_from_box(x1, y1, x2, y2)

    st.session_state.update({
        "image_data": img,
        "product_mask": product_mask,
        "occupied_cells": sorted(occ),
        "excluded_cells": [],
        "target_cells": []
    })

# -------------------- UI 表示 --------------------
img_data = st.session_state.get("image_data")
if img_data is not None:
    col1, col2 = st.columns([1.1, 1.2])

    # ---------- 左カラム ----------
    with col1:
        occupied_set = set(st.session_state["occupied_cells"])
        excluded_set = set(st.session_state["excluded_cells"])
        target_set   = set(st.session_state["target_cells"])
        final_cells  = (occupied_set - excluded_set) | target_set
        ratio = round(len(final_cells))  # 100 セル固定なので count==%
        status = "⭕️ 合格" if ratio <= 20 else ("▲ 注意" if ratio <= 30 else "❌ 不合格")

        st.markdown(f"📊 **テキスト占有率: {ratio}%**  {status}")
        if st.button("🔄 リセット"):
            reset_image()
            st.stop()


        overlay = draw_overlay(
            img_data,
            occupied_set,
            target_set,
            excluded_set,
            st.session_state["product_mask"]
        )
        st.image(overlay, caption="OCR + セルマップ", width=int(IMAGE_SIZE * 0.8))

    # ---------- 右カラム ----------
    with col2:
        # ---- 除外マスフォーム ----
        st.markdown("### 🛠️ 除外マスを選択")
        with st.form("form_exclusion"):
            if "temp_excluded" not in st.session_state:
                st.session_state["temp_excluded"] = list(excluded_set)

            for row_cells in group_cells_by_row(occupied_set):
                if not row_cells:
                    continue
                cols = st.columns([0.1] * len(row_cells), gap="small")
                for i, cid in enumerate(row_cells):
                    with cols[i]:
                        checked = cid in st.session_state["temp_excluded"]
                        val = st.checkbox(cid, value=checked, key=f"exclude_{cid}")
                        if val and cid not in st.session_state["temp_excluded"]:
                            st.session_state["temp_excluded"].append(cid)
                        if not val and cid in st.session_state["temp_excluded"]:
                            st.session_state["temp_excluded"].remove(cid)

            if st.form_submit_button("🔄 除外反映"):
                st.session_state["excluded_cells"] = list(st.session_state["temp_excluded"])

        st.markdown("---")
        # ---- 対象マスフォーム ----
        st.markdown("### 🛠️ 対象マスを選択")
        with st.form("form_target"):
            if "temp_target" not in st.session_state:
                st.session_state["temp_target"] = list(target_set)

            candidate_cells = sorted(get_all_cells() - occupied_set)
            for row_cells in group_cells_by_row(candidate_cells):
                if not row_cells:
                    continue
                cols = st.columns([0.1] * len(row_cells), gap="small")
                for i, cid in enumerate(row_cells):
                    with cols[i]:
                        checked = cid in st.session_state["temp_target"]
                        val = st.checkbox(cid, value=checked, key=f"target_{cid}")
                        if val and cid not in st.session_state["temp_target"]:
                            st.session_state["temp_target"].append(cid)
                        if not val and cid in st.session_state["temp_target"]:
                            st.session_state["temp_target"].remove(cid)

            if st.form_submit_button("🔄 対象反映"):
                st.session_state["target_cells"] = list(st.session_state["temp_target"])

else:
    st.info("画像がアップロードされていません。")
