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
st.title("📏 テキスト占有率チェッカー（反映一発版）")

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
            ix1, iy1 = max(x1, cx1), max(y1, cy1)
            ix2, iy2 = min(x2, cx2), min(y2, cy2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area = (x2 - x1) * (y2 - y1)
            if area > 0 and inter / area >= threshold:
                cells.add(f"{row+1}-{col+1}")
    return cells

# === 行ベースでセルを整列 ===
def group_cells_by_row(cells):
    d = {str(r): [] for r in range(1, GRID_SIZE + 1)}
    for c in sorted(cells, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
        r, _ = c.split('-')
        d[r].append(c)
    return list(d.values())

# === オーバーレイ描画 ===
def draw_overlay(img, occupied, excluded):
    vis = np.array(img).copy()
    overlay = vis.copy()
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            cid = f"{row+1}-{col+1}"
            if cid in occupied:
                color = (255, 100, 100) if cid in excluded else (100, 180, 255)
                cv2.rectangle(overlay, (x, y), (x + CELL_SIZE, y + CELL_SIZE), color, -1)
            cv2.rectangle(vis, (x, y), (x + CELL_SIZE, y + CELL_SIZE), (0, 255, 0), 1)
            cv2.putText(vis, cid, (x + 4, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

# === 除外セル反映コールバック ===
def apply_excluded():
    st.session_state["excluded_cells"] = st.session_state.get("temp_excluded", [])

# === リセット処理（対象キーの削除のみ） ===
def reset_image():
    keys_to_delete = [
        "uploaded",       # アップロードされたファイルの情報
        "image_data",
        "occupied_cells",
        "excluded_cells",
        "temp_excluded"
    ]
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    # ※ 再実行は行わず、以下の UI 表示の if でキーの存在を確認します

# --- ファイルアップローダー ---
uploaded = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])
if uploaded:
    st.session_state["uploaded"] = uploaded

# --- 画像アップロードと OCR 結果の作成 ---
if st.session_state.get("uploaded") and st.session_state.get("image_data") is None:
    img = Image.open(st.session_state["uploaded"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img)
    results = reader.readtext(arr)
    occ = set()
    for bbox, text, conf in results:
        if not text.strip():
            continue
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        occ |= get_cells_from_box(x1, y1, x2, y2)
    st.session_state["image_data"]     = img
    st.session_state["occupied_cells"] = sorted(occ)
    st.session_state["excluded_cells"] = []

# --- UI 表示部 ---
if st.session_state.get("image_data"):
    # 左側：画像・判定結果等
    col1, col2 = st.columns([1.1, 1.2])
    with col1:
        # 占有セルから最終判定結果の算出
        occupied = st.session_state.get("occupied_cells", [])
        excluded = st.session_state.get("excluded_cells", [])
        final = set(occupied) - set(excluded)
        ratio = round(len(final) / (GRID_SIZE * GRID_SIZE) * 100)
        status = "⭕️ 合格" if ratio <= 20 else ("▲ 注意" if ratio <= 30 else "❌ 不合格")
        st.markdown(f"📊 **テキスト占有率： {ratio}%**")
        # 「最終判定結果」とリセットボタンを同一行に配置
        c_status, c_reset = st.columns([4, 1])
        with c_status:
            st.markdown(f"📝 **最終判定結果： {status}**")
        with c_reset:
            if st.button("🔄 リセット"):
                reset_image()
        # 画像の表示（80% 縮小：約640px 幅）
        img = st.session_state.get("image_data")
        if img is not None:
            overlay_img = draw_overlay(
                img,
                st.session_state.get("occupied_cells", []),
                st.session_state.get("excluded_cells", [])
            )
            st.image(overlay_img, caption="OCR + セルマップ", width=int(IMAGE_SIZE * 0.8))
    # 右側：チェックボックス群
    with col2:
        st.markdown("### 🛠️ 除外マスを選択")
        if "temp_excluded" not in st.session_state:
            st.session_state["temp_excluded"] = list(st.session_state.get("excluded_cells", []))
        for row_cells in group_cells_by_row(st.session_state.get("occupied_cells", [])):
            if not row_cells:
                continue
            with st.container():
                widths = [0.1] * len(row_cells)
                cols = st.columns(widths, gap="small")
                for i, cid in enumerate(row_cells):
                    with cols[i]:
                        checked = cid in st.session_state.get("temp_excluded", [])
                        val = st.checkbox(cid, value=checked, key=f"tmp_{cid}")
                        if val and cid not in st.session_state.get("temp_excluded", []):
                            st.session_state["temp_excluded"].append(cid)
                        if not val and cid in st.session_state.get("temp_excluded", []):
                            st.session_state["temp_excluded"].remove(cid)
        st.button("🔄 反映", on_click=apply_excluded)
else:
    st.info("画像がアップロードされていません。")
