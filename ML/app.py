import os
import joblib
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------------
# Processamento de imagem do Canvas
# -------------------------------------------------------------
def process_canvas(canvas_result):
    if canvas_result is None:
        return None, "canvas_result é None"

    # Tenta extrair a matriz bruta de pixels
    try:
        raw_data = canvas_result.image_data
    except Exception as e:
        return None, f"Erro ao acessar image_data: {e}"

    if raw_data is None:
        return None, "image_data é None"

    img_array = np.array(raw_data, dtype=np.uint8)

    if img_array.size == 0:
        return None, "Array vazio"

    # Achata a matriz de 3D/4D para 2D pegando o maior valor entre os canais
    if len(img_array.shape) == 3:
        img_gray = np.max(img_array, axis=2)
    else:
        img_gray = img_array

    max_val = np.max(img_gray)
    if max_val == 0:
        return None, f"Tela sem desenhos (max pixel = {max_val})"

    # Processa para 28x28 EMNIST
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    img_transposed = cv2.transpose(img_resized)
    processed_tensor = img_transposed.reshape(1, -1) / 255.0

    return processed_tensor, f"OK (Max pixel: {max_val})"


# -------------------------------------------------------------
# Carregamento dos Modelos .pkl
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(cenario):
    model_path = os.path.join(BASE_DIR, "models", f"melhor_modelo_{cenario}.pkl")
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado em: {model_path}")
        return None
    return joblib.load(model_path)


# -------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------
st.set_page_config(page_title="Classificador EMNIST", layout="wide")
st.title("Classificador de Símbolos EMNIST")

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_vf = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas_vf_debug"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed, debug_msg = process_canvas(canvas_vf)
        
        # Prints para depuração do fluxo no Streamlit
        st.caption(f"**Status Canvas:** {debug_msg}")

        if img_processed is not None:
            model = load_model("VF")
            if model:
                pred = model.predict(img_processed)[0]
                resultado_texto = "V" if str(pred).strip() in ["1", "V", "True", "1.0"] else "F"
                st.success(f"Predição: **{resultado_texto}**")
        else:
            st.warning("Desenhe no quadro para acionar a leitura.")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_num = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas_num_debug"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed, debug_msg = process_canvas(canvas_num)
        st.caption(f"**Status Canvas:** {debug_msg}")

        if img_processed is not None:
            model = load_model("1_5")
            if model:
                pred = model.predict(img_processed)[0]
                st.success(f"Predição: **{pred}**")
        else:
            st.warning("Desenhe no quadro para acionar a leitura.")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_let = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas_let_debug"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed, debug_msg = process_canvas(canvas_let)
        st.caption(f"**Status Canvas:** {debug_msg}")

        if img_processed is not None:
            model = load_model("A_E")
            if model:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Predição: **{letra_pred}**")
        else:
            st.warning("Desenhe no quadro para acionar a leitura.")
