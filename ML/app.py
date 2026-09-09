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
    """Extrai e processa a imagem do canvas em ambiente de nuvem."""
    if canvas_result is None or canvas_result.image_data is None:
        return None

    img_array = np.array(canvas_result.image_data, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Captura a maior intensidade entre os canais RGB e o canal Alpha
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        rgb_max = np.max(img_array[:, :, :3], axis=2)
        alpha = img_array[:, :, 3]
        img_gray = np.maximum(rgb_max, alpha)
    elif len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Se não houver nenhum pixel desenhado (tudo zero)
    if np.max(img_gray) == 0:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Alinha a matriz ao formato EMNIST (transposição)
    img_transposed = cv2.transpose(img_resized)

    # Normaliza entre 0 e 1 e achata para formato 1x784
    return img_transposed.reshape(1, -1) / 255.0


# -------------------------------------------------------------
# Carregamento dos Modelos
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(cenario):
    """Carrega o modelo .pkl correspondente ao cenário."""
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
st.write("Desenhe o caractere no quadro abaixo. A predição é realizada automaticamente assim que você solta o mouse.")

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    
    if "key_vf" not in st.session_state:
        st.session_state["key_vf"] = 0

    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_vf = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_vf_{st.session_state['key_vf']}"
        )
        if st.button("Limpar Tela", key="clear_vf"):
            st.session_state["key_vf"] += 1
            st.rerun()

    with col_pred:
        st.subheader("Resultado da Predição")
        img_processed = process_canvas(canvas_vf)
        if img_processed is not None:
            model = load_model("VF")
            if model:
                pred = model.predict(img_processed)[0]
                res = "Verdadeiro (V)" if pred == 1 else "Falso (F)"
                st.success(f"Classificação: **{res}**")
        else:
            st.info("Aguardando desenho no canvas...")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    
    if "key_num" not in st.session_state:
        st.session_state["key_num"] = 0

    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_num = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_num_{st.session_state['key_num']}"
        )
        if st.button("Limpar Tela", key="clear_num"):
            st.session_state["key_num"] += 1
            st.rerun()

    with col_pred:
        st.subheader("Resultado da Predição")
        img_processed = process_canvas(canvas_num)
        if img_processed is not None:
            model = load_model("1_5")
            if model:
                pred = model.predict(img_processed)[0]
                st.success(f"Dígito Predito: **{pred}**")
        else:
            st.info("Aguardando desenho no canvas...")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    
    if "key_let" not in st.session_state:
        st.session_state["key_let"] = 0

    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_let = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_let_{st.session_state['key_let']}"
        )
        if st.button("Limpar Tela", key="clear_let"):
            st.session_state["key_let"] += 1
            st.rerun()

    with col_pred:
        st.subheader("Resultado da Predição")
        img_processed = process_canvas(canvas_let)
        if img_processed is not None:
            model = load_model("A_E")
            if model:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Letra Predita: **{letra_pred}**")
        else:
            st.info("Aguardando desenho no canvas...")
