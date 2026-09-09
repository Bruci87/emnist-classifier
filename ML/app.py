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
    """Extrai a imagem do canvas e formata para o modelo EMNIST (1, 784)."""
    if canvas_result is None or canvas_result.image_data is None:
        return None

    img_array = np.array(canvas_result.image_data, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Extrai o valor máximo dos canais para identificar o traço branco
    if len(img_array.shape) == 3:
        img_gray = np.max(img_array, axis=2)
    else:
        img_gray = img_array

    # Se nada foi desenhado (brilho máximo inferior a 10)
    if np.max(img_gray) < 10:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Transposição para alinhar a rotação das colunas/linhas com o dataset EMNIST
    img_transposed = cv2.transpose(img_resized)

    # Normalização entre 0 e 1 e vetorização para (1, 784)
    return img_transposed.reshape(1, -1) / 255.0


# -------------------------------------------------------------
# Carregamento do Modelo .pkl
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
st.write("Desenhe o caractere no quadro e o resultado será exibido ao lado.")

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    
    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_vf = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas_vf_app"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed = process_canvas(canvas_vf)
        if img_processed is not None:
            model = load_model("VF")
            if model:
                pred = model.predict(img_processed)[0]
                # Retorna V se a classe predita for 1, senão F
                resultado_texto = "V" if str(pred) in ["1", "V", "True"] else "F"
                st.success(f"Resultado: **{resultado_texto}**")
        else:
            st.info("Aguardando desenho...")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    
    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_num = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas_num_app"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed = process_canvas(canvas_num)
        if img_processed is not None:
            model = load_model("1_5")
            if model:
                pred = model.predict(img_processed)[0]
                st.success(f"Resultado: **{pred}**")
        else:
            st.info("Aguardando desenho...")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    
    col_canvas, col_pred = st.columns([1, 1])
    
    with col_canvas:
        canvas_let = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key="canvas_let_app"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed = process_canvas(canvas_let)
        if img_processed is not None:
            model = load_model("A_E")
            if model:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Resultado: **{letra_pred}**")
        else:
            st.info("Aguardando desenho...")
