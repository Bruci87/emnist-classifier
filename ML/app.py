import os
import joblib
import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------------
# Processamento de Imagem do Canvas
# -------------------------------------------------------------
def process_canvas(canvas_result):
    """Extrai os dados do canvas com tratamento seguro para evitar RuntimeError."""
    if canvas_result is None:
        return None

    # Captura segura da propriedade image_data
    try:
        raw_data = canvas_result.image_data
    except Exception:
        return None

    if raw_data is None:
        return None

    img_array = np.array(raw_data, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Extrai a intensidade do traço cobrindo os canais de cor/alpha
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Pega a cor branca nos canais RGB ou a transparência do traço
        rgb_max = np.max(img_array[:, :, :3], axis=2)
        alpha = img_array[:, :, 3]
        img_gray = np.maximum(rgb_max, alpha)
    elif len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Se a tela estiver vazia (tudo zero)
    if np.max(img_gray) == 0:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Transposição de matriz para ajustar orientação do EMNIST
    img_transposed = cv2.transpose(img_resized)

    # Normaliza entre 0 e 1 e remodela para formato (1, 784)
    return img_transposed.reshape(1, -1) / 255.0


# -------------------------------------------------------------
# Carregamento dos Modelos .pkl
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
st.write("Desenhe o caractere no quadro abaixo. A predição é exibida na caixa ao lado.")

# Formulário único para evitar incompatibilidade entre abas e renderização do canvas
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
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_vf_main"
        )

    with col_pred:
        st.subheader("Resultado")
        img_processed = process_canvas(canvas_vf)
        if img_processed is not None:
            model = load_model("VF")
            if model:
                pred = model.predict(img_processed)[0]
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
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_num_main"
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
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas_let_main"
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
          
