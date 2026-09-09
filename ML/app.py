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
    """Garante a extração dos pixels do traço independente do formato RGBA/RGB do navegador."""
    if canvas_result is None:
        return None

    try:
        raw_data = canvas_result.image_data
    except Exception:
        return None

    if raw_data is None:
        return None

    img_array = np.array(raw_data, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Se a matriz possui múltiplos canais (ex: RGBA ou RGB)
    if len(img_array.shape) == 3:
        # Pega a maior intensidade pixel a pixel entre todos os canais (RGB + Alpha)
        img_gray = np.max(img_array, axis=2)
    else:
        img_gray = img_array

    # Se não houver nenhum pixel ativo (tudo preto/transparente)
    if np.max(img_gray) == 0:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Transposição para alinhar orientação do EMNIST
    img_transposed = cv2.transpose(img_resized)

    # Normalização entre 0 e 1 e reshaping para (1, 784)
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
st.write("Desenhe no quadro e clique no botão **Classificar**.")

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
            key="canvas_vf_v3"
        )
        btn_vf = st.button("Classificar (V/F)", key="btn_vf_run")

    with col_pred:
        st.subheader("Resultado")
        img_processed = process_canvas(canvas_vf)
        
        if img_processed is not None:
            model = load_model("VF")
            if model:
                pred = model.predict(img_processed)[0]
                res_str = str(pred).strip()
                resultado_texto = "V" if res_str in ["1", "V", "True", "1.0"] else "F"
                st.success(f"Resultado: **{resultado_texto}**")
        else:
            st.info("Aguardando desenho no canvas...")

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
            key="canvas_num_v3"
        )
        btn_num = st.button("Classificar Dígito", key="btn_num_run")

    with col_pred:
        st.subheader("Resultado")
        img_processed = process_canvas(canvas_num)
        
        if img_processed is not None:
            model = load_model("1_5")
            if model:
                pred = model.predict(img_processed)[0]
                st.success(f"Resultado: **{pred}**")
        else:
            st.info("Aguardando desenho no canvas...")

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
            key="canvas_let_v3"
        )
        btn_let = st.button("Classificar Letra", key="btn_let_run")

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
            st.info("Aguardando desenho no canvas...")
