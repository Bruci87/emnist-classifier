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
    """Extrai o traço do canvas convertendo RGBA/Alpha para escala de cinza."""
    if canvas_result is None:
        return None

    try:
        raw_image = canvas_result.image_data
    except Exception:
        return None

    if raw_image is None:
        return None

    img_array = np.array(raw_image, dtype=np.uint8)

    if img_array.size == 0:
        return None

    # Extrai o traço: em canvas RGBA com fundo preto, o pincel branco preenche R, G, B
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Pega a cor branca dos canais RGB
        img_gray = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Verifica se há qualquer pixel diferente de zero (linha desenhada)
    if np.max(img_gray) == 0 and np.sum(img_array[:, :, 3]) > 0:
        # Fallback: se a cor estiver no Alpha
        img_gray = img_array[:, :, 3]

    if np.max(img_gray) == 0:
        return None

    # Redimensiona para 28x28 (padrão EMNIST)
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Transposição para alinhar orientação do EMNIST
    img_transposed = cv2.transpose(img_resized)

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
st.write("Desenhe o caractere no quadro e clique em **Classificar**.")

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
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_vf_{st.session_state['key_vf']}"
        )
        c1, c2 = st.columns(2)
        btn_class = c1.button("Classificar", key="btn_vf")
        btn_clear = c2.button("Limpar Tela", key="clear_vf")
        
        if btn_clear:
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
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_num_{st.session_state['key_num']}"
        )
        c1, c2 = st.columns(2)
        btn_class = c1.button("Classificar", key="btn_num")
        btn_clear = c2.button("Limpar Tela", key="clear_num")
        
        if btn_clear:
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
            fill_color="#000000",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=f"canvas_let_{st.session_state['key_let']}"
        )
        c1, c2 = st.columns(2)
        btn_class = c1.button("Classificar", key="btn_let")
        btn_clear = c2.button("Limpar Tela", key="clear_let")
        
        if btn_clear:
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
