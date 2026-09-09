import os
import joblib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from utils import process_canvas

# Carrega o modelo de forma segura usando o diretório do próprio arquivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(cenario):
    """Carrega o modelo .pkl correspondente ao cenário."""
    model_path = os.path.join(BASE_DIR, "models", f"melhor_modelo_{cenario}.pkl")
    if not os.path.exists(model_path):
        st.error(f"Modelo não encontrado em: {model_path}")
        return None
    return joblib.load(model_path)

# Configuração da página
st.set_page_config(page_title="Classificador EMNIST", layout="wide")
st.title("Classificador de Símbolos EMNIST")
st.write("Desenhe o caractere no canvas e clique no botão correspondente para realizar a predição.")

# Abas por cenário
tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    canvas_vf = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_vf"
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Classificar V/F", key="btn_vf"):
            model = load_model("VF")
            if model and canvas_vf.image_data is not None:
                img_processed = process_canvas(canvas_vf.image_data)
                if img_processed is not None:
                    pred = model.predict(img_processed)[0]
                    res = "Verdadeiro (V)" if pred == 1 else "Falso (F)"
                    st.success(f"Resultado: {res}")
                else:
                    st.warning("Por favor, desenhe algo no canvas antes de classificar.")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    canvas_num = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_num"
    )
    
    if st.button("Classificar Dígito", key="btn_num"):
        model = load_model("1_5")
        if model and canvas_num.image_data is not None:
            img_processed = process_canvas(canvas_num.image_data)
            if img_processed is not None:
                pred = model.predict(img_processed)[0]
                st.success(f"Dígito Predito: {pred}")
            else:
                st.warning("Por favor, desenhe algo no canvas antes de classificar.")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    canvas_let = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_let"
    )
    
    if st.button("Classificar Letra", key="btn_let"):
        model = load_model("A_E")
        if model and canvas_let.image_data is not None:
            img_processed = process_canvas(canvas_let.image_data)
            if img_processed is not None:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Letra Predita: {letra_pred}")
            else:
                st.warning("Por favor, desenhe algo no canvas antes de classificar.")
