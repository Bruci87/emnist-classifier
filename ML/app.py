import streamlit as st
import joblib
from streamlit_drawable_canvas import st_canvas
from src.utils import process_canvas

st.set_page_config(page_title="Classificador EMNIST - Multiprova", layout="wide")

st.title("Classificador de Símbolos EMNIST")
st.write("Desenhe o caractere no canvas e clique no botão correspondente para realizar a predição.")

# Inicializa chaves no session_state para controlar o reset dos canvas
for key in ["key_vf", "key_1_5", "key_A_E"]:
    if key not in st.session_state:
        st.session_state[key] = 0

# Cria as 3 abas
tab1, tab2, tab3 = st.tabs([
    "Classificador Binário (V/F)", 
    "Classificador Dígitos (1 a 5)", 
    "Classificador Letras (A a E)"
])

# Configurações padrão para a área de desenho
CANVAS_CONFIG = {
    "stroke_width": 20,
    "stroke_color": "#FFFFFF",
    "background_color": "#000000",
    "height": 280,
    "width": 280,
    "drawing_mode": "freedraw",
    "return_image_data": True
}

# --- Aba 1: Verdadeiro / Falso ---
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    canvas_vf = st_canvas(**CANVAS_CONFIG, key=f"canvas_vf_{st.session_state['key_vf']}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Classificar V/F"):
            img_vector = process_canvas(canvas_vf.image_data)
            if img_vector is not None:
                try:
                    model = joblib.load("models/melhor_modelo_VF.pkl")
                    pred = model.predict(img_vector)[0]
                    resultado = "Verdadeiro (V)" if pred == 1 else "Falso (F)"
                    st.success(f"**Resultado:** {resultado}")
                except Exception as e:
                    st.error("Modelo não encontrado. Execute `python train_pipeline.py` no terminal.")
            else:
                st.warning("Desenhe um caractere antes de classificar.")
    with col2:
        if st.button("Limpar Canvas", key="btn_clear_vf"):
            st.session_state["key_vf"] += 1
            st.rerun()

# --- Aba 2: Dígitos 1 a 5 ---
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    canvas_num = st_canvas(**CANVAS_CONFIG, key=f"canvas_1_5_{st.session_state['key_1_5']}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Classificar Dígito"):
            img_vector = process_canvas(canvas_num.image_data)
            if img_vector is not None:
                try:
                    model = joblib.load("models/melhor_modelo_1_5.pkl")
                    pred = model.predict(img_vector)[0]
                    st.success(f"**Dígito Predito:** {pred}")
                except Exception as e:
                    st.error("Modelo não encontrado. Execute `python train_pipeline.py` no terminal.")
            else:
                st.warning("Desenhe um dígito antes de classificar.")
    with col2:
        if st.button("Limpar Canvas", key="btn_clear_1_5"):
            st.session_state["key_1_5"] += 1
            st.rerun()

# --- Aba 3: Letras A a E ---
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    canvas_letra = st_canvas(**CANVAS_CONFIG, key=f"canvas_A_E_{st.session_state['key_A_E']}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Classificar Letra"):
            img_vector = process_canvas(canvas_letra.image_data)
            if img_vector is not None:
                try:
                    model = joblib.load("models/melhor_modelo_A_E.pkl")
                    pred = model.predict(img_vector)[0]
                    mapeamento = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                    letra = mapeamento.get(pred, str(pred))
                    st.success(f"**Letra Predita:** {letra}")
                except Exception as e:
                    st.error("Modelo não encontrado. Execute `python train_pipeline.py` no terminal.")
            else:
                st.warning("Desenhe uma letra antes de classificar.")
    with col2:
        if st.button("Limpar Canvas", key="btn_clear_A_E"):
            st.session_state["key_A_E"] += 1
            st.rerun()