import os
import joblib
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -------------------------------------------------------------
# Processamento de Imagem Enviada
# -------------------------------------------------------------
def process_uploaded_image(uploaded_file, modo_orientacao="Padrão (Sem rotação)"):
    if uploaded_file is None:
        return None, None

    # Abre a imagem enviada em escala de cinza
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image, dtype=np.uint8)

    # Inverte se o fundo for branco e o desenho preto (EMNIST espera fundo preto)
    if np.mean(img_array) > 127:
        img_array = cv2.bitwise_not(img_array)

    # Redimensiona para 28x28 (padrão EMNIST)
    img_28x28 = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    # Ajustes de orientação conforme seleção
    if modo_orientacao == "Apenas Transpose":
        img_final = cv2.transpose(img_28x28)
    elif modo_orientacao == "90° Anti-horário + Flip":
        img_final = cv2.rotate(img_28x28, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_final = cv2.flip(img_final, 0)
    else:  # Padrão
        img_final = img_28x28

    # Normalização entre 0 e 1 e reshaping para (1, 784)
    processed_tensor = img_final.reshape(1, -1) / 255.0

    return processed_tensor, img_final


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
st.write("Faça o upload de uma imagem do caractere para realizar a predição.")

# Barra lateral para ajustar o alinhamento da imagem
st.sidebar.header("Configurações de Alinhamento")
orientacao_selecionada = st.sidebar.radio(
    "Ajuste da Orientação EMNIST:",
    ["Padrão (Sem rotação)", "Apenas Transpose", "90° Anti-horário + Flip"],
    index=0
)

tab1, tab2, tab3 = st.tabs(["Classificador Binário (V/F)", "Classificador Dígitos (1 a 5)", "Classificador Letras (A a E)"])

# -------------------------------------------------------------
# TAB 1: V/F
# -------------------------------------------------------------
with tab1:
    st.header("1. Classificador Verdadeiro (V) / Falso (F)")
    col_input, col_pred, col_debug = st.columns([1, 1, 1])
    
    with col_input:
        file_vf = st.file_uploader("Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_vf")

    with col_pred:
        st.subheader("Resultado")
        if file_vf is not None:
            img_processed, img_28 = process_uploaded_image(file_vf, orientacao_selecionada)
            model = load_model("VF")
            if model and img_processed is not None:
                pred = model.predict(img_processed)[0]
                resultado_texto = "V" if str(pred).strip() in ["1", "V", "True", "1.0"] else "F"
                st.success(f"Resultado: **{resultado_texto}**")
        else:
            st.info("Aguardando arquivo de imagem...")

    with col_debug:
        st.subheader("Entrada do Modelo (28x28)")
        if file_vf is not None and 'img_28' in locals() and img_28 is not None:
            st.image(img_28, width=140, caption="Como o modelo enxerga")

# -------------------------------------------------------------
# TAB 2: Dígitos 1 a 5
# -------------------------------------------------------------
with tab2:
    st.header("2. Classificador de Dígitos (1 a 5)")
    col_input, col_pred, col_debug = st.columns([1, 1, 1])
    
    with col_input:
        file_num = st.file_uploader("Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_num")

    with col_pred:
        st.subheader("Resultado")
        if file_num is not None:
            img_processed, img_28 = process_uploaded_image(file_num, orientacao_selecionada)
            model = load_model("1_5")
            if model and img_processed is not None:
                pred = model.predict(img_processed)[0]
                st.success(f"Resultado: **{pred}**")
        else:
            st.info("Aguardando arquivo de imagem...")

    with col_debug:
        st.subheader("Entrada do Modelo (28x28)")
        if file_num is not None and 'img_28' in locals() and img_28 is not None:
            st.image(img_28, width=140, caption="Como o modelo enxerga")

# -------------------------------------------------------------
# TAB 3: Letras A a E
# -------------------------------------------------------------
with tab3:
    st.header("3. Classificador de Letras (A a E)")
    col_input, col_pred, col_debug = st.columns([1, 1, 1])
    
    with col_input:
        file_let = st.file_uploader("Envie a imagem (PNG/JPG)", type=["png", "jpg", "jpeg"], key="file_let")

    with col_pred:
        st.subheader("Resultado")
        if file_let is not None:
            img_processed, img_28 = process_uploaded_image(file_let, orientacao_selecionada)
            model = load_model("A_E")
            if model and img_processed is not None:
                pred = model.predict(img_processed)[0]
                letras_map = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E'}
                letra_pred = letras_map.get(pred, str(pred))
                st.success(f"Resultado: **{letra_pred}**")
        else:
            st.info("Aguardando arquivo de imagem...")

    with col_debug:
        st.subheader("Entrada do Modelo (28x28)")
        if file_let is not None and 'img_28' in locals() and img_28 is not None:
            st.image(img_28, width=140, caption="Como o modelo enxerga")
