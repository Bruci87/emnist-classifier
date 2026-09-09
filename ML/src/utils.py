import cv2
import numpy as np

def process_canvas(canvas_data):
    """Ajusta o desenho do canvas para a resolução 28x28 e vetor 1x784."""
    if canvas_data is None:
        return None

    img_array = np.array(canvas_data)

    if img_array.max() == 0:
        return None

    if len(img_array.shape) == 3:
        img_gray = img_array[:, :, 0]
    else:
        img_gray = img_array

    # Redimensiona para 28x28
    img_resized = cv2.resize(img_gray.astype(np.uint8), (28, 28), interpolation=cv2.INTER_AREA)

    # Normaliza entre 0 e 1 e transforma em vetor 1x784
    return img_resized.reshape(1, -1) / 255.0
