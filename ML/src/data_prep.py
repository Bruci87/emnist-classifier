import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class EMNISTDataPrep:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.X = None
        self.y = None

    def load_data(self):
        """Carrega, reorienta para a vertical e normaliza os dados do EMNIST."""
        print("Carregando dataset EMNIST Balanced (pode demorar alguns segundos na primeira vez)...")
        emnist = fetch_openml('EMNIST_Balanced', version=1, cache=True, as_frame=False)
        
        # 1. Converte o vetor 1D para matrizes 28x28
        X_imgs = emnist.data.reshape(-1, 28, 28)
        
        # 2. Transpõe cada matriz (.T) para corrigir a orientação Fortran/column-major do EMNIST
        X_fixed = np.array([img.T for img in X_imgs])
        
        # 3. Re-achata em vetor 1x784 e normaliza entre 0 e 1
        self.X = X_fixed.reshape(-1, 784) / 255.0
        self.y = emnist.target.astype(int)

    def get_split_1_5(self):
        """Filtra e divide para os dígitos de 1 a 5."""
        mask = np.isin(self.y, [1, 2, 3, 4, 5])
        return train_test_split(self.X[mask], self.y[mask], test_size=self.test_size, random_state=self.random_state)

    def get_split_A_E(self):
        """Filtra e divide para as letras A a E (Rótulos 10 a 14 no EMNIST Balanced)."""
        mask = np.isin(self.y, [10, 11, 12, 13, 14])
        return train_test_split(self.X[mask], self.y[mask], test_size=self.test_size, random_state=self.random_state)

    def get_split_VF(self):
        """Filtra e divide para V e F (F=15 Mapeado para 0, V=31 Mapeado para 1)."""
        mask = np.isin(self.y, [15, 31])
        X_vf, y_vf = self.X[mask], self.y[mask]
        y_binary = np.where(y_vf == 31, 1, 0)
        return train_test_split(X_vf, y_binary, test_size=self.test_size, random_state=self.random_state)