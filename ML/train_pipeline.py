import os
import pandas as pd
from src.data_prep import EMNISTDataPrep
from src.trainer import ModelTrainer

def main():
    os.makedirs("data", exist_ok=True)

    prep = EMNISTDataPrep()
    prep.load_data()

    trainer = ModelTrainer()
    resultados = []

    # 1. Dataset Binário V/F
    X_tr, X_te, y_tr, y_te = prep.get_split_VF()
    resultados.append(trainer.run(X_tr, X_te, y_tr, y_te, "VF"))

    # 2. Dataset Dígitos 1 a 5
    X_tr, X_te, y_tr, y_te = prep.get_split_1_5()
    resultados.append(trainer.run(X_tr, X_te, y_tr, y_te, "1_5"))

    # 3. Dataset Letras A a E
    X_tr, X_te, y_tr, y_te = prep.get_split_A_E()
    resultados.append(trainer.run(X_tr, X_te, y_tr, y_te, "A_E"))

    # Consolida e salva o relatório .csv exigido (Item 5)
    df_final = pd.concat(resultados)
    df_final.to_csv("data/relatorio_resultados.csv")
    print("Treinamento finalizado! Relatório salvo em data/relatorio_resultados.csv.")

if __name__ == "__main__":
    main()