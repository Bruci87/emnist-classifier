import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class ModelTrainer:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "NaiveBayes": GaussianNB(),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(kernel='rbf', probability=True, random_state=42),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
        }

    def run(self, X_train, X_test, y_train, y_test, cenario_nome):
        relatorios = []
        melhor_modelo = None
        melhor_acc_test = -1.0

        for name, model in self.models.items():
            print(f"Treinando {name} para o cenário [{cenario_nome}]...")
            model.fit(X_train, y_train)

            acc_train = model.score(X_train, y_train)
            acc_test = model.score(X_test, y_test)

            y_pred = model.predict(X_test)
            report_dict = classification_report(y_test, y_pred, output_dict=True)

            df_rep = pd.DataFrame(report_dict).transpose()
            df_rep['modelo'] = name
            df_rep['cenario'] = cenario_nome
            df_rep['acc_train'] = acc_train
            df_rep['acc_test'] = acc_test
            relatorios.append(df_rep)

            # Seleciona o modelo com maior acurácia no conjunto de teste
            if acc_test > melhor_acc_test:
                melhor_acc_test = acc_test
                melhor_modelo = model

        os.makedirs("models", exist_ok=True)
        joblib.dump(melhor_modelo, f"models/melhor_modelo_{cenario_nome}.pkl")
        return pd.concat(relatorios)