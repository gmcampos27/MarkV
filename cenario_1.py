# -*- coding: utf-8 -*-
"""stat.ipynb"""

#Bibliotecas de manipulação de dados
import pandas as pd
import numpy as np

#Biblioteca de visualização de plots
import matplotlib.pyplot as plt
import seaborn as sns

#Bibliotecas resposnsáveis pelo Machine Learning

from sklearn.metrics import *
from sklearn import metrics
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression

from sklearn.utils import resample
from tqdm import tqdm
import numpy as np

from itertools import cycle

from google.colab import drive
drive.mount('/content/drive',  force_remount=True)
df = pd.read_csv('/path/dataset_criancas_V3.csv') #Dataset que será utilizado

df.head(5) #Primeiras 5 linhas do dataset

df = df.drop('qseqid',axis =1) #retirar coluna dos IDs
df.head()

"""**Decision Tree**"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Separando as features (X) e o rótulo (y)
X = df.drop(columns=['taxname'])
y = df['taxname']

# Convertendo rótulos categóricos para numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Criando e treinando o modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = model.predict(X_test)

# Criando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - DT')
plt.show()

# @title Texto de título padrão
# Avaliar o modelo
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# Acurácia
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Calcular outras métricas
print(classification_report(y_test, y_pred))


# Overfitting?
if train_accuracy > test_accuracy + 0.15 :
    print("Aviso: O modelo pode estar superajustado. A precisão do treinamento é significativamente maior do que a precisão do teste.")
elif train_accuracy < test_accuracy:
  print("Aviso: O modelo está tendo melhor desempenho em dados não vistos. Verifique se há erros")
else:
    print("O modelo não parece estar com sobreajuste significativo.")

from sklearn.tree import export_graphviz
import graphviz

# Exportando a árvore em formato DOT
dot_data = export_graphviz(
    model, out_file=None,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True, rounded=True, special_characters=True
)

# Visualizando
graph = graphviz.Source(dot_data)

# Renderizando a imagem e salvando no formato JPEG
graph.render("decision_tree", format="jpeg")

# Exibindo o gráfico
graph.view("decision_tree")

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(30,15))  # Ajuste os valores conforme necessário
tree.plot_tree(model, feature_names=X.columns, class_names=label_encoder.classes_, filled=True)
#plt.savefig("pathDT_Tree.jpeg", dpi = 500)
plt.show()

from sklearn.utils import resample
n_bootstrap = 1000
random_state = 42

# Listas para armazenar as métricas
accuracy_scores = []
f1_scores = []
precision = []
MCC = []
recall = []
train_accuracy_scores = []
train_f1_scores = []
feature_importances_dt = []

# Realizar o bootstrap
for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
    # Reamostragem com reposição
    X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state + i)

    # Treinar o modelo na amostra bootstrap
    dt = DecisionTreeClassifier(random_state=random_state + i)
    dt.fit(X_resampled, y_resampled)

    y_pred = dt.predict(X_test)

    #Métricas Treino
    train_accuracy_scores.append(accuracy_score(y_resampled, dt.predict(X_resampled)))
    train_f1_scores.append(f1_score(y_resampled, dt.predict(X_resampled), average="weighted"))

    # Calcular as métricas teste
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
    precision.append(precision_score(y_test, y_pred,average = 'weighted'))
    recall.append(recall_score(y_test, y_pred, average='weighted'))
    MCC.append(matthews_corrcoef(y_test, y_pred))

    feature_importances_dt.append(dt.feature_importances_)

# Calcular a média das importâncias de recursos
average_feature_importances = np.median(feature_importances_dt, axis=0)

feature_names = X_train.columns

plt.figure(figsize=(10, 6))

# Create horizontal bar plot
bars = plt.barh(feature_names, average_feature_importances, color='skyblue')  # Default color

# Add dotted line at 0.050
#plt.axvline(x=0.050, color='gray', linestyle='--', label='Threshold (0.050)')

# Paint bars exceeding the threshold in red
for bar, importance in zip(bars, average_feature_importances):
    if importance > 0.00:
        bar.set_color('red')

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importance")
#plt.legend()
plt.tight_layout()
plt.savefig("pathDT_Features.png", dpi = 900)
plt.show()

# Calcular estatísticas
accuracy_median = np.median(accuracy_scores)
accuracy_ci = (np.percentile(accuracy_scores, 2.5), np.percentile(accuracy_scores, 97.5))

f1_median = np.median(f1_scores)
f1_ci = (np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5))

precision_median = np.median(precision)
precision_ci = (np.percentile(precision, 2.5), np.percentile(precision, 97.5))

recall_median = np.median(recall)
recall_ci = (np.percentile(recall, 2.5), np.percentile(recall, 97.5))

MCC_median = np.median(MCC)
MCC_ci= (np.percentile(MCC, 2.5), np.percentile(MCC, 97.5))

# Exibir resultados
print(f"Accuracy: {accuracy_median:.4f} (95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
print(f"Precision: {precision_median:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
print(f"Recall: {recall_median:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
print(f"F1-Score: {f1_median:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
print(f"MCC: {MCC_median:.4f} (95% CI: {MCC_ci[0]:.4f} - {MCC_ci[1]:.4f})")


# Visualização
import matplotlib.pyplot as plt

# Plot para acurácia
plt.figure(figsize=(12, 6))
plt.hist(accuracy_scores, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(accuracy_median, color="red", linestyle="--", label="median Accuracy")
plt.axvline(accuracy_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(accuracy_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Precisão
plt.figure(figsize=(12, 6))
plt.hist(precision, bins=30, color="green", alpha=0.7, edgecolor="black")
plt.axvline(precision_median, color="red", linestyle="--", label="median Precision")
plt.axvline(precision_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(precision_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Precision")
plt.xlabel("Precision")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Revocação
plt.figure(figsize=(12, 6))
plt.hist(recall, bins=30, color="yellow", alpha=0.7, edgecolor="black")
plt.axvline(recall_median, color="red", linestyle="--", label="median Recall")
plt.axvline(recall_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(recall_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Recall")
plt.xlabel("Recall")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot para F1-score
plt.figure(figsize=(12, 6))
plt.hist(f1_scores, bins=30, color="orange", alpha=0.7, edgecolor="black")
plt.axvline(f1_median, color="red", linestyle="--", label="median F1-Score")
plt.axvline(f1_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(f1_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of F1-Score")
plt.xlabel("F1-Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#MCC
plt.figure(figsize=(12, 6))
plt.hist(MCC, bins=30, color="purple", alpha=0.7, edgecolor="black")
plt.axvline(MCC_median, color="red", linestyle="--", label="median MCC")
plt.axvline(MCC_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(MCC_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of MCC")
plt.xlabel("MCC")
plt.ylabel("Frequency")
plt.legend()
plt.show()

"""RANDOM FOREST"""

y = df['taxname']
X = df.drop("taxname", axis=1)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_test.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_test.shape))

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state =  42)
RF.fit(X_train, y_train)

y_pred = RF.predict(X_test)

# Criando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - RF')
plt.show()

from sklearn.utils import resample
n_bootstrap = 1000
random_state = 42

from sklearn.ensemble import RandomForestClassifier

# Listas para armazenar as métricas
accuracy_rf = []
f1_rf = []
precision_rf = []
MCC_rf = []
recall_rf = []
feature_importances_rf = []

# Realizar o bootstrap
for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
    # Reamostragem com reposição
    X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state + i)

    # Treinar o modelo na amostra bootstrap
    rf = RandomForestClassifier(random_state =  random_state + 1)
    rf.fit(X_resampled, y_resampled)

    feature_importances_rf.append(rf.feature_importances_)

    # Avaliar o modelo no conjunto de teste original
    y_pred = rf.predict(X_test)

    # Calcular as métricas
    accuracy_rf.append(accuracy_score(y_test, y_pred))
    f1_rf.append(f1_score(y_test, y_pred, average="weighted"))
    precision_rf.append(precision_score(y_test, y_pred,average = 'weighted'))
    recall_rf.append(recall_score(y_test, y_pred, average='weighted'))
    MCC_rf.append(matthews_corrcoef(y_test, y_pred))

import matplotlib.pyplot as plt
import numpy as np

# Calcular a média das importâncias de recursos
average_feature_importances = np.median(feature_importances_rf, axis=0)

feature_names = X_train.columns

plt.figure(figsize=(10, 6))

# Create horizontal bar plot
bars = plt.barh(feature_names, average_feature_importances, color='skyblue')  # Default color

# Add dotted line at 0.050
plt.axvline(x=0.050, color='gray', linestyle='--', label='Threshold (0.050)')

# Paint bars exceeding the threshold in red
for bar, importance in zip(bars, average_feature_importances):
    if importance > 0.050:
        bar.set_color('red')

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
#plt.legend()
plt.tight_layout()
plt.savefig("pathRF_Features.png", dpi = 900)
plt.show()

# Calcular estatísticas
accuracy_median = np.median(accuracy_rf)
accuracy_ci = (np.percentile(accuracy_rf, 2.5), np.percentile(accuracy_rf, 97.5))

f1_median = np.median(f1_rf)
f1_ci = (np.percentile(f1_rf, 2.5), np.percentile(f1_rf, 97.5))

precision_median = np.median(precision_rf)
precision_ci = (np.percentile(precision_rf, 2.5), np.percentile(precision_rf, 97.5))

recall_median = np.median(recall_rf)
recall_ci = (np.percentile(recall_rf, 2.5), np.percentile(recall_rf, 97.5))

MCC_median = np.median(MCC_rf)
MCC_ci= (np.percentile(MCC_rf, 2.5), np.percentile(MCC_rf, 97.5))

# Exibir resultados
print(f"Accuracy: {accuracy_median:.4f} (95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
print(f"Precision: {precision_median:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
print(f"Recall: {recall_median:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
print(f"F1-Score: {f1_median:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
print(f"MCC: {MCC_median:.4f} (95% CI: {MCC_ci[0]:.4f} - {MCC_ci[1]:.4f})")


# Visualização
import matplotlib.pyplot as plt

# Plot para acurácia
plt.figure(figsize=(12, 6))
plt.hist(accuracy_rf, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(accuracy_median, color="red", linestyle="--", label="median Accuracy")
plt.axvline(accuracy_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(accuracy_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Precisão
plt.figure(figsize=(12, 6))
plt.hist(precision_rf, bins=30, color="green", alpha=0.7, edgecolor="black")
plt.axvline(precision_median, color="red", linestyle="--", label="median Precision")
plt.axvline(precision_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(precision_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Precision")
plt.xlabel("Precision")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Revocação
plt.figure(figsize=(12, 6))
plt.hist(recall_rf, bins=30, color="yellow", alpha=0.7, edgecolor="black")
plt.axvline(recall_median, color="red", linestyle="--", label="median Recall")
plt.axvline(recall_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(recall_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Recall")
plt.xlabel("Recall")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot para F1-score
plt.figure(figsize=(12, 6))
plt.hist(f1_rf, bins=30, color="orange", alpha=0.7, edgecolor="black")
plt.axvline(f1_median, color="red", linestyle="--", label="median F1-Score")
plt.axvline(f1_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(f1_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of F1-Score")
plt.xlabel("F1-Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#MCC
plt.figure(figsize=(12, 6))
plt.hist(MCC_rf, bins=30, color="purple", alpha=0.7, edgecolor="black")
plt.axvline(MCC_median, color="red", linestyle="--", label="median MCC")
plt.axvline(MCC_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(MCC_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of MCC")
plt.xlabel("MCC")
plt.ylabel("Frequency")
plt.legend()
plt.show()

"""XGBOOST"""

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train_encoded)

# prompt: please create a confusion matrix, decoding the labels

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred are already defined from your XGBoost model
y_pred_xgb = xgb.predict(X_test)
y_test_encoded = label_encoder.transform(y_test)


cm = confusion_matrix(y_test_encoded, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - XGBoost')
plt.show()

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tqdm import tqdm
print(xgb.__version__)

random_state = 42
accuracy_xgb = []
f1_xgb = []
precision_xgb = []
MCC_xgb = []
recall_xgb = []
feature_importances_xgb = []

n_bootstrap =  1000
# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Realizar o bootstrap
for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
    # Reamostragem com reposição
    X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state + i)

    # Encode the target variable using LabelEncoder
    y_resampled_encoded = label_encoder.fit_transform(y_resampled)

    # Treinar o modelo na amostra bootstrap
    xgb = XGBClassifier(eval_metric='logloss', random_state=random_state + i) #disable the internal label encoder
    xgb.fit(X_resampled, y_resampled_encoded) # fit with the encoded data

    # Avaliar o modelo no conjunto de teste original
    y_pred = xgb.predict(X_test)

    # Decode the predicted values back to original labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Calcular as métricas using the original labels
    accuracy_xgb.append(accuracy_score(y_test, y_pred_decoded))
    f1_xgb.append(f1_score(y_test, y_pred_decoded, average="weighted"))
    precision_xgb.append(precision_score(y_test, y_pred_decoded, average="weighted"))
    recall_xgb.append(recall_score(y_test, y_pred_decoded, average="weighted"))
    MCC_xgb.append(matthews_corrcoef(y_test, y_pred_decoded))

    feature_importances_xgb.append(xgb.feature_importances_)

# Calcular estatísticas
accuracy_median = np.median(accuracy_xgb)
accuracy_ci = (np.percentile(accuracy_xgb, 2.5), np.percentile(accuracy_xgb, 97.5))

f1_median = np.median(f1_xgb)
f1_ci = (np.percentile(f1_xgb, 2.5), np.percentile(f1_xgb, 97.5))

precision_median = np.median(precision_xgb)
precision_ci = (np.percentile(precision_xgb, 2.5), np.percentile(precision_xgb, 97.5))

recall_median = np.median(recall_xgb)
recall_ci = (np.percentile(recall_xgb, 2.5), np.percentile(recall_xgb, 97.5))

MCC_median = np.median(MCC_xgb)
MCC_ci= (np.percentile(MCC_xgb, 2.5), np.percentile(MCC_xgb, 97.5))

# Exibir resultados
print(f"Accuracy: {accuracy_median:.4f} (95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
print(f"Precision: {precision_median:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
print(f"Recall: {recall_median:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
print(f"F1-Score: {f1_median:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
print(f"MCC: {MCC_median:.4f} (95% CI: {MCC_ci[0]:.4f} - {MCC_ci[1]:.4f})")


# Visualização
import matplotlib.pyplot as plt

# Plot para acurácia
plt.figure(figsize=(12, 6))
plt.hist(accuracy_xgb, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(accuracy_median, color="red", linestyle="--", label="median Accuracy")
plt.axvline(accuracy_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(accuracy_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Precisão
plt.figure(figsize=(12, 6))
plt.hist(precision_xgb, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(precision_median, color="red", linestyle="--", label="median Precision")
plt.axvline(precision_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(precision_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Precision")
plt.xlabel("Precision")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Revocação
plt.figure(figsize=(12, 6))
plt.hist(recall_xgb, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(recall_median, color="red", linestyle="--", label="median Recall")
plt.axvline(recall_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(recall_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Recall")
plt.xlabel("Recall")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot para F1-score
plt.figure(figsize=(12, 6))
plt.hist(f1_xgb, bins=30, color="orange", alpha=0.7, edgecolor="black")
plt.axvline(f1_median, color="red", linestyle="--", label="median F1-Score")
plt.axvline(f1_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(f1_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of F1-Score")
plt.xlabel("F1-Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#MCC
plt.figure(figsize=(12, 6))
plt.hist(MCC_xgb, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(MCC_median, color="red", linestyle="--", label="median MCC")
plt.axvline(MCC_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(MCC_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of MCC")
plt.xlabel("MCC")
plt.ylabel("Frequency")
plt.legend()
plt.show()

average_feature_importances = np.median(feature_importances_xgb, axis=0)

feature_names = X_train.columns

plt.figure(figsize=(10, 6))

# Create horizontal bar plot
bars = plt.barh(feature_names, average_feature_importances, color='skyblue')  # Default color

# Add dotted line at 0.050
#plt.axvline(x=0.00, color='gray', linestyle='--', label='Threshold (0.050)')

# Paint bars exceeding the threshold in red
for bar, importance in zip(bars, average_feature_importances):
    if importance > 0.00:
        bar.set_color('red')

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
#plt.legend()
plt.tight_layout()
plt.savefig('pathxgboost_feature_importance.png', dpi = 900)
plt.show()

"""**LGBM**"""

import lightgbm as lgb
lgbm = lgb.LGBMClassifier(random_state=42)  # Use LGBMClassifier
lgbm.fit(X_train, y_train_encoded)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred are already defined from your XGBoost model
y_pred_lgbm = lgbm.predict(X_test)
y_test_encoded = label_encoder.transform(y_test)


cm = confusion_matrix(y_test_encoded, y_pred_lgbm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predição')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - LGBM')
plt.show()

import lightgbm as lgb

random_state = 42
accuracy_lgbm = []
f1_lgbm = []
precision_lgbm = []
MCC_lgbm = []
recall_lgbm = []
feature_importances_lgbm = []

n_bootstrap = 1000

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Realizar o bootstrap
for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
    # Reamostragem com reposição
    X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state + i)

    # Encode the target variable using LabelEncoder
    y_resampled_encoded = label_encoder.fit_transform(y_resampled)

    # Treinar o modelo na amostra bootstrap
    lgbm = lgb.LGBMClassifier(random_state=random_state + i)  # Use LGBMClassifier
    lgbm.fit(X_resampled, y_resampled_encoded)  # Fit with encoded data

    # Avaliar o modelo no conjunto de teste original
    y_pred_encoded = lgbm.predict(X_test)  # Predict on X_test

    # Decode the predicted values back to original labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)  # Decode predictions

    # Calcular as métricas using the original labels
    accuracy_lgbm.append(accuracy_score(y_test, y_pred_decoded))  # Use original labels
    f1_lgbm.append(f1_score(y_test, y_pred_decoded, average="weighted"))
    precision_lgbm.append(precision_score(y_test, y_pred_decoded, average="weighted"))
    recall_lgbm.append(recall_score(y_test, y_pred_decoded, average="weighted"))
    MCC_lgbm.append(matthews_corrcoef(y_test, y_pred_decoded))

    feature_importances_lgbm.append(lgbm.feature_importances_)

average_feature_importances_lgbm = np.median(feature_importances_lgbm, axis=0)

feature_names = X_train.columns

plt.figure(figsize=(10, 6))

# Create horizontal bar plot
bars = plt.barh(feature_names, average_feature_importances_lgbm, color='skyblue')  # Default color

# Add dotted line at 0.050
plt.axvline(x=0.050, color='gray', linestyle='--', label='Threshold (0.050)')

# Paint bars exceeding the threshold in red
for bar, importance in zip(bars, average_feature_importances_lgbm):
    if importance > 0.050:
        bar.set_color('red')

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Light GBM Feature Importance")
#plt.legend()
plt.tight_layout()
plt.savefig('pathlgbm_feature_importance.png', dpi = 900)
plt.show()

# Calcular estatísticas
accuracy_median = np.median(accuracy_lgbm)
accuracy_ci = (np.percentile(accuracy_lgbm, 2.5), np.percentile(accuracy_lgbm, 97.5))

f1_median = np.median(f1_lgbm)
f1_ci = (np.percentile(f1_lgbm, 2.5), np.percentile(f1_lgbm, 97.5))

precision_median = np.median(precision_lgbm)
precision_ci = (np.percentile(precision_lgbm, 2.5), np.percentile(precision_lgbm, 97.5))

recall_median = np.median(recall_lgbm)
recall_ci = (np.percentile(recall_lgbm, 2.5), np.percentile(recall_lgbm, 97.5))

MCC_median = np.median(MCC_lgbm)
MCC_ci= (np.percentile(MCC_lgbm, 2.5), np.percentile(MCC_lgbm, 97.5))

# Exibir resultados
print(f"Accuracy: {accuracy_median:.4f} (95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
print(f"Precision: {precision_median:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
print(f"Recall: {recall_median:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
print(f"F1-Score: {f1_median:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
print(f"MCC: {MCC_median:.4f} (95% CI: {MCC_ci[0]:.4f} - {MCC_ci[1]:.4f})")


# Visualização
import matplotlib.pyplot as plt

# Plot para acurácia
plt.figure(figsize=(12, 6))
plt.hist(accuracy_lgbm, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(accuracy_median, color="red", linestyle="--", label="median Accuracy")
plt.axvline(accuracy_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(accuracy_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Precisão
plt.figure(figsize=(12, 6))
plt.hist(precision_lgbm, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(precision_median, color="red", linestyle="--", label="median Precision")
plt.axvline(precision_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(precision_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Precision")
plt.xlabel("Precision")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#Revocação
plt.figure(figsize=(12, 6))
plt.hist(recall_lgbm, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(recall_median, color="red", linestyle="--", label="median Recall")
plt.axvline(recall_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(recall_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of Recall")
plt.xlabel("Recall")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Plot para F1-score
plt.figure(figsize=(12, 6))
plt.hist(f1_lgbm, bins=30, color="orange", alpha=0.7, edgecolor="black")
plt.axvline(f1_median, color="red", linestyle="--", label="median F1-Score")
plt.axvline(f1_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(f1_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of F1-Score")
plt.xlabel("F1-Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#MCC
plt.figure(figsize=(12, 6))
plt.hist(MCC_lgbm, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
plt.axvline(MCC_median, color="red", linestyle="--", label="median MCC")
plt.axvline(MCC_ci[0], color="green", linestyle="--", label="95% CI Lower")
plt.axvline(MCC_ci[1], color="green", linestyle="--", label="95% CI Upper")
plt.title("Bootstrap Distribution of MCC")
plt.xlabel("MCC")
plt.ylabel("Frequency")
plt.legend()
plt.show()

data_ac = {'DT': accuracy_scores,
           'RF': accuracy_rf,
          'XGB': accuracy_xgb,
          'LGBM': accuracy_lgbm}
data_ac = pd.DataFrame(data_ac).assign(metrica='Acurácia')
pd.DataFrame(data_ac).assign(metrica='Acurácia')

data_rv = {'DT': recall,
           'RF': recall_rf,
          'XGB': recall_xgb,
          'LGBM': recall_lgbm}
data_rv = pd.DataFrame(data_rv).assign(metrica='Revocação')
pd.DataFrame(data_rv).assign(metrica='Revocação')

data_pr = {'DT': precision,
           'RF': precision_rf,
          'XGB': precision_xgb,
          'LGBM': precision_lgbm}
data_pr = pd.DataFrame(data_pr).assign(metrica='Precisão')
pd.DataFrame(data_pr).assign(metrica='Precisão')

data_f1 = {'DT': f1_scores,
           'RF': f1_rf,
          'XGB': f1_xgb,
          'LGBM': f1_lgbm}
data_f1 = pd.DataFrame(data_f1).assign(metrica='F1')
pd.DataFrame(data_f1).assign(metrica='F1')

data_mcc = {'DT': MCC,
           'RF': MCC_rf,
           'XGB':MCC_xgb,
           'LGBM': MCC_lgbm}
data_mcc = pd.DataFrame(data_mcc).assign(metrica='MCC')
pd.DataFrame(data_mcc).assign(metrica='MCC')

cdf = pd.concat([data_f1, data_ac, data_pr, data_mcc, data_rv])
mdf = pd.melt(cdf, id_vars=['metrica'], var_name='Classifier')
print(mdf.head())

ax = sns.boxplot(x="metrica", y="value",
                 hue="Classifier",
                 data=mdf,
                 palette=sns.color_palette("Set2"),
                 flierprops={"marker": "."})

# Customize the median display after creation
for line in ax.lines:
    if line.get_label() == 'medians':  # Assuming the default label for medians
        line.set_marker('o')
        line.set_markerfacecolor('black')
        line.set_markeredgecolor('black')
        line.set_markersize(2)

plt.xlabel("")
plt.ylabel("")
# plt.savefig('pathboxplots.png', transparent=None, dpi=900, format='png')
plt.show()

df_anova = mdf[mdf["metrica"] == "Acurácia"]
df_anova

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Filtrar para uma métrica específica, por exemplo, "Accuracy"
metrica_escolhida = "Acurácia"
df_anova = mdf[mdf["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Teste de Tukey
tukey = pairwise_tukeyhsd(endog=df_anova["value"], groups=df_anova["Classifier"], alpha=0.05)

# Exibir resultado
print(tukey)

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Filtrar para uma métrica específica, por exemplo, "Accuracy"
metrica_escolhida = "MCC"
df_anova = mdf[mdf["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Teste de Tukey
tukey = pairwise_tukeyhsd(endog=df_anova["value"], groups=df_anova["Classifier"], alpha=0.05)

# Exibir resultado
print(tukey)

metrica_escolhida = "Precisão"
df_anova = mdf[mdf["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Teste de Tukey
tukey = pairwise_tukeyhsd(endog=df_anova["value"], groups=df_anova["Classifier"], alpha=0.05)

# Exibir resultado
print(tukey)

metrica_escolhida = "Revocação"
df_anova = mdf[mdf["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Teste de Tukey
tukey = pairwise_tukeyhsd(endog=df_anova["value"], groups=df_anova["Classifier"], alpha=0.05)

# Exibir resultado
print(tukey)

metrica_escolhida = "F1"
df_anova = mdf[mdf["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Teste de Tukey
tukey = pairwise_tukeyhsd(endog=df_anova["value"], groups=df_anova["Classifier"], alpha=0.05)

# Exibir resultado
print(tukey)

