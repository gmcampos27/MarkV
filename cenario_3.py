#Autor: Gabriel Montenegro de Campos
#Bibliotecas
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from sklearn.utils import resample
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import pickle


#Diretório dados
df = pd.read_csv("path/Final_df.csv")

#Manipulação Dados
colunas_remover = ["Unnamed: 0.1", "Unnamed: 0", "Accession", "Organism_Name", "Species", "Genus", "Molecule_type", "Country", "Host", "Query_ID"] #Deixar apenas a coluna categórica Família
df = df.drop(columns=colunas_remover)
df = df.fillna(0)
print("Arquivo lido")


# Divisão dos dados
X = df.drop('Family', axis=1)
y = df['Family']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
print("Divisão dos dados")
#Distribuição
print("Distribuição original:")
print(y.value_counts())

# Distribuição no conjunto de treinamento
print("\nDistribuição no conjunto de treinamento:")
print(y_train.value_counts())

# Distribuição no conjunto de teste
print("\nDistribuição no conjunto de teste:")
print(y_test.value_counts())

#XGBoost

# Realizar o bootstrap
import xgboost as xgb
from xgboost import XGBClassifier
from tqdm import tqdm
print(xgb.__version__)
print("XGBoost")

label_encoder = LabelEncoder()

n_bootstrap = 1000
random_state = 42

f1_xgb = []
precision_xgb = []
MCC_xgb = []
recall_xgb = []
accuracy_xgb = []
feature_importances_xgb = []

# Realizar o bootstrap
for i in tqdm(range(n_bootstrap), desc="Bootstrapping"):
    # Reamostragem com reposição
    X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state + i)

    #LabelEnconder
    y_resampled_encoded = label_encoder.fit_transform(y_resampled)

    # Treinar o modelo na amostra bootstrap
    xgb = XGBClassifier(eval_metric='logloss', random_state=random_state + i)
    xgb.fit(X_resampled, y_resampled_encoded)

     # Avaliar o modelo no conjunto de teste original
    y_pred = xgb.predict(X_test)
    feature_importances_xgb.append(xgb.feature_importances_)

    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    accuracy_xgb.append(accuracy_score(y_test, y_pred_decoded))
    f1_xgb.append(f1_score(y_test, y_pred_decoded, average="weighted"))
    precision_xgb.append(precision_score(y_test, y_pred_decoded, average="weighted"))
    recall_xgb.append(recall_score(y_test, y_pred_decoded, average="weighted"))
    MCC_xgb.append(matthews_corrcoef(y_test, y_pred_decoded))

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

average_feature_importances = np.median(feature_importances_xgb, axis=0)

feature_names = X_train.columns

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': average_feature_importances
}).sort_values(by='Importance', ascending=False)

# Mostrar os valores das features
importance_df.head(10)

# Treinando o modelo
print("Random Forest")

# Parâmetros do bootstrap
n_bootstrap = 1000
random_state = 42

print("Iniciando Bootstrap")
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
    rf = RandomForestClassifier(class_weight="balanced", random_state= random_state + i)
    rf.fit(X_resampled, y_resampled)
    
    # Avaliar o modelo no conjunto de teste original
    y_pred = rf.predict(X_test)
    feature_importances_rf.append(rf.feature_importances_)
    
    # Calcular as métricas
    accuracy_rf.append(accuracy_score(y_test, y_pred))
    f1_rf.append(f1_score(y_test, y_pred, average="weighted"))
    precision_rf.append(precision_score(y_test, y_pred,average = 'weighted'))
    recall_rf.append(recall_score(y_test, y_pred, average='weighted'))
    MCC_rf.append(matthews_corrcoef(y_test, y_pred))

# Calcular estatísticas
accuracy_mean = np.mean(accuracy_rf)
accuracy_ci = (np.percentile(accuracy_rf, 2.5), np.percentile(accuracy_rf, 97.5))

f1_mean = np.mean(f1_rf)
f1_ci = (np.percentile(f1_rf, 2.5), np.percentile(f1_rf, 97.5))

precision_mean = np.mean(precision_rf)
precision_ci = (np.percentile(precision_rf, 2.5), np.percentile(precision_rf, 97.5))

recall_mean = np.mean(recall_rf)
recall_ci = (np.percentile(recall_rf, 2.5), np.percentile(recall_rf, 97.5))

MCC_mean = np.mean(MCC_rf)
MCC_ci= (np.percentile(MCC_rf, 2.5), np.percentile(MCC_rf, 97.5))

# Exibir resultados
print(f"Accuracy: {accuracy_mean:.4f} (95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})")
print(f"Precision: {precision_mean:.4f} (95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})")
print(f"Recall: {recall_mean:.4f} (95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})")
print(f"F1-Score: {f1_mean:.4f} (95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})")
print(f"MCC: {MCC_mean:.4f} (95% CI: {MCC_ci[0]:.4f} - {MCC_ci[1]:.4f})")

average_feature_importances_rf = np.median(feature_importances_rf, axis=0)
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': average_feature_importances_rf
}).sort_values(by='Importance', ascending=False)
importance_df.head(10)


data_ac = {'RF': accuracy_rf,
          'XGB': accuracy_xgb}
data_ac = pd.DataFrame(data_ac).assign(metrica='Acurácia')
pd.DataFrame(data_ac).assign(metrica='Acurácia')

data_rv = {'RF': recall_rf,
          'XGB': recall_xgb}
data_rv = pd.DataFrame(data_rv).assign(metrica='Recall')
pd.DataFrame(data_rv).assign(metrica='Recall')
data_rv

data_pr = {'RF': precision_rf,
          'XGB': precision_xgb}
data_pr = pd.DataFrame(data_pr).assign(metrica='Precisão')

data_f1 = {'RF': f1_rf,
          'XGB': f1_xgb}
data_f1 = pd.DataFrame(data_f1).assign(metrica='F1')

data_mcc = {'RF': MCC_rf,
          'XGB': MCC_xgb}
data_mcc = pd.DataFrame(data_mcc).assign(metrica='MCC')
pd.DataFrame(data_mcc).assign(metrica='MCC')

cdf = pd.concat([data_f1, data_ac, data_pr, data_mcc, data_rv])
mdf = pd.melt(cdf, id_vars=['metrica'], var_name='Classifier')
print(mdf.head())
mdf.to_csv("mdf_metrica_results.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))  # Definir tamanho da figura

ax = sns.boxplot(
    x="metrica", 
    y="value",
    hue="Classifier",
    data=mdf,
    palette=sns.color_palette("hls", 2),
    showmeans=True,
    flierprops={"marker": "."},
    meanprops={"marker": "o",
               "markerfacecolor": "black",
               "markeredgecolor": "black",
               "markersize": 5}
)

plt.xlabel("Métrica")
plt.ylabel("Valor")
plt.title("")
plt.legend(title="Classificador")
plt.savefig("Boxplot.svg", transparent=False, dpi=900, format="svg")
plt.savefig("Boxplot.png", dpi=300, format="png")

import scipy.stats as stats

# Shapiro-Wilk test
stats.shapiro(metricas['value'])

# Histogram and KDE plot
fig, ax = plt.subplots(figsize=(6, 4))
metricas.plot(kind='hist', density=True, ax=ax, alpha=0.7) 
metricas.plot(kind='kde', ax=ax)
plt.title("Distribuição das Métricas")
plt.xlabel("Valor da Métrica")
plt.ylabel("Densidade")
plt.show()

# QQ-plots
fig, ax = plt.subplots(figsize=(14, 4), ncols=2)

# Selecting only the numeric columns for QQ plots
numeric_cols = metricas.select_dtypes(include=np.number).columns

# Plotting for the first numeric column (if available)
if len(numeric_cols) > 0:
    sm.qqplot(metricas[numeric_cols[0]], line='s', ax=ax[0])
    ax[0].set_title(f"QQ-plot {numeric_cols[0]}")

# Plotting for the second numeric column (if available)
if len(numeric_cols) > 1:
    sm.qqplot(metricas[numeric_cols[1]], line='s', ax=ax[1])
    ax[1].set_title(f"QQ-plot {numeric_cols[1]}")

plt.tight_layout()
plt.show()

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Filtrar para uma métrica específica, por exemplo, "Accuracy"
metrica_escolhida = "Acurácia"
df_anova = metricas[metricas["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

metrica_escolhida = "Recall"
df_anova = metricas[metricas["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

metrica_escolhida = "MCC"
df_anova = metricas[metricas["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

metrica_escolhida = "F1"
df_anova = metricas[metricas["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)

metrica_escolhida = "Precisão"
df_anova = metricas[metricas["metrica"] == metrica_escolhida]

# Rodar o modelo ANOVA
modelo = ols("value ~ C(Classifier)", data=df_anova).fit()
anova_tabela = sm.stats.anova_lm(modelo, typ=2)

print(anova_tabela)
