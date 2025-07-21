import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



df_ibovespa = pd.read_csv("Historico_4_Anos_Ibovespa.csv", sep = ",", decimal = ",")

def categoriza_dia(df_var_dia):
    if df_var_dia > 0:
        return "1"
    else:
        return "0"


#tratando VAR e VOL retirando ultimo digiro e trocando , por .
df_ibovespa['Vol.'] = df_ibovespa['Vol.'].str[:-1]
df_ibovespa['Var%'] = df_ibovespa['Var%'].str[:-1]
df_ibovespa['Vol.'] = df_ibovespa['Vol.'].str.replace(',','.')
df_ibovespa['Var%'] = df_ibovespa['Var%'].str.replace(',','.')
df_ibovespa['Vol.'] = df_ibovespa['Vol.'].astype(float)
df_ibovespa['Var%'] = df_ibovespa['Var%'].astype(float)


#alterando formato da data e convertendo colunas de string para float
df_ibovespa["Data"] = pd.to_datetime(df_ibovespa["Data"], format="%d.%m.%Y")

#removendo pontos das colunas de valores e alterando para float
df_ibovespa["Último"] = df_ibovespa["Último"].str.replace(".", "").astype(float)
df_ibovespa["Abertura"] = df_ibovespa["Abertura"].str.replace(".", "").astype(float)
df_ibovespa["Máxima"] = df_ibovespa["Máxima"].str.replace(".", "").astype(float)
df_ibovespa["Mínima"] = df_ibovespa["Mínima"].str.replace(".", "").astype(float) 


#ordenando por data 
df_ibovespa = df_ibovespa.sort_values('Data')

#teste criando bariaveis lag do dia anterior

df_ibovespa['Abertura_d-1'] = df_ibovespa['Abertura'].shift(1)
df_ibovespa['Máxima_d-1'] = df_ibovespa['Máxima'].shift(1)
df_ibovespa['Mínima_d-1'] = df_ibovespa['Mínima'].shift(1)


# Criando minha variavel target, VAR% > 0 = 1 se não = 0
df_ibovespa["Target"] = df_ibovespa["Var%"].apply(categoriza_dia)

df_ibovespa = df_ibovespa.dropna()

# df_ibovespa.tail(3)

# Divisao efetuada conforme é pedido no TC, usando os ultimos 30 dias como teste
train = df_ibovespa.iloc[:-30]
test = df_ibovespa.iloc[-30:]

#selecionando as colunas de features e target

X_train = train[["Abertura","Máxima","Mínima","Abertura_d-1","Máxima_d-1","Mínima_d-1"]]
y_train = train['Target'].astype(int)

X_test = test[["Abertura","Máxima","Mínima","Abertura_d-1","Máxima_d-1","Mínima_d-1"]]
y_test = test['Target'].astype(int)


#criar o modelo de regressão logistica

model = LogisticRegression()


#Adicionando StandardScale

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#trenar o modelo 

model.fit(X_train_scaled, y_train)

#Efetuando previsao

y_pred = model.predict(X_test_scaled)

#testes com valores de dias que não existem na base
#y_pred = model.predict([[139.586,140.049,138.384,138.855,139.695,138.855]]) #02/07/2025
#y_pred = model.predict([[139.051,141.304,139.051,139.586,140.049,138.384]]) #03/07/2025
#y_pred = model.predict([[140.928,141.564,140.597,139.051,141.304,139.051]]) #04/07/2025
#y_pred = model.predict([[141.265,141.342,139.295,140.928,141.564,140.597]]) #05/07/2025

#y_pred = model.predict([[135.298,136.022,134.380,136.187,136.187,134.840]]) #15/07/2025

#print(y_pred)


acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc:.2f} ({acc*100:.1f}%)")



print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Queda (0)", "Alta (1)"]))


# Calcula a matriz de confusão novamente (só por garantia)
cm = confusion_matrix(y_test, y_pred)

#matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

#########################################################################################################

# # Evolução Histórica da base

# plt.figure(figsize=(14, 6))
# plt.plot(df_ibovespa['Data'], df_ibovespa['Último'], label='Fechamento (Último)')
# plt.title('Evolução Histórica do Fechamento do IBOVESPA')
# plt.xlabel('Data')
# plt.ylabel('Índice')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #Distribuição da Variação Diária (% Var)
# plt.figure(figsize=(8, 4))
# sns.histplot(df_ibovespa['Var%'], bins=50, kde=True, color='skyblue')
# plt.title('Distribuição das Variações Diárias do IBOVESPA')
# plt.xlabel('Variação Percentual Diária')
# plt.ylabel('Frequência')
# plt.tight_layout()
# plt.show()


# # Distribuição do Target Clases
# sns.countplot(x='Target', data=df_ibovespa)
# plt.title('Distribuição das Classes (Target)')
# plt.xlabel('Tendência (0 = Queda, 1 = Alta)')
# plt.ylabel('Quantidade de Dias')
# plt.tight_layout()
# plt.show()