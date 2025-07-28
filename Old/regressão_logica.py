import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('Historico_4_Anos_Ibovespa.csv')

# Convertendo a coluna 'Data' para o formato datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d.%m.%Y')
# Ordenando o DataFrame pela coluna 'Data'  
df = df.sort_values('Data')

df["Target"] = (df["Último"].shift(-1) > df["Último"]).astype(int)

#Loop para retirar . de milhares e converter as colunas de string para float
for col in ['Último', 'Abertura', 'Máxima', 'Mínima']:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False) # remove separador de milhar
    df[col] = df[col].astype(float)

# Removendo sinal de %, criando nova coluna target e convertendo para float
df['Var%'] = df['Var%'].str.replace(r'%', '', regex=True)
df['Var%'] = df['Var%'].str.replace(r',', '.', regex=True).astype(float)

# Removendo letras e convertendo a coluna de volume para float
df['Vol.'] = df['Vol.'].astype(str).str.replace(r'[A-Za-z]', '', regex=True)
df['Vol.'] = df['Vol.'].str.replace(r',', '.', regex=True).astype(float)

#Ordenando o df pela coluna data do menor para o maior
df = df.sort_values('Data')

#print(df)


# df.info()

# print(df.isna().sum())


df= df[:-1]
# #visualizar
# print(df[['Último', 'Target']].head(5))


# X = df[['Abertura', 'Máxima', 'Mínima']]
# y = df['Target']

# # Dividindo os dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#forma de separa conforme o tech Challinger

#df.iloc[:-30] O : significa "todas as linhas", e o -30  exemplo de 100 linhas ele pega 70
#df.iloc[-30:] fazer o inverso pega apenas as 30 últimas linhas
train = df.iloc[:-30]
test = df.iloc[-30:]

X_train = train[['Abertura', 'Máxima', 'Mínima']]
y_train = train['Target'].astype(int)

X_test = test[['Abertura', 'Máxima', 'Mínima']]
y_test = test['Target'].astype(int)

# Treinando o modelo
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

#calculando a acurácia
accuracy = model.score(X_test, y_test)
print(f"Acurácia do modelo: {accuracy*100:.2f}%")