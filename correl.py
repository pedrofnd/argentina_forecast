import pandas as pd
from config import path, teste_shapiro, plotar_distribuicao_residuos
import os.path
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


##path com o banco de dados em CSV com o formato extraido a partir do projeto da API-CAMMESA
# path2 = r'BD_historico'
path2 = "DB"
DB_carga = 'balanco_mensal.csv'
DB_temp = 'temperatura_mensal.csv'
path_complete_carga = os.path.join(path, path2, DB_carga)
path_complete_temp = os.path.join(path, path2, DB_temp)
#ler o dataframe balanco:
df_carga = pd.read_csv(path_complete_carga, sep=';', header = 0, dtype = str)
df_temp = pd.read_csv(path_complete_temp, sep=';', header = 0, dtype = str)
#transformar a data para data e energia(mwmed) para valor
columns_to_convert_carga = ['energia(mwh)', 'energia(mwmed)'] #valores
columns_to_convert_temp = ['temperatura'] #valores
for col in columns_to_convert_carga:
    df_carga[col] = pd.to_numeric(df_carga[col].str.replace(',', ''), errors='coerce')
for col in columns_to_convert_temp:
    df_temp[col] = pd.to_numeric(df_temp[col].str.replace(',', ''), errors='coerce')
#data (tem que adicionar o seculo para o pandas identificar)
# Converte a coluna 'data' para o formato de data correto ('%d/%m/%y')
df_carga['data'] = pd.to_datetime(df_carga['data'], format='%d/%m/%Y')
df_temp['data'] = pd.to_datetime(df_temp['data'], format='%d/%m/%Y')
#Pegar so os dados que contem 'OFERTA':
colunas = ['data', 'balanco', 'tipo', 'energia(mwmed)']
df_carga = df_carga[colunas]
df_oferta = df_carga.loc[df_carga['balanco'] == 'OFERTA']
df_oferta.to_csv('db_oferta.csv', index=False, sep= ';')

#gerar por tipo de geracao
grouped = df_oferta.groupby(['data'])
df_oferta_data = grouped.agg({'energia(mwmed)':'sum'})
df_oferta_data = df_oferta_data.reset_index()
df_oferta_data.to_csv('db_oferta_groupdata.csv', index=False, sep= ';')

#criar o dataframe conjunto com a temperatura:
df_oferta_data_temp = df_oferta_data.merge(df_temp, on='data', how='left')

# ------------------- start dos procedimentos de correlacao ------------------- #
#ajustando o dataset para correlacao
df_corr = df_oferta_data_temp.drop(columns='data')
# Renomeie a coluna 'energia(mwmed)' para algo sem parênteses
df_corr.rename(columns={'energia(mwmed)': 'energia_mwmed'}, inplace=True)
print('dataframe antes da remocao de dados',df_corr.describe())

#--processo para retirada de dados anomalos--#
# Calculando z-scores para temp e energia
z_scores_temp = (df_corr['temperatura'] - df_corr['temperatura'].mean()) / df_corr['temperatura'].std()
z_scores_energia = (df_corr['energia_mwmed'] - df_corr['energia_mwmed'].mean()) / df_corr['energia_mwmed'].std()
# Adicionar os valores de z-score ao DataFrame
df_corr['z_score_temp'] = z_scores_temp
df_corr['z_score_energia'] = z_scores_energia
# Definir o limiar para detecção de anomalias
threshold = 2.5
# Identificar as linhas que excedem o limiar e depois excluir as linhas
linhas_anomalas = df_corr[abs(z_scores_energia) >= threshold]
df_corr = df_corr[(np.abs(z_scores_energia) < threshold)]
# Exiba as linhas excluídas em uma única linha
print(f"Linhas com z-score acima do limiar:\n{linhas_anomalas}")
print('dataframe apos a remocao de dados',df_corr.describe())

#Estimando a regressão múltipla OLS
modelo_OLS = sm.OLS.from_formula("energia_mwmed ~ temperatura", df_corr).fit()
##Parâmetros do modelo com intervalos de confiança Nível de significância de 5% / Nível de confiança de 95%
modelo_OLS.conf_int(alpha=0.05)
# Previsões dos modelos
y_pred_OLS = modelo_OLS.predict()
df_corr['y_pred_OLS'] = y_pred_OLS
#Parâmetros do modelo
print(modelo_OLS.summary())

### Transformação de Box-Cox
#Para o cálculo do lambda de Box-Cox
#xt é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
xt, lmbda = boxcox(df_corr['energia_mwmed'])
print("Lambda: ",lmbda)
df_corr['bc_energia'] = xt
# Estimando um novo modelo múltiplo com variável dependente
#transformada por Box-Cox
modelo_bc = sm.OLS.from_formula('bc_energia ~  temperatura', df_corr).fit()
# Previsões dos modelos
y_pred_bc = modelo_bc.predict()
df_corr['y_pred_bc'] = y_pred_bc
y_pred_bc_inv = inv_boxcox(y_pred_bc, lmbda)
df_corr['y_pred_bc_inv'] = y_pred_bc_inv

# Parâmetros do modelo
print(modelo_bc.summary())

# Fit the OLS model using statsmodels
modelo_qd = sm.OLS.from_formula("energia_mwmed ~ temperatura + np.power(temperatura, 2)", data=df_corr).fit()
##Parâmetros do modelo com intervalos de confiança Nível de significância de 5% / Nível de confiança de 95%
modelo_qd.conf_int(alpha=0.05)
# Previsões dos modelos
y_pred_qd = modelo_qd.predict()
df_corr['y_pred_qd'] = y_pred_qd
# Parâmetros do modelo
print(modelo_qd.summary())

# Dados originais para auxiliar na plotagem
x = df_corr['temperatura']
y = df_corr['energia_mwmed']

#calcular o r2 de todos os modelos:
r2_OLS = modelo_OLS.rsquared
r2_bc = modelo_bc.rsquared
r2_qd = modelo_qd.rsquared
print(f"R² do modelo OLS: {r2_OLS}\nR² do modelo Box-Cox: {r2_bc}\nR² do modelo OLS Quadrático: {r2_qd}")

# calcular as metricas
mse = mean_squared_error(y, y_pred_OLS)
mae = mean_absolute_error(y, y_pred_bc_inv)
mape = mean_absolute_percentage_error(y, y_pred_qd)
print("Comecar as mensuracoes dos:Erro Quadrático Médio:", mse, "Erro Absoluto Médio:", mae, "MAPE", mape)

# Calcular os resíduos para ambos os modelos
residuos_ols = y - y_pred_OLS
df_corr['residuos_ols'] = residuos_ols
residuos_bc = y - y_pred_bc_inv
df_corr['residuos_bc'] = residuos_bc
residuos_qd = y - y_pred_qd
df_corr['residuos_qd'] = residuos_qd

#realizar os testes shapiro que estao no modulo config
teste_shapiro(residuos_ols,alpha=0.05,metodo="OLS linear")
teste_shapiro(residuos_bc,alpha=0.05,metodo="OLS linear - Box-Cox")
teste_shapiro(residuos_qd,alpha=0.05,metodo="OLS quadratico")

# Crie uma grade de valores de temperatura para plotar a equação de segundo grau
temperatura_grid = np.linspace(df_corr['temperatura'].min(), df_corr['temperatura'].max(), 100)
# Previsões do modelo OLS Quadrático para a grade de temperatura
y_pred_qd_grid = modelo_qd.predict({'temperatura': temperatura_grid})

# Configurações do estilo do gráfico (opcional, pode personalizar conforme necessário)
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Plot dos dados originais
plt.scatter(x, y, label="Observações", color="b", alpha=0.5)

# Plot das previsões dos modelos
plt.plot(x, y_pred_OLS, label="OLS", color="r")
# Reverta a transformação Box-Cox nos valores ajustados
# y_pred_bc_original = np.exp(y_pred_bc)
plt.plot(x, y_pred_bc_inv, label="Box-Cox", color="g")
plt.plot(temperatura_grid, y_pred_qd_grid, label="OLS Quadrático", color="m")

# Configurações do gráfico
plt.title("Previsões vs. Observações")
plt.xlabel("Temperatura")
plt.ylabel("Energia (mwmed)")
plt.legend()

# Mostrar o gráfico
plt.show()

#plotar a distribuicao dos residuos que estao no modulo config
plotar_distribuicao_residuos(residuos_ols,residuos_qd)