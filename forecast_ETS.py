#objetivo eh utilizar as previsões estatisticas (primeiramente mensais) para determinar carga
import os.path
import pandas as pd
from config import datai, dataf, arquivo1, arquivo2, path
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose



##path com o banco de dados em CSV com o formato extraido a partir do projeto da API-CAMMESA
# path2 = r'BD_historico'
path2 = ""
DB = 'balanco_mensal.csv'
path_complete = os.path.join(path,path2,DB)
print(path_complete)
#ler o dataframe balanco:
df = pd.read_csv(path_complete, sep= ';', header = 0, dtype = str)
#transformar a data para data e energia(mwmed) para valor
columns_to_convert = ['energia(mwh)', 'energia(mwmed)'] #valores
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
#data (tem que adicionar o seculo para o pandas identificar)
# Converte a coluna 'data' para o formato de data correto ('%d/%m/%y')
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
#Pegar so os dados que contem 'OFERTA':
colunas = ['data', 'balanco', 'tipo', 'energia(mwmed)']
df = df[colunas]
df_oferta = df.loc[df['balanco'] == 'OFERTA']
df_oferta.to_csv('db_oferta.csv', index=False, sep= ';')

#gerar por tipo de geracao
grouped = df_oferta.groupby(['data'])
df_oferta_data = grouped.agg({'energia(mwmed)':'sum'})
df_oferta_data = df_oferta_data.reset_index()
df_oferta_data.to_csv('db_oferta_groupdata.csv', index=False, sep= ';')

# ------------------- start dos procedimentos de suavizacao exponencial ------------------- #
#Modelos de suavizacao exponencial
#1)Suavização Exponencial Simples (SES)
#2)Suavização Exponencial de Holt (SEH)
#3)Suavização Exponencial de Holt-Winters (HW - Aditivo/Multiplicativo)
#4)Modelo ETS (Error, Trend, Seasonal)

def plot_four_quadrants(data, title):
    fig = plt.figure(figsize=(12, 8))
    # Primeiro e segundo quadrantes - Série temporal completa
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.plot(data.index, data['energia(mwmed)'])
    ax1.set_title('Série Temporal Completa')
    ax1.set_xlabel('Data')
    ax1.set_ylabel('Energia (mwmed)')
    # Terceiro quadrante - Gráfico por mês e ano
    ax2 = fig.add_subplot(2, 2, 3)
    grouped_monthly = data.groupby([data.index.month, data.index.year]).sum()
    grouped_monthly['energia(mwmed)'].unstack().plot(ax=ax2, marker='o')
    ax2.set_title('Energia por Mês e Ano')
    ax2.set_xlabel('Mês')
    ax2.set_ylabel('Energia (mwmed)')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
    # Quarto quadrante - Histograma
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.hist(data['energia(mwmed)'], bins=20)
    ax3.set_title('Histograma')
    ax3.set_xlabel('Energia (mwmed)')
    ax3.set_ylabel('Frequência')
    # Ajustar layout para evitar sobreposição
    plt.tight_layout()
    # Exibir os gráficos
    plt.show()

def plot_exponential_forecasts(train_data, test_data, predictions_dict, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data['energia(mwmed)'], label='Treinamento')
    plt.plot(test_data.index, test_data['energia(mwmed)'], label='Teste')
    for method, predictions in predictions_dict.items():
        plt.plot(test_data.index, predictions, label=f'Previsões {method}', linestyle='dashed')
    plt.xlabel('Data')
    plt.ylabel('Energia (mwmed)')
    plt.title(title)
    plt.legend()
    plt.show()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#converter para timeseries
df_oferta_data['data'] = pd.to_datetime(df_oferta_data['data'])
df_oferta_data.set_index('data', inplace=True)
df_oferta_data = df_oferta_data.asfreq('MS')
#separar as bases em treino e teste
train_end_date = '2022-12-01'
train_data = df_oferta_data.loc['2016-01-01':'2022-06-01']
test_data = df_oferta_data.loc['2022-07-01':'2023-06-30']

# Criar um dicionário para armazenar as métricas de erro de cada modelo
metrics_dict = {}

#primeiro passo: plotar o grafico temporais (seguindo a ordem data e outro agrupando por meses)
plot_four_quadrants(df_oferta_data, "analise inicial serie temporal")

#---1--- modelos de Suavização Exponencial Simples (SES):
# 1) Treinar o modelo SES nos dados de treinamento a partir da criacao de ums instancia referente a classe 'SimpleExpSmoothing'
ses_model = ExponentialSmoothing(train_data['energia(mwmed)'], trend=None, seasonal=None)
ses_fit = ses_model.fit() #seta os parametros a partir do treino pela base de treino (nível inicial (initial_level)/ Parâmetro de suavização (smoothing_level)
# 2) Fazer previsões nos dados de teste
ses_predictions = ses_fit.predict(start=test_data.index[0], end=test_data.index[-1])
# 3) Calcular a precisão (RMSE) das previsões
metrics_ses = {
    'RMSE': mean_squared_error(test_data['energia(mwmed)'], ses_predictions, squared=False),
    'MAE': mean_absolute_error(test_data['energia(mwmed)'], ses_predictions),
    'MAPE': np.mean(np.abs((test_data['energia(mwmed)'] - ses_predictions) / test_data['energia(mwmed)'])) * 100
}
metrics_dict['SES'] = metrics_ses
# 4) plotar os gráficos (usar funcao)
# plot_exponential_forecasts(train_data, test_data, {'SES': ses_predictions}, 'Previsões de Energia usando Métodos Exponenciais')

#---2---aplicar o 2 modelos de Suavização Exponencial de Holt (SEH):
# 1) Treinar o modelo SES nos dados de treinamento a partir da criacao de ums instancia referente a classe 'SimpleExpSmoothing'
seh_model = ExponentialSmoothing(train_data['energia(mwmed)'], trend='add', seasonal=None)
seh_fit = seh_model.fit() #seta os parametros a partir do treino pela base de treino (nível inicial (initial_level)/ Parâmetro de suavização (smoothing_level)
# 2) Fazer previsões nos dados de teste
seh_predictions = seh_fit.predict(start=test_data.index[0], end=test_data.index[-1])
metrics_seh = {
    'RMSE': mean_squared_error(test_data['energia(mwmed)'], seh_predictions, squared=False),
    'MAE': mean_absolute_error(test_data['energia(mwmed)'], seh_predictions),
    'MAPE': np.mean(np.abs((test_data['energia(mwmed)'] - seh_predictions) / test_data['energia(mwmed)'])) * 100
}
metrics_dict['SEH'] = metrics_seh
# 4) plotar os gráficos (usar funcao)
# plot_exponential_forecasts(train_data, test_data, {'SES': ses_predictions, 'SEH': seh_predictions}, 'Previsões de Energia usando Métodos Exponenciais')

#---3.1---aplicar o 3 modelos de Suavização Exponencial de Holt-Winters (HW - Aditivo):
# 1) Treinar o modelo HW nos dados de treinamento - Aditivo
hw_aditivo_model = ExponentialSmoothing(train_data['energia(mwmed)'], trend='add', seasonal='add', seasonal_periods=12)
hw_aditivo_fit = hw_aditivo_model.fit()
# 2) Fazer previsões nos dados de teste - Aditivo
hw_aditivo_predictions = hw_aditivo_fit.predict(start=test_data.index[0], end=test_data.index[-1])
# 3) Calcular a precisão (RMSE) das previsões - Aditivo
metrics_hw_aditivo = {
    'RMSE': mean_squared_error(test_data['energia(mwmed)'], hw_aditivo_predictions, squared=False),
    'MAE': mean_absolute_error(test_data['energia(mwmed)'], hw_aditivo_predictions),
    'MAPE': np.mean(np.abs((test_data['energia(mwmed)'] - hw_aditivo_predictions) / test_data['energia(mwmed)'])) * 100
}
metrics_dict['HW - Aditivo'] = metrics_hw_aditivo
# 4) plotar os gráficos (usar funcao)
# plot_exponential_forecasts(train_data, test_data, {'SES': ses_predictions, 'SEH': seh_predictions, 'HW - Aditivo': hw_aditivo_predictions}, 'Previsões de Energia usando Métodos Exponenciais')

#---3.2---aplicar o 3 modelos de Suavização Exponencial de Holt-Winters (HW - Multiplicativo):
# 1) Treinar o modelo HW nos dados de treinamento - Multiplicativo
hw_multiplicativo_model = ExponentialSmoothing(train_data['energia(mwmed)'], trend='add', seasonal='mul', seasonal_periods=12)
hw_multiplicativo_fit = hw_multiplicativo_model.fit()
# 2) Fazer previsões nos dados de teste - Multiplicativo
hw_multiplicativo_predictions = hw_multiplicativo_fit.predict(start=test_data.index[0], end=test_data.index[-1])
# 3) Calcular a precisão (RMSE) das previsões - Multiplicativo
metrics_hw_multiplicativo = {
    'RMSE': mean_squared_error(test_data['energia(mwmed)'], hw_multiplicativo_predictions, squared=False),
    'MAE': mean_absolute_error(test_data['energia(mwmed)'], hw_multiplicativo_predictions),
    'MAPE': np.mean(np.abs((test_data['energia(mwmed)'] - hw_multiplicativo_predictions) / test_data['energia(mwmed)'])) * 100
}
metrics_dict['HW - Multiplicativo'] = metrics_hw_multiplicativo
# 4) plotar os gráficos (usar funcao)
# plot_exponential_forecasts(train_data, test_data, {'HW - Aditivo': hw_aditivo_predictions, 'HW - Multiplicativo': hw_multiplicativo_predictions}, 'Previsões de Energia usando Métodos Exponenciais - HW')

#---4---aplicar o 4 modelos ETS (Error, Trend, Seasonal):
# 1) Treinar o modelo ETS nos dados de treinamento
ets_model = ExponentialSmoothing(train_data['energia(mwmed)'], trend='add', seasonal='add', seasonal_periods=12)
ets_fit = ets_model.fit()
print("Parâmetros do modelo ETS:", ets_fit.params)
# 2) Fazer previsões nos dados de teste
ets_predictions = ets_fit.predict(start=test_data.index[0], end=test_data.index[-1])
# 3) Calcular a precisão (RMSE,MAE e MAPE) das previsões
metrics_ets = {
    'RMSE': mean_squared_error(test_data['energia(mwmed)'], ets_predictions, squared=False),
    'MAE': mean_absolute_error(test_data['energia(mwmed)'], ets_predictions),
    'MAPE': np.mean(np.abs((test_data['energia(mwmed)'] - ets_predictions) / test_data['energia(mwmed)'])) * 100
}
metrics_dict['ETS'] = metrics_ets
# 4) plotar os gráficos (usar funcao)
plot_exponential_forecasts(train_data, test_data, {'SES': ses_predictions, 'SEH': seh_predictions, 'HW - Multiplicativo': hw_multiplicativo_predictions, 'ETS': ets_predictions}, 'Previsões de Energia usando Métodos Exponenciais')


# Criar uma figura com quatro subplots e as componentes de decomposicao do ETS
fig, axes = plt.subplots(4, 1, figsize=(10, 10))
decomposition = seasonal_decompose(ets_fit.fittedvalues, model='additive', period=12)
# Plotar a componente observada
decomposition.observed.plot(ax=axes[0], legend=False, color='blue')
axes[0].set_ylabel('Observado')
axes[0].set_title('Decomposição dos Dados - Observado')
# Plotar a componente de tendência
decomposition.trend.plot(ax=axes[1], legend=False, color='red')
axes[1].set_ylabel('Tendência')
axes[1].set_title('Decomposição dos Dados - Tendência')
# Plotar a componente sazonal
decomposition.seasonal.plot(ax=axes[2], legend=False, color='green')
axes[2].set_ylabel('Sazonalidade')
axes[2].set_title('Decomposição dos Dados - Sazonalidade')
# Plotar os resíduos
decomposition.resid.plot(ax=axes[3], legend=False, color='purple')
axes[3].set_ylabel('Resíduos')
axes[3].set_title('Decomposição dos Dados - Resíduos')
# Ajustar o layout para evitar sobreposição
plt.tight_layout()
plt.show()


# Criar o DataFrame a partir do dicionário de métricas e plotar grafico:
metrics_df = pd.DataFrame(metrics_dict)
print(metrics_df)
# Extrair as métricas de interesse para cada modelo
rmse_values = [metrics_dict[model]['RMSE'] for model in metrics_dict]
mae_values = [metrics_dict[model]['MAE'] for model in metrics_dict]
mape_values = [metrics_dict[model]['MAPE'] for model in metrics_dict]

# ----- Modelos para o eixo x: plotagem dos parametros de erro -----
models = list(metrics_dict.keys())
# Criar subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
# Gráfico de barras para RMSE
axs[0].bar(models, rmse_values, color='blue')
axs[0].set_title('Comparação de RMSE entre Modelos')
axs[0].set_xlabel('Modelos')
axs[0].set_ylabel('RMSE')
# Gráfico de barras para MAE
axs[1].bar(models, mae_values, color='green')
axs[1].set_title('Comparação de MAE entre Modelos')
axs[1].set_xlabel('Modelos')
axs[1].set_ylabel('MAE')
# Gráfico de barras para MAPE
axs[2].bar(models, mape_values, color='orange')
axs[2].set_title('Comparação de MAPE entre Modelos')
axs[2].set_xlabel('Modelos')
axs[2].set_ylabel('MAPE')
# Ajustar layout
plt.tight_layout()
# Mostrar os gráficos
plt.show()


#----- testagem da previsao ETS:-----
# Plotar resíduos - ETS
residuals_ets = test_data['energia(mwmed)'] - ets_predictions
plt.figure(figsize=(10, 4))
plt.plot(test_data.index, residuals_ets, label='Resíduos - ETS', color='green')
plt.axhline(y=0, color='gray', linestyle='dashed')
plt.xlabel('Data')
plt.ylabel('Resíduos')
plt.title('Resíduos do Modelo ETS')
plt.legend()
plt.show()
# Plotar um histograma dos resíduos
plt.figure(figsize=(10, 6))
plt.hist(residuals_ets, bins=30, color='blue', edgecolor='black')
plt.title('Histograma dos Resíduos do Modelo ETS')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()
# Plotar Avaliar autocorrelação dos resíduos - ETS
plot_acf(residuals_ets, lags=5)
plt.xlabel('Lags')
plt.ylabel('Autocorrelação')
plt.title('Autocorrelação dos Resíduos - ETS')
plt.show()
# Teste de Ljung-Box para verificar autocorrelação dos resíduos - ETS
lb_test_ets = acorr_ljungbox(residuals_ets, lags=5)
print("lb_test_ets:", lb_test_ets)

##problema de convergencia !!!!
##modelo ETS: A inclusão do parâmetro 'damping_trend'
