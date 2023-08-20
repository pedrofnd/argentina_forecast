#objetivo eh utilizar as previsões estatisticas (primeiramente mensais) para determinar carga
import os.path
import pandas as pd
from config import path, mean_absolute_percentage_error
from config import plot_exponential_forecasts, plot_four_quadrants, plot_ets_decomposition, plot_metrics_comparison
from config import analyze_ets_residuals
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
path2 = "DB"
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

# Chamando a função com o objeto 'ets_fit'
plot_ets_decomposition(ets_fit)

# Criar o DataFrame a partir do dicionário de métricas e plotar grafico:
metrics_df = pd.DataFrame(metrics_dict)
print(metrics_df)
# Extrair as métricas de interesse para cada modelo
rmse_values = [metrics_dict[model]['RMSE'] for model in metrics_dict]
mae_values = [metrics_dict[model]['MAE'] for model in metrics_dict]
mape_values = [metrics_dict[model]['MAPE'] for model in metrics_dict]

# Chamando a função com o dicionário de métricas
plot_metrics_comparison(metrics_dict)

# Chamando a função com os dados de teste e previsões do modelo ETS
analyze_ets_residuals(test_data, ets_predictions)

##problema de convergencia !!!!
##modelo ETS: A inclusão do parâmetro 'damping_trend'
