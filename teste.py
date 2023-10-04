#importar as funcoes do config
from config import path
from config import plot_exponential_forecasts, plot_dado_mes_histograma, plot_serie_decomposition, plot_acf_pacf, plot_residual_analysis
from config import test_stationarity_dickey_fuller
from config import find_best_sarima_params
from config import mean_absolute_percentage_error
from config import test_residuals
#outras LIBS
import os.path
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
import ast
from pmdarima import auto_arima


##path com o banco de dados em CSV com o formato extraido a partir do projeto da API-CAMMESA
path2 = r'DB'
# path2 = ""
DB = 'balanco_temp_diario.csv'
path_complete = os.path.join(path,path2,DB)
print(path_complete)
#ler o dataframe balanco:
df = pd.read_csv(path_complete, sep= ';', header = 0, dtype = str)
#transformar a data para data e energia(mwmed) para valor
columns_to_convert = ['mean'] #valores
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')
#data (tem que adicionar o seculo para o pandas identificar)
# Converte a coluna 'data' para o formato de data correto ('%d/%m/%y')
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
#Pegar so os dados que contem 'data':
colunas = ['data', 'mean']
df = df[colunas]
# df_oferta = df.loc[df['balanco'] == 'OFERTA']
# df_oferta.to_csv('db_oferta.csv', index=False, sep= ';')

#gerar por tipo de geracao
grouped = df.groupby(['data'])
df_oferta_data_ts = grouped.agg({'mean': 'sum'})
#AJUSTE PRA NAO MUDAR TUDO:
column_rename_mapping = {'data': 'data','mean': 'energia(mwmed)'}
df_oferta_data_ts = df_oferta_data_ts.rename(columns=column_rename_mapping)
df_oferta_data_ts = df_oferta_data_ts.reset_index()
df_oferta_data_ts.to_csv('db_balanco_groupped.csv', index=False, sep=';')

#converter para timeseries
df_oferta_data_ts['data'] = pd.to_datetime(df_oferta_data_ts['data'])
df_oferta_data_ts.set_index('data', inplace=True)
df_oferta_data_ts = df_oferta_data_ts.asfreq('D')
print(df_oferta_data_ts)

#separar as bases em treino e teste
train_end_date = '2022-12-01'
train_data = df_oferta_data_ts.loc['2020-05-01':'2022-12-31']
test_data = df_oferta_data_ts.loc['2023-01-01':'2023-08-10']

# ------------------- start dos procedimentos de ARIMA - metodologia BOX-JENKINS ------------------- #

# Aplicar os metodos "I" ndiff de diferenciacao para remover a estacionaridade das series(m-1 e m-11):
diferenca_dia = 1
first_diff_train_data = train_data.diff(periods=1).dropna() #serie ndiff

# ------- definir qual a data de diferenca que vai usar ---------- IMPORTANTEEEE
diff_data = first_diff_train_data

seasonal_period = 30  #periodo sazonal

def func_auto_arima_diff(train_data, diff_data, test_data, seasonal_period, diferenca_dia):

    #ajustando as bases de dados
    train_data_base = train_data.copy()
    train_data = diff_data

    #rodar o autoarima:
    autoarima_model_treino = auto_arima(train_data['energia(mwmed)'], seasonal=True, m=seasonal_period, stepwise=True,suppress_warnings=False, trace=False)
    autoarima_result_treino = autoarima_model_treino.fit(train_data['energia(mwmed)'])
    print(autoarima_model_treino.summary())

    # Valores previstos de diferenças sazonais
    predicted_diff_values = autoarima_result_treino.predict(n_periods=len(test_data))
    # Reverter a diferenciação sazonal para os dados originais de teste
    last_seasonal_values = train_data_base['energia(mwmed)'].iloc[-diferenca_dia:]
    # Criar um DataFrame para armazenar os valores previstos e alinhar os índices
    predicted_values_df = pd.DataFrame({'predicted_diff_values': predicted_diff_values}, index=test_data.index)

    # Adicionar os valores previstos aos valores sazonais originais
    predicted_values_df['predicted_values'] = last_seasonal_values.values + predicted_values_df['predicted_diff_values']
    predicted_values = predicted_values_df['predicted_values']

    #continuando
    test_values = test_data['energia(mwmed)']
    # calcular os residuos:
    residuals = test_values - predicted_values

    # calcular as metricas
    mse = mean_squared_error(test_values, predicted_values)
    mae = mean_absolute_error(test_values, predicted_values)
    mape = mean_absolute_percentage_error(test_values, predicted_values)
    print("Comecar as mensuracoes dos:Erro Quadrático Médio:", mse, "Erro Absoluto Médio:", mae, "MAPE", mape)
    # Chamar a função para testar os resíduos
    print('vamos rodar a funcao test_residuals')
    test_residuals(residuals)

    # Chamar a função para plotar a análise dos resíduos
    print('vamos rodar a funcao plot_residual_analysis')
    plot_residual_analysis(test_values, predicted_values, residuals)
    return

func_auto_arima_diff(train_data, diff_data, test_data, seasonal_period, diferenca_dia)