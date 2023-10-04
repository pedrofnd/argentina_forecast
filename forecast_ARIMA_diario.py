#importar as funcoes do config
from config import path
from config import plot_exponential_forecasts, plot_dado_mes_histograma, plot_serie_decomposition, plot_acf_pacf, plot_residual_analysis
from config import test_stationarity_dickey_fuller
from config import find_best_sarima_params
from config import mean_absolute_percentage_error
from config import test_residuals, func_peter_arima_traindata
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

#gerar por tipo de geracao:
grouped = df.groupby(['data'])
df_oferta_data_ts = grouped.agg({'mean': 'sum'})
#AJUSTE PRA NAO MUDAR TUDO:
column_rename_mapping = {'data': 'data','mean': 'energia(mwmed)'}
df_oferta_data_ts = df_oferta_data_ts.rename(columns=column_rename_mapping)
df_oferta_data_ts = df_oferta_data_ts.reset_index()
df_oferta_data_ts.to_csv('db_balanco_groupped.csv', index=False, sep=';')

#converter para timeseries:
df_oferta_data_ts['data'] = pd.to_datetime(df_oferta_data_ts['data'])
df_oferta_data_ts.set_index('data', inplace=True)
df_oferta_data_ts = df_oferta_data_ts.asfreq('D')

#definir a sazonalidade das series:
seasonal_period = 360
#separar as bases em treino e teste
train_end_date = '2022-12-01'
train_data = df_oferta_data_ts.loc['2020-05-01':'2022-12-31']
test_data = df_oferta_data_ts.loc['2023-01-01':'2023-08-10']

# ------------------- start dos procedimentos de ARIMA - metodologia BOX-JENKINS ------------------- #
#-----Inicio do codigo especifico ARIMA

#primeiro passo: plotar o grafico temporais (seguindo a ordem data e outro agrupando por meses)
# plot_dado_mes_histograma(df_oferta_data_ts, "analise inicial serie temporal")

#separar componentes dos dados, rodando para casos aditivos(sazo constante ao longo da serie):
plot_serie_decomposition(df_oferta_data_ts, model='additive', period=12)
# plot_serie_decomposition(df_oferta_data, model='multiplicative', period=12)

#fazer a testagem de autocorrelacao da serie ACF e PACF:
# plot_acf_pacf(train_data, lags=10)

# Aplicando o teste de estacionaridade na série de treino
test_stationarity_dickey_fuller(train_data,"serie treino")

# Aplicar os metodos "I" ndiff de diferenciacao para remover a estacionaridade das series(m-1 e m-11):
diff_period1 = 1
diff_period2 = 2
diff_period3 = 12

#aplicando os testes para os periodos acima
first_diff_train_data = train_data.diff(periods=diff_period1).dropna() #serie ndiff1
second_diff_train_data = train_data.diff(periods=diff_period2).dropna() #serie ndiff ndiff2
seasonal_diff_train_data = train_data.diff(periods=diff_period3).dropna() #serie ndiff sazo

# Teste para estacionariedade
test_stationarity_dickey_fuller(first_diff_train_data,f"serie diferenciacao m-1")
test_stationarity_dickey_fuller(second_diff_train_data,f"serie diferenciacao m-12")
test_stationarity_dickey_fuller(seasonal_diff_train_data,f"serie diferenciacao m-12")

#aplicar o metodo BOX-COX para tentar deixar estacionaria
# box_cox_train_data, lambda_value = boxcox(train_data['energia(mwmed)']) #serie BOX-COX
# test_stationarity_dickey_fuller(box_cox_train_data,"serie BOX-COX")

def func_auto_arima(train_data, test_data,seasonal_period):
    #rodar o autoarima:
    autoarima_model_treino = auto_arima(train_data['energia(mwmed)'], seasonal=True, m=seasonal_period, stepwise=True,suppress_warnings=False, trace=False)
    autoarima_result_treino = autoarima_model_treino.fit(train_data['energia(mwmed)'])
    print(autoarima_model_treino.summary())
    # Valores previstos de diferenças sazonais
    predicted_values = autoarima_result_treino.predict(n_periods=len(test_data))
    print('-----NUMERO DE PREVISOES -------',len(test_data))

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


#rodar funcao com a serie normal sem tratamentos:
func_auto_arima(train_data, test_data,seasonal_period)

#rodar funcao com a serie normal sem tratamentos:
# func_peter_arima_traindata(train_data, test_data, i=1,diferenca)