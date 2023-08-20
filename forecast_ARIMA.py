#importar as funcoes do config
from config import path
from config import plot_exponential_forecasts, plot_dado_mes_histograma, plot_serie_decomposition, plot_acf_pacf
from config import test_stationarity_dickey_fuller
from config import find_best_sarima_params
from config import mean_absolute_percentage_error
#outras LIBS
import os.path
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import ks_2samp
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import ks_2samp
from arch import arch_model
from itertools import product
from scipy.stats import kstest

##path com o banco de dados em CSV com o formato extraido a partir do projeto da API-CAMMESA
path2 = r'DB'
# path2 = ""
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
df_oferta_data_ts = grouped.agg({'energia(mwmed)': 'sum'})
df_oferta_data_ts = df_oferta_data_ts.reset_index()
df_oferta_data_ts.to_csv('db_oferta_groupdata.csv', index=False, sep=';')

#converter para timeseries
df_oferta_data_ts['data'] = pd.to_datetime(df_oferta_data_ts['data'])
df_oferta_data_ts.set_index('data', inplace=True)
df_oferta_data_ts = df_oferta_data_ts.asfreq('MS')
#separar as bases em treino e teste
train_end_date = '2022-12-01'
train_data = df_oferta_data_ts.loc['2016-01-01':'2022-06-01']
test_data = df_oferta_data_ts.loc['2022-07-01':'2023-06-30']

# ------------------- start dos procedimentos de ARIMA - metodologia BOX-JENKINS ------------------- #
#Modelos ARIMA
#1. Obter base de dados [--ok--]
#2. Transformar em séries temporais [--ok--]
#3. Plotar as séries [--ok--]
##a. Séries temporais contínuas / mensais / histograma
##b. Séries temporais decompostas: continua / tendencia / sazo / erro
#4. Separar as bases em treino e teste [--ok--]
#5. Testagem da autocorrelacao dos dados (Tentar tirar sugestão de algum modelo SARIMA) [--ok--]
##a. ACF (correlaciona com LAGS anteriores da para identificar padroes de tendencia e sazo)
##b. PACF (correlacao direta do LAG x com atual para identificar padroes de tendencia e sazo)
#6. Testagem da Estacionaridade
##a. Aplicação do método das Ndiffs
##b. Transformação BOX-COX
#7. Aplicar um modelo com as seguintes informações:
##a. 5.a) a partir do ACF e PACF
##b. Caso a testagem da estacionaridade acuse algo:
###i. Método 6.a)
###ii. Método 6.b)
#8. Aplicar um modelo com o auto-arima
#9. Testagem dos resíduos
##a. Plot resíduos / histograma / ACF (autocorrelação dos erros)
##b. Teste de Ljung-Box
##c. Teste de Normalidade (Ks)
##d. Arch-Test nos resíduos
#10. Nos modelos de testes finais:
##a. RMSE / MAE / MAPE (caso MAE seja com percentual muito elevado, aplicar outros modelos ARCH ou usar método Ndiffs e BOX-COX)
##b. Checar amplitude do intervalo de confiança

# Criar um dicionário para armazenar as métricas de erro de cada modelo
# metrics_dict = {}


#-----Inicio do codigo especifico ARIMA

#primeiro passo: plotar o grafico temporais (seguindo a ordem data e outro agrupando por meses)
plot_dado_mes_histograma(df_oferta_data_ts, "analise inicial serie temporal")

#separar componentes dos dados, rodando para casos aditivos(sazo constante ao longo da serie):
plot_serie_decomposition(df_oferta_data_ts, model='additive', period=12)
# plot_serie_decomposition(df_oferta_data, model='multiplicative', period=12)

#fazer a testagem de autocorrelacao da serie ACF e PACF:
plot_acf_pacf(train_data, lags=17)

# Aplicando o teste de estacionaridade na série de treino
test_stationarity_dickey_fuller(train_data,"serie treino")

# Aplicar os metodos "I" ndiff de diferenciacao para remover a estacionaridade das series(m-1 e m-11):
seasonal_period = 12 #periodo sazonal
first_diff_train_data = train_data.diff(periods=1).dropna() #serie ndiff
seasonal_diff_train_data = train_data.diff(periods=seasonal_period).dropna() #serie ndiff sazo
test_stationarity_dickey_fuller(first_diff_train_data,"serie diferenciacao m-1")
test_stationarity_dickey_fuller(seasonal_diff_train_data,"serie diferenciacao m-12")
#aplicar o metodo BOX-COX para tentar deixar estacionaria
box_cox_train_data, lambda_value = boxcox(train_data['energia(mwmed)']) #serie BOX-COX
test_stationarity_dickey_fuller(box_cox_train_data,"serie BOX-COX")

# --- agora com todos os testes aplicar o auto-arima na base de treino
# print("----autoarima_model_treino")
# autoarima_model_treino = auto_arima(train_data['energia(mwmed)'], seasonal=True, m=12,stepwise=True,suppress_warnings=False, trace=False)
# autoarima_model_treino.fit(train_data['energia(mwmed)'])
# print(autoarima_model_treino.summary())

#agora com todos os testes aplicar o auto-arima na base de ndiff
# print("----autoarima_model_ndiff")
# autoarima_model_ndiff = auto_arima(first_diff_train_data['energia(mwmed)'], seasonal=True, m=12,stepwise=True,suppress_warnings=False, trace=False)
# autoarima_model_ndiff.fit(first_diff_train_data['energia(mwmed)'])
# print(autoarima_model_ndiff.summary())

# Descobrir os melhores parametros do SARIMA:
best_params = find_best_sarima_params(train_data, test_data)
print("Melhores parâmetros:", best_params)

# --- Definir os parâmetros do modelo SARIMA
p = 2  # Ordem do termo autorregressivo
d = 0  # Ordem de diferenciação
q = 1  # Ordem do termo de média móvel
ps = 1  # Ordem do termo autorregressivo
ds = 0  # Ordem de diferenciação
qs = 1  # Ordem do termo de média móvel
s = 12  # Periodicidade sazonal (12 para sazonalidade anual)
max_iter = 1000
# Criar o modelo SARIMA para a série de treino original (train_data)
sarima_model_treino = SARIMAX(train_data['energia(mwmed)'], order=(p, d, q), seasonal_order=(ps, ds, qs, s))
sarima_result_treino = sarima_model_treino.fit(max_iter=max_iter)
# Valores previstos:
predictions_test = sarima_result_treino.get_forecast(steps=len(test_data))
predicted_values = predictions_test.predicted_mean
test_values = test_data['energia(mwmed)']
#residuos:
residuals = test_values - predicted_values
#metricas
mse = mean_squared_error(test_values, predicted_values)
mae = mean_absolute_error(test_values, predicted_values)
mape = mean_absolute_percentage_error(test_values, predicted_values)
print("Erro Quadrático Médio:", mse, "Erro Absoluto Médio:", mae, "MAPE", mape )
# Plotar os dados reais de teste
plt.figure(figsize=(12, 6))
plt.plot(test_values.index, test_values, label='Dados de Teste', color='blue')
# Plotar as previsões do modelo
plt.plot(predicted_values.index, predicted_values, label='Previsões do Modelo', color='red')
# Plotar a diferença entre os dados reais e as previsões
plt.plot(residuals.index, residuals, label='Diferença (Resíduos)', color='green')
plt.xlabel('Data')
plt.ylabel('Energia (mwmed)')
plt.title('Comparação entre Dados de Teste e Previsões do Modelo SARIMA')
plt.legend()
plt.show()

# Sumário dos resultados dos modelos
# print("Resultado do modelo SARIMA para a série de treino original:")
# print(sarima_result_treino.summary())
# # Criar o modelo SARIMA para a série após o procedimento de diferenciação ndiff (first_diff_train_data)
# sarima_model_ndiff = SARIMAX(first_diff_train_data['energia(mwmed)'], order=(p, d, q), seasonal_order=(p, d, q, s))
# sarima_result_ndiff = sarima_model_ndiff.fit()
# # Sumário dos resultados dos modelos
# print("\nResultado do modelo SARIMA para a série após diferenciação ndiff:")
# print(sarima_result_ndiff.summary())


# --- Teste de Ljung-Box para autocorrelação serial nos resíduos
lb_test = acorr_ljungbox(residuals, lags=6)
p_values_ljung_box = lb_test['lb_pvalue']
min_p_value_idx = lb_test['lb_pvalue'].idxmin()
min_p_value = lb_test['lb_pvalue'].min()
if all(p > 0.01 for p in p_values_ljung_box):
    print("Não há evidência de autocorrelação serial nos resíduos. LAG",min_p_value_idx,'valor',min_p_value)
else:
    print("Há evidência de autocorrelação serial nos resíduos. LAG",min_p_value_idx,'valor',min_p_value)

# Teste de Kolmogorov-Smirnov para normalidade dos resíduos
ks_statistic, ks_p_value = kstest(residuals, 'norm')
alpha = 0.05  # Nível de significância
if ks_p_value > alpha:
    print("Os resíduos seguem uma distribuição normal (p-value:", ks_p_value, ")")
else:
    print("Os resíduos não seguem uma distribuição normal (p-value:", ks_p_value, ")")
# Plotar os resíduos e o histograma
plt.figure(figsize=(10, 8))
# Resíduos
plt.subplot(2, 1, 1)
plt.plot(residuals, marker='o', linestyle='-', color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Resíduos do Modelo SARIMA")
plt.ylabel("Resíduos")
plt.grid(True)
# Histograma dos resíduos
plt.subplot(2, 1, 2)
plt.hist(residuals, bins=20, color='blue', alpha=0.7)
plt.title("Histograma dos Resíduos")
plt.xlabel("Valor dos Resíduos")
plt.ylabel("Frequência")
plt.grid(True)
plt.tight_layout()
plt.show()

# Teste ARCH para efeitos de heteroscedasticidade condicional nos resíduos
arch_test = arch_model(residuals)
arch_test_result = arch_test.fit()
if arch_test_result.pvalues[-1] > 0.01:
    print("Não há evidência de efeitos ARCH nos resíduos.")
else:
    print("Há evidência de efeitos ARCH nos resíduos.")

#testagem dos residuos:
# print(autoarima_model_treino.resid)
# test_and_plot_resid(autoarima_model_treino)
# test_and_plot_resid(autoarima_model_ndiff)