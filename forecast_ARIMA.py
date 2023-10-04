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

#sazonalidade
seasonal_period = 12

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
plot_serie_decomposition(df_oferta_data_ts, model='multiplicative', period=12)

#fazer a testagem de autocorrelacao da serie ACF e PACF:
plot_acf_pacf(train_data, lags=17)

# Aplicando o teste de estacionaridade na série de treino
test_stationarity_dickey_fuller(train_data,"serie treino")

# Aplicar os metodos "I" ndiff de diferenciacao para remover a estacionaridade das series(m-1 e m-11):
diff_period1 = 1
diff_period2 = 2
diff_period3 = 12
#aplicando os testes para os periodos acima
first_diff_train_data = train_data.diff(periods=diff_period1).dropna() #serie ndiff1
# second_diff_train_data = train_data.diff(periods=diff_period2).dropna() #serie ndiff ndiff2
# seasonal_diff_train_data = train_data.diff(periods=diff_period3).dropna() #serie ndiff sazo
# Teste para estacionariedade
test_stationarity_dickey_fuller(first_diff_train_data,f"serie diferenciacao m-1")
# test_stationarity_dickey_fuller(second_diff_train_data,f"serie diferenciacao m-12")
# test_stationarity_dickey_fuller(seasonal_diff_train_data,f"serie diferenciacao m-12")



# #aplicar o metodo BOX-COX para tentar deixar estacionaria
# box_cox_train_data, lambda_value = boxcox(train_data['energia(mwmed)']) #serie BOX-COX
# test_stationarity_dickey_fuller(box_cox_train_data,"serie BOX-COX")


def func_auto_arima(train_data, test_data,seasonal_period):

    #rodar o autoarima:
    autoarima_model_treino = auto_arima(train_data['energia(mwmed)'], seasonal=True, m=seasonal_period, stepwise=True,suppress_warnings=False, trace=False)
    autoarima_result_treino = autoarima_model_treino.fit(train_data['energia(mwmed)'])
    print(autoarima_model_treino.summary())
    # Valores previstos de diferenças sazonais
    predicted_values = autoarima_result_treino.predict(n_periods=len(test_data))

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
    test_results = test_residuals(residuals)
    autocorrelation_result, normality_result, arch_result = test_residuals(residuals)

    # Chamar a função para plotar a análise dos resíduos
    print('vamos rodar a funcao plot_residual_analysis')
    plot_residual_analysis(test_values, predicted_values, residuals)
    return


def func_peter_arima_traindata(train_data, test_data,seasonal_period,i):
    # Descobrir os melhores parametros do SARIMA:
    nome_arima_df= "resultados_arima.csv"
    if os.path.exists(nome_arima_df):
        print("Encontramos o arquivo CSV com os dados 'resultados_arima.csv'")
        best_params_df = pd.read_csv(nome_arima_df, sep=';', decimal='.')
    else:
        print("Não encontramos o arquivo CSV, então vamos rodar a função 'find_best_sarima_params'")
        best_params_df = find_best_sarima_params(train_data, test_data, nome_arima_df, s=seasonal_period, top_n=20)  # Define o número de melhores parâmetros a serem considerados
        best_params_df = pd.read_csv(nome_arima_df, sep=';', decimal='.')

    # Limitar o número de iterações
    num_iterations = 1

    # Criar um dataframe vazio para armazenar os resultados
    results_df = pd.DataFrame(columns=['iteracao', 'parametros', 'mse', 'mae', 'mape', 'predicted_values_str', 'autocorrelacao_erros', 'normalidade_erro', 'arch_erro'])

    # Iterar sobre os 'x' melhores parâmetros do DataFrame
    # i=1
    for idx, row in best_params_df.head(num_iterations).iterrows():
        print(f"------------ ITERACAO DE NUMERO {i} ------------------")
        best_params_str = row['Parametros']
        best_params = ast.literal_eval(best_params_str)  # Convertendo a string de tupla para uma tupla
        params = ast.literal_eval(row['Parametros'])

        # Desempacotar os melhores parâmetros e definir os parametros para ficar o modelo
        p, d, q, ps, ds, qs = best_params
        # Linha para definir manualmente os parâmetros (comente se não for usar)
        # p, d, q, ps, ds, qs = 2, 0, 1, 1, 0, 1
        s = 12  # Periodicidade sazonal (12 para sazonalidade anual)
        max_iter = 500

        # Criar o modelo SARIMA para a série de treino original (train_data)
        sarima_model_treino = SARIMAX(train_data['energia(mwmed)'], order=(p, d, q), seasonal_order=(ps, ds, qs, s))
        sarima_result_treino = sarima_model_treino.fit(max_iter=max_iter)

        # Valores previstos:
        predictions_test = sarima_result_treino.get_forecast(steps=len(test_data))
        predicted_values = predictions_test.predicted_mean
        test_values = test_data['energia(mwmed)']
        # Converter a lista de predicted_values em uma string para salvar no dataframe
        predicted_values_str = "\n".join(predicted_values.apply(str))
        #calcular os residuos:
        residuals = test_values - predicted_values

        #calcular as metricas
        mse = mean_squared_error(test_values, predicted_values)
        mae = mean_absolute_error(test_values, predicted_values)
        mape = mean_absolute_percentage_error(test_values, predicted_values)
        print("Comecar as mensuracoes dos:Erro Quadrático Médio:", mse, "Erro Absoluto Médio:", mae, "MAPE", mape )

        # Chamar a função para testar os resíduos
        print('vamos rodar a funcao test_residuals')
        test_residuals(residuals)
        #colocar a resposta dos residuos em variaveis para salvar no dataframe
        test_results = test_residuals(residuals)
        autocorrelation_result, normality_result, arch_result = test_residuals(residuals)

        # Chamar a função para plotar a análise dos resíduos
        print('vamos rodar a funcao plot_residual_analysis')
        plot_residual_analysis(test_values, predicted_values, residuals)

        results_df.loc[idx] = [idx, params, mse, mae, mape, predicted_values_str, autocorrelation_result, normality_result, arch_result]
        results_df.to_csv('resultados_arima_completo.csv', index=False, sep=';', decimal='.')
        i=i+1
    return

#rodar funcao com a serie normal sem tratamentos:
func_auto_arima(train_data, test_data,seasonal_period)

#rodar funcao com a serie normal sem tratamentos:
# func_peter_arima_traindata(train_data, test_data,seasonal_period, i=1)