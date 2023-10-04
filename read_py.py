import os
import pandas as pd
from config import path
from config import test_normality, plot_exponential_forecasts, plot_dado_mes_histograma, plot_serie_decomposition, plot_acf_pacf, plot_residual_analysis
from config import test_stationarity_dickey_fuller
from config import find_best_sarima_params
from config import mean_absolute_percentage_error
from config import test_residuals
import statsmodels.api as sm
from scipy.stats import kstest
from scipy import stats


##path com o banco de dados em CSV de OFERTA/CARGA com o formato extraido a partir do projeto da API-CAMMESA
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
df_oferta_data_ts.to_csv('db_carga_groupdata.csv', index=False, sep=';')

#converter para timeseries
df_oferta_data_ts['data'] = pd.to_datetime(df_oferta_data_ts['data'])
df_oferta_data_ts.set_index('data', inplace=True)
df_oferta_data_ts = df_oferta_data_ts.asfreq('MS')

#importar dataframe com as temperaturas mensais:
path2 = r'DB'
# path2 = ""
DB = 'temperatura_mensal.csv'
path_complete = os.path.join(path,path2,DB)
print(path_complete)
#ler o dataframe balanco:
df_temperatura = pd.read_csv(path_complete, sep= ';', header = 0, dtype = str)
#transformar a data para data e energia(mwmed) para valor
columns_to_convert = ['temperatura'] #valores
for col in columns_to_convert:
    df_temperatura[col] = pd.to_numeric(df_temperatura[col].str.replace(',', ''), errors='coerce')
#data (tem que adicionar o seculo para o pandas identificar)
# Converte a coluna 'data' para o formato de data correto ('%d/%m/%y')
df_temperatura['data'] = pd.to_datetime(df_temperatura['data'], format='%d/%m/%Y')

#converter para timeseries
df_temperatura['data'] = pd.to_datetime(df_temperatura['data'])
df_temperatura.set_index('data', inplace=True)
df_temperatura_data_ts = df_temperatura.asfreq('MS')

#plotar os graficos:
# plot_dado_mes_histograma(df_oferta_data_ts, "analise inicial serie temporal - carga")
# plot_dado_mes_histograma(df_temperatura_data_ts, "analise inicial serie temporal - temperatura")
# plotar a decomposicao temporal
# plot_serie_decomposition(df_oferta_data_ts, model='multiplicative', period=12, title="analise inicial serie temporal - carga")
# plot_serie_decomposition(df_temperatura_data_ts, model='multiplicative', period=12,title="analise inicial serie temporal - carga")


# Exemplo de uso:
test_normality(df_oferta_data_ts, alpha=0.05)
test_normality(df_temperatura_data_ts, alpha=0.05)

#estatisticas descritivas:
estatisticas_descritivas_carga = df_oferta_data_ts.describe()
estatisticas_descritivas_temp = df_temperatura_data_ts.describe()
print("estatistica descritiva carga",estatisticas_descritivas_carga)
print("estatistica descritiva temperatura",estatisticas_descritivas_temp)


#correlação entre os dados:
# Suponhamos que 'dados_x' e 'dados_y' sejam seus conjuntos de dados
# Para a correlação de Pearson

correlacao_pearson_oferta_temp = df_oferta_data_ts['energia(mwmed)'].corr(df_temperatura_data_ts['temperatura'], method='pearson')
# Para a correlação de Spearman
correlacao_spearman_oferta_temp = df_oferta_data_ts['energia(mwmed)'].corr(df_temperatura_data_ts['temperatura'], method='spearman')
print("Correlação de Pearson entre OFERTA e TEMPERATURA:")
print(correlacao_pearson_oferta_temp)
print("\nCorrelação de Spearman entre OFERTA e TEMPERATURA:")
print(correlacao_spearman_oferta_temp)