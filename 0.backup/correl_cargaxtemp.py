import os.path
import pandas as pd
from config import datai, dataf, arquivo1, arquivo2, path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

##path com o banco de dados em CSV com o formato extraido a partir do projeto da API-CAMMESA
# path2 = r'BD_historico'
path2 = ""
DB = 'balanco.csv'
path_complete = os.path.join(path,path2,DB)
print(path_complete)
#ler o dataframe balanco:
df = pd.read_csv(path_complete, sep= ';', header = 0, dtype = str)
#criar um dataframe auxiliar para transformar os valores horarios no formato numerico
df_cargamean_temp = df.iloc[:, 4:28]
colunas = df_cargamean_temp.columns
for col in colunas:
    df_cargamean_temp.loc[:, col] = pd.to_numeric(df_cargamean_temp[col], errors='coerce')
#fazer a media das colunas de [4:28]
df.loc[:,'mean'] = df_cargamean_temp.mean(axis=1)
#substituir no original
common_columns = df.columns.intersection(df_cargamean_temp.columns)
df[common_columns] = df_cargamean_temp[common_columns]
# ajustar a 'DATE' para o formato data
df_balanco_carga = df.iloc[:, 0]
df_balanco_carga = pd.to_datetime(df_balanco_carga, format='%d/%m/%Y')
df['data'] = df_balanco_carga
df.to_csv('df_carga_media.csv',index=False)


##ler o arquivo de temperatura da argentina
base_path_temp = r'Z:\Comercializadora de Energia\6. MIDDLE\38.DATABASE'
path2_temp = 'temperatura.csv'
columns_to_read = ['data','temp_arg', 'temp_arg_min', 'temp_arg_max']
path_complete = os.path.join(base_path_temp,path2_temp)
df_temperatura = pd.read_csv(path_complete, sep= ';', header = 0, dtype = str, usecols=columns_to_read)
#converter as colunas de interesse para numerico
columns_to_convert = ['temp_arg', 'temp_arg_min', 'temp_arg_max']
for col in columns_to_convert:
    df_temperatura[col] = pd.to_numeric(df_temperatura[col], errors='coerce')
#ajustar a coluna DATA:
df3_temp = df_temperatura.iloc[:, 0]
df3_temp = pd.to_datetime(df3_temp, format='%d/%m/%Y')
df_temperatura['data'] = df3_temp
df_temperatura.to_csv('df_temperatura.csv', index=False)

##merge nos dois dataframes:
df_merge = df.merge(df_temperatura, on='data', how='right')
df_merge_fin = df_merge[['data', 'geracao', 'tipo', 'regiao', 'mean', 'temp_arg', 'temp_arg_min', 'temp_arg_max']]
df_merge_fin.to_csv('db_balanco_temp.csv', index=False)

#gerar por tipo de geracao
grouped = df_merge_fin.groupby(['data', 'tipo'])
df_final = grouped.agg({'mean':'sum', 'temp_arg':'mean', 'temp_arg_max':'mean', 'temp_arg_min':'mean'})
df_final = df_final.reset_index()
df_final.to_csv('db_balanco_temp_tipo.csv', index=False)
#gerar pela geracao e importacao
grouped = df_merge_fin.groupby(['data'])
df_final = grouped.agg({'mean':'sum',  'temp_arg':'mean', 'temp_arg_max':'mean', 'temp_arg_min':'mean'})
df_final = df_final.reset_index()
df_final.to_csv('db_balanco_temp_geracao.csv', index=False)

# Extrair os dados das colunas 'temp_arg' e 'mean'
temp_arg_data = df_final['temp_arg']
mean_data = df_final['mean']

# Função para ajustar uma curva gaussiana (normal) aos dados
def gaussian(x, mu, sigma, A):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Criar histograma para 'temp_arg'
plt.figure(figsize=(10, 6))
hist_values, bins, _ = plt.hist(temp_arg_data, bins=15, density=True, alpha=0.6, color='blue', label='temp_arg Histogram')
plt.xlabel('temp_arg')
plt.ylabel('Density')
plt.title('Histogram for temp_arg')

# Ajustar a curva gaussiana aos dados de 'temp_arg'
bin_centers = (bins[:-1] + bins[1:]) / 2
params_temp_arg, _ = curve_fit(gaussian, bin_centers, hist_values, p0=[temp_arg_data.mean(), temp_arg_data.std(), 1])
mu_temp_arg_fit, sigma_temp_arg_fit, A_temp_arg_fit = params_temp_arg
x_range = np.linspace(temp_arg_data.min(), temp_arg_data.max(), 100)
y_gaussian_temp_arg = gaussian(x_range, mu_temp_arg_fit, sigma_temp_arg_fit, A_temp_arg_fit)
plt.plot(x_range, y_gaussian_temp_arg, color='blue', linestyle='--', label='Gaussian Fit')
plt.legend()

# Criar histograma para 'mean'
plt.figure(figsize=(10, 6))
hist_values_mean, bins_mean, _ = plt.hist(mean_data, bins=15, density=True, alpha=0.6, color='orange', label='mean Histogram')
plt.xlabel('mean')
plt.ylabel('Density')
plt.title('Histogram for mean')

# Ajustar a curva gaussiana aos dados de 'mean'
bin_centers_mean = (bins_mean[:-1] + bins_mean[1:]) / 2
params_mean, _ = curve_fit(gaussian, bin_centers_mean, hist_values_mean, p0=[mean_data.mean(), mean_data.std(), 1])
mu_mean_fit, sigma_mean_fit, A_mean_fit = params_mean
x_range_mean = np.linspace(mean_data.min(), mean_data.max(), 100)
y_gaussian_mean = gaussian(x_range_mean, mu_mean_fit, sigma_mean_fit, A_mean_fit)
plt.plot(x_range_mean, y_gaussian_mean, color='orange', linestyle='--', label='Gaussian Fit')
plt.legend()
plt.show()

# Criar histograma para 'temp_arg'
plt.figure(figsize=(10, 6))
plt.hist(temp_arg_data, bins=15, density=True, alpha=0.6, color='blue', label='temp_arg Histogram')
plt.xlabel('temp_arg')
plt.ylabel('Density')
plt.title('Histogram for temp_arg')
plt.legend()
plt.show()

# Criar histograma para 'mean'
plt.figure(figsize=(10, 6))
plt.hist(mean_data, bins=15, density=True, alpha=0.6, color='orange', label='mean Histogram')
plt.xlabel('mean')
plt.ylabel('Density')
plt.title('Histogram for mean')
plt.legend()
plt.show()