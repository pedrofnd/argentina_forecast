import datetime
import os
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
path = os.path.abspath(os.path.dirname(__file__))


###------------------------------------------------------------###
#------------ FUNCOES PARA O CODIGO forecast_ARIMA---------------#
###------------------------------------------------------------###

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

def plot_dado_mes_histograma(data, title):
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

def plot_serie_decomposition(data, model, period=12):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    decomposition = seasonal_decompose(data, model=model, period=period)
    # Plotar a componente observada
    axes[0].plot(data.index, decomposition.observed)
    axes[0].set_ylabel('Observado')
    axes[0].set_title('Decomposição dos Dados - Observado')
    # Plotar a componente de tendência
    axes[1].plot(data.index, decomposition.trend)
    axes[1].set_ylabel('Tendência')
    axes[1].set_title('Decomposição dos Dados - Tendência')
    # Plotar a componente sazonal
    axes[2].plot(data.index, decomposition.seasonal)
    axes[2].set_ylabel('Sazonalidade')
    axes[2].set_title('Decomposição dos Dados - Sazonalidade')
    # Plotar os resíduos
    axes[3].plot(data.index, decomposition.resid)
    axes[3].set_ylabel('Resíduos')
    axes[3].set_title('Decomposição dos Dados - Resíduos')
    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(data, lags=20):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Plot da função de autocorrelação
    plot_acf(data, lags=lags, ax=axes[0])
    axes[0].set_title('Função de Autocorrelação (ACF)')
    # Plot da função de autocorrelação parcial
    plot_pacf(data, lags=lags, ax=axes[1])
    axes[1].set_title('Função de Autocorrelação Parcial (PACF)')
    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()

def test_stationarity_dickey_fuller(data,serie):
    result = adfuller(data, autolag='AIC')
    print('---Resultado do Teste de Dickey-Fuller:',serie)
    print(f'Valor do teste: {result[0]}')
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print(f'A série {serie} é estacionária. ')
    else:
        print(f'A série {serie} não é estacionária. ')

def find_best_sarima_params(train_data, validation_data, s=12):
    # Definir a grade de parametrizações
    p_values = range(0, 3)  # Ordem do termo autorregressivo
    d_values = range(0, 2)  # Ordem de diferenciação
    q_values = range(0, 3)  # Ordem do termo de média móvel
    P_values = range(0, 3)  # Ordem do termo autorregressivo sazonal
    D_values = range(0, 2)  # Ordem de diferenciação sazonal
    Q_values = range(0, 3)  # Ordem do termo de média móvel sazonal
    # Criar todas as combinações possíveis de parâmetros
    param_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))
    best_mse = float('inf')
    best_params = None
    # Iterar sobre as combinações de parâmetros
    for params in param_combinations:
        p, d, q, P, D, Q = params
        try:
            # Ajustar o modelo SARIMA aos dados de treinamento
            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
            model_fit = model.fit(disp=False)
            # Fazer previsões nos dados de validação
            predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
            # Calcular o erro quadrático médio
            mse = mean_squared_error(validation_data, predictions)
            # Atualizar os melhores parâmetros se necessário
            if mse < best_mse:
                best_mse = mse
                best_params = params
        except:
            continue
    return best_params

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100