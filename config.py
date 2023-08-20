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


def plot_residual_analysis(test_values, predicted_values, residuals):
    plt.figure(figsize=(12, 10))

    # Plotar os dados reais de teste
    plt.subplot(3, 1, 1)
    plt.plot(test_values.index, test_values, label='Dados de Teste', color='blue')
    plt.plot(predicted_values.index, predicted_values, label='Previsões do Modelo', color='red')
    plt.xlabel('Data')
    plt.ylabel('Energia (mwmed)')
    plt.title('Comparação entre Dados de Teste e Previsões do Modelo SARIMA')
    plt.legend()

    # Plotar a diferença entre os dados reais e as previsões (resíduos)
    plt.subplot(3, 1, 2)
    plt.plot(residuals.index, residuals, label='Diferença (Resíduos)', color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Energia (mwmed)')
    plt.title('Resíduos do Modelo SARIMA')
    plt.legend()

    # Histograma dos resíduos
    plt.subplot(3, 1, 3)
    plt.hist(residuals, bins=20, color='blue', alpha=0.7)
    plt.title("Histograma dos Resíduos")
    plt.xlabel("Valor dos Resíduos")
    plt.ylabel("Frequência")

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()

def test_residuals(residuals):
    # Teste de Ljung-Box para autocorrelação serial nos resíduos
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

    # Teste ARCH para efeitos de heteroscedasticidade condicional nos resíduos
    arch_test = arch_model(residuals)
    arch_test_result = arch_test.fit()
    if arch_test_result.pvalues[-1] > 0.01:
        print("Não há evidência de efeitos ARCH nos resíduos.")
    else:
        print("Há evidência de efeitos ARCH nos resíduos.")


###------------------------------------------------------------###
#------------- FUNCOES PARA O CODIGO forecast_ETS----------------#
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

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_ets_decomposition(ets_fit):
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


def plot_metrics_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    rmse_values = [metrics_dict[model]['RMSE'] for model in models]
    mae_values = [metrics_dict[model]['MAE'] for model in models]
    mape_values = [metrics_dict[model]['MAPE'] for model in models]
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

def analyze_ets_residuals(test_data, ets_predictions):
    # Calcular os resíduos
    residuals_ets = test_data['energia(mwmed)'] - ets_predictions
    # Plotar gráfico dos resíduos
    plt.figure(figsize=(10, 4))
    plt.plot(test_data.index, residuals_ets, label='Resíduos - ETS', color='green')
    plt.axhline(y=0, color='gray', linestyle='dashed')
    plt.xlabel('Data')
    plt.ylabel('Resíduos')
    plt.title('Resíduos do Modelo ETS')
    plt.legend()
    plt.show()
    # Plotar histograma dos resíduos
    plt.figure(figsize=(10, 6))
    plt.hist(residuals_ets, bins=30, color='blue', edgecolor='black')
    plt.title('Histograma dos Resíduos do Modelo ETS')
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()
    # Plotar a autocorrelação dos resíduos
    plot_acf(residuals_ets, lags=5)
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelação')
    plt.title('Autocorrelação dos Resíduos - ETS')
    plt.show()
    # Realizar o teste de Ljung-Box para verificar a autocorrelação dos resíduos
    lb_test_ets = acorr_ljungbox(residuals_ets, lags=5)
    print("lb_test_ets:", lb_test_ets)

