import pandas as pd
import matplotlib.pyplot as plt

#ler dataframe e excluir NAs
df = pd.read_csv('DB_geracao.csv',delimiter=";")
df = df.dropna(subset=["mean"])
df = df[(df['mean'] > 5)]
df = df.sort_values('DATA')

#ver as estatisticas
correlacao = df['mean'].corr(df['temp_media'],method='pearson')
descricao = df['mean'].describe()
print("Estatísticas Descritivas:")
print(descricao)
print("Correlação de Pearson:")
print(correlacao)

#GRAFICO DE LINHA:
coluna1 = df['DATA']
coluna2 = df['mean']
# Plotar o gráfico de dispersão
plt.plot(coluna1, coluna2)
plt.xlabel('DATA')
plt.ylabel('mean')
plt.title(f'carga')
# Exibir o gráfico
plt.show()

#GRAFICO DE DISPERSAO:
coluna1 = df['temp_media']
coluna2 = df['mean']
# Plotar o gráfico de dispersão
plt.scatter(coluna1, coluna2)
plt.xlabel('temp_media')
plt.ylabel('mean')
plt.title(f'temperatura x carga')
# Exibir o gráfico
plt.show()

# Plotar o histograma
plt.hist(df['mean'], bins=30, edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.title('Histograma')
# Exibir o histograma
plt.show()