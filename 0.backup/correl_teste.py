import os.path
import pandas as pd
from config import datai, dataf, arquivo1, arquivo2, path

#ler o arquivo de temperatura da argentina
base_path = r'J:\SEDE\Comercializadora de Energia\6. MIDDLE\38.DATABASE'
temp = 'Temperatura.csv'
path_complete = os.path.join(base_path,temp)
df = pd.read_csv(path_complete, sep= ';', header = 0, dtype = str)
print(df)
