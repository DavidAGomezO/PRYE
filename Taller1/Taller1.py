import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def CheckEncoding(path):
  encoding_list = ['ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737'
                  , 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862'
                  , 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950'
                  , 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254'
                  , 'cp1255', 'cp1256', 'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr'
                  , 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2'
                  , 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2'
                  , 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9'
                  , 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab'
                  , 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2'
                  , 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf_32'
                  , 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8', 'utf_8_sig']

  for encoding in encoding_list:
      worked = True
      try:
          df = pd.read_csv(path, encoding=encoding, nrows=5)
      except:
          worked = False
      if worked:
          print(encoding, ':\n', df.head())
          break

# Funcionó para cp437

data = pd.read_csv("PRYE/Taller1/DatosTaller1.csv", encoding='cp437')
df = pd.DataFrame(data)

# Extraer empresas
Empresas = list(set(df["Publisher"]))

# Ventas totales
VentasTotales = {Empresa: 0 for Empresa in Empresas}
Valores = []

for Name in Empresas:
  sum = np.sum(df.loc[df['Publisher'] == Name]["Global_Sales"])
  VentasTotales[Name] = sum
  Valores += [sum]


InvVentas = {Ventas: [] for Ventas in Valores}
for Empresa in Empresas:
  n = VentasTotales[Empresa]
  InvVentas[n] += [Empresa]


Valores.sort(reverse=True)

TopVeinteVentas = {i:[] for i in range(1,21)}

for i in range(1,21):
   TopVeinteVentas[i] += InvVentas[Valores[i-1]] + [VentasTotales[InvVentas[Valores[i-1]][0]]]

# Como hay solo una empresa en estos 20 se puede armar un DF

dfV = pd.DataFrame(TopVeinteVentas).transpose()

print(dfV)

xVals = list(dfV[0])
yVals = list(dfV[1])

fig = plt.figure(figsize=(15,5))

plt.barh(xVals,yVals)

plt.show()