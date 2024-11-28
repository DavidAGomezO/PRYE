import math
import numpy as np
import pandas as pd

## Intervalos de Confianza

#   Media dada la desviación estándar poblacional

def X_norm(x,h_0,s,n,z_a2):
  """Intervalo de confianza para probar el valor de una media dada la
  Desviación estándar poblacional

  Parameters
  ----------
  x : float
      media muestral
  h_0 : float
      media a probar
  s : float
      desviación estándar poblacional
  n : int
      tamaño de muestra
  z_a2 : float
      Valor tal que P(Z>z_a2)=a/2 con Z dist. Normal estándar
  """
  sqrn = math.sqrt(n)
  Z = z_a2*s/sqrn
  return (x - Z, x + Z)

#   Media dada la desviación estándar muestral
def T_stud(x,h_0,S,n,t_a2):
  """Intervalo de confianza para probar el valor de una media dada la
  desviación estándar muestral

  Parameters
  ----------
  x : float
      media muestral
  h_0 : float
      media a probar
  S : float
      desviación estándar muestral
  n : int
      tamaño de muestra
  t_a2 : float
      Valor tal que P(T(n-1)>t_a2)=a/2
  """
  T = t_a2*S/math.sqrt(n)
  return (x-T,x+T)


## Prueba de bondad de ajuste

#   Prueba Chi cuadrado

def GradosDeLibertad(DatosObs):
  return max((len(DatosObs[0])-1),1)*max((len(DatosObs)-1),1)

def BondadAjuste(DatosObs,funcMasa,n=0):
  """Prueba de bondad de ajuste para comprobar la distribución de ciertos datos

  Parameters
  ----------
  DatosObs : [[float],[float]]
      Consta de los datos observados en la forma [[X][Y]] siendo X los
      valores posibles y Y la frecencia de dicho valor.
  funcMasa : func(float)->float
      Función de la distribución discreta que se quiere probar
  """

  # Total de datos si no es dado
  if n == 0:
    n = np.sum([np.sum(Dato) for Dato in DatosObs])

  E = [n*funcMasa(x) for x in DatosObs[0]]

  X2 = np.sum(
    [((DatosObs[1][i] - E[i])**2)/E[i] for i in range(len(E))]
  )

  return X2

## Prueba de independencia (categórica)

#   Para dos variables

def PruebaIndependencia(DatosObs,lens=True):
  """Prueba de Independencia para dos variables

  Parameters
  ----------
  DatosObs : [[float]]
      Datos observados a forma de matriz
  n : Bool, optional
      Indica si en los datos vienen incluidas la fila de totales y
      columna de totales
  """
  
  if not(lens):
    for Dato in DatosObs:
      Dato + [np.sum(Dato)]
    TotalCol = [
      np.sum([DatosObs[j][i] for j in range(len(DatosObs))]) for i in range(len(DatosObs[0]))
    ]

    DatosObs + TotalCol
  TotalCol = DatosObs[-1]
  TotalFila = [DatosObs[i][-1] for i in range(len(DatosObs))]

  n = DatosObs[-1][-1]

  E = [
    [TotalFila[i]*TotalCol[j]/n for j in range(len(DatosObs[0])-1)] for i in range(len(DatosObs)-1)
  ]

  X2 = np.sum(
    [
      [((DatosObs[i][j] - E[i][j])**2)/E[i][j] for j in range(len(DatosObs[0])-1)]
      for i in range(len(DatosObs)-1)
    ]
  )
  return X2

Datos = [
  [0,1,2,3],
  [684,255,50,11]
]

def f(x:int):
  return (math.e**(-0.4))*((0.4)**x)/math.factorial(x)

print(BondadAjuste(Datos,f))