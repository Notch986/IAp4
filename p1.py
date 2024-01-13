# Importación de bibliotecas necesarias
import pandas as pd                 # Importa pandas para trabajar con datos tabulares
import numpy as np                  # Importa numpy para operaciones matemáticas
from sklearn.linear_model import LinearRegression   # Importa LinearRegression para regresión lineal
from sklearn.impute import SimpleImputer           # Importa SimpleImputer para manejar valores faltantes
import matplotlib.pyplot as plt       # Importa matplotlib para graficar

# Carga datos desde un archivo Excel ('data1.xlsx')
df = pd.read_excel('data1.xlsx')

# Define las variables predictoras (X) y la variable de respuesta (y)
X = df[['DC_VIATICOS_COSTO_PASAJES_N']]    # Variable predictora: Costo de pasajes
y = df['DC_VIATICOS_VIA_N']                # Variable de respuesta: Costo total del viático

# Crea un imputador que reemplace los valores faltantes (NaN) con la media de la columna correspondiente
imputador = SimpleImputer(strategy='mean')
X = imputador.fit_transform(X)             # Imputa los valores faltantes en X
y = imputador.fit_transform(y.values.reshape(-1, 1))  # Imputa los valores faltantes en y

# Crea un modelo de regresión lineal
modelo = LinearRegression()

# Ajusta el modelo a los datos de X e y
modelo.fit(X, y)

# Calcula el coeficiente de la pendiente (m) y el término independiente (b)
pendiente = modelo.coef_[0]
intercepto = modelo.intercept_

# Imprime la ecuación de la regresión
print(f"La ecuación de la regresión es: DC_VIATICOS_VIA_N = {pendiente} * DC_VIATICOS_COSTO_PASAJES_N + {intercepto}")

# Grafica los datos y la regresión
plt.scatter(X, y, label="Datos")        # Graficar los datos
plt.plot(X, modelo.predict(X), color="red", label="Regresión")  # Graficar la regresión
plt.title("Regresión Lineal Simple")     # Título del gráfico
plt.xlabel("Costo de Pasajes (DC_VIATICOS_COSTO_PASAJES_N)")  # Etiqueta del eje X
plt.ylabel("Costo Total del Viático (DC_VIATICOS_VIA_N)")      # Etiqueta del eje Y
plt.legend()                             # Mostrar leyenda
plt.show()                               # Mostrar el gráfico
