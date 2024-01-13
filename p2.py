# Importa las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importa una proyección 3D para gráficos

# Importa las bibliotecas necesarias para el procesamiento de datos y el modelado
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Carga los datos desde el archivo Excel (reemplaza 'data2.xlsx' con tu archivo)
data = pd.read_excel('data2.xlsx')

# Define las variables predictoras y la variable de respuesta
predictor_variables = ['VC_VIATICOS_AREA', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'CH_VIATICOS_TIPO', 'DC_VIATICOS_COSTO_PASAJES_N', 'DC_VIATICOS_VIA_N']
variable_respuesta = 'DC_VIATICOS_TOTAL_N'

# Divide las variables en categóricas y numéricas
variables_categoricas = ['VC_VIATICOS_AREA', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'CH_VIATICOS_TIPO']
variables_numericas = ['DC_VIATICOS_COSTO_PASAJES_N', 'DC_VIATICOS_VIA_N']

# Preprocesamiento de variables categóricas: codificación one-hot
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Preprocesamiento de variables numéricas: escalado e imputación de valores faltantes
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# ColumnTransformer para aplicar transformaciones a las variables adecuadas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, variables_categoricas),
        ('num', numeric_transformer, variables_numericas)
    ])

# Crea un modelo de regresión lineal
model = LinearRegression()

# Define el pipeline que incluye preprocesamiento y modelo
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Define las variables predictoras y la variable de respuesta
X = data[predictor_variables]
y = data[variable_respuesta]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrena el modelo con los datos de entrenamiento
clf.fit(X_train, y_train)

# Realiza predicciones en los datos de prueba
y_pred = clf.predict(X_test)

# Calcula el error cuadrático medio (MSE) para evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')
# Predicción de costo de viático estándar
new_data = pd.DataFrame({
    'VC_VIATICOS_AREA': ['FACULTAD DE TRABAJO SOCIAL'],
    'DEPARTAMENTO': ['PUNO'],
    'PROVINCIA': ['PUNO'],
    'DISTRITO': ['PUNO'],
    'CH_VIATICOS_TIPO': [2],
    'DC_VIATICOS_COSTO_PASAJES_N': [1500],
    'DC_VIATICOS_VIA_N': [500]
})

predicted_cost = clf.predict(new_data)
print(f'Costo total de viático estimado: {predicted_cost[0]}')

# Crea un gráfico de regresión múltiple 3D (ejemplo con dos variables predictoras)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['DC_VIATICOS_COSTO_PASAJES_N'], X_test['DC_VIATICOS_VIA_N'], y_test, c='b', marker='o', label='Datos reales')
ax.set_xlabel('Costo de Pasajes')
ax.set_ylabel('Viajes')
ax.set_zlabel('Costo Total de Viático')
ax.plot_trisurf(X_test['DC_VIATICOS_COSTO_PASAJES_N'], X_test['DC_VIATICOS_VIA_N'], y_pred, color='r', alpha=0.5, label='Regresión')
plt.legend()
plt.show()
