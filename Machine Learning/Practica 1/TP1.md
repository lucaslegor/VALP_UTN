## Guía de ejercicios prácticos 1

- 1. ¿Qué es Pandas? ¿Cuál es su relación con numpy?

- 2. ¿Por qué motivo un arreglo en numpy (ndarray) es más eficiente que una lista de Python?

- 3. Investigue si existen otras alternativas a Pandas.

- 4. A partir del dataset disponible en https://www.kaggle.com/datasets/abhishek14398/salary-dataset-sim ple-linear-regression: [URL 🔗](https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression)

- a) Cargue el dataset en un dataframe de Pandas.

- b) Imprima en pantalla un breve resumen estadístico del dataframe (Pista: describe()).

- c) ¿Qué información se incluye en el resumen?

- d) ¿Cuál es la descripción de cada atributo?

- e) Tomando como partida las columnas ”YearsExperience” y ”Salary” realice un gráfico de dispersión. ¿Cuál es la relación, a simple vista, entre ambas variables?

- 5. ¿Qué es scikit-learn?

- 6. Enumere 5 librerías similares y describa al menos 2.

- 7. Liste al menos 5 algoritmos de Aprendizaje Automático disponibles en scikit-learn. Describa uno.

## Imputación de valores faltantes

Por diversas razones, muchos conjuntos de datos del mundo real contienen valores faltantes, que a menudo se codifican como espacios en blanco, NaN (Not a Number), etc. Una estrategia básica para utilizar conjuntos de datos incompletos consiste en descartar filas y/o columnas completas que contengan valores faltantes. Sin embargo, esto implica la pérdida de datos que podrían ser valiosos (aunque estén incompletos). Una estrategia mejor es imputar los valores faltantes, es decir, inferirlos a partir de la parte conocida de los datos. https://scikit-learn.org/stable/modules/impute.html [URL 🔗](https://scikit-learn.org/stable/modules/impute.html)

- 8. ¿Cuál es la diferencia entre imputación univariable (SimpleImputer) y multivariable (IterativeImputer)?

- 9. A partir del dataset disponible en https://www.kaggle.com/datasets/sharmagayatri/data-science-job-c sv?select=data_science_job.csv: [URL 🔗](https://www.kaggle.com/datasets/sharmagayatri/data-science-job-csv?select=data_science_job.csv)

- a) Cargue el dataset en un dataframe de Pandas.

- b) ¿El dataset contiene valores NaN? Pista: isna() y sum().

- c) ¿Qué hace el siguiente fragmento de código? Compare la salida obtenida en este inciso con la salida del anterior.

- 1 from sklearn.impute import MissingIndicator

- 2 mask = MissingIndicator(features="all").fit_transform(df)

- 3 print(pd.DataFrame(mask, columns=df.columns).sum())

- d) Describa el propósito del siguiente fragmento de código. Señale las principales diferencias entre ambos mecanismos de imputación.


```
1 import pandas as pd
2 from sklearn.experimental import enable_iterative_imputer
3 from sklearn.impute import SimpleImputer, IterativeImputer
4 X = df[["city_development_index", "experience", "training_hours"]]
5 simple = SimpleImputer(strategy="median").fit_transform(X)
6 iterative = IterativeImputer(random_state=0).fit_transform(X)
7 rows, cols = np.where(X.isna().to_numpy())
8 imputed = pd.DataFrame({
9 "row": X.index[rows],
10 "column": X.columns[cols],
11 "original": X.to_numpy()[rows, cols],
12 "simple": simple[rows, cols],
13 "iterative": iterative[rows, cols],
14 })
15 print(len(imputed))
16 print(imputed.head(12))
```
