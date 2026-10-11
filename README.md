# Machine Learning ? Aplicaci?n Pr?ctica 1

Clasificaci?n supervisada no probabil?stica de la rotaci?n de empleados
(*employee attrition*) y comparaci?n de estrategias de selecci?n de caracter?sticas.
El proyecto incluye los experimentos terminados y un informe cient?fico en LaTeX.

## Datos

Se utiliza **IBM HR Analytics Employee Attrition & Performance**, disponible en
[Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).

- 1.470 observaciones y 35 columnas originales, incluida la variable objetivo.
- Objetivo: `Attrition`, codificado como `No = 0` y `Yes = 1`.
- Distribuci?n: 1.233 casos `No` (83,88 %) y 237 casos `Yes` (16,12 %).
- Sin valores ausentes.
- Tras eliminar `EmployeeNumber`, `EmployeeCount`, `Over18` y `StandardHours`,
  quedan 30 predictores: 23 num?ricos y 7 categ?ricos.

Los predictores num?ricos se estandarizan con `StandardScaler` y los categ?ricos
se codifican con `OneHotEncoder(handle_unknown="ignore")`. Las variables enteras
ordinales conservan su representaci?n num?rica ordenada. La representaci?n
completa contiene 51 caracter?sticas transformadas.

## Estructura

```text
data/
    raw/           # Dataset original
    processed/     # Carpeta prevista para datos procesados
notebooks/         # An?lisis, preprocesamiento y experimentos
figures/           # Figuras exportadas en PDF
results/           # M?tricas, subconjuntos y comparaciones en CSV
report/
    main.tex       # Documento principal
    main.pdf       # Informe compilado
    references.bib # Bibliograf?a
    sections/      # Secciones del informe
    tables/        # Tablas exportadas en LaTeX
README.md
requirements.txt
```

El c?digo experimental est? en los notebooks. No se necesita una carpeta `src/`.

## Notebooks

| Orden | Archivo | Contenido |
|---|---|---|
| 1 | `01_data_understanding.ipynb` | Estructura, tipos de variables y calidad de los datos |
| 2 | `02_eda.ipynb` | An?lisis exploratorio y asociaciones con la rotaci?n |
| 3 | `03_preprocessing.ipynb` | Definici?n e inspecci?n del preprocesamiento |
| 4 | `04_baseline_models.ipynb` | Evaluaci?n con todos los predictores v?lidos |
| 5 | `05_univariate_selection.ipynb` | Filtro univariante |
| 6 | `06_multivariate_selection.ipynb` | Filtro multivariante |
| 7 | `07_wrapper_selection.ipynb` | Selecci?n mediante wrapper |
| 8 | `08_final_analysis.ipynb` | Comparaci?n final, solapamiento, figuras y tablas |

## Dise?o experimental

Se comparan cuatro clasificadores: K-NN, ?rbol de Decisi?n, SVM y Red Neuronal.
Se mantienen sus configuraciones predeterminadas o m?nimas:

```python
KNeighborsClassifier()
DecisionTreeClassifier(random_state=42)
SVC()
MLPClassifier(random_state=42, max_iter=2000)
```

Las cuatro representaciones evaluadas son:

1. **Todas las caracter?sticas:** 51 caracter?sticas transformadas.
2. **Filtro univariante:** `SelectKBest(score_func=f_classif, k=15)`.
3. **Filtro multivariante:** selector voraz inspirado en mRMR, con relevancia
   medida mediante informaci?n mutua y redundancia mediante correlaci?n absoluta.
   Retiene 15 caracter?sticas; es una implementaci?n propia, no el mRMR can?nico.
4. **Wrapper:** selecci?n secuencial hacia delante de 15 caracter?sticas,
   utilizando un ?rbol de Decisi?n, puntuaci?n F1 y validaci?n interna de 5 folds.

La evaluaci?n externa utiliza `StratifiedKFold(n_splits=10, shuffle=True,
random_state=42)`. El preprocesamiento y la selecci?n se ajustan dentro de cada
fold de entrenamiento externo mediante pipelines. En el wrapper, el
preprocesamiento se ajusta sobre ese fold externo antes de la validaci?n interna
del selector; el fold de prueba externo queda excluido durante todo el ajuste.

Se registran accuracy, precision, recall, F1 y balanced accuracy. Precision,
recall y F1 corresponden a `Attrition=Yes`. Por el desequilibrio de clases,
la comparaci?n prioriza F1 y balanced accuracy, con atenci?n al recall.
No se aplican ajuste de hiperpar?metros, remuestreo ni ponderaci?n de clases.

Los listados de caracter?sticas seleccionadas proceden de ajustes separados
sobre todos los datos para interpretaci?n. Su solapamiento indica coincidencia
entre estrategias, no frecuencia de selecci?n entre folds. Las categor?as
one-hot se agrupan por variable original para interpretarlas.

## Resultados principales

La tabla muestra el clasificador con mayor F1 medio en cada estrategia:

| Estrategia | Clasificador | F1 | Balanced accuracy |
|---|---|---:|---:|
| Todas las caracter?sticas | Red Neuronal | 0.4738 | 0.6765 |
| Filtro univariante | ?rbol de Decisi?n | 0.3809 | 0.6347 |
| Filtro multivariante | Red Neuronal | 0.3902 | 0.6341 |
| Wrapper | Red Neuronal | 0.3919 | 0.6286 |

La Red Neuronal con todas las caracter?sticas obtiene el mejor F1 y balanced
accuracy observados. SVM con todas las caracter?sticas alcanza la mayor accuracy
(0.8653) y precision (0.8363), pero su recall es 0.2241.

La selecci?n reduce la dimensionalidad de 51 a 15 caracter?sticas (70,59 %).
Beneficia a algunos clasificadores, especialmente K-NN con el filtro univariante
y el wrapper, pero no supera al mejor modelo con la representaci?n completa.
Ninguna estrategia de selecci?n es universalmente la mejor.

Las variables originales seleccionadas por las tres estrategias son
`BusinessTravel`, `JobInvolvement`, `JobRole`, `MaritalStatus`, `MonthlyIncome`,
`OverTime` y `StockOptionLevel`. Estas coincidencias describen asociaciones
relevantes para la clasificaci?n; no implican causalidad.

## Entorno y reproducci?n

Desde la ra?z del proyecto, crea y activa un entorno virtual si lo necesitas:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para ejecutar los notebooks con sus rutas relativas, inicia Jupyter desde
`notebooks/`:

```powershell
cd notebooks
jupyter notebook
```

Ejecuta los notebooks en el orden indicado para reproducir el flujo completo.
Las salidas ya est?n disponibles en `results/`, `figures/` y `report/tables/`;
no hace falta volver a entrenar modelos para consultar o compilar el informe.
La ejecuci?n de los experimentos vuelve a escribir sus archivos de salida.
El wrapper puede requerir m?s tiempo por sus evaluaciones internas.

`requirements.txt` enumera las dependencias sin fijar versiones. Por ello,
la reproducci?n en otro entorno puede presentar diferencias entre versiones.

## Informe cient?fico

El informe utiliza un formato de una columna y 11 puntos, con introducci?n,
descripci?n del problema, metodolog?a, resultados, discusi?n, conclusi?n y
referencias. El PDF actual tiene seis p?ginas. PA2 todav?a no est? incluida.

Para compilar, necesitas una distribuci?n LaTeX con `latexmk`, `pdflatex` y
BibTeX. Desde la ra?z del proyecto:

```powershell
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

El resultado es `report/main.pdf`. La compilaci?n regenera los archivos
auxiliares de LaTeX. El documento reutiliza `figures/f1_comparison.pdf` y las
tablas `best_per_experiment.tex` y `selected_variables.tex`.

La bibliograf?a incluye la fuente del dataset, el art?culo de scikit-learn,
la documentaci?n oficial de selecci?n de caracter?sticas y el art?culo original
de mRMR. Las referencias completas est?n en `report/references.bib`.

## Alcance

Los resultados corresponden a un dataset sint?tico/ficticio y al dise?o
experimental descrito. Los subconjuntos se fijaron en 15 caracter?sticas
transformadas para compararlos. No se encontr? en el repositorio un enunciado
oficial ni una lista de algoritmos de clase con los que comprobar requisitos
adicionales a los proporcionados para esta pr?ctica.
