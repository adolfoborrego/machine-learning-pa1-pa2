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

Para los cuatro clasificadores originales, los predictores num?ricos se estandarizan con `StandardScaler` y los categ?ricos
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

El código experimental está en los notebooks. `notebooks/pa1/pa1_utils.py`
comparte la evaluación de RIPPER y el selector mRMR, sin datos ni estado ajustado al importar.

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
| 8 | `08_final_analysis.ipynb` | Comparación final, reglas, figuras y tablas |

## Dise?o experimental

Los cinco clasificadores son K-NN, Árbol de Decisión, SVM, Red Neuronal
y RIPPER. Se mantienen sus configuraciones predeterminadas o mínimas:

```python
KNeighborsClassifier()
DecisionTreeClassifier(random_state=42)
SVC()
MLPClassifier(random_state=42, max_iter=2000)
RIPPER(random_state=42)  # wittgenstein==0.3.5
```

Las cuatro estrategias experimentales son:

1. **Todas las características:** los 30 predictores válidos; 51 columnas
   transformadas para K-NN, Árbol de Decisión, SVM y Red Neuronal.
2. **Filtro univariante:** `SelectKBest(score_func=f_classif, k=15)`.
3. **Filtro multivariante:** selector voraz inspirado en mRMR, con relevancia
   medida mediante informaci?n mutua y redundancia mediante correlaci?n absoluta.
   Retiene 15 caracter?sticas; es una implementaci?n propia, no el mRMR can?nico.
4. **Wrapper:** selección secuencial hacia delante. Para los cuatro modelos
   de scikit-learn usa 15 columnas transformadas, Árbol de Decisión,
   puntuación F1 y cinco folds internos; RIPPER busca predictores originales
   con F1, dos folds internos y hasta tres adiciones.

La evaluaci?n externa utiliza `StratifiedKFold(n_splits=10, shuffle=True,
random_state=42)`. El preprocesamiento y la selecci?n se ajustan dentro de cada
fold de entrenamiento externo; los modelos de scikit-learn usan pipelines y
RIPPER usa ajustes expl?citos sobre DataFrames. En el wrapper basado en ?rbol, el
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


### Representaciones y reproducción

RIPPER induce reglas IF-THEN y recibe variables nominales como strings y
variables numéricas sin escalar. Aprende sus intervalos numéricos solo
con datos de entrenamiento. En los filtros usa las mismas 15 columnas
seleccionadas que los otros clasificadores: las numéricas recuperan sus
unidades originales y los indicadores dummy permanecen binarios. Su wrapper
selecciona predictores originales; por ello, sus recuentos de variables no
son directamente comparables con los de columnas transformadas. Cada
candidato se ajusta dentro del entrenamiento interno, sin usar el fold
externo de prueba.

Cada notebook 04-07 exporta `*_results.csv` y `*_summary.csv` con los cinco
clasificadores. `results/pa1/ripper_fold_results.csv` conserva las 40
evaluaciones de RIPPER con reglas, selecciones y métricas para interpretar
los modelos. Después se ejecuta `08_final_analysis.ipynb`, que combina las
cuatro estrategias para generar tablas y figuras.

Los notebooks se ejecutan desde `notebooks/pa1/`, donde está el módulo
compartido. Cada uno carga sus datos sin depender del estado de otro kernel;
las comparaciones de 05-07 requieren los CSV de las etapas anteriores.

## Resultados principales

La tabla muestra el clasificador con mayor F1 medio en cada estrategia:

| Estrategia | Clasificador | F1 | Balanced accuracy |
|---|---|---:|---:|
| Todas las caracter?sticas | Red Neuronal | 0.4738 | 0.6765 |
| Filtro univariante | ?rbol de Decisi?n | 0.3809 | 0.6347 |
| Filtro multivariante | Red Neuronal | 0.3902 | 0.6341 |
| Wrapper | RIPPER | 0.3996 | 0.6557 |

La Red Neuronal con todas las caracter?sticas obtiene el mejor F1 y balanced
accuracy observados. SVM con todas las caracter?sticas alcanza la mayor accuracy
(0.8653) y precision (0.8363), pero su recall es 0.2241.

Para K-NN, Árbol de Decisión, SVM y Red Neuronal, la selección reduce
la dimensionalidad de 51 a 15 columnas transformadas (70,59 %).
Beneficia a algunos clasificadores, especialmente K-NN con el filtro univariante
y el wrapper, pero no supera al mejor modelo con la representaci?n completa.
Ninguna estrategia de selecci?n es universalmente la mejor.

Las variables originales seleccionadas por las tres estrategias son
`BusinessTravel`, `JobInvolvement`, `JobRole`, `MaritalStatus`, `MonthlyIncome`,
`OverTime` y `StockOptionLevel`. Estas coincidencias describen asociaciones
relevantes para la clasificaci?n; no implican causalidad.

El wrapper de RIPPER logra el mayor F1 entre las configuraciones reducidas
(0.3996) con una media de 2.6 predictores originales por fold. La Red
Neuronal con todos los predictores mantiene el mayor F1 global (0.4738).

## Entorno y reproducci?n

Desde la ra?z del proyecto, crea y activa un entorno virtual si lo necesitas:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para ejecutar los notebooks con sus rutas relativas, inicia Jupyter desde
`notebooks/pa1/`:

```powershell
cd notebooks/pa1
jupyter notebook
```

Ejecuta los notebooks en el orden indicado para reproducir el flujo completo.
Las salidas ya est?n disponibles en `results/`, `figures/` y `report/tables/`;
no hace falta volver a entrenar modelos para consultar o compilar el informe.
La ejecuci?n de los experimentos vuelve a escribir sus archivos de salida.
El wrapper puede requerir m?s tiempo por sus evaluaciones internas.

`requirements.txt` fija `wittgenstein==0.3.5`; las demás dependencias no fijan versiones. Por ello,
la reproducci?n en otro entorno puede presentar diferencias entre versiones.

## Informe cient?fico

El informe utiliza un formato de una columna y 11 puntos, con introducci?n,
descripci?n del problema, metodolog?a, resultados, discusi?n, conclusi?n y
referencias. El documento integra PA1 y PA2; el límite es de 20 páginas principales y dos de apéndices.

Para compilar, necesitas una distribuci?n LaTeX con `latexmk`, `pdflatex` y
BibTeX. Desde la ra?z del proyecto:

```powershell
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

El resultado es `report/main.pdf`. La compilaci?n regenera los archivos
auxiliares de LaTeX. El documento reutiliza `figures/pa1/f1_comparison.pdf` y las
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
