# TFM_StockForecast

## Estructura de Proyecto
```
├── config
│   ├── built_config.py
│   ├── conda.yaml
│   ├── model_clustering.json
│   ├── model_stock.json
│   ├── rejected_products.csv
│   └── source_data.json
├── data
│   ├── clean
│   │   ├── clustering.csv
│   │   ├── festivos.csv
│   │   ├── prevision.csv
│   │   ├── promos.csv
│   │   ├── promos_range.csv
│   │   ├── stock.csv
│   │   ├── stock_all.csv
│   │   ├── stock_byprod.csv
│   │   ├── ventas.csv
│   │   └── ventas_byprod.csv
│   └── raw
│       ├── 01_TablaVentas.csv
│       ├── 02_TablaPromos.csv
│       ├── 03_TablaStock.csv
│       ├── 04_PrevisionEmpresa.csv
│       └── 05_Festivos.csv
├── notebooks/*
├── reports/*
├── src
│   ├── builder.py
│   ├── cleaner.py
│   ├── cleaner_datasets.py
│   ├── cleaner_utils.py
│   ├── create_variables.py
│   ├── feature_selection.py
│   ├── helpers_mlflow.py
│   ├── keras_utils.py
│   ├── load_data.py
│   ├── model.py
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── read_config.py
│   ├── trainer.py
│   ├── trainer_clustering.py
│   ├──utils.py
│   └── validate.py
├── README.md
├── .gitignore
├── run_eda.py
├── report_missings.py
└── train.py
```


## MAIN

### RUN_TRAIN.py

Script para lanzar un entrenamiento de modelo.

### RUN_EDA.py

Script para lanzar un informe de EDA de los datos limpios para su exploración.

## SOURCE

Carpeta con el código fuente del proyecto:

### READ_CONFIG.py

Funciones auxiliares para leer los JSON de configuración de los modelos

### LOAD_DATA.py

Funciones auxiliares para leer los datos fuente (CSVs de data) a partir del JSON de configuración source_data.json

## DATA

Carpeta con los datos del proyecto:

- Ventas
- Promociones
- Stock
- Previsión de la demanda
- Festivos

## CONFIG

Archivos de configuración para el proyecto

### CONDA.YAML

Dependencias de librerías de python del proyecto

### SOURCE_DATA.JSON

JSON de configuración de los datos fuente del proyecto (CSV de la carpeta data)

