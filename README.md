# Unfollow Model 🕸️📉

Este repositorio contiene el código y los experimentos del proyecto **Polarización por Ruptura de Lazos**. 

La idea principal es simular cómo interactúan los usuarios en una red social. Para esto, partimos del **modelo de Deffuant** (dinámicas de opinión) y le agregamos una capa "coevolutiva". ¿Qué significa esto? Que los agentes no solo actualizan sus opiniones al interactuar, sino que pueden hacer "unfollow" (romper la conexión) si la diferencia de opinión con el otro nodo es muy grande. Además, la red intenta reconectarse buscando amigos de amigos con opiniones más parecidas (homofilia), simulando cómo se terminan formando las burbujas o cámaras de eco.

## Estructura del repo

- `src/`: Donde vive la lógica dura de la simulación.
  - `models/`: Implementación matemática de los modelos (`deffuant.py` para el clásico y `deffuant_coevo.py` para el de unfollow).
  - `sim/`: Los runners para ejecutar múltiples pasos de la simulación y guardar el estado.
  - `graphs/`: Funciones para generar las redes iniciales.
  - `metrics/` y `viz/`: Herramientas para medir la polarización y graficar.
- `notebooks/`: Jupyter notebooks con los primeros análisis, visualizaciones y notas de progreso.
- `experiments/`: Scripts preparados para lanzar las simulaciones más pesadas.
- `paper/`: Borradores y notas de la investigación.

## Instalación

Todo está hecho en Python, dependiendo fuerte de `numpy` y `networkx`. Para instalar todo lo necesario, crear un entorno virtual e instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Estado actual

Ya están implementados los modelos base y se pueden correr los primeros análisis comparando el modelo clásico vs el modelo con unfollow + burbuja.
