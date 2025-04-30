# TSP Routing System - Traveling Salesman on Road Networks

Este proyecto es una implementación de prueba de concepto para resolver el **Problema del Viajante (TSP)** sobre una red vial utilizando diferentes enfoques algorítmicos. Forma parte del proyecto final del curso **Análisis de Algoritmos**, hecho por Diego Martinez, Eugenia Dayoub, Gabriela Rojas y Luis Gutiérrez.

## Descripción

La solución permite:
- Cargar una red vial (grafo con pesos).
- Definir un conjunto de ubicaciones/puntos a visitar.
- Calcular rutas óptimas usando tres algoritmos distintos:
  1. **Fuerza Bruta (Brute Force)**
  2. **Vecino Más Cercano (Nearest Neighbor)**
  3. **2-opt (Mejora local sobre una heurística inicial)**

## ⚙️ Tecnologías utilizadas

- **Python 3.10+**
- `networkx` – manejo de grafos
- `numpy` – soporte matemático
- `itertools` y `time` – para permutaciones y mediciones de tiempo
- `FastAPI` – API REST
- `Leaflet.js` – visualización de mapas

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/ProyectoADA.git
   cd ProyectoADA
   ```

2. Crea un entorno virtual y actívalo:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Estructura del proyecto

```
ProyectoADA/
├── main.py               # Servidor FastAPI
├── tsp_router.py         # Clase principal con lógica TSP
├── static/              # Archivos frontend
│   ├── index.html      # Interfaz web
│   └── app.js          # Lógica frontend
├── data/               # Archivos de prueba
│   └── example.json    # Ejemplo de grafo y puntos
├── tests/              # Pruebas unitarias
│   └── test_tsp_router.py
├── README.md
└── requirements.txt
```
