# TSP Routing System - Traveling Salesman on Road Networks

Este proyecto es una implementación de prueba de concepto para resolver el **Problema del Viajante (TSP)** sobre una red vial utilizando diferentes enfoques algorítmicos. Forma parte del proyecto final del curso **Análisis de Algoritmos**, hecho por Diego Martinez, Eugenia Dayoub, Gabriela Rojas y Luis Gutiérrez.

---

## Descripción

La solución permite:
- Cargar una red vial (grafo con pesos).
- Definir un conjunto de ubicaciones/puntos a visitar.
- Calcular rutas óptimas usando tres algoritmos distintos:
  1. **Fuerza Bruta (Brute Force)**
  2. **Vecino Más Cercano (Nearest Neighbor)**
  3. **2-opt (Mejora local sobre una heurística inicial)**

Posteriormente, se integrará un frontend web interactivo para visualizar los resultados en un mapa (Leaflet u OpenLayers).

---

## Tecnologías utilizadas

- **Python 3.10+**
- `networkx` – manejo de grafos
- `numpy` – soporte matemático
- `itertools` y `time` – para permutaciones y mediciones de tiempo

---


