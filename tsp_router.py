import networkx as nx              # Para trabajar con grafos
import itertools                   # Para generar permutaciones (usado en fuerza bruta)
import time                        # Para medir tiempo de ejecución
import numpy as np                 # Para operaciones numéricas
from typing import List, Tuple, Dict, Any  # Tipado para anotar funciones

class TSPRouter:
    """
    Clase que resuelve el Problema del Viajante (TSP) sobre una red vial representada como grafo.
    Implementa tres algoritmos: fuerza bruta, vecino más cercano, y mejora con 2-opt.
    """

    def __init__(self):
        """Inicializa un grafo vacío y una lista de puntos a visitar."""
        self.graph = nx.Graph()  # Grafo que almacena nodos y aristas
        self.points = []         # Lista de nodos que representan los puntos a visitar

    def load_graph(self, edges: List[Tuple[str, str, float]]) -> None:
        """
        Carga un grafo a partir de una lista de aristas.
        Cada arista es una tupla: (nodo1, nodo2, peso)
        """
        self.graph.add_weighted_edges_from(edges)

    def set_points(self, points: List[str]) -> None:
        """
        Establece la lista de puntos que se deben visitar.
        """
        self.points = points

    def distance(self, u: str, v: str) -> float:
        """
        Calcula la distancia más corta entre dos nodos en el grafo,
        utilizando el peso de las aristas (e.g., distancia real o tiempo).
        """
        return nx.shortest_path_length(self.graph, u, v, weight='weight')

    def brute_force_tsp(self) -> Dict[str, Any]:
        """
        Resuelve el TSP probando todas las permutaciones posibles (fuerza bruta).
        Solo recomendable para pocas ubicaciones (<10), ya que la complejidad es factorial.
        """
        start = time.time()            # Inicia temporizador
        best_path = None              # Guarda el mejor camino encontrado
        min_cost = float('inf')       # Guarda el costo mínimo

        # Genera todas las permutaciones posibles del conjunto de puntos
        for perm in itertools.permutations(self.points):
            cost = 0
            # Suma las distancias entre cada par consecutivo en la permutación
            for i in range(len(perm) - 1):
                cost += self.distance(perm[i], perm[i + 1])
            # Suma la distancia para regresar al punto de inicio
            cost += self.distance(perm[-1], perm[0])

            # Si el costo actual es menor que el anterior, se actualiza
            if cost < min_cost:
                min_cost = cost
                best_path = perm

        end = time.time()  # Finaliza temporizador

        return {
            "algorithm": "Brute Force",
            "path": list(best_path),     # Ruta óptima encontrada
            "cost": min_cost,            # Costo total
            "time": round(end - start, 4)  # Tiempo de ejecución
        }

    def nearest_neighbor(self, start_node: str = None) -> Dict[str, Any]:
        """
        Algoritmo del vecino más cercano (heurístico).
        Comienza en un punto y siempre elige el nodo no visitado más cercano.
        """
        start = time.time()
        unvisited = set(self.points)             # Conjunto de nodos aún no visitados
        if not start_node:
            start_node = self.points[0]          # Usa el primero como punto de inicio si no se da otro
        path = [start_node]                      # Ruta inicial con el punto de partida
        unvisited.remove(start_node)

        # Recorre el grafo eligiendo siempre el nodo no visitado más cercano
        while unvisited:
            last = path[-1]
            next_node = min(unvisited, key=lambda x: self.distance(last, x))
            path.append(next_node)
            unvisited.remove(next_node)

        # Regresa al nodo de inicio para cerrar el ciclo
        path.append(start_node)

        # Calcula el costo total de la ruta
        cost = sum(self.distance(path[i], path[i+1]) for i in range(len(path) - 1))
        end = time.time()

        return {
            "algorithm": "Nearest Neighbor",
            "path": path,
            "cost": cost,
            "time": round(end - start, 4)
        }

    def two_opt(self, initial_path: List[str]) -> Dict[str, Any]:
        """
        Mejora una solución inicial al TSP usando el algoritmo de 2-opt.
        Intercambia segmentos de la ruta si reduce el costo total.
        """
        def two_opt_swap(route: List[str], i: int, k: int) -> List[str]:
            """Invierte una sección del recorrido entre los índices i y k."""
            return route[:i] + route[i:k+1][::-1] + route[k+1:]

        start = time.time()
        best = initial_path[:-1]  # Elimina el nodo duplicado final (inicio/reinicio)
        improved = True

        # Itera mientras se sigan encontrando mejoras
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for k in range(i + 1, len(best)):
                    new_route = two_opt_swap(best, i, k)            # Nueva ruta candidata
                    new_cost = self._route_cost(new_route + [new_route[0]])  # Cierra el ciclo
                    old_cost = self._route_cost(best + [best[0]])            # Ruta actual
                    if new_cost < old_cost:
                        best = new_route
                        improved = True

        # Se añade el nodo de inicio al final para cerrar el ciclo
        path = best + [best[0]]
        end = time.time()

        return {
            "algorithm": "2-opt",
            "path": path,
            "cost": self._route_cost(path),
            "time": round(end - start, 4)
        }

    def _route_cost(self, route: List[str]) -> float:
        """
        Calcula el costo total de una ruta completa.
        Suma las distancias entre cada par consecutivo de nodos.
        """
        return sum(self.distance(route[i], route[i+1]) for i in range(len(route) - 1))
