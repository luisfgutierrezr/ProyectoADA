import networkx as nx              # Para trabajar con grafos
import itertools                   # Para generar permutaciones (usado en fuerza bruta)
import time                        # Para medir tiempo de ejecución
import numpy as np                 # Para operaciones numéricas
from typing import List, Tuple, Dict, Any  # Tipado para anotar funciones
from datetime import datetime

class TSPRouter:
    """
    Clase que resuelve el Problema del Viajante (TSP) sobre una red vial representada como grafo.
    Implementa tres algoritmos: fuerza bruta, vecino más cercano, y mejora con 2-opt.
    """

    def __init__(self):
        """Inicializa un grafo vacío y una lista de puntos a visitar."""
        self.graph = nx.Graph()  # Grafo que almacena nodos y aristas
        self.points = []         # Lista de nodos que representan los puntos a visitar
        self.distance_matrix = None
        self.path_matrix = None  # Matriz que almacena los caminos entre puntos

    def load_graph(self, edges: List[Tuple[int, int, float]]) -> None:
        """
        Carga un grafo a partir de una lista de aristas.
        Cada arista es una tupla: (nodo1, nodo2, peso)
        """
        self.graph.add_weighted_edges_from(edges)

    def set_points(self, points: List[int]) -> None:
        """
        Establece la lista de puntos que se deben visitar y calcula la matriz de distancias.
        """
        self.points = points
        self._calculate_distance_and_path_matrices()

    def _calculate_distance_and_path_matrices(self):
        """
        Calcula la matriz de distancias y caminos entre todos los puntos usando la longitud de las aristas.
        """
        n = len(self.points)
        self.distance_matrix = np.zeros((n, n))
        self.path_matrix = [[None for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                try:
                    # Calcula la longitud del camino más corto entre los puntos
                    path = nx.shortest_path(
                        self.graph,
                        source=self.points[i],
                        target=self.points[j],
                        weight='length'
                    )
                    path_length = nx.shortest_path_length(
                        self.graph,
                        source=self.points[i],
                        target=self.points[j],
                        weight='length'
                    )
                    self.distance_matrix[i][j] = path_length
                    self.distance_matrix[j][i] = path_length
                    self.path_matrix[i][j] = path
                    self.path_matrix[j][i] = path[::-1]  # Camino inverso
                except nx.NetworkXNoPath:
                    # Si no existe camino, usa un número grande
                    self.distance_matrix[i][j] = float('inf')
                    self.distance_matrix[j][i] = float('inf')
                    self.path_matrix[i][j] = None
                    self.path_matrix[j][i] = None

    def get_path_between_points(self, i: int, j: int) -> List[int]:
        """
        Retorna el camino entre dos puntos usando la matriz de caminos.
        """
        return self.path_matrix[i][j]

    def brute_force_tsp(self) -> Dict[str, Any]:
        """
        Resuelve el TSP probando todas las permutaciones posibles (fuerza bruta).
        Solo recomendable para pocas ubicaciones (<10), ya que la complejidad es factorial.
        """
        start_time = time.time()
        n = len(self.points)
        print(f"[DEBUG] brute_force_tsp: n = {n}, points = {self.points}")
        if n == 0:
            return {"error": "No points to visit", "path": [], "distance": 0, "time": 0}
        # Genera todas las permutaciones posibles del conjunto de puntos
        min_path = None
        min_dist = float('inf')
        # Generar todas las permutaciones empezando desde el primer punto
        first_point = 0
        remaining_points = list(range(1, n))
        for perm in itertools.permutations(remaining_points):
            print(f"[DEBUG] brute_force_tsp: trying permutation {perm}")
            # Construir el camino completo incluyendo el primer punto
            path = [first_point] + list(perm)
            # Calcular la distancia total
            dist = 0
            for i in range(len(path)-1):
                dist += self.distance_matrix[path[i]][path[i+1]]
            dist += self.distance_matrix[path[-1]][path[0]]  # Regresar al punto de inicio
            if dist < min_dist:
                min_dist = dist
                min_path = path
        end_time = time.time()
        if min_path is None:
            return {"error": "No valid path found", "path": [], "distance": 0, "time": 0}
        # Construir el camino completo usando los caminos entre puntos
        full_path = []
        for i in range(len(min_path)-1):
            path_segment = self.get_path_between_points(min_path[i], min_path[i+1])
            if path_segment is None:
                return {"error": f"No path found between points {min_path[i]} and {min_path[i+1]}", 
                       "path": [], "distance": 0, "time": 0}
            full_path.extend(path_segment[:-1])  # Exclude last point to avoid duplicates
        # Add the path back to the start
        final_segment = self.get_path_between_points(min_path[-1], min_path[0])
        if final_segment is None:
            return {"error": f"No path found between points {min_path[-1]} and {min_path[0]}", 
                   "path": [], "distance": 0, "time": 0}
        full_path.extend(final_segment)
        return {
            "path": full_path,
            "distance": min_dist,
            "time": end_time - start_time
        }

    def nearest_neighbor(self) -> Dict[str, Any]:
        """
        Resuelve el TSP usando el algoritmo del vecino más cercano.
        Comienza desde un nodo inicial y siempre se mueve al nodo más cercano no visitado.
        """
        start_time = time.perf_counter()
        n = len(self.points)
        if n == 0:
            return {"error": "No points to visit", "path": [], "distance": 0, "time": 0}
            
        unvisited = set(range(n))
        current = 0
        path = [current]
        unvisited.remove(current)
        
        while unvisited:
            # Encontrar el nodo más cercano al nodo actual
            next_point = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            path.append(next_point)
            unvisited.remove(next_point)
            current = next_point
            
        # Calcular la distancia total
        total_dist = sum(self.distance_matrix[path[i]][path[i+1]] for i in range(n-1))
        total_dist += self.distance_matrix[path[-1]][path[0]]  # Regresar al punto de inicio
        
        # Construir el camino completo usando los caminos entre puntos
        full_path = []
        for i in range(len(path)-1):
            full_path.extend(self.get_path_between_points(path[i], path[i+1])[:-1])
        full_path.extend(self.get_path_between_points(path[-1], path[0]))
        
        end_time = time.perf_counter()
        
        return {
            "path": full_path,
            "distance": total_dist,
            "time": end_time - start_time
        }

    def genetic_algorithm_tsp(self) -> Dict[str, Any]:
        """
        Resuelve el TSP usando un algoritmo genético.
        """
        start_time = time.perf_counter()
        n = len(self.points)
        if n == 0:
            return {"error": "No points to visit", "path": [], "distance": 0, "time": 0}
        # Parámetros mejorados del algoritmo genético
        population_size = 200
        generations = 300
        mutation_rate = 0.1
        # Inicializar población
        population = [list(np.random.permutation(n)) for _ in range(population_size)]
        for _ in range(generations):
            # Calcular fitness penalizando rutas imposibles
            fitness = [self._calculate_path_distance(p) for p in population]
            # Penalizar rutas imposibles (distancia infinita)
            fitness = [f if np.isfinite(f) else 1e12 for f in fitness]
            # Seleccionar padres
            parents = []
            for _ in range(population_size):
                tournament = np.random.choice(population_size, 3, replace=False)
                winner = min(tournament, key=lambda x: fitness[x])
                parents.append(population[winner])
            # Crear nueva generación
            new_population = []
            for i in range(0, population_size, 2):
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                if np.random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < mutation_rate:
                    child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            population = new_population
        # Obtener mejor solución
        best_path = min(population, key=self._calculate_path_distance)
        best_distance = self._calculate_path_distance(best_path)
        print(f"[DEBUG] genetic best_path: {best_path}")
        print(f"[DEBUG] genetic best_distance: {best_distance}")
        print(f"[DEBUG] path length: {len(best_path)} (should be igual a n)")
        print(f"[DEBUG] unique nodes in path: {len(set(best_path))}")
        if not np.isfinite(best_distance):
            print("[WARNING] La mejor ruta encontrada por el genético es inválida (distancia infinita). Puede que la red no esté completamente conectada entre los puntos.")
        # Construir el camino completo usando los caminos entre puntos
        full_path = []
        for i in range(len(best_path)-1):
            full_path.extend(self.get_path_between_points(best_path[i], best_path[i+1])[:-1])
        full_path.extend(self.get_path_between_points(best_path[-1], best_path[0]))
        end_time = time.perf_counter()
        return {
            "path": full_path,
            "distance": best_distance,
            "time": end_time - start_time
        }
        
    def _calculate_path_distance(self, path: List[int]) -> float:
        """
        Calcula la distancia total de una ruta.
        """
        total = sum(self.distance_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        total += self.distance_matrix[path[-1]][path[0]]  # Regresar al inicio
        return total
        
    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Realiza cruce entre dos padres.
        """
        n = len(parent1)
        point = np.random.randint(0, n)
        child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
        child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
        return child1, child2
        
    def _mutate(self, path: List[int]) -> List[int]:
        """
        Realiza mutación en una ruta.
        """
        i, j = np.random.choice(len(path), 2, replace=False)
        path[i], path[j] = path[j], path[i]
        return path

"""     def two_opt(self, initial_path: List[str]) -> Dict[str, Any]:
        
        Mejora una solución inicial al TSP usando el algoritmo de 2-opt.
        Intercambia segmentos de la ruta si reduce el costo total.
        
        def two_opt_swap(route: List[str], i: int, k: int) -> List[str]:
            Invierte una sección del recorrido entre los índices i y k
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
        } """

