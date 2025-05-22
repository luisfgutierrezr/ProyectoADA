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

    def load_graph(self, edges: List[Tuple[int, int, float]]) -> None:
        """
        Carga un grafo a partir de una lista de aristas.
        Cada arista es una tupla: (nodo1, nodo2, peso)
        """
        self.graph.add_weighted_edges_from(edges)

    def set_points(self, points: List[int]) -> None:
        """
        Establece la lista de puntos que se deben visitar.
        """
        self.points = points

    def distance(self, u: int, v: int) -> float:
        """
        Calcula la distancia más corta entre dos nodos en el grafo,
        utilizando el peso de las aristas (e.g., distancia real o tiempo).
        """
        try:
            return nx.shortest_path_length(self.graph, u, v, weight='length')
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between nodes {u} and {v}")
            return float('inf')

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

    def nearest_neighbor(self, start_node: int = None) -> Dict[str, Any]:
        """
        Resuelve el TSP usando el algoritmo del vecino más cercano.
        Comienza desde un nodo inicial y siempre se mueve al nodo más cercano no visitado.
        """
        start = time.time()  # Iniciar conteo de tiempo para medir rendimiento

        # Si no se especifica nodo inicial, se toma el primero de la lista de puntos
        if not start_node:
            start_node = self.points[0]

        print(f"\nStarting nearest neighbor algorithm from node {start_node}")
        print(f"Total points to visit: {len(self.points)}")

        current_node = start_node  # Nodo desde el que se comienza el recorrido

        # Crear conjunto con los puntos no visitados (se eliminan a medida que se recorren)
        unvisited = set(self.points)
        unvisited.remove(current_node)  # Quitamos el nodo inicial porque ya lo estamos visitando

        path = [current_node]  # Lista para guardar la ruta en orden de visita
        iteration = 0
        max_iterations = len(self.points) * 2  # Safety check to prevent infinite loops

        # Repetir hasta que se hayan visitado todos los puntos
        while unvisited and iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}:")
            print(f"Current node: {current_node}")
            print(f"Unvisited nodes remaining: {len(unvisited)}")

            # Encontrar el nodo más cercano al nodo actual (según distancia más corta en el grafo)
            try:
                next_node = min(unvisited, key=lambda node: self.distance(current_node, node))
                dist = self.distance(current_node, next_node)
                print(f"Found next node: {next_node} at distance {dist}")

                # Agregar el nodo más cercano a la ruta
                path.append(next_node)

                # Eliminar el nodo recién visitado del conjunto de no visitados
                unvisited.remove(next_node)

                # Actualizar el nodo actual al nuevo nodo visitado (simulando el movimiento)
                current_node = next_node
            except ValueError as e:
                print(f"Error finding next node: {e}")
                print("Current unvisited nodes:", unvisited)
                print("Current node:", current_node)
                break

        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached, algorithm may not have completed")

        # Al finalizar, cerrar el ciclo regresando al punto de partida
        path.append(start_node)

        # Calcular el costo total del recorrido (suma de distancias entre nodos consecutivos)
        cost = sum(self.distance(path[i], path[i+1]) for i in range(len(path) - 1))

        end = time.time()  # Terminar conteo de tiempo

        print(f"\nAlgorithm completed:")
        print(f"Path length: {len(path)}")
        print(f"Total cost: {cost}")
        print(f"Time taken: {round(end - start, 4)} seconds")

        return {
            "algorithm": "Nearest Neighbor",
            "path": path,                     # Ruta seguida
            "cost": cost,                     # Costo total (distancia)
            "time": round(end - start, 4)     # Tiempo de ejecución en segundos
        }

    def _route_cost(self, route: List[int]) -> float:
        """
        Calcula el costo total de una ruta completa.
        Suma las distancias entre cada par consecutivo de nodos.
        """
        return sum(self.distance(route[i], route[i+1]) for i in range(len(route) - 1))

    def genetic_algorithm_tsp(self, population_size=50, generations=100, mutation_rate=0.1) -> Dict[str, Any]:
        """
        Resuelve el TSP usando un algoritmo genético.
        population_size: número de rutas en cada generación
        generations: número de generaciones
        mutation_rate: probabilidad de mutación por individuo
        """
        import random
        import time
        start = time.time()
        points = self.points
        n = len(points)
        if n < 2:
            return {"algorithm": "Genetic Algorithm", "path": points, "cost": 0, "time": 0}

        # Inicializar población con permutaciones aleatorias
        population = [random.sample(points, n) for _ in range(population_size)]

        def fitness(route):
            return self._route_cost(route + [route[0]])

        def crossover(parent1, parent2):
            # Order crossover (OX)
            a, b = sorted(random.sample(range(n), 2))
            child = [None]*n
            child[a:b] = parent1[a:b]
            fill = [item for item in parent2 if item not in child]
            idx = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            return child

        def mutate(route):
            i, j = random.sample(range(n), 2)
            route[i], route[j] = route[j], route[i]
            return route

        for _ in range(generations):
            # Evaluar fitness
            scored = [(fitness(route), route) for route in population]
            scored.sort(key=lambda x: x[0])
            population = [route for _, route in scored]
            # Elitismo: mantener los mejores
            new_population = population[:2]
            # Rellenar el resto
            while len(new_population) < population_size:
                # Selección por torneo
                contenders = random.sample(population, 5)
                parent1 = min(contenders, key=fitness)
                parent2 = min(random.sample(population, 5), key=fitness)
                child = crossover(parent1, parent2)
                if random.random() < mutation_rate:
                    child = mutate(child)
                new_population.append(child)
            population = new_population

        # Mejor individuo final
        best_route = min(population, key=lambda r: fitness(r))
        best_cost = fitness(best_route)
        end = time.time()
        return {
            "algorithm": "Genetic Algorithm",
            "path": best_route + [best_route[0]],
            "cost": best_cost,
            "time": round(end - start, 4)
        }

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

