import networkx as nx
import itertools
import time
import numpy as np
from typing import List, Tuple, Dict, Any

class TSPRouter:
    """
    Class to handle TSP (Traveling Salesman Problem) routing on road networks.
    Implements three different algorithms: brute force, nearest neighbor, and 2-opt.
    """
    
    def __init__(self):
        """Initialize an empty graph and points list."""
        self.graph = nx.Graph()
        self.points = []

    def load_graph(self, edges: List[Tuple[str, str, float]]) -> None:
        """
        Load a graph from a list of edges.
        
        Args:
            edges: List of tuples (node1, node2, weight) representing the road network
        """
        self.graph.add_weighted_edges_from(edges)

    def set_points(self, points: List[str]) -> None:
        """
        Set the points to visit in the TSP problem.
        
        Args:
            points: List of node IDs to visit
        """
        self.points = points

    def distance(self, u: str, v: str) -> float:
        """
        Calculate the shortest path distance between two nodes.
        
        Args:
            u: Source node
            v: Target node
            
        Returns:
            float: Shortest path distance
        """
        return nx.shortest_path_length(self.graph, u, v, weight='weight')

    def brute_force_tsp(self) -> Dict[str, Any]:
        """
        Solve TSP using brute force (tries all possible permutations).
        Only suitable for small instances (n < 10).
        
        Returns:
            Dict containing algorithm name, path, cost, and execution time
        """
        start = time.time()
        best_path = None
        min_cost = float('inf')

        # Try all possible permutations of points
        for perm in itertools.permutations(self.points):
            cost = 0
            # Calculate cost for this permutation
            for i in range(len(perm) - 1):
                cost += self.distance(perm[i], perm[i + 1])
            # Add cost to return to start
            cost += self.distance(perm[-1], perm[0])

            if cost < min_cost:
                min_cost = cost
                best_path = perm

        end = time.time()
        return {
            "algorithm": "Brute Force",
            "path": list(best_path),
            "cost": min_cost,
            "time": round(end - start, 4)
        }

    def nearest_neighbor(self, start_node: str = None) -> Dict[str, Any]:
        """
        Solve TSP using the nearest neighbor heuristic.
        A greedy algorithm that always moves to the closest unvisited node.
        
        Args:
            start_node: Optional starting node. If None, uses first point.
            
        Returns:
            Dict containing algorithm name, path, cost, and execution time
        """
        start = time.time()
        unvisited = set(self.points)
        if not start_node:
            start_node = self.points[0]
        path = [start_node]
        unvisited.remove(start_node)

        # Build path by always choosing nearest unvisited node
        while unvisited:
            last = path[-1]
            next_node = min(unvisited, key=lambda x: self.distance(last, x))
            path.append(next_node)
            unvisited.remove(next_node)

        # Return to start
        path.append(start_node)
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
        Improve a TSP solution using 2-opt local search.
        Repeatedly removes two edges and reconnects the resulting paths.
        
        Args:
            initial_path: Initial path to improve
            
        Returns:
            Dict containing algorithm name, improved path, cost, and execution time
        """
        def two_opt_swap(route: List[str], i: int, k: int) -> List[str]:
            """Helper function to perform 2-opt swap."""
            return route[:i] + route[i:k+1][::-1] + route[k+1:]

        start = time.time()
        best = initial_path[:-1]  # Remove last node (return to start)
        improved = True

        # Keep improving until no better solution is found
        while improved:
            improved = False
            for i in range(1, len(best) - 2):
                for k in range(i + 1, len(best)):
                    new_route = two_opt_swap(best, i, k)
                    new_cost = self._route_cost(new_route + [new_route[0]])
                    old_cost = self._route_cost(best + [best[0]])
                    if new_cost < old_cost:
                        best = new_route
                        improved = True

        path = best + [best[0]]  # Add return to start
        end = time.time()
        return {
            "algorithm": "2-opt",
            "path": path,
            "cost": self._route_cost(path),
            "time": round(end - start, 4)
        }

    def _route_cost(self, route: List[str]) -> float:
        """
        Calculate the total cost of a route.
        
        Args:
            route: List of nodes in the route
            
        Returns:
            float: Total cost of the route
        """
        return sum(self.distance(route[i], route[i+1]) for i in range(len(route) - 1)) 