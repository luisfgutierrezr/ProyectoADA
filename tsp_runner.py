import osmnx as ox
import pandas as pd
import networkx as nx
from tsp_router import TSPRouter
from typing import List, Tuple, Dict, Any
from shapely.geometry import Point
from pyproj import Transformer

class TSPRunner:
    def __init__(self):
        """Initialize the TSP runner with empty graph and points."""
        self.graph = None
        self.points = None
        self.tsp_router = TSPRouter()
        self.transformer = None

    def load_osm_graph(self, osm_file: str) -> None:
        """Load the road network from an OSM file."""
        print(f"Loading OSM file: {osm_file}")
        
        # Load the graph with all nodes and edges
        self.graph = ox.graph_from_xml(
            osm_file,
            simplify=False,  # Don't simplify the graph
            retain_all=True  # Keep all nodes, even if they're not connected
        )
        
        # Project the graph immediately after loading
        self.graph = ox.project_graph(self.graph)
        
        # Create coordinate transformer
        self.transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 (lat/lon)
            self.graph.graph['crs'],  # Projected CRS
            always_xy=True
        )
        
        # Get graph bounds from nodes
        nodes = list(self.graph.nodes(data=True))
        if nodes:
            lats = [data['y'] for _, data in nodes]
            lons = [data['x'] for _, data in nodes]
            bounds = (min(lons), min(lats), max(lons), max(lats))
            print(f"Graph bounds: {bounds}")
            print(f"Number of nodes in original OSM: {len(nodes)}")
        else:
            print("Warning: Graph has no nodes")
        
        # Convert to undirected graph for simplicity
        self.graph = self.graph.to_undirected()
        
        # Add edge weights (length in meters)
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if 'length' not in data:
                data['length'] = 1.0  # Default length if not present
        
        print(f"Graph has {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        
        # Print some sample node data
        if nodes:
            print("\nSample node data:")
            for i, (node_id, data) in enumerate(nodes[:3]):
                print(f"Node {node_id}: {data}")
            print("...")

    def _find_nearest_node(self, x: float, y: float) -> int:
        """Find the nearest node to the given coordinates using simple distance calculation."""
        min_dist = float('inf')
        nearest_node = None
        
        # Convert input coordinates to the same projection as the graph
        point_x, point_y = self.transformer.transform(x, y)
        
        # Debug: Print the first few nodes to verify data structure
        print(f"\nSearching for nearest node to ({x}, {y}) -> projected to ({point_x}, {point_y})")
        print("First few nodes in graph:")
        for i, (node, data) in enumerate(self.graph.nodes(data=True)):
            if i < 3:  # Print first 3 nodes
                print(f"Node {node}: x={data.get('x')}, y={data.get('y')}")
            if 'x' not in data or 'y' not in data:
                print(f"Warning: Node {node} missing coordinates")
                continue
            # Calculate Euclidean distance in projected coordinates
            dist = ((point_x - data['x']) ** 2 + (point_y - data['y']) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        if nearest_node is not None:
            print(f"Found nearest node {nearest_node} at distance {min_dist}")
            return nearest_node  # Return as integer
        else:
            print("No nearest node found")
            return None

    def load_points(self, points_file: str) -> None:
        """Load points from TSV file and convert to nearest graph nodes."""
        if not self.graph:
            raise ValueError("Graph must be loaded before loading points")
            
        # Read points from TSV
        df = pd.read_csv(points_file, sep='\t')
        
        # Print bounds of points
        min_x, max_x = df['X'].min(), df['X'].max()
        min_y, max_y = df['Y'].min(), df['Y'].max()
        print(f"Points bounds: X({min_x}, {max_x}), Y({min_y}, {max_y})")
        
        # Convert points to nearest graph nodes
        self.points = []
        for idx, row in df.iterrows():
            try:
                # Find nearest node using our custom function
                nearest_node = self._find_nearest_node(row['X'], row['Y'])
                if nearest_node is not None:
                    self.points.append(nearest_node)
                    print(f"Successfully mapped point {idx} to node {nearest_node}")
                else:
                    print(f"Warning: Could not find valid nearest node for point {idx} at ({row['X']}, {row['Y']})")
            except Exception as e:
                print(f"Warning: Error processing point {idx} at ({row['X']}, {row['Y']}): {str(e)}")
                continue
        
        if not self.points:
            raise ValueError("No valid points could be mapped to the graph")
        
        print(f"Successfully mapped {len(self.points)} points to graph nodes")

    def run_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Run all TSP algorithms and return results."""
        if not self.graph or not self.points:
            raise ValueError("Graph and points must be loaded first")

        # Set up the TSP router with the graph and points
        self.tsp_router.graph = self.graph.copy(as_view=False)  # Create a deep copy of the graph
        self.tsp_router.points = self.points.copy()  # Create a copy of the points
        
        # Calculate distance matrix before running algorithms
        self.tsp_router._calculate_distance_and_path_matrices()

        results = {}
        
        # Run nearest neighbor algorithm
        print("\nRunning nearest neighbor algorithm...")
        results["nearest_neighbor"] = self.tsp_router.nearest_neighbor()
        
        # Run genetic algorithm
        print("\nRunning genetic algorithm...")
        results["genetic"] = self.tsp_router.genetic_algorithm_tsp()

        # Only run brute force for small numbers of points (n <= 10)
        if len(self.points) <= 10:
            print("\nRunning brute force algorithm (this may take a while for n > 8)...")
            results["brute_force"] = self.tsp_router.brute_force_tsp()
        else:
            print(f"\nSkipping brute force algorithm for {len(self.points)} points (too many points)")

        return results

# ESTO ES TEMPORAL POR MIENTRAS SE HACE EL SERVER Y EL FRONT (PARA PROBAR)

def main():
    # Create runner instance
    runner = TSPRunner()
    
    # Load data
    runner.load_osm_graph("data/chapinero.osm")
    runner.load_points("data/points.tsv")
    
    # Run algorithms
    results = runner.run_algorithms()
    
    # Print results
    for algo, result in results.items():
        print(f"\n{algo.upper()} Results:")
        print(f"Path: {' -> '.join(str(node) for node in result['path'])}")
        print(f"Distance: {result['distance']:.2f} meters")  # Using distance instead of cost
        print(f"Time: {result['time']:.4f} seconds")

if __name__ == "__main__":
    main()