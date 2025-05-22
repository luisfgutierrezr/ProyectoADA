# Acá se hace el backend
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
from typing import List, Dict
import json
import os
from datetime import datetime
import uvicorn
import itertools
from tsp_router import TSPRouter

# Initialize FastAPI app
app = FastAPI(title="TSP Router")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables to store network and points
road_network = None
points = None
results = {}

# Replace NetworkData with TSPRouter
router = TSPRouter()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page with the map interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/network")
async def upload_network(file: UploadFile = File(...)):
    """Upload and process road network file (.csv or .osm)"""
    try:
        filename = file.filename.lower()
        if filename.endswith('.osm'):
            temp_path = f"temp_{filename}"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            G = ox.graph.graph_from_xml(temp_path, simplify=True)
            os.remove(temp_path)
            # Convert to edge list with weights (length)
            edges = [(u, v, d.get('length', 1.0)) for u, v, d in G.edges(data=True)]
            router.graph = G
            router.load_graph(edges)
        else:
            df = pd.read_csv(file.file)
            edges = [(row['source'], row['target'], row['weight'] if 'weight' in row else 1) for _, row in df.iterrows()]
            router.load_graph(edges)
        return {"message": "Network loaded successfully", "nodes": len(router.graph.nodes()), "edges": len(router.graph.edges())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload/points")
async def upload_points(file: UploadFile = File(...)):
    """Upload and process points file (.csv or .tsv) and map to nearest node IDs"""
    try:
        filename = file.filename.lower()
        if filename.endswith('.tsv'):
            df = pd.read_csv(file.file, sep='\t')
        else:
            df = pd.read_csv(file.file)
        # Map (X, Y) to nearest node in the graph
        if router.graph is None or len(router.graph.nodes) == 0:
            raise HTTPException(status_code=400, detail="Network must be loaded before points.")
        points_xy = list(zip(df['X'], df['Y']))
        # Use osmnx.get_nearest_node for each point
        node_ids = [ox.distance.nearest_nodes(router.graph, x, y) for x, y in points_xy]
        router.set_points(node_ids)
        return {"message": "Points loaded successfully", "points": len(node_ids)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run-algorithms")
async def run_algorithms():
    """Run TSP algorithms using TSPRouter and return results"""
    if router.graph is None or not router.points:
        raise HTTPException(status_code=400, detail="Network and points must be loaded first")
    try:
        results = {
            "brute_force": router.brute_force_tsp(),
            "nearest_neighbor": router.nearest_neighbor(),
            "genetic": router.genetic_algorithm_tsp()
        }
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_brute_force(points):
    """Brute force TSP implementation"""
    start_time = datetime.now()
    
    # Calculate all pairwise distances
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i][j] = calculate_distance(points[i], points[j])
    
    # Find shortest path
    min_path = None
    min_dist = float('inf')
    
    for path in itertools.permutations(range(n)):
        dist = sum(distances[path[i]][path[i+1]] for i in range(n-1))
        dist += distances[path[-1]][path[0]]  # Return to start
        if dist < min_dist:
            min_dist = dist
            min_path = path
    
    end_time = datetime.now()
    
    return {
        "path": min_path,
        "distance": min_dist,
        "time": (end_time - start_time).total_seconds()
    }

def run_nearest_neighbor(points):
    """Nearest neighbor TSP implementation"""
    start_time = datetime.now()
    
    n = len(points)
    unvisited = set(range(n))
    current = 0
    path = [current]
    unvisited.remove(current)
    
    while unvisited:
        next_point = min(unvisited, 
                        key=lambda x: calculate_distance(points[current], points[x]))
        path.append(next_point)
        unvisited.remove(next_point)
        current = next_point
    
    # Calculate total distance
    total_dist = sum(calculate_distance(points[path[i]], points[path[i+1]]) 
                    for i in range(len(path)-1))
    total_dist += calculate_distance(points[path[-1]], points[path[0]])
    
    end_time = datetime.now()
    
    return {
        "path": path,
        "distance": total_dist,
        "time": (end_time - start_time).total_seconds()
    }

def run_genetic_algorithm(points):
    """Genetic algorithm TSP implementation"""
    start_time = datetime.now()
    
    # Implementation of genetic algorithm
    # This is a simplified version - in practice, you'd want a more robust implementation
    population_size = 50
    generations = 100
    mutation_rate = 0.1
    
    n = len(points)
    population = [list(np.random.permutation(n)) for _ in range(population_size)]
    
    for _ in range(generations):
        # Calculate fitness
        fitness = [calculate_path_distance(p, points) for p in population]
        
        # Select parents
        parents = []
        for _ in range(population_size):
            tournament = np.random.choice(population_size, 3, replace=False)
            winner = min(tournament, key=lambda x: fitness[x])
            parents.append(population[winner])
        
        # Create new generation
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            
            if np.random.random() < mutation_rate:
                child1 = mutate(child1)
            if np.random.random() < mutation_rate:
                child2 = mutate(child2)
                
            new_population.extend([child1, child2])
        
        population = new_population
    
    # Get best solution
    best_path = min(population, key=lambda p: calculate_path_distance(p, points))
    best_distance = calculate_path_distance(best_path, points)
    
    end_time = datetime.now()
    
    return {
        "path": best_path,
        "distance": best_distance,
        "time": (end_time - start_time).total_seconds()
    }

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_path_distance(path, points):
    """Calculate total distance of a path"""
    total = 0
    for i in range(len(path)-1):
        total += calculate_distance(points[path[i]], points[path[i+1]])
    total += calculate_distance(points[path[-1]], points[path[0]])
    return total

def crossover(parent1, parent2):
    """Perform crossover between two parents"""
    n = len(parent1)
    point = np.random.randint(0, n)
    child1 = parent1[:point] + [x for x in parent2 if x not in parent1[:point]]
    child2 = parent2[:point] + [x for x in parent1 if x not in parent2[:point]]
    return child1, child2

def mutate(path):
    """Perform mutation on a path"""
    i, j = np.random.choice(len(path), 2, replace=False)
    path[i], path[j] = path[j], path[i]
    return path

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)