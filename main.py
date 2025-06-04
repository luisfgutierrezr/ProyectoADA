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
point_index_to_node = []  # Maps original point index to nearest node id
node_coordinates = {}  # Maps node IDs to their coordinates

# Replace NetworkData with TSPRouter
router = TSPRouter()

# Maximum points for brute force
MAX_BRUTE_FORCE_POINTS = 10

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page with the map interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/network")
async def upload_network(file: UploadFile = File(...)):
    """Upload and process road network file (.osm)"""
    try:
        filename = file.filename.lower()
        if not filename.endswith('.osm'):
            raise HTTPException(status_code=400, detail="Only .osm files are supported")
            
        temp_path = f"temp_{filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Load and simplify the graph
        G = ox.graph.graph_from_xml(temp_path, simplify=True)
        os.remove(temp_path)
        
        # Store node coordinates
        global node_coordinates
        node_coordinates = {node: {'lat': data['y'], 'lon': data['x']} 
                          for node, data in G.nodes(data=True)}
        
        # Store the graph in the router
        router.graph = G
        
        # Calculate bounds from graph nodes
        lats = [data['y'] for _, data in G.nodes(data=True)]
        lons = [data['x'] for _, data in G.nodes(data=True)]
        bounds = {
            'minLat': min(lats),
            'maxLat': max(lats),
            'minLon': min(lons),
            'maxLon': max(lons)
        }
        
        # Return graph metadata and bounds
        return {
            "message": "Network loaded successfully",
            "nodes": len(G.nodes()),
            "edges": len(G.edges()),
            "bounds": bounds
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload/points")
async def upload_points(file: UploadFile = File(...)):
    """Upload and process points file (.tsv) and map to nearest node IDs"""
    try:
        filename = file.filename.lower()
        if not filename.endswith('.tsv'):
            raise HTTPException(status_code=400, detail="Only .tsv files are supported")
            
        df = pd.read_csv(file.file, sep='\t')
        
        if router.graph is None or len(router.graph.nodes) == 0:
            raise HTTPException(status_code=400, detail="Network must be loaded before points.")
            
        # Map (X, Y) to nearest node in the graph
        points_xy = list(zip(df['X'], df['Y']))
        node_ids = []
        node_coords = []
        
        for x, y in points_xy:
            nearest_node = ox.distance.nearest_nodes(router.graph, x, y)
            node_ids.append(nearest_node)
            node_coords.append({
                'id': nearest_node,
                'latitude': y,
                'longitude': x
            })
        
        router.set_points(node_ids)
        
        # Store points for frontend
        global points
        points = node_coords
        global point_index_to_node
        point_index_to_node = node_ids.copy()
        
        return {
            "message": "Points loaded successfully",
            "points": len(node_ids),
            "coordinates": node_coords
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run-algorithms")
async def run_algorithms():
    """Run TSP algorithms using TSPRouter and return results"""
    if router.graph is None or not router.points:
        raise HTTPException(status_code=400, detail="Network and points must be loaded first")
        
    try:
        # Check if we have too many points for brute force
        if len(router.points) > MAX_BRUTE_FORCE_POINTS:
            results = {
                "nearest_neighbor": router.nearest_neighbor(),
                "genetic": router.genetic_algorithm_tsp()
            }
            results["brute_force"] = {
                "error": f"Too many points for brute force (max {MAX_BRUTE_FORCE_POINTS})",
                "path": [],
                "distance": 0,
                "time": 0
            }
        else:
            results = {
                "brute_force": router.brute_force_tsp(),
                "nearest_neighbor": router.nearest_neighbor(),
                "genetic": router.genetic_algorithm_tsp()
            }
            
        # Add coordinates to paths for visualization
        for algo in results:
            if "error" not in results[algo]:
                # Convert node IDs to coordinates
                path_coordinates = []
                for node_id in results[algo]["path"]:
                    if node_id in node_coordinates:
                        path_coordinates.append({
                            'latitude': node_coordinates[node_id]['lat'],
                            'longitude': node_coordinates[node_id]['lon']
                        })
                results[algo]["path_coordinates"] = path_coordinates
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)