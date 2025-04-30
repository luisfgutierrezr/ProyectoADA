from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from tsp_router import TSPRouter

app = FastAPI(title="TSP Routing API")

# Data models for API
class Edge(BaseModel):
    node1: str
    node2: str
    weight: float

class GraphInput(BaseModel):
    edges: List[Edge]
    points: List[str]

# Create router instance
router = TSPRouter()

@app.post("/solve")
async def solve_tsp(data: GraphInput):
    """
    Solve TSP using all three algorithms.
    """
    try:
        edges = [(e.node1, e.node2, e.weight) for e in data.edges]
        router.load_graph(edges)
        router.set_points(data.points)

        results = {
            "brute_force": router.brute_force_tsp(),
            "nearest_neighbor": router.nearest_neighbor(),
            "two_opt": router.two_opt(router.nearest_neighbor()["path"])
        }
        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "TSP Routing API",
        "version": "1.0.0",
        "description": "API for solving TSP on road networks using multiple algorithms"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
