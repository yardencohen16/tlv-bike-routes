"""
server.py — FastAPI server wrapping the flat-route cycling engine.

Startup loads the graph once into memory (cached to disk after first run).

Run locally:
    python3 server.py

Endpoint:
    GET /route?start_lat=32.0853&start_lon=34.7818&end_lat=32.0679&end_lon=34.7751
"""

import os
import pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

from flat_route import load_graph, fetch_elevations, annotate_elevation_gain, get_flat_route, compute_route_stats

CACHE_FILE = "graph_cache.pkl"

# ---------------------------------------------------------------------------
# Graph — loaded once at startup, cached to disk
# ---------------------------------------------------------------------------

state: dict = {}


def load_or_build_graph():
    if os.path.exists(CACHE_FILE):
        print("Loading graph from cache (fast)...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("First run — downloading graph and fetching elevations (this takes a few minutes)...")
    G = load_graph()
    G = fetch_elevations(G)
    G = annotate_elevation_gain(G)

    print("Saving graph to cache for future restarts...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(G, f)

    return G


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["graph"] = load_or_build_graph()
    print("Graph ready. Server is up.")
    yield
    state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="TLV Flat Bike Router", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Serve the UI
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("index.html")


# ---------------------------------------------------------------------------
# Route endpoint
# ---------------------------------------------------------------------------

@app.get("/route")
def route(
    start_lat: float = Query(..., description="Start latitude"),
    start_lon: float = Query(..., description="Start longitude"),
    end_lat:   float = Query(..., description="End latitude"),
    end_lon:   float = Query(..., description="End longitude"),
):
    G = state.get("graph")
    if G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")

    try:
        coords = get_flat_route(G, (start_lat, start_lon), (end_lat, end_lon))
        distance_km, elevation_gain_m = compute_route_stats(G, coords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "distance_km": round(distance_km, 3),
        "elevation_gain_m": round(elevation_gain_m, 1),
        "waypoints": len(coords),
        "coordinates": [{"lat": lat, "lon": lon} for lat, lon in coords],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
