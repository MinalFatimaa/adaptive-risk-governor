from datetime import datetime
from pydantic import BaseModel


class GraphNode(BaseModel):
    node_id: str

    node_type: str

    attributes: dict = {}


class GraphEdge(BaseModel):
    edge_id: str

    source_id: str

    target_id: str

    edge_type: str

    first_seen: datetime

    last_seen: datetime

    weight: float = 1.0


class GraphSnapshot(BaseModel):
    nodes: list[GraphNode] = []

    edges: list[GraphEdge] = []

    customer_component: list[str] = []

    relationship_counts: dict[str, int] = {}