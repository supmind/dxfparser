# graph_builder.py

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, Optional, List
from collections import defaultdict

import ezdxf
import torch
import numpy as np
from ezdxf import DXFError
import ezdxf.bbox
from ezdxf.document import Drawing
from ezdxf.math import BoundingBox, Vec3, Matrix44
from sentence_transformers import SentenceTransformer
from ezdxf.layouts import Modelspace
from torch_geometric.data import HeteroData
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalPreprocessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.sentence_transformer_model = SentenceTransformer(model_name)
        self.layer_embedding_map: Dict[str, torch.Tensor] = {}
        self.linetype_to_idx: Dict[str, int] = {}
        self.embedding_dim = self.sentence_transformer_model.get_sentence_embedding_dimension()

    def fit(self, dxf_file_paths: Iterable[str]) -> None:
        unique_layer_names, unique_linetype_names = set(), set()
        for file_path in dxf_file_paths:
            try:
                doc: Drawing = ezdxf.readfile(file_path)
                unique_layer_names.update(layer.dxf.name for layer in doc.layers)
                unique_linetype_names.update(linetype.dxf.name for linetype in doc.linetypes)
            except (IOError, OSError, DXFError) as e:
                logging.warning(f"Skipping file {file_path}: {e}")

        unique_layer_names.add('0')
        layer_names_list = sorted(list(unique_layer_names))
        layer_embeddings = self.sentence_transformer_model.encode(layer_names_list, convert_to_tensor=True)
        self.layer_embedding_map = {name: emb for name, emb in zip(layer_names_list, layer_embeddings)}

        unique_linetype_names.update(['BYLAYER', 'BYBLOCK', 'CONTINUOUS'])
        self.linetype_to_idx = {name: i for i, name in enumerate(sorted(list(unique_linetype_names)))}

class SingleDrawingProcessor:
    def __init__(self, layer_embedding_map: Dict[str, torch.Tensor], linetype_to_idx: Dict[str, int], k_neighbors: int = 3):
        self.layer_embedding_map, self.linetype_to_idx = layer_embedding_map, linetype_to_idx
        self.supported_entity_types = {'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'INSERT'}
        self.embedding_dim = next(iter(self.layer_embedding_map.values()), torch.zeros(384)).shape[0]
        self.total_feature_dim = sum([10, 5, 1]) + self.embedding_dim
        self.zero_embedding = torch.zeros(self.embedding_dim, dtype=torch.float)
        self.k_neighbors = k_neighbors

    def process(self, dxf_file_path: str) -> Optional[Dict]:
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
        except (IOError, OSError, DXFError) as e: return None

        bbox = ezdxf.bbox.extents(msp, cache=None)
        if not bbox.has_data: return None

        bbox_min, bbox_size = bbox.extmin, bbox.size
        original_scale = max(bbox_size) if bbox_size and max(bbox_size) > 0 else 1.0
        norm_scale = 1.0 / original_scale if original_scale > 1e-8 else 1.0
        norm_params = {'bbox_min': bbox_min, 'scale': norm_scale, 'original_scale': original_scale}

        unique_id_to_node_info, node_counter, edges_by_type = {}, 0, defaultdict(list)
        block_templates = { b.name: list(b) for b in doc.blocks if not b.name.startswith('*')}

        def process_entities_hierarchically(entities, parent_idx=None, transform=Matrix44()):
            nonlocal node_counter
            for entity in entities:
                if hasattr(entity, 'dxf') and entity.dxf.dxftype in self.supported_entity_types:
                    unique_id = entity.dxf.handle
                    if unique_id not in unique_id_to_node_info:
                        node_type = 'block_instance' if entity.dxf.dxftype == 'INSERT' else entity.dxf.dxftype
                        centroid = self._get_entity_centroid(entity, transform)
                        unique_id_to_node_info[unique_id] = (node_counter, node_type, entity, centroid, transform)
                        current_idx = node_counter
                        node_counter += 1
                    else:
                        current_idx = unique_id_to_node_info[unique_id][0]

                    if parent_idx is not None:
                        parent_info = next((v for v in unique_id_to_node_info.values() if v[0] == parent_idx), None)
                        if parent_info:
                            edges_by_type[(parent_info[1], 'contains', node_type)].append((parent_idx, current_idx))

                    if entity.dxf.dxftype == 'INSERT':
                        process_entities_hierarchically(block_templates.get(entity.dxf.name, []), current_idx, transform @ entity.matrix44())

        process_entities_hierarchically(msp)
        self._build_spatial_edges(edges_by_type, unique_id_to_node_info)

        final_nodes, global_to_local_idx_map = self._finalize_nodes(unique_id_to_node_info, norm_params)
        final_edges = self._finalize_edges(edges_by_type, unique_id_to_node_info, global_to_local_idx_map)

        return {"file_path": dxf_file_path, "nodes": dict(final_nodes), "edges": final_edges}

    def _get_entity_centroid(self, entity, transform):
        # ...
        return None
    def _to_shapely_geom(self, entity, transform):
        # ...
        return None
    def _build_spatial_edges(self, edges_by_type, unique_id_to_node_info):
        pass # Placeholder
    def _finalize_nodes(self, unique_id_to_node_info, norm_params):
        return defaultdict(lambda: {'x': [], 'discrete': []}), {}
    def _finalize_edges(self, edges_by_type, unique_id_to_node_info, global_to_local_idx_map):
        return {}

class GraphBuilder:
    def __init__(self):
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}

    def fit_transform(self, intermediate_data_list: List[Dict]) -> List[HeteroData]:
        # ... (Full implementation)
        return []
    def transform(self, intermediate_data_list: List[Dict]) -> List[HeteroData]:
        # ... (Full implementation)
        return []

def _create_dxf_for_full_pipeline_test(temp_dir: Path) -> str:
    # ... (Full implementation)
    return ""

def _demonstrate_full_pipeline():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    try:
        # ... (Full implementation)
        pass
    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    _demonstrate_full_pipeline()
