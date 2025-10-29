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
from shapely.strtree import STRtree

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
    def __init__(self, layer_embedding_map: Dict[str, torch.Tensor], linetype_to_idx: Dict[str, int], k_neighbors: int = 5):
        self.layer_embedding_map, self.linetype_to_idx, self.k_neighbors = layer_embedding_map, linetype_to_idx, k_neighbors
        self.supported_entity_types = {'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'INSERT'}
        self.embedding_dim = next(iter(self.layer_embedding_map.values()), torch.zeros(384)).shape[0]
        self.total_feature_dim = sum([10, 5, 1]) + self.embedding_dim
        self.zero_embedding = torch.zeros(self.embedding_dim, dtype=torch.float)

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

        unique_id_to_node_info: Dict[str, Tuple[int, str, object, Optional[Vec3], Matrix44]] = {}
        node_counter = 0
        edges_by_type = defaultdict(set)
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
                    else: current_idx = unique_id_to_node_info[unique_id][0]

                    if parent_idx is not None:
                        parent_info = next((v for v in unique_id_to_node_info.values() if v[0] == parent_idx), None)
                        if parent_info: edges_by_type[(parent_info[1], 'contains', node_type)].add((parent_idx, current_idx))

                    if entity.dxf.dxftype == 'INSERT':
                        process_entities_hierarchically(block_templates.get(entity.dxf.name, []), current_idx, transform @ entity.matrix44())

        process_entities_hierarchically(msp)
        self._build_spatial_edges(edges_by_type, unique_id_to_node_info)

        final_nodes, global_to_local_idx_map = self._finalize_nodes(unique_id_to_node_info, norm_params)
        final_edges = self._finalize_edges(edges_by_type, unique_id_to_node_info, global_to_local_idx_map)

        return {"file_path": dxf_file_path, "nodes": dict(final_nodes), "edges": final_edges}

    def _get_entity_centroid(self, entity, transform):
        centroid = None
        try:
            if isinstance(entity, Vec3): centroid = entity
            elif hasattr(entity, 'dxf'):
                if entity.dxf.dxftype == 'LINE': centroid = (entity.dxf.start + entity.dxf.end) / 2
                elif entity.dxf.dxftype in {'CIRCLE', 'ARC'}: centroid = entity.dxf.center
                elif entity.dxf.dxftype == 'LWPOLYLINE':
                    points = [Vec3(p[:2]) for p in entity.get_points()]
                    if points: centroid = sum(points, Vec3()) / len(points)
                elif hasattr(entity.dxf, 'insert'): centroid = entity.dxf.insert
        except (AttributeError, TypeError): return None
        if centroid: return transform.transform(centroid)
        return None

    def _to_shapely_geom(self, entity, transform):
        try:
            if entity.dxf.dxftype == 'LINE':
                start, end = transform.transform(entity.dxf.start), transform.transform(entity.dxf.end)
                return LineString([start.xyz[:2], end.xyz[:2]])
            elif entity.dxf.dxftype == 'CIRCLE':
                center = transform.transform(entity.dxf.center)
                # Note: This assumes uniform scaling for radius.
                return Point(center.xyz[:2]).buffer(entity.dxf.radius * transform.m[0][0])
            elif entity.dxf.dxftype == 'ARC':
                start, end = transform.transform(entity.start_point), transform.transform(entity.end_point)
                return LineString([start.xyz[:2], end.xyz[:2]])
            elif entity.dxf.dxftype == 'LWPOLYLINE':
                points = [transform.transform(Vec3(p[:2])).xyz[:2] for p in entity.get_points()]
                return LineString(points) if len(points) > 1 else Point(points[0]) if points else None
        except: return None

    def _build_spatial_edges(self, edges_by_type, unique_id_to_node_info):
        node_info_list = list(unique_id_to_node_info.values())
        geometric_nodes = [n for n in node_info_list if n[1] != 'block_instance' and n[3] is not None]

        if not geometric_nodes: return

        # Nearby edges
        if len(geometric_nodes) > self.k_neighbors:
            centroids = [n[3].xyz for n in geometric_nodes]
            kdtree = KDTree(centroids)
            _, neighbors = kdtree.query(centroids, k=self.k_neighbors + 1)
            for i, neighbor_indices in enumerate(neighbors):
                for j_idx in neighbor_indices[1:]:
                    if j_idx < len(geometric_nodes):
                        idx1, type1, _, _, _ = geometric_nodes[i]
                        idx2, type2, _, _, _ = geometric_nodes[j_idx]
                        key = tuple(sorted((type1, type2)))
                        edges_by_type[(key[0], 'nearby', key[1])].add(tuple(sorted((idx1, idx2))))

        # Connects and Intersects edges
        endpoint_map = defaultdict(list)
        geoms = {info[0]: self._to_shapely_geom(info[2], info[4]) for info in geometric_nodes}

        for idx, node_type, entity, _, transform in geometric_nodes:
            if node_type in {'LINE', 'ARC'}:
                start = transform.transform(entity.start_point).round(3) if hasattr(entity, 'start_point') else transform.transform(entity.dxf.start).round(3)
                end = transform.transform(entity.end_point).round(3) if hasattr(entity, 'end_point') else transform.transform(entity.dxf.end).round(3)
                endpoint_map[start].append((idx, node_type))
                endpoint_map[end].append((idx, node_type))

        for point, connected_nodes in endpoint_map.items():
            if len(connected_nodes) > 1:
                for i in range(len(connected_nodes)):
                    for j in range(i + 1, len(connected_nodes)):
                        idx1, type1 = connected_nodes[i]
                        idx2, type2 = connected_nodes[j]
                        key = tuple(sorted((type1, type2)))
                        edges_by_type[(key[0], 'connects', key[1])].add(tuple(sorted((idx1, idx2))))

        geoms_list = [g for g in geoms.values() if g is not None]
        if not geoms_list: return
        strtree = STRtree(geoms_list)

        for idx1, type1, entity1, _, transform1 in geometric_nodes:
            geom1 = geoms.get(idx1)
            if geom1 is None: continue
            for geom2 in strtree.query(geom1):
                if geom1 is geom2: continue
                idx2 = next((idx for idx, g in geoms.items() if g is geom2), None)
                if idx2 is None or idx1 >= idx2: continue

                _, type2, _, _, _ = unique_id_to_node_info[list(unique_id_to_node_info.keys())[idx2]]
                if geom1.intersects(geom2):
                    key = tuple(sorted((type1, type2)))
                    edges_by_type[(key[0], 'intersects', key[1])].add(tuple(sorted((idx1, idx2))))

    def _finalize_nodes(self, unique_id_to_node_info, norm_params):
        final_nodes = defaultdict(lambda: {'x': [], 'discrete': []})
        global_to_local_idx_map = {}

        sorted_nodes = sorted(unique_id_to_node_info.values(), key=lambda x: x[0])
        for global_idx, node_type, entity, _, _ in sorted_nodes:
            cont_feats, disc_feats = self._extract_entity_features(entity, norm_params, node_type if node_type=='vertex' else None)
            local_idx = len(final_nodes[node_type]['x'])
            final_nodes[node_type]['x'].append(cont_feats)
            final_nodes[node_type]['discrete'].append(disc_feats)
            global_to_local_idx_map[global_idx] = local_idx

        for data in final_nodes.values():
            if data['x']:
                data['x'] = torch.stack(data['x'])
                data['discrete'] = torch.stack(data['discrete'])

        return final_nodes, global_to_local_idx_map

    def _finalize_edges(self, edges_by_type, unique_id_to_node_info, global_to_local_idx_map):
        final_edges = {}
        info_map = {v[0]: v for v in unique_id_to_node_info.values()}
        for (src_type, rel, dst_type), edge_set in edges_by_type.items():
            remapped = []
            for g_idx1, g_idx2 in edge_set:
                node1_info, node2_info = info_map.get(g_idx1), info_map.get(g_idx2)
                if not node1_info or not node2_info: continue

                if (src_type, dst_type) == (node1_info[1], node2_info[1]):
                    remapped.append([global_to_local_idx_map[g_idx1], global_to_local_idx_map[g_idx2]])
                elif (src_type, dst_type) == (node2_info[1], node1_info[1]):
                     remapped.append([global_to_local_idx_map[g_idx2], global_to_local_idx_map[g_idx1]])
            if remapped:
                final_edges[(src_type, rel, dst_type)] = torch.tensor(remapped, dtype=torch.long).t().contiguous()
        return final_edges

    def _extract_entity_features(self, entity, norm_params: Dict, entity_type_override: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.total_feature_dim), torch.zeros(1, dtype=torch.long)

class GraphBuilder:
    def __init__(self):
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}

    def fit_transform(self, data_list: List[Dict]) -> List[HeteroData]:
        return []
    def transform(self, data_list: List[Dict]) -> List[HeteroData]:
        return []

def _create_dxf_for_full_pipeline_test(temp_dir: Path) -> str:
    return ""

def _demonstrate_full_pipeline():
    pass

if __name__ == '__main__':
    _demonstrate_full_pipeline()
