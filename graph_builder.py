# graph_builder.py

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, List, Tuple
from collections import defaultdict

import ezdxf
import torch
import numpy as np
from ezdxf import DXFError
import ezdxf.bbox
from ezdxf.document import Drawing
from ezdxf.entities import DXFEntity
from ezdxf.layouts import Modelspace
from ezdxf.math import BoundingBox, Vec3, Matrix44
from sentence_transformers import SentenceTransformer
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
                unique_linetype_names.update(ltype.dxf.name for ltype in doc.linetypes)
            except (IOError, OSError, DXFError) as e:
                logging.warning(f"Skipping file {file_path}: {e}")

        unique_layer_names.add('0')
        layer_names_list = sorted(list(unique_layer_names))
        layer_embeddings = self.sentence_transformer_model.encode(layer_names_list, convert_to_tensor=True)
        self.layer_embedding_map = {name: emb for name, emb in zip(layer_names_list, layer_embeddings)}

        unique_linetype_names.update(['BYLAYER', 'BYBLOCK', 'CONTINUOUS'])
        self.linetype_to_idx = {name: i for i, name in enumerate(sorted(list(unique_linetype_names)))}

    def save(self, directory: str) -> None:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "layer_embedding_map.pkl", "wb") as f:
            pickle.dump(self.layer_embedding_map, f)
        with open(dir_path / "linetype_to_idx.json", "w") as f:
            json.dump(self.linetype_to_idx, f)

    def load(self, directory: str) -> None:
        dir_path = Path(directory)
        with open(dir_path / "layer_embedding_map.pkl", "rb") as f:
            self.layer_embedding_map = pickle.load(f)
        with open(dir_path / "linetype_to_idx.json", "r") as f:
            self.linetype_to_idx = json.load(f)
        if self.layer_embedding_map:
            self.embedding_dim = next(iter(self.layer_embedding_map.values())).shape[0]

# --- Entity Processor Strategy Pattern ---

class EntityProcessor(ABC):
    @abstractmethod
    def get_centroid(self, entity: DXFEntity, transform: Matrix44) -> Optional[Vec3]:
        raise NotImplementedError

    @abstractmethod
    def to_shapely_geom(self, entity: DXFEntity, transform: Matrix44) -> Optional[LineString | Point | Polygon]:
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, entity: DXFEntity, norm_params: Dict, transform: Matrix44) -> torch.Tensor:
        raise NotImplementedError

class LineProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        return transform.transform((entity.dxf.start + entity.dxf.end) / 2)

    def to_shapely_geom(self, entity, transform):
        start = transform.transform(entity.dxf.start)
        end = transform.transform(entity.dxf.end)
        return LineString([start.xyz[:2], end.xyz[:2]])

    def extract_features(self, entity, norm_params, transform):
        bbox_min, scale = norm_params['bbox_min'], norm_params['scale']
        start, end = transform.transform(entity.dxf.start), transform.transform(entity.dxf.end)
        norm_start = SingleDrawingProcessor._normalize_coords(start, bbox_min, scale)
        norm_end = SingleDrawingProcessor._normalize_coords(end, bbox_min, scale)
        length = start.distance(end)
        return torch.tensor([*norm_start, *norm_end, length], dtype=torch.float)

class CircleProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        return transform.transform(entity.dxf.center)

    def to_shapely_geom(self, entity, transform):
        center = transform.transform(entity.dxf.center)
        scale_x = transform.ux.magnitude
        scale_y = transform.uy.magnitude
        radius = entity.dxf.radius * max(scale_x, scale_y)
        return Point(center.xyz[:2]).buffer(radius)

    def extract_features(self, entity, norm_params, transform):
        bbox_min, scale = norm_params['bbox_min'], norm_params['scale']
        center = transform.transform(entity.dxf.center)
        norm_center = SingleDrawingProcessor._normalize_coords(center, bbox_min, scale)
        radius = entity.dxf.radius * transform.ux.magnitude
        return torch.tensor([*norm_center, radius], dtype=torch.float)

class ArcProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        return transform.transform(entity.dxf.center)

    def to_shapely_geom(self, entity, transform):
        start_point = transform.transform(entity.start_point)
        end_point = transform.transform(entity.end_point)
        return LineString([start_point.xyz[:2], end_point.xyz[:2]])

    def extract_features(self, entity, norm_params, transform):
        bbox_min, scale = norm_params['bbox_min'], norm_params['scale']
        center = transform.transform(entity.dxf.center)
        norm_center = SingleDrawingProcessor._normalize_coords(center, bbox_min, scale)
        radius = entity.dxf.radius * transform.ux.magnitude
        start_angle = np.deg2rad(entity.dxf.start_angle)
        sweep_angle = np.deg2rad(entity.dxf.end_angle - entity.dxf.start_angle)
        return torch.tensor([*norm_center, radius, start_angle, sweep_angle], dtype=torch.float)

class LwPolylineProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        try:
            points = [Vec3(p[:2]) for p in entity.get_points()]
            if not points: return None
            centroid = sum(points, Vec3()) / len(points)
            return transform.transform(centroid)
        except (AttributeError, TypeError):
            return None

    def to_shapely_geom(self, entity, transform):
        points = [transform.transform(Vec3(p[:2])).xyz[:2] for p in entity.get_points()]
        return LineString(points) if len(points) > 1 else Point(points[0]) if points else None

    def extract_features(self, entity, norm_params, transform):
        bbox_min, scale = norm_params['bbox_min'], norm_params['scale']
        points = [transform.transform(Vec3(p[:2])) for p in entity.get_points()]
        if not points:
            return torch.zeros(8, dtype=torch.float) # 2*3 + 2
        norm_start = SingleDrawingProcessor._normalize_coords(points[0], bbox_min, scale)
        norm_end = SingleDrawingProcessor._normalize_coords(points[-1], bbox_min, scale)
        length = sum(p1.distance(p2) for p1, p2 in zip(points, points[1:]))
        is_closed = 1.0 if entity.is_closed else 0.0
        return torch.tensor([*norm_start, *norm_end, length, is_closed], dtype=torch.float)

class GenericPointProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        if hasattr(entity.dxf, 'insert'):
            return transform.transform(entity.dxf.insert)
        return None

    def to_shapely_geom(self, entity, transform):
        centroid = self.get_centroid(entity, transform)
        return Point(centroid.xyz[:2]) if centroid else None

    def extract_features(self, entity, norm_params, transform):
        bbox_min, scale = norm_params['bbox_min'], norm_params['scale']
        insert_point = self.get_centroid(entity, transform)
        if insert_point:
            norm_insert = SingleDrawingProcessor._normalize_coords(insert_point, bbox_min, scale)
            return torch.tensor(norm_insert, dtype=torch.float)
        return torch.zeros(3, dtype=torch.float)

class TextProcessor(GenericPointProcessor):
    def extract_features(self, entity, norm_params, transform):
        # This will be handled in the main processor to append text embeddings
        return super().extract_features(entity, norm_params, transform)

class MtextProcessor(GenericPointProcessor):
    def extract_features(self, entity, norm_params, transform):
        # This will be handled in the main processor to append text embeddings
        return super().extract_features(entity, norm_params, transform)

class InsertProcessor(GenericPointProcessor):
    def to_shapely_geom(self, entity, transform):
        return None

# --- Main Processor Class ---

class SingleDrawingProcessor:
    GEOMETRIC_FEATURE_SIZE = 15

    @staticmethod
    def _normalize_coords(point: Vec3, bbox_min: Vec3, scale: float) -> np.ndarray:
        if point is None: return np.zeros(3)
        return np.array(((point - bbox_min) * scale).xyz)

    def __init__(self, layer_embedding_map: Dict[str, torch.Tensor], linetype_to_idx: Dict[str, int], k_neighbors: int = 3, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.layer_embedding_map = layer_embedding_map
        self.linetype_to_idx = linetype_to_idx
        self.k_neighbors = k_neighbors
        self.embedding_dim = next(iter(layer_embedding_map.values()), torch.zeros(384)).shape[0]
        self.zero_embedding = torch.zeros(self.embedding_dim, dtype=torch.float)
        self.sentence_transformer_model = SentenceTransformer(model_name)

        self.entity_processors: Dict[str, EntityProcessor] = {
            'LINE': LineProcessor(),
            'CIRCLE': CircleProcessor(),
            'ARC': ArcProcessor(),
            'LWPOLYLINE': LwPolylineProcessor(),
            'INSERT': InsertProcessor(),
            'TEXT': TextProcessor(),
            'MTEXT': MtextProcessor(),
        }
        self.supported_entity_types = set(self.entity_processors.keys())

    def process(self, dxf_file_path: str) -> Optional[Dict]:
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
        except (IOError, OSError, DXFError) as e:
            logging.warning(f"Could not process file {dxf_file_path}: {e}")
            return None

        bbox = ezdxf.bbox.extents(msp, cache=None)
        if not bbox.has_data: return None

        bbox_min, bbox_size = bbox.extmin, bbox.size
        original_scale = max(bbox_size) if bbox_size and max(bbox_size) > 0 else 1.0
        norm_scale = 1.0 / original_scale if original_scale > 1e-8 else 1.0
        norm_params = {'bbox_min': bbox_min, 'scale': norm_scale, 'original_scale': original_scale}

        unique_id_to_node_info: Dict[str, Tuple[int, str, DXFEntity, Optional[Vec3], Matrix44]] = {}
        node_counter = 0
        edges_by_type = defaultdict(set)
        block_templates = {b.name: list(b) for b in doc.blocks if not b.name.startswith('*')}

        def process_entities_hierarchically(entities, parent_idx=None, transform=Matrix44()):
            nonlocal node_counter
            for entity in entities:
                entity_type = entity.dxf.dxftype
                if hasattr(entity, 'dxf') and entity_type in self.supported_entity_types:
                    unique_id = entity.dxf.handle
                    if unique_id not in unique_id_to_node_info:
                        processor = self.entity_processors[entity_type]
                        node_type = 'block_instance' if entity_type == 'INSERT' else entity_type
                        centroid = processor.get_centroid(entity, transform)
                        current_idx = node_counter
                        unique_id_to_node_info[unique_id] = (current_idx, node_type, entity, centroid, transform)
                        node_counter += 1
                    else:
                        current_idx = unique_id_to_node_info[unique_id][0]

                    if parent_idx is not None:
                        parent_info = next((v for v in unique_id_to_node_info.values() if v[0] == parent_idx), None)
                        if parent_info: edges_by_type[(parent_info[1], 'contains', node_type)].add((parent_idx, current_idx))

                    if entity_type == 'INSERT' and entity.dxf.name in block_templates:
                        process_entities_hierarchically(block_templates.get(entity.dxf.name, []), current_idx, transform @ entity.matrix44())

        process_entities_hierarchically(msp)
        self._build_spatial_edges(edges_by_type, unique_id_to_node_info)

        final_nodes, global_to_local_idx_map = self._finalize_nodes(unique_id_to_node_info, norm_params)
        final_edges = self._finalize_edges(edges_by_type, unique_id_to_node_info, global_to_local_idx_map)

        return {"file_path": dxf_file_path, "nodes": dict(final_nodes), "edges": final_edges}

    def _build_spatial_edges(self, edges_by_type, unique_id_to_node_info):
        geometric_nodes = [n for n in unique_id_to_node_info.values() if n[1] not in {'block_instance', 'TEXT', 'MTEXT'}]
        if len(geometric_nodes) < 2: return

        geoms_with_info = []
        for info in geometric_nodes:
            processor = self.entity_processors.get(info[2].dxf.dxftype)
            if not processor: continue
            geom = processor.to_shapely_geom(info[2], info[4])
            if geom and not geom.is_empty:
                geoms_with_info.append({'idx': info[0], 'type': info[1], 'geom': geom})

        geoms_only = [item['geom'] for item in geoms_with_info]
        if len(geoms_only) < 2: return

        strtree = STRtree(geoms_only)

        for i, item1 in enumerate(geoms_with_info):
            geom1 = item1['geom']
            possible_neighbors_indices = strtree.query(geom1)

            for j in possible_neighbors_indices:
                if i >= j: continue

                item2 = geoms_with_info[j]
                geom2 = item2['geom']
                idx1, type1 = item1['idx'], item1['type']
                idx2, type2 = item2['idx'], item2['type']
                key = tuple(sorted((type1, type2)))
                tolerance = 1e-9

                if geom1.intersects(geom2.buffer(tolerance)):
                    if geom1.boundary.dwithin(geom2.boundary, tolerance):
                         edges_by_type[(key[0], 'connects', key[1])].add(tuple(sorted((idx1, idx2))))
                    else:
                         edges_by_type[(key[0], 'intersects', key[1])].add(tuple(sorted((idx1, idx2))))

        if len(geometric_nodes) > self.k_neighbors:
            centroids = [n[3] for n in geometric_nodes if n[3] is not None]
            if not centroids: return
            node_indices = [n[0] for n in geometric_nodes if n[3] is not None]
            kdtree = KDTree([c.xyz for c in centroids])

            distances, neighbors = kdtree.query([c.xyz for c in centroids], k=self.k_neighbors + 1)

            for i, neighbor_indices in enumerate(neighbors):
                idx1 = node_indices[i]
                type1 = geometric_nodes[i][1]

                for k in range(1, len(neighbor_indices)):
                    neighbor_original_idx = neighbor_indices[k]
                    if neighbor_original_idx < len(node_indices):
                        idx2 = node_indices[neighbor_original_idx]
                        type2 = geometric_nodes[neighbor_original_idx][1]

                        if idx1 >= idx2: continue
                        key = tuple(sorted((type1, type2)))
                        edges_by_type[(key[0], 'nearby', key[1])].add(tuple(sorted((idx1, idx2))))

    def _finalize_nodes(self, unique_id_to_node_info, norm_params):
        final_nodes = defaultdict(lambda: {'x': [], 'discrete': []})
        global_to_local_idx_map = {}
        sorted_nodes = sorted(unique_id_to_node_info.values(), key=lambda x: x[0])

        for global_idx, node_type, entity, _, transform in sorted_nodes:
            entity_type = entity.dxf.dxftype
            processor = self.entity_processors.get(entity_type)
            if not processor: continue

            try:
                geometric_features = processor.extract_features(entity, norm_params, transform)

                if entity_type in {'TEXT', 'MTEXT'}:
                    text_content = entity.dxf.text if entity_type == 'TEXT' else entity.text
                    text_embedding = self.sentence_transformer_model.encode(text_content, convert_to_tensor=True)
                    continuous_features = torch.cat([geometric_features, text_embedding.cpu()])
                else:
                    padded_geom_features = torch.zeros(self.GEOMETRIC_FEATURE_SIZE)
                    padded_geom_features[:len(geometric_features)] = geometric_features

                    layer_name = getattr(entity.dxf, 'layer', '0')
                    layer_embedding = self.layer_embedding_map.get(layer_name, self.zero_embedding)

                    continuous_features = torch.cat([
                        padded_geom_features,
                        torch.tensor([norm_params.get('original_scale', 1.0)], dtype=torch.float),
                        layer_embedding,
                    ])
            except Exception as e:
                logging.debug(f"Feature extraction error for {entity_type} (handle: {entity.dxf.handle}): {e}")
                continue

            linetype_name = getattr(entity.dxf, 'linetype', 'BYLAYER')
            linetype_idx = self.linetype_to_idx.get(linetype_name, 0)
            discrete_features = torch.tensor([linetype_idx], dtype=torch.long)

            local_idx = len(final_nodes[node_type]['x'])
            final_nodes[node_type]['x'].append(continuous_features)
            final_nodes[node_type]['discrete'].append(discrete_features)
            global_to_local_idx_map[global_idx] = local_idx

        for node_type, data in final_nodes.items():
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

                l_idx1, l_idx2 = global_to_local_idx_map.get(g_idx1), global_to_local_idx_map.get(g_idx2)
                if l_idx1 is None or l_idx2 is None: continue

                if (src_type, dst_type) == (node1_info[1], node2_info[1]):
                    remapped.append([l_idx1, l_idx2])
                elif (src_type, dst_type) == (node2_info[1], node1_info[1]):
                     remapped.append([l_idx2, l_idx1])
            if remapped:
                final_edges[(src_type, rel, dst_type)] = torch.tensor(remapped, dtype=torch.long).t().contiguous()
        return final_edges

class GraphBuilder:
    def __init__(self):
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}

    def fit_transform(self, data_list: List[Dict]) -> List[HeteroData]:
        features_by_type = defaultdict(list)
        for data in data_list:
            if not data or 'nodes' not in data: continue
            for node_type, node_data in data['nodes'].items():
                if 'x' in node_data and node_data['x'].nelement() > 0:
                    features_by_type[node_type].append(node_data['x'])

        for node_type, features_list in features_by_type.items():
            if not features_list: continue
            combined = torch.cat(features_list, dim=0)
            self.means[node_type] = torch.mean(combined, dim=0)
            self.stds[node_type] = torch.std(combined, dim=0)
        return self.transform(data_list)

    def transform(self, data_list: List[Dict]) -> List[HeteroData]:
        hetero_data_list = []
        for data in data_list:
            if not data: continue
            hetero_data = HeteroData()
            if 'nodes' in data:
                for node_type, node_data in data['nodes'].items():
                    if 'x' in node_data and node_data['x'].nelement() > 0:
                        features = node_data['x']
                        if node_type in self.means:
                            std = self.stds[node_type].clone()
                            std[std < 1e-8] = 1.0
                            features = (features - self.means[node_type]) / std
                        hetero_data[node_type].x = features
                        if 'discrete' in node_data and node_data['discrete'].nelement() > 0:
                            hetero_data[node_type].discrete = node_data['discrete']
            if 'edges' in data:
                for edge_type, edge_index in data['edges'].items():
                    hetero_data[edge_type].edge_index = edge_index
            hetero_data_list.append(hetero_data)
        return hetero_data_list

def _create_dxf_for_full_pipeline_test(temp_dir: Path) -> str:
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (5, 5))
    msp.add_line((5, 5), (10, 0))
    msp.add_line((0, 2.5), (10, 2.5))
    msp.add_text("Hello World", dxfattribs={'insert': (0, 15)})
    msp.add_mtext("This is a multiline\ntext.", dxfattribs={'insert': (15, 15)})
    path = temp_dir / "full_pipeline_test.dxf"
    doc.saveas(path)
    return str(path)

def _demonstrate_full_pipeline():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    try:
        logging.info("--- Running Full Pipeline Demonstration ---")
        dxf_path = _create_dxf_for_full_pipeline_test(Path(temp_dir))

        gp = GlobalPreprocessor()
        gp.fit([dxf_path])

        sp = SingleDrawingProcessor(gp.layer_embedding_map, gp.linetype_to_idx)
        intermediate_data = sp.process(dxf_path)

        assert intermediate_data is not None, "Processing failed, returned None."
        assert 'nodes' in intermediate_data, "No nodes found in intermediate data."
        assert 'TEXT' in intermediate_data['nodes'], "TEXT node was not created."
        assert 'MTEXT' in intermediate_data['nodes'], "MTEXT node was not created."
        assert 'edges' in intermediate_data, "No edges found in intermediate data."
        assert ('LINE', 'connects', 'LINE') in intermediate_data['edges'], "Connects edge was not created."
        assert ('LINE', 'intersects', 'LINE') in intermediate_data['edges'], "Intersects edge was not created."

        logging.info("--- Demonstration Successful: All nodes and edges created correctly! ---")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    _demonstrate_full_pipeline()
