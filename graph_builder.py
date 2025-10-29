# graph_builder.py

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

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

@dataclass
class NodeInfo:
    global_idx: int
    node_type: str
    entity: DXFEntity
    centroid: Optional[Vec3]
    transform: Matrix44

class GlobalPreprocessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.sentence_transformer_model = SentenceTransformer(model_name)
        self.layer_embedding_map: Dict[str, torch.Tensor] = {}
        self.linetype_to_idx: Dict[str, int] = {}
        self.layer_to_idx: Dict[str, int] = {}
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

        # Continuous feature: layer name embeddings
        layer_embeddings = self.sentence_transformer_model.encode(layer_names_list, convert_to_tensor=True)
        self.layer_embedding_map = {name: emb for name, emb in zip(layer_names_list, layer_embeddings)}

        # Discrete feature: layer name to integer index
        self.layer_to_idx = {name: i for i, name in enumerate(layer_names_list)}

        unique_linetype_names.update(['BYLAYER', 'BYBLOCK', 'CONTINUOUS'])
        self.linetype_to_idx = {name: i for i, name in enumerate(sorted(list(unique_linetype_names)))}

    def save(self, directory: str) -> None:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "layer_embedding_map.pkl", "wb") as f:
            pickle.dump(self.layer_embedding_map, f)
        with open(dir_path / "linetype_to_idx.json", "w") as f:
            json.dump(self.linetype_to_idx, f)
        with open(dir_path / "layer_to_idx.json", "w") as f:
            json.dump(self.layer_to_idx, f)

    def load(self, directory: str) -> None:
        dir_path = Path(directory)
        with open(dir_path / "layer_embedding_map.pkl", "rb") as f:
            self.layer_embedding_map = pickle.load(f)
        with open(dir_path / "linetype_to_idx.json", "r") as f:
            self.linetype_to_idx = json.load(f)
        with open(dir_path / "layer_to_idx.json", "r") as f:
            self.layer_to_idx = json.load(f)
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
        center, scale, original_scale = norm_params['center'], norm_params['scale'], norm_params['original_scale']
        start, end = transform.transform(entity.dxf.start), transform.transform(entity.dxf.end)
        norm_start = SingleDrawingProcessor._normalize_coords(start, center, scale)
        norm_end = SingleDrawingProcessor._normalize_coords(end, center, scale)
        length = start.distance(end) / original_scale
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
        center_param, scale, original_scale = norm_params['center'], norm_params['scale'], norm_params['original_scale']
        center = transform.transform(entity.dxf.center)
        norm_center = SingleDrawingProcessor._normalize_coords(center, center_param, scale)
        radius = (entity.dxf.radius * transform.ux.magnitude) / original_scale
        return torch.tensor([*norm_center, radius], dtype=torch.float)

class ArcProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        # The centroid of an arc is more complex, for now, we use the center of the circle from which the arc is derived
        return transform.transform(entity.dxf.center)

    def to_shapely_geom(self, entity, transform):
        # Flattening provides a good approximation of the arc's geometry
        points = [transform.transform(p).xyz[:2] for p in entity.flattening(sagitta=0.1)]
        return LineString(points) if len(points) > 1 else None

    def extract_features(self, entity, norm_params, transform):
        center_param, scale, original_scale = norm_params['center'], norm_params['scale'], norm_params['original_scale']
        center = transform.transform(entity.dxf.center)
        norm_center = SingleDrawingProcessor._normalize_coords(center, center_param, scale)
        radius = (entity.dxf.radius * transform.ux.magnitude) / original_scale
        # Normalize angles by dividing by 360. DXF angles can exceed 360 or be negative.
        # This initial normalization will be further standardized by the GraphBuilder.
        start_angle = entity.dxf.start_angle / 360.0
        end_angle = entity.dxf.end_angle / 360.0
        return torch.tensor([*norm_center, radius, start_angle, end_angle], dtype=torch.float)


class DimensionProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        # Centroid of a dimension is not well-defined, return insert point
        return transform.transform(entity.dxf.insert)

    def to_shapely_geom(self, entity, transform):
        # A dimension does not have a primary geometry, return None
        return None

    def extract_features(self, entity, norm_params, transform):
        center_param, scale = norm_params['center'], norm_params['scale']
        insert = transform.transform(entity.dxf.insert)
        norm_insert = SingleDrawingProcessor._normalize_coords(insert, center_param, scale)
        return torch.tensor(norm_insert, dtype=torch.float)


class LwPolylineProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        try:
            # First, transform all points to the world coordinate system
            points = list(entity.flattening(distance=0.1))
            if not points: return None
            transformed_points = [transform.transform(p) for p in points]
            # Then, calculate the centroid from the transformed points
            centroid = sum(transformed_points, Vec3()) / len(transformed_points)
            return centroid
        except (AttributeError, TypeError):
            return None

    def to_shapely_geom(self, entity, transform):
        # Use flattening to accurately represent polylines with bulges
        points = [transform.transform(p).xyz[:2] for p in entity.flattening(distance=0.1)]
        return LineString(points) if len(points) > 1 else Point(points[0]) if points else None

    def extract_features(self, entity, norm_params, transform):
        center, scale, original_scale = norm_params['center'], norm_params['scale'], norm_params['original_scale']

        # Use flattening for geometric accuracy
        points = list(entity.flattening(distance=0.1))
        transformed_points = [transform.transform(p) for p in points]

        if not transformed_points:
            return torch.zeros(9, dtype=torch.float) # 2*3 + 3 (length, closed, width)

        norm_start = SingleDrawingProcessor._normalize_coords(transformed_points[0], center, scale)
        norm_end = SingleDrawingProcessor._normalize_coords(transformed_points[-1], center, scale)
        length = sum(p1.distance(p2) for p1, p2 in zip(transformed_points, transformed_points[1:])) / original_scale
        is_closed = 1.0 if entity.is_closed else 0.0

        # Extract and average width information
        widths = [p[2] for p in entity.points] + [p[3] for p in entity.points]
        avg_width = (sum(widths) / len(widths) if widths else 0.0) / original_scale

        return torch.tensor([*norm_start, *norm_end, length, is_closed, avg_width], dtype=torch.float)

class GenericPointProcessor(EntityProcessor):
    def get_centroid(self, entity, transform):
        if hasattr(entity.dxf, 'insert'):
            return transform.transform(entity.dxf.insert)
        return None

    def to_shapely_geom(self, entity, transform):
        centroid = self.get_centroid(entity, transform)
        return Point(centroid.xyz[:2]) if centroid else None

    def extract_features(self, entity, norm_params, transform):
        center, scale = norm_params['center'], norm_params['scale']
        insert_point = self.get_centroid(entity, transform)
        if insert_point:
            norm_insert = SingleDrawingProcessor._normalize_coords(insert_point, center, scale)
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
    def _normalize_coords(point: Vec3, center: Vec3, scale: float) -> np.ndarray:
        if point is None: return np.zeros(3)
        return np.array(((point - center) * scale).xyz)

    def __init__(self, layer_embedding_map: Dict[str, torch.Tensor], linetype_to_idx: Dict[str, int], layer_to_idx: Dict[str, int], sentence_transformer_model: SentenceTransformer, k_neighbors: int = 3):
        self.layer_embedding_map = layer_embedding_map
        self.linetype_to_idx = linetype_to_idx
        self.layer_to_idx = layer_to_idx
        self.k_neighbors = k_neighbors
        self.embedding_dim = next(iter(layer_embedding_map.values()), torch.zeros(384)).shape[0]
        self.zero_embedding = torch.zeros(self.embedding_dim, dtype=torch.float)
        self.sentence_transformer_model = sentence_transformer_model

        self.entity_processors: Dict[str, EntityProcessor] = {
            'LINE': LineProcessor(),
            'CIRCLE': CircleProcessor(),
            'ARC': ArcProcessor(),
            'LWPOLYLINE': LwPolylineProcessor(),
            'INSERT': InsertProcessor(),
            'TEXT': TextProcessor(),
            'MTEXT': MtextProcessor(),
            'DIMENSION': DimensionProcessor(),
        }
        self.supported_entity_types = set(self.entity_processors.keys())

    def process(self, dxf_file_path: str) -> Optional[Dict]:
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
        except (IOError, OSError, DXFError) as e:
            logging.warning(f"Could not process file {dxf_file_path}: {e}")
            return None

        sanitized_entities = self._sanitize_entities(msp)
        bbox = ezdxf.bbox.extents(sanitized_entities, cache=None)
        if not bbox.has_data: return None

        bbox_center, bbox_size = bbox.center, bbox.size
        original_scale = max(bbox_size) if bbox_size and max(bbox_size) > 0 else 1.0
        norm_scale = 1.0 / original_scale if original_scale > 1e-8 else 1.0
        norm_params = {'center': bbox_center, 'scale': norm_scale, 'original_scale': original_scale}

        unique_id_to_node_info: Dict[str, NodeInfo] = {}
        node_counter = 0
        edges_by_type = defaultdict(set)
        block_templates = {b.name: list(b) for b in doc.blocks if not b.name.startswith('*')}

        def process_entities_hierarchically(entities, parent_idx=None, transform=Matrix44()):
            nonlocal node_counter
            for entity in entities:
                entity_type = entity.dxf.dxftype
                if hasattr(entity, 'dxf') and entity_type in self.supported_entity_types:
                    unique_id = entity.dxf.handle if entity.dxf.handle is not None else f"decomposed_{node_counter}"
                    if unique_id not in unique_id_to_node_info:
                        processor = self.entity_processors[entity_type]
                        node_type = 'block_instance' if entity_type == 'INSERT' else entity_type
                        centroid = processor.get_centroid(entity, transform)
                        current_idx = node_counter
                        unique_id_to_node_info[unique_id] = NodeInfo(
                            global_idx=current_idx,
                            node_type=node_type,
                            entity=entity,
                            centroid=centroid,
                            transform=transform
                        )
                        node_counter += 1
                    else:
                        current_idx = unique_id_to_node_info[unique_id].global_idx

                    if parent_idx is not None:
                        parent_info = next((v for v in unique_id_to_node_info.values() if v.global_idx == parent_idx), None)
                        if parent_info: edges_by_type[(parent_info.node_type, 'contains', node_type)].add((parent_idx, current_idx))

                    if entity_type == 'INSERT' and entity.dxf.name in block_templates:
                        process_entities_hierarchically(block_templates.get(entity.dxf.name, []), current_idx, transform @ entity.matrix44())

        process_entities_hierarchically(sanitized_entities)
        self._build_spatial_edges(edges_by_type, unique_id_to_node_info)

        final_nodes, global_to_local_idx_map = self._finalize_nodes(unique_id_to_node_info, norm_params)
        final_edges = self._finalize_edges(edges_by_type, unique_id_to_node_info, global_to_local_idx_map)

        return {"file_path": dxf_file_path, "nodes": dict(final_nodes), "edges": final_edges}

    def _build_spatial_edges(self, edges_by_type, unique_id_to_node_info: Dict[str, NodeInfo]):
        geometric_nodes = [n for n in unique_id_to_node_info.values() if n.node_type not in {'block_instance', 'TEXT', 'MTEXT'}]
        if len(geometric_nodes) < 2: return

        geoms_with_info = []
        for info in geometric_nodes:
            processor = self.entity_processors.get(info.entity.dxf.dxftype)
            if not processor: continue
            geom = processor.to_shapely_geom(info.entity, info.transform)
            if geom and not geom.is_empty:
                geoms_with_info.append({'idx': info.global_idx, 'type': info.node_type, 'geom': geom})

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
            centroids = [n.centroid for n in geometric_nodes if n.centroid is not None]
            if not centroids: return
            node_indices = [n.global_idx for n in geometric_nodes if n.centroid is not None]
            kdtree = KDTree([c.xyz for c in centroids])

            distances, neighbors = kdtree.query([c.xyz for c in centroids], k=self.k_neighbors + 1)

            for i, neighbor_indices in enumerate(neighbors):
                idx1 = node_indices[i]
                type1 = geometric_nodes[i].node_type

                for k in range(1, len(neighbor_indices)):
                    neighbor_original_idx = neighbor_indices[k]
                    if neighbor_original_idx < len(node_indices):
                        idx2 = node_indices[neighbor_original_idx]
                        type2 = geometric_nodes[neighbor_original_idx].node_type

                        if idx1 >= idx2: continue
                        key = tuple(sorted((type1, type2)))
                        edges_by_type[(key[0], 'nearby', key[1])].add(tuple(sorted((idx1, idx2))))

    def _finalize_nodes(self, unique_id_to_node_info: Dict[str, NodeInfo], norm_params: Dict):
        final_nodes = defaultdict(lambda: {'x': [], 'discrete': []})
        global_to_local_idx_map = {}
        sorted_nodes = sorted(unique_id_to_node_info.values(), key=lambda x: x.global_idx)

        for node in sorted_nodes:
            entity_type = node.entity.dxf.dxftype
            processor = self.entity_processors.get(entity_type)
            if not processor: continue

            try:
                # --- Continuous Features ---
                geometric_features = processor.extract_features(node.entity, norm_params, node.transform)

                # Handle RGB color (already resolved for BYLAYER/BYBLOCK)
                rgb = node.entity.rgb if node.entity.rgb is not None else (255, 255, 255)
                normalized_rgb = torch.tensor(rgb, dtype=torch.float) / 255.0

                if entity_type in {'TEXT', 'MTEXT'}:
                    text_content = node.entity.dxf.text if entity_type == 'TEXT' else node.entity.text
                    text_embedding = self.sentence_transformer_model.encode(text_content, convert_to_tensor=True)
                    continuous_features = torch.cat([geometric_features, normalized_rgb, text_embedding.cpu()])
                else:
                    padded_geom_features = torch.zeros(self.GEOMETRIC_FEATURE_SIZE)
                    padded_geom_features[:len(geometric_features)] = geometric_features

                    layer_name = getattr(node.entity.dxf, 'layer', '0')
                    layer_embedding = self.layer_embedding_map.get(layer_name, self.zero_embedding)

                    continuous_features = torch.cat([
                        padded_geom_features,
                        torch.tensor([norm_params.get('original_scale', 1.0)], dtype=torch.float),
                        normalized_rgb,
                        layer_embedding,
                    ])

                # --- Discrete Features ---
                layer_name = node.entity.dxf.layer
                layer_idx = self.layer_to_idx.get(layer_name, 0)

                # Resolve BYLAYER linetype
                linetype_name = node.entity.dxf.linetype
                if linetype_name.upper() == 'BYLAYER':
                    layer = node.entity.doc.layers.get(layer_name)
                    linetype_name = layer.dxf.linetype
                linetype_idx = self.linetype_to_idx.get(linetype_name, 0)

                color_aci = node.entity.dxf.color # ACI color index

                # Resolve BYLAYER lineweight
                lineweight = node.entity.dxf.lineweight
                if lineweight == -1: # -1 is BYLAYER
                    layer = node.entity.doc.layers.get(layer_name)
                    lineweight = layer.dxf.lineweight

                discrete_features = torch.tensor([layer_idx, linetype_idx, color_aci, lineweight], dtype=torch.long)

            except Exception as e:
                logging.debug(f"Feature extraction error for {entity_type} (handle: {node.entity.dxf.handle}): {e}")
                continue

            local_idx = len(final_nodes[node.node_type]['x'])
            final_nodes[node.node_type]['x'].append(continuous_features)
            final_nodes[node.node_type]['discrete'].append(discrete_features)
            global_to_local_idx_map[node.global_idx] = local_idx

        for node_type, data in final_nodes.items():
            if data['x']:
                data['x'] = torch.stack(data['x'])
                data['discrete'] = torch.stack(data['discrete'])
        return final_nodes, global_to_local_idx_map

    def _finalize_edges(self, edges_by_type, unique_id_to_node_info: Dict[str, NodeInfo], global_to_local_idx_map: Dict[int, int]):
        final_edges = {}
        info_map = {v.global_idx: v for v in unique_id_to_node_info.values()}
        for (src_type, rel, dst_type), edge_set in edges_by_type.items():
            remapped = []
            for g_idx1, g_idx2 in edge_set:
                node1_info, node2_info = info_map.get(g_idx1), info_map.get(g_idx2)
                if not node1_info or not node2_info: continue

                l_idx1, l_idx2 = global_to_local_idx_map.get(g_idx1), global_to_local_idx_map.get(g_idx2)
                if l_idx1 is None or l_idx2 is None: continue

                if (src_type, dst_type) == (node1_info.node_type, node2_info.node_type):
                    remapped.append([l_idx1, l_idx2])
                elif (src_type, dst_type) == (node2_info.node_type, node1_info.node_type):
                     remapped.append([l_idx2, l_idx1])
            if remapped:
                final_edges[(src_type, rel, dst_type)] = torch.tensor(remapped, dtype=torch.long).t().contiguous()
        return final_edges

    def _sanitize_entities(self, msp: Modelspace) -> List[DXFEntity]:
        sanitized_entities = []
        for entity in msp:
            if entity.dxf.dxftype == 'DIMENSION':
                try:
                    # Decompose dimension into atomic entities
                    decomposed = list(entity.virtual_entities())
                    # Inherit properties from the parent dimension
                    for sub_entity in decomposed:
                        sub_entity.dxf.layer = entity.dxf.layer
                        sub_entity.dxf.color = entity.dxf.color
                        # Note: Not all properties are applicable or exist on sub-entities
                    sanitized_entities.extend(decomposed)
                except Exception as e:
                    logging.warning(f"Could not decompose DIMENSION (handle: {entity.dxf.handle}): {e}")
                    sanitized_entities.append(entity) # Keep the original if decomposition fails
            else:
                sanitized_entities.append(entity)
        return sanitized_entities


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
    # Define layers with different properties
    doc.layers.new(name="GEOMETRY", dxfattribs={"color": 1})  # Blue
    doc.layers.new(name="TEXT", dxfattribs={"color": 2})  # Yellow
    doc.layers.new(name="BLOCKS", dxfattribs={"color": 3})  # Green

    # Create a simple base block (Block A)
    block_a = doc.blocks.new(name="BLOCK_A")
    block_a.add_line((0, 0), (1, 1), dxfattribs={"layer": "GEOMETRY"})
    block_a.add_circle((0.5, 0.5), 0.25, dxfattribs={"layer": "GEOMETRY"})

    # Create a nested block (Block B) that contains Block A
    block_b = doc.blocks.new(name="BLOCK_B")
    block_b.add_blockref("BLOCK_A", insert=(0, 0), dxfattribs={
        "layer": "BLOCKS",
        "rotation": 45,
        "xscale": 2.0,
        "yscale": 2.0,
    })
    block_b.add_text("Nested", dxfattribs={"insert": (1, 1), "layer": "TEXT"})

    msp = doc.modelspace()
    # Add entities to modelspace to create different relationships
    # 1. A line that connects with the arc
    msp.add_line((5, 0), (7, 0), dxfattribs={"layer": "GEOMETRY"}) # Connects with arc endpoint
    # 2. An arc
    msp.add_arc(center=(0, 0), radius=5, start_angle=0, end_angle=90, dxfattribs={"layer": "GEOMETRY"})
    # 3. A line that intersects with the first line
    msp.add_line((6, -1), (6, 1), dxfattribs={"layer": "GEOMETRY"}) # Intersects line 1
    # 4. A standalone circle (nearby)
    msp.add_circle((10, 10), 1, dxfattribs={"layer": "GEOMETRY"})
    # 5. Insert the nested block
    msp.add_blockref("BLOCK_B", insert=(15, 15))
    # 6. Add text entities
    msp.add_text("Hello", dxfattribs={'insert': (0, 15), "layer": "TEXT"})
    msp.add_mtext("World", dxfattribs={'insert': (15, 10), "layer": "TEXT"})

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

        # Pre-load the model to be passed into the processor
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        sp = SingleDrawingProcessor(gp.layer_embedding_map, gp.linetype_to_idx, gp.layer_to_idx, model)
        intermediate_data = sp.process(dxf_path)

        assert intermediate_data is not None, "Processing failed, returned None."
        logging.info("--- Graph Data Summary ---")

        # Node Summary
        logging.info("\n[Node Summary]")
        if 'nodes' in intermediate_data and intermediate_data['nodes']:
            for node_type, node_data in intermediate_data['nodes'].items():
                num_nodes = len(node_data.get('x', []))
                x_shape = node_data.get('x', torch.Tensor()).shape
                discrete_shape = node_data.get('discrete', torch.Tensor()).shape
                logging.info(f"  - Node Type: {node_type:<15} | Count: {num_nodes:<5} | Feature Shape: {str(x_shape):<20} | Discrete Shape: {str(discrete_shape)}")
        else:
            logging.info("  No nodes found.")

        # Edge Summary
        logging.info("\n[Edge Summary]")
        if 'edges' in intermediate_data and intermediate_data['edges']:
            for edge_type, edge_index in intermediate_data['edges'].items():
                num_edges = edge_index.shape[1]
                logging.info(f"  - Edge Type: {str(edge_type):<40} | Count: {num_edges}")
        else:
            logging.info("  No edges found.")

        # Print features as a JSON string
        logging.info("\n[Graph Features as JSON]")

        def convert_tensors_to_lists(data):
            if isinstance(data, torch.Tensor):
                return data.tolist()
            if isinstance(data, dict):
                return {str(k) if isinstance(k, tuple) else k: convert_tensors_to_lists(v) for k, v in data.items()}
            if isinstance(data, list):
                return [convert_tensors_to_lists(i) for i in data]
            return data

        exportable_data = convert_tensors_to_lists(intermediate_data)
        json_string = json.dumps(exportable_data, indent=2)
        logging.info(json_string)

        logging.info("\n--- Demonstration Complete ---")

    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    _demonstrate_full_pipeline()
