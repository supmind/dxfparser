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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalPreprocessor:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.sentence_transformer_model = SentenceTransformer(model_name)
        self.layer_embedding_map: Dict[str, torch.Tensor] = {}
        self.linetype_to_idx: Dict[str, int] = {}
        self.embedding_dim = self.sentence_transformer_model.get_sentence_embedding_dimension()

    def fit(self, dxf_file_paths: Iterable[str]) -> None:
        unique_layer_names: Set[str] = set()
        unique_linetype_names: Set[str] = set()
        for file_path in dxf_file_paths:
            try:
                doc: Drawing = ezdxf.readfile(file_path)
                unique_layer_names.update(layer.dxf.name for layer in doc.layers)
                unique_linetype_names.update(linetype.dxf.name for linetype in doc.linetypes)
            except (IOError, OSError, DXFError) as e:
                logging.warning(f"Skipping file {file_path} due to error: {e}")

        unique_layer_names.add('0')
        layer_names_list = sorted(list(unique_layer_names))
        layer_embeddings = self.sentence_transformer_model.encode(layer_names_list, convert_to_tensor=True)
        self.layer_embedding_map = {name: emb for name, emb in zip(layer_names_list, layer_embeddings)}

        unique_linetype_names.update(['BYLAYER', 'BYBLOCK', 'CONTINUOUS'])
        linetype_names_list = sorted(list(unique_linetype_names))
        self.linetype_to_idx = {name: i for i, name in enumerate(linetype_names_list)}

class SingleDrawingProcessor:
    def __init__(self, layer_embedding_map: Dict[str, torch.Tensor], linetype_to_idx: Dict[str, int]):
        self.layer_embedding_map = layer_embedding_map
        self.linetype_to_idx = linetype_to_idx
        self.supported_entity_types = {'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT', 'MTEXT', 'DIMENSION'}
        self.embedding_dim = next(iter(self.layer_embedding_map.values()), torch.zeros(384)).shape[0]
        self.total_feature_dim = sum([10, 5, 1]) + self.embedding_dim
        self.zero_embedding = torch.zeros(self.embedding_dim, dtype=torch.float)

    def _get_true_bounding_box(self, msp: Modelspace) -> BoundingBox:
        try:
            return ezdxf.bbox.extents(msp, cache=None)
        except Exception:
            return BoundingBox()

    def _normalize_coords(self, point: Vec3, bbox_min: Vec3, scale: float) -> np.ndarray:
        if point is None: return np.zeros(3)
        return np.array(((point - bbox_min) * scale).xyz)

    def _extract_entity_features(self, entity, norm_params: Dict, entity_type_override: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bbox_min, scale = norm_params['bbox_min'], norm_params['scale']
        coords_vec, dims_vec = np.zeros(10), np.zeros(5)
        entity_type = entity_type_override or (entity.dxf.dxftype if hasattr(entity, 'dxf') else 'vertex')

        try:
            if entity_type == 'LINE':
                start_norm, end_norm = self._normalize_coords(entity.dxf.start, bbox_min, scale), self._normalize_coords(entity.dxf.end, bbox_min, scale)
                coords_vec[:6] = [*start_norm, *end_norm]
                dims_vec[0] = entity.dxf.start.distance(entity.dxf.end)
            elif entity_type == 'CIRCLE':
                center_norm = self._normalize_coords(entity.dxf.center, bbox_min, scale)
                coords_vec[:3] = center_norm
                dims_vec[0] = entity.dxf.radius
            elif entity_type == 'vertex':
                coords_vec[:3] = self._normalize_coords(entity, bbox_min, scale)
        except Exception as e:
            logging.warning(f"Feature extraction error: {e}")

        layer_name = getattr(getattr(entity, 'dxf', {}), 'layer', '0')
        layer_embedding = self.layer_embedding_map.get(layer_name, self.zero_embedding)

        continuous_features = torch.cat([
            torch.tensor(coords_vec, dtype=torch.float), torch.tensor(dims_vec, dtype=torch.float),
            torch.tensor([norm_params['original_scale']], dtype=torch.float), layer_embedding,
        ])

        linetype_name = getattr(getattr(entity, 'dxf', {}), 'linetype', 'BYLAYER')
        linetype_idx = self.linetype_to_idx.get(linetype_name, 0)
        return continuous_features, torch.tensor([linetype_idx], dtype=torch.long)

    def process(self, dxf_file_path: str) -> Optional[Dict]:
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
        except (IOError, OSError, DXFError) as e:
            return None

        bbox = self._get_true_bounding_box(msp)
        if not bbox.has_data: return None

        bbox_min, bbox_size = bbox.extmin, bbox.size
        original_scale = max(bbox_size) if bbox_size and max(bbox_size) > 0 else 1.0
        norm_scale = 1.0 / original_scale if original_scale > 1e-8 else 1.0
        norm_params = {'bbox_min': bbox_min, 'scale': norm_scale, 'original_scale': original_scale}

        unique_id_to_node_info = {}
        node_counter = 0
        edges_by_type = defaultdict(list)

        def register_node(entity, node_type_override=None):
            nonlocal node_counter
            is_virtual = not hasattr(entity, 'dxf') or not hasattr(entity.dxf, 'handle')
            unique_id = f"virtual_{id(entity)}" if is_virtual else entity.dxf.handle

            if unique_id not in unique_id_to_node_info:
                node_type = node_type_override or entity.dxf.dxftype
                unique_id_to_node_info[unique_id] = (node_counter, node_type, entity)
                node_counter += 1
            return unique_id_to_node_info[unique_id][0]

        def process_entities(entities):
            for entity in entities:
                if hasattr(entity, 'dxf') and entity.dxf.dxftype in self.supported_entity_types:
                    current_idx = register_node(entity)

                    if entity.dxf.dxftype == 'LWPOLYLINE':
                        points = [Vec3(p[:2]) for p in entity.get_points()]
                        vertex_indices = [register_node(p, 'vertex') for p in points]
                        for idx in vertex_indices: edges_by_type[('LWPOLYLINE', 'has_vertex', 'vertex')].append((current_idx, idx))
                        for i in range(len(vertex_indices) - 1): edges_by_type[('vertex', 'segment', 'vertex')].append((vertex_indices[i], vertex_indices[i+1]))
                        if entity.is_closed and len(vertex_indices) > 1: edges_by_type[('vertex', 'segment', 'vertex')].append((vertex_indices[-1], vertex_indices[0]))

                    elif entity.dxf.dxftype == 'DIMENSION':
                        for sub in entity.virtual_entities():
                            sub_idx = register_node(sub)
                            edges_by_type[('DIMENSION', 'decomposes_to', sub.dxf.dxftype)].append((current_idx, sub_idx))

        process_entities(msp)

        final_nodes, global_to_local_idx_map = self._finalize_nodes(unique_id_to_node_info, norm_params)
        final_edges = self._finalize_edges(edges_by_type, unique_id_to_node_info, global_to_local_idx_map)

        return {"file_path": dxf_file_path, "nodes": dict(final_nodes), "edges": final_edges}

    def _finalize_nodes(self, unique_id_to_node_info, norm_params):
        final_nodes = defaultdict(lambda: {'x': [], 'discrete': []})
        global_to_local_idx_map = {}

        sorted_nodes = sorted(unique_id_to_node_info.values(), key=lambda x: x[0])
        for global_idx, node_type, entity in sorted_nodes:
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
        for (src_type, rel, dst_type), edge_list in edges_by_type.items():
            remapped = []
            for g_idx1, g_idx2 in edge_list:
                node1_info, node2_info = info_map.get(g_idx1), info_map.get(g_idx2)
                if not node1_info or not node2_info: continue

                if (src_type, dst_type) == (node1_info[1], node2_info[1]):
                    remapped.append([global_to_local_idx_map[g_idx1], global_to_local_idx_map[g_idx2]])
                elif (src_type, dst_type) == (node2_info[1], node1_info[1]):
                     remapped.append([global_to_local_idx_map[g_idx2], global_to_local_idx_map[g_idx1]])
            if remapped:
                final_edges[(src_type, rel, dst_type)] = torch.tensor(remapped, dtype=torch.long).t().contiguous()
        return final_edges

class GraphBuilder:
    def __init__(self):
        self.means: Dict[str, torch.Tensor] = {}
        self.stds: Dict[str, torch.Tensor] = {}

    def fit_transform(self, intermediate_data_list: List[Dict]) -> List[HeteroData]:
        features_by_type = defaultdict(list)
        for data in intermediate_data_list:
            if data and 'nodes' in data:
                for node_type, node_data in data['nodes'].items():
                    if 'x' in node_data and node_data['x'].numel() > 0:
                        features_by_type[node_type].append(node_data['x'])

        for node_type, features_list in features_by_type.items():
            combined = torch.cat(features_list, dim=0)
            self.means[node_type] = torch.mean(combined, dim=0)
            self.stds[node_type] = torch.std(combined, dim=0)

        return self.transform(intermediate_data_list)

    def transform(self, intermediate_data_list: List[Dict]) -> List[HeteroData]:
        hetero_data_list = []
        for data in intermediate_data_list:
            if not data or 'nodes' not in data: continue
            hetero_data = HeteroData()
            for node_type, node_data in data['nodes'].items():
                if 'x' in node_data and node_data['x'].numel() > 0:
                    features = node_data['x']
                    if node_type in self.means:
                        std = self.stds[node_type].clone()
                        std[std < 1e-8] = 1.0
                        features = (features - self.means[node_type]) / std
                    hetero_data[node_type].x = features
            hetero_data_list.append(hetero_data)
        return hetero_data_list

def _create_dxf_for_full_pipeline_test(temp_dir: Path) -> str:
    doc = ezdxf.new()
    msp = doc.modelspace()
    doc.layers.add("MY_LAYER")
    msp.add_lwpolyline([(0,0), (10,0), (10,10)], dxfattribs={"layer": "MY_LAYER"})
    dimstyle = doc.dimstyles.new("TestDim")
    msp.add_linear_dim(base=(5, 15), p1=(0,0), p2=(10,0), dimstyle="TestDim").render()
    path = temp_dir / "full_pipeline_test.dxf"
    doc.saveas(path)
    return str(path)

def _demonstrate_full_pipeline():
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp()
    try:
        logging.info("\n--- 完整处理流程演示 ---")
        dxf_path = _create_dxf_for_full_pipeline_test(Path(temp_dir))

        gp = GlobalPreprocessor()
        gp.fit([dxf_path])

        sp = SingleDrawingProcessor(gp.layer_embedding_map, gp.linetype_to_idx)
        intermediate_data = sp.process(dxf_path)

        assert "LWPOLYLINE" in intermediate_data['nodes']
        assert "DIMENSION" in intermediate_data['nodes']
        assert ('vertex', 'segment', 'vertex') in intermediate_data['edges']

        gb = GraphBuilder()
        hetero_data_list = gb.fit_transform([intermediate_data])

        assert len(hetero_data_list) == 1
        h_data = hetero_data_list[0]
        assert "LWPOLYLINE" in h_data.node_types
        logging.info("--- 完整处理流程演示成功 ---")
    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    _demonstrate_full_pipeline()
