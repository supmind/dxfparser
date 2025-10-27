# Standard library imports
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json

# Third-party imports
import torch
import ezdxf
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace
from ezdxf.entities import DXFEntity, Dimension
from ezdxf.math import BoundingBox
from torch_geometric.data import HeteroData
from shapely.geometry import LineString, Polygon, Point
from shapely.errors import GEOSException

# Set up a logger for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DxfParser:
    """
    Parses a DXF file to extract geometric and annotation entities,
    and prepares the data for a Heterogeneous Graph Neural Network (HGT).
    """
    SUPPORTED_GEOMETRIES: List[str] = [
        'LINE', 'LWPOLYLINE', 'CIRCLE', 'ARC', 'HATCH', 'INSERT'
    ]
    SUPPORTED_ANNOTATIONS: List[str] = [
        'TEXT', 'MTEXT', 'DIMENSION', 'LEADER'
    ]

    def __init__(self, dxf_path: str):
        self.dxf_path = Path(dxf_path)
        self.doc, self.modelspace = self._load_dxf()

    def _load_dxf(self) -> Tuple[Drawing, Modelspace]:
        if not self.dxf_path.is_file():
            raise FileNotFoundError(f"DXF file not found at: {self.dxf_path}")
        try:
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            logger.info(f"Successfully loaded DXF file: {self.dxf_path}")
            return doc, msp
        except UnicodeDecodeError:
            logger.warning(f"Decoding failed, trying 'gbk'.")
            return ezdxf.readfile(self.dxf_path, encoding='gbk'), doc.modelspace()
        except ezdxf.DXFStructureError as e:
            logger.error(f"Corrupt DXF file: {self.dxf_path}. Error: {e}")
            raise

    def process(self, stage: int) -> Tuple[Optional[HeteroData], Dict[str, Any]]:
        entities = self._extract_and_explode_entities(stage)
        if not entities:
            logger.warning("No entities extracted. Aborting.")
            return None, {}

        scale_info = self._analyze_scale_and_unit(entities)
        entity_to_scale = self._partition_regions(entities, scale_info)
        transform_params = self._calculate_transform_params(entities, stage, entity_to_scale)
        transformed_entities = self._apply_transformations(entities, transform_params)

        graph_data = self._build_hetero_graph(transformed_entities)

        meta_data = {
            'dxf_path': str(self.dxf_path),
            'stage': stage,
            'entity_count': len(transformed_entities),
            'scale_info': scale_info,
            'transform_params': transform_params
        }
        logger.info(f"Processing for stage {stage} complete.")
        return graph_data, meta_data

    def _extract_and_explode_entities(self, stage: int) -> List[DXFEntity]:
        allowed_types = self.SUPPORTED_GEOMETRIES
        if stage == 2:
            allowed_types.extend(self.SUPPORTED_ANNOTATIONS)

        initial_entities = [e for e in self.modelspace if e.dxf.dxftype in allowed_types]
        exploded_entities = []
        for entity in initial_entities:
            if entity.dxf.dxftype == 'INSERT':
                exploded_entities.extend(self._handle_insert_entity(entity))
            else:
                exploded_entities.append(entity)

        if stage == 1:
            return [e for e in exploded_entities if e.dxf.dxftype in self.SUPPORTED_GEOMETRIES and e.dxf.dxftype != 'INSERT']
        return exploded_entities

    def _handle_insert_entity(self, insert_entity: DXFEntity) -> List[DXFEntity]:
        final_entities = []
        if insert_entity.block() is None:
            return []

        try:
            virtuals = list(insert_entity.virtual_entities())
            for entity in virtuals:
                if entity.dxf.dxftype == 'INSERT':
                    final_entities.extend(self._handle_insert_entity(entity))
                elif entity.dxf.dxftype != 'ATTDEF':
                    final_entities.append(entity)
        except Exception as e:
            logger.error(f"Error processing block '{insert_entity.dxf.name}': {e}")

        final_entities.extend(insert_entity.attribs)
        return final_entities

    def _analyze_scale_and_unit(self, entities: List[DXFEntity]) -> Dict[str, Any]:
        dims = [e for e in entities if e.dxf.dxftype == 'DIMENSION']
        if not dims:
            return {"unit": "mm", "scales": {1.0: len(entities)}, "reliable_dims": []}

        reliable_dims, scale_factors = [], []
        for dim in dims:
            if dim.dxf.text in {"", "<>"}:
                measurement = dim.get_measurement()
                geom_length = self._get_dimension_geom_length(dim)
                if geom_length and measurement > 1e-6 and geom_length > 1e-6:
                    scale_factors.append(measurement / geom_length)
                    reliable_dims.append(dim)

        if not scale_factors:
            return {"unit": "mm", "scales": {1.0: len(entities)}, "reliable_dims": []}

        db = DBSCAN(eps=0.5, min_samples=2).fit(np.array(scale_factors).reshape(-1, 1))
        labels = db.labels_
        scales = defaultdict(int)
        unique_labels = set(labels) - {-1}

        if not unique_labels:
            scales[np.median(scale_factors)] = len(scale_factors)
        else:
            for label in unique_labels:
                class_members = np.array(scale_factors)[labels == label]
                scales[np.median(class_members)] = len(class_members)

        main_scale_value = max(scales, key=scales.get)
        main_scale_dims = [
            d for d in reliable_dims
            if abs((d.get_measurement() / self._get_dimension_geom_length(d)) - main_scale_value) < 0.5
        ]
        avg_measurement = np.mean([d.get_measurement() for d in main_scale_dims])
        unit = "mm" if avg_measurement > 1000 else "m"

        return {"unit": unit, "scales": dict(scales), "reliable_dims": reliable_dims}

    def _get_dimension_geom_length(self, dim: Dimension) -> Optional[float]:
        try:
            p1, p2 = dim.dxf.defpoint, dim.dxf.defpoint2
            return p1.distance(p2)
        except:
            return None

    def _partition_regions(self, entities: List[DXFEntity], scale_info: Dict[str, Any]) -> Dict[str, float]:
        entity_to_scale = {}
        reliable_dims = scale_info.get("reliable_dims", [])

        if not reliable_dims:
            main_scale = max(scale_info.get("scales", {}), key=scale_info["scales"].get) if scale_info.get("scales") else 1.0
            for entity in entities:
                entity_to_scale[entity.dxf.handle] = main_scale
            return entity_to_scale

        dim_scales = {
            dim.dxf.handle: dim.get_measurement() / self._get_dimension_geom_length(dim)
            for dim in reliable_dims if self._get_dimension_geom_length(dim)
        }

        seed_points = np.array([self._get_entity_bbox(dim).center for dim in reliable_dims if self._get_entity_bbox(dim)])
        seed_scales = np.array(list(dim_scales.values()))

        for entity in entities:
            handle = entity.dxf.handle
            if handle in dim_scales:
                entity_to_scale[handle] = dim_scales[handle]
                continue

            bbox = self._get_entity_bbox(entity)
            if bbox:
                distances = np.linalg.norm(seed_points - np.array(bbox.center), axis=1)
                if distances.size > 0:
                    entity_to_scale[handle] = seed_scales[np.argmin(distances)]
        return entity_to_scale

    def _calculate_transform_params(self, entities: List[DXFEntity], stage: int, entity_to_scale: Dict[str, float]) -> Dict[str, Any]:
        if not entities: return {'translate': (0, 0), 'scale': 1.0}

        entities_to_measure = entities
        if stage == 2:
            temp_entities = []
            for entity in entities:
                scale = entity_to_scale.get(entity.dxf.handle, 1.0)
                if abs(scale - 1.0) > 1e-6:
                    e_copy = entity.copy()
                    e_copy.transform(ezdxf.math.Matrix44.scale(scale))
                    temp_entities.append(e_copy)
                else:
                    temp_entities.append(entity)
            entities_to_measure = temp_entities

        global_bbox = BoundingBox()
        for entity in entities_to_measure:
            bbox = self._get_entity_bbox(entity)
            if bbox and bbox.has_data:
                global_bbox.extend(bbox)

        if not global_bbox.has_data: return {'translate': (0, 0), 'scale': 1.0}

        if stage == 1:
            size = global_bbox.size
            scale = 1.0 / max(size) if max(size) > 1e-6 else 1.0
            return {'translate': -global_bbox.extmin, 'scale': scale}
        else: # Stage 2
            return {'translate': -global_bbox.center, 'scale': 1.0}

    def _apply_transformations(self, entities: List[DXFEntity], transform_params: Dict[str, Any]) -> List[DXFEntity]:
        translate = transform_params.get('translate', (0, 0, 0))
        scale = transform_params.get('scale', 1.0)
        transform_matrix = ezdxf.math.Matrix44.chain(
            ezdxf.math.Matrix44.scale(scale),
            ezdxf.math.Matrix44.translate(*translate)
        )
        transformed_entities = []
        for entity in entities:
            try:
                transformed_entity = entity.copy()
                transformed_entity.transform(transform_matrix)
                bbox = self._get_entity_bbox(transformed_entity)
                if bbox:
                    transformed_entity.transformed_bounds = bbox
                transformed_entities.append(transformed_entity)
            except Exception as e:
                logger.warning(f"Failed to transform {entity.dxf.dxftype}: {e}")
        return transformed_entities

    def _get_entity_bbox(self, entity: DXFEntity) -> Optional[BoundingBox]:
        try:
            # This is a simplified approach. ezdxf.render.Extents is more robust.
            if hasattr(entity, 'transformed_bounds'):
                return entity.transformed_bounds
            if entity.dxf.dxftype in {'LINE', 'LWPOLYLINE', 'POLYLINE'}:
                return BoundingBox(entity.vertices_in_wcs())
            elif entity.dxf.dxftype in {'CIRCLE', 'ARC'}:
                center = entity.dxf.center
                radius = entity.dxf.radius
                return BoundingBox((center.x - radius, center.y - radius), (center.x + radius, center.y + radius))
            elif hasattr(entity.dxf, 'insert'):
                return BoundingBox([entity.dxf.insert, entity.dxf.insert])
        except (AttributeError, ValueError, GEOSException) as e:
            logger.debug(f"Could not get bounding box for {entity.dxf.dxftype}: {e}")
        return None

    def _get_sampled_points(self, entity: DXFEntity, num_samples: int = 10) -> np.ndarray:
        """为线性实体生成采样点。"""
        points = []
        try:
            if entity.dxf.dxftype in {'LINE', 'LWPOLYLINE'}:
                # 对于直线和多段线，我们可以沿着线段采样
                length = entity.length
                if length > 1e-6:
                    for i in range(num_samples):
                        points.append(entity.point(i / (num_samples - 1)))
                else: # 如果线段长度为0，则只使用其起点
                    points.append(entity.start_point)

            elif entity.dxf.dxftype == 'ARC':
                # 对于圆弧，我们可以在角度范围内采样
                for i in range(num_samples):
                    angle = entity.dxf.start_angle + (entity.dxf.end_angle - entity.dxf.start_angle) * (i / (num_samples - 1))
                    points.append(entity.point_at_angle(angle))

            elif entity.dxf.dxftype == 'CIRCLE':
                # 对于圆，我们在圆周上采样
                for i in range(num_samples):
                    angle = 2 * np.pi * (i / num_samples)
                    points.append(entity.point_at_angle(np.degrees(angle)))
            else:
                # 对于其他实体，使用其边界框中心作为代表点
                bbox = self._get_entity_bbox(entity)
                if bbox and bbox.has_data:
                    points.append(bbox.center)

        except (AttributeError, ValueError):
            bbox = self._get_entity_bbox(entity)
            if bbox and bbox.has_data:
                points.append(bbox.center)

        return np.array(points) if points else np.array([])


    def _entity_to_shapely(self, entity: DXFEntity):
        """将ezdxf实体转换为Shapely几何对象。"""
        try:
            if entity.dxf.dxftype in {'LINE', 'LWPOLYLINE'}:
                return LineString([p for p in entity.vertices_in_wcs()])
            elif entity.dxf.dxftype == 'CIRCLE':
                return Point(entity.dxf.center).buffer(entity.dxf.radius)
            # 可以根据需要添加对ARC, HATCH等的支持
        except (AttributeError, GEOSException):
            return None
        return None

    def _build_hetero_graph(self, entities: List[DXFEntity], k_neighbors: int = 15) -> HeteroData:
        data = HeteroData()
        node_features, node_positions, entity_map = {}, {}, {}

        # 1. 创建节点和特征
        for i, entity in enumerate(entities):
            etype = entity.dxf.dxftype
            if etype not in node_features:
                node_features[etype], node_positions[etype] = [], []

            bbox = self._get_entity_bbox(entity)
            if not (bbox and bbox.has_data): continue

            # 特征编码 (简化版)
            pos = torch.tensor(bbox.center, dtype=torch.float32)
            features = pos # 简化特征

            node_features[etype].append(features)
            node_positions[etype].append(pos)
            entity_map[entity.dxf.handle] = {'type': etype, 'idx': len(node_positions[etype]) - 1}

        for etype, features_list in node_features.items():
            max_len = max(f.shape[0] for f in features_list) if features_list else 0
            padded_features = [torch.nn.functional.pad(f, (0, max_len - f.shape[0])) for f in features_list]
            if padded_features:
                data[etype].x = torch.stack(padded_features)
                data[etype].pos = torch.stack(node_positions[etype])

        # 2. 构建边
        edge_indices = defaultdict(list)

        # 2.1 基于采样点的K-NN边
        from scipy.spatial import KDTree
        all_sampled_points, point_to_entity_idx = [], []
        for i, entity in enumerate(entities):
            sampled = self._get_sampled_points(entity)
            if sampled.size > 0:
                all_sampled_points.append(sampled)
                point_to_entity_idx.extend([i] * len(sampled))

        if all_sampled_points:
            kdtree = KDTree(np.vstack(all_sampled_points))

            for i, entity in enumerate(entities):
                sampled = self._get_sampled_points(entity)
                if sampled.size == 0: continue

                dist, idx = kdtree.query(sampled, k=k_neighbors + 1)
                neighbor_entity_indices = set()
                for p_idx in idx.flatten():
                    neighbor_entity_indices.add(point_to_entity_idx[p_idx])

                src_info = entity_map.get(entity.dxf.handle)
                if not src_info: continue

                for neighbor_idx in neighbor_entity_indices:
                    if neighbor_idx == i: continue
                    neighbor_entity = entities[neighbor_idx]
                    dst_info = entity_map.get(neighbor_entity.dxf.handle)
                    if not dst_info: continue

                    src_type, dst_type = src_info['type'], dst_info['type']
                    # 保持边类型的方向一致性
                    if src_type <= dst_type:
                        edge_indices[(src_type, f'nearby_{k_neighbors}', dst_type)].append([src_info['idx'], dst_info['idx']])
                    else:
                        edge_indices[(dst_type, f'nearby_{k_neighbors}', src_type)].append([dst_info['idx'], src_info['idx']])

        # 2.2 拓扑边
        shapely_geoms = {e.dxf.handle: self._entity_to_shapely(e) for e in entities}

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                h1, h2 = e1.dxf.handle, e2.dxf.handle
                s1, s2 = shapely_geoms.get(h1), shapely_geoms.get(h2)

                if not (s1 and s2): continue

                src_info, dst_info = entity_map.get(h1), entity_map.get(h2)
                if not (src_info and dst_info): continue
                src_type, dst_type = src_info['type'], dst_info['type']

                # 定义一个辅助函数来保持边的一致性
                def add_edge(relation: str):
                    if src_type <= dst_type:
                        edge_indices[(src_type, relation, dst_type)].append([src_info['idx'], dst_info['idx']])
                    else:
                        edge_indices[(dst_type, relation, src_type)].append([dst_info['idx'], src_info['idx']])

                if s1.touches(s2):
                    add_edge('connected_to')

                if s1.intersects(s2) and not s1.touches(s2):
                    add_edge('intersects')

                # intelligent contains (确保方向)
                area_ratio_s1_in_s2 = s1.area / s2.area if s2.area > 1e-6 else float('inf')
                area_ratio_s2_in_s1 = s2.area / s1.area if s1.area > 1e-6 else float('inf')

                if s2.contains(s1) and area_ratio_s1_in_s2 < 0.5:
                    edge_indices[(dst_type, 'contains', src_type)].append([dst_info['idx'], src_info['idx']])
                elif s1.contains(s2) and area_ratio_s2_in_s1 < 0.5:
                    edge_indices[(src_type, 'contains', dst_type)].append([src_info['idx'], dst_info['idx']])

        for edge_type, indices in edge_indices.items():
            if indices:
                unique_indices = torch.tensor(indices, dtype=torch.long).unique(dim=0)
                data[edge_type].edge_index = unique_indices.t().contiguous()

        logger.info(f"Successfully built advanced hetero graph with {len(data.node_types)} node types and {len(data.edge_types)} edge types.")
        return data

    def save_verification_files(self, output_dir: str) -> None:
        """生成并保存一系列用于可视化验证的DXF文件。"""
        logger.info("Generating verification files...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        entities = self._extract_and_explode_entities(stage=2)
        scale_info = self._analyze_scale_and_unit(entities)
        entity_to_scale = self._partition_regions(entities, scale_info)

        # 1. 区域着色验证
        doc_regions = ezdxf.new()
        msp_regions = doc_regions.modelspace()
        unique_scales = sorted(list(set(entity_to_scale.values())))
        colors = [i + 1 for i in range(len(unique_scales))]
        scale_to_color = {scale: color for scale, color in zip(unique_scales, colors)}

        for entity in entities:
            scale = entity_to_scale.get(entity.dxf.handle)
            if scale:
                e_copy = entity.copy()
                e_copy.dxf.color = scale_to_color.get(scale, 0)
                msp_regions.add_entity(e_copy)
        doc_regions.saveas(output_dir / "verification_regions.dxf")

        # 2. 变换验证
        params_s1 = self._calculate_transform_params(entities, 1, entity_to_scale)
        params_s1['scale'] *= 0.1 # 缩小10倍以便查看
        transformed_s1 = self._apply_transformations(entities, params_s1)
        doc_s1 = ezdxf.new(); msp_s1 = doc_s1.modelspace()
        for entity in transformed_s1: msp_s1.add_entity(entity)
        doc_s1.saveas(output_dir / "verification_transformed_stage1.dxf")

        params_s2 = self._calculate_transform_params(entities, 2, entity_to_scale)
        transformed_s2 = self._apply_transformations(entities, params_s2)
        doc_s2 = ezdxf.new(); msp_s2 = doc_s2.modelspace()
        for entity in transformed_s2: msp_s2.add_entity(entity)
        doc_s2.saveas(output_dir / "verification_transformed_stage2.dxf")

        # 3. 拓扑关系验证
        doc_topo = ezdxf.new(); msp_topo = doc_topo.modelspace()
        for e in entities: msp_topo.add_entity(e.copy()) # 先添加所有实体

        shapely_geoms = {e.dxf.handle: self._entity_to_shapely(e) for e in entities}

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                s1, s2 = shapely_geoms.get(e1.dxf.handle), shapely_geoms.get(e2.dxf.handle)
                if not (s1 and s2): continue

                p1, p2 = self._get_entity_bbox(e1).center, self._get_entity_bbox(e2).center

                if s1.touches(s2):
                    msp_topo.add_line(p1, p2, dxfattribs={'color': 1}) # Red
                if s1.intersects(s2) and not s1.touches(s2):
                    msp_topo.add_line(p1, p2, dxfattribs={'color': 5}) # Blue
                if s1.area > 1e-6 and s1.contains(s2) and (s2.area / s1.area) < 0.5:
                    msp_topo.add_line(p1, p2, dxfattribs={'color': 3}) # Green

        doc_topo.saveas(output_dir / "verification_topology.dxf")
        logger.info(f"Verification files saved to '{output_dir}'.")

if __name__ == '__main__':
    INPUT_DXF_PATH = 'path/to/your/input.dxf'
    OUTPUT_DIR = 'output'

    input_path = Path(INPUT_DXF_PATH)
    if not input_path.is_file():
        logger.error(f"Test failed: Input file not found at '{INPUT_DXF_PATH}'")
    else:
        logger.info(f"Starting processing for DXF file: {INPUT_DXF_PATH}")
        parser = DxfParser(dxf_path=INPUT_DXF_PATH)

        # 1. 生成并保存验证文件
        parser.save_verification_files(output_dir=OUTPUT_DIR)

        # 2. 运行完整的Stage 2处理流程并保存最终的图数据
        graph, metadata = parser.process(stage=2)

        # Save outputs
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        if graph:
            output_graph_path = output_dir / f"{input_path.stem}_graph.pt"
            torch.save(graph, output_graph_path)
            logger.info(f"Graph data saved to: {output_graph_path}")

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.generic)): return obj.tolist()
                if isinstance(obj, BoundingBox): return str(obj)
                if hasattr(obj, 'dxf'): return f"DXFEntity({obj.dxf.dxftype})"
                return str(obj)

        output_meta_path = output_dir / f"{input_path.stem}_metadata.json"
        with open(output_meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4, cls=CustomEncoder)
        logger.info(f"Metadata saved to: {output_meta_path}")

        logger.info("All processing and validation flows completed.")
