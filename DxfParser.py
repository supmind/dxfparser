# Standard library imports
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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
        """
        Initializes the DxfParser with the path to a DXF file.

        Args:
            dxf_path: The file path to the DXF file.
        """
        self.dxf_path = Path(dxf_path)
        self.doc: Drawing
        self.modelspace: Modelspace
        self.doc, self.modelspace = self._load_dxf()

    def _load_dxf(self) -> Tuple[Drawing, Modelspace]:
        """
        Safely loads a DXF file, handling potential encoding and structure errors.

        Returns:
            A tuple containing the loaded DXF document and its modelspace.

        Raises:
            FileNotFoundError: If the DXF file is not found.
            ezdxf.DXFStructureError: If the DXF file is corrupt.
        """
        if not self.dxf_path.is_file():
            logger.error(f"DXF file not found at: {self.dxf_path}")
            raise FileNotFoundError(f"DXF file not found at: {self.dxf_path}")

        try:
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()
            logger.info(f"Successfully loaded DXF file: {self.dxf_path}")
            return doc, msp
        except UnicodeDecodeError:
            logger.warning(f"Could not decode DXF file {self.dxf_path} with default encoding. Trying 'gbk'.")
            try:
                doc = ezdxf.readfile(self.dxf_path, encoding='gbk')
                msp = doc.modelspace()
                logger.info(f"Successfully loaded DXF file with 'gbk' encoding: {self.dxf_path}")
                return doc, msp
            except Exception as e:
                logger.error(f"Failed to load DXF file {self.dxf_path} with fallback encoding 'gbk'. Error: {e}")
                raise e
        except ezdxf.DXFStructureError as e:
            logger.error(f"Corrupt DXF file: {self.dxf_path}. Error: {e}")
            raise e

    def process(self, stage: int) -> Tuple[List[DXFEntity], Dict[str, Any]]:
        """
        运行完整处理流水线的主公共方法。

        Args:
            stage: 一个整数（1或2），用于确定处理阶段。
                   阶段1专注于全局归一化的几何形状。
                   阶段2处理所有实体以进行最终的图构建。

        Returns:
            一个包含实体列表和元数据的元组。
        """
        # 1. 根据阶段提取和过滤实体
        entities = self._extract_and_explode_entities(stage)

        # 2. 分析尺度和单位
        scale_info = self._analyze_scale_and_unit(entities)

        # 3. 划分区域
        entity_to_scale = self._partition_regions(entities, scale_info)

        # 4. 根据阶段计算变换参数
        transform_params = self._calculate_transform_params(entities, stage, entity_to_scale)

        # 5. 应用变换
        transformed_entities = self._apply_transformations(entities, transform_params)

        # 6. 准备元数据
        meta_data = {
            'dxf_path': str(self.dxf_path),
            'stage': stage,
            'entity_count': len(transformed_entities),
            'scale_info': scale_info,
            'entity_to_scale': entity_to_scale,
            'transform_params': transform_params
        }

        # 7. 构建异构图
        graph_data = self._build_hetero_graph(transformed_entities)

        # 8. 返回图数据和元数据
        logger.info(f"阶段 {stage} 的处理完成。")
        return graph_data, meta_data

    def _extract_and_explode_entities(self, stage: int) -> List[DXFEntity]:
        """
        此方法根据`stage`筛选实体类型，并调用`_handle_insert_entity`处理块。
        在stage 1模式下，会进行二次过滤以确保只返回纯几何实体。
        """
        if stage == 1:
            allowed_types = self.SUPPORTED_GEOMETRIES
        elif stage == 2:
            allowed_types = self.SUPPORTED_GEOMETRIES + self.SUPPORTED_ANNOTATIONS
        else:
            logger.warning(f"提供了无效的阶段 '{stage}'。将不会提取任何实体。")
            return []

        logger.info(f"阶段 {stage}: 正在提取以下类型的实体: {allowed_types}")
        # Note: 'INSERT' is included in SUPPORTED_GEOMETRIES, so it's always considered.
        initial_entities = [e for e in self.modelspace if e.dxf.dxftype in allowed_types]
        logger.info(f"在模型空间中找到 {len(initial_entities)} 个初始实体。")

        exploded_entities: List[DXFEntity] = []
        for entity in initial_entities:
            if entity.dxf.dxftype == 'INSERT':
                exploded_entities.extend(self._handle_insert_entity(entity))
            else:
                exploded_entities.append(entity)

        # Secondary filtering for Stage 1 to ensure pure geometry.
        # This removes entities like ATTRIB that may have been exploded from blocks.
        if stage == 1:
            # We must also filter out 'INSERT' itself, as it's a container, not a primitive geometry.
            final_geometries = [
                e for e in exploded_entities
                if e.dxf.dxftype in self.SUPPORTED_GEOMETRIES and e.dxf.dxftype != 'INSERT'
            ]
            logger.info(f"阶段 1 二次过滤后，剩余纯几何实体: {len(final_geometries)}")
            return final_geometries

        logger.info(f"块分解后的实体总数: {len(exploded_entities)}")
        return exploded_entities

    def _handle_insert_entity(self, insert_entity: DXFEntity) -> List[DXFEntity]:
        """
        使用健壮的只读方法递归处理INSERT实体（块）。

        此方法使用`virtual_entities()`，它比`explode()`更安全，因为它可以在不修改文档实体数据库的情况下生成变换后的虚拟实体。
        它能优雅地处理匿名块和其他边缘情况。

        附着的ATTRIB实体被单独处理，因为它们已经处于全局坐标中，不应被变换。

        Args:
            insert_entity: 要处理的INSERT实体。

        Returns:
            一个由块处理产生的扁平化基础实体列表。
        """
        final_entities: List[DXFEntity] = []
        block_def = insert_entity.block()

        # 1. 对有效的块定义进行防御性检查
        if block_def is None:
            logger.warning(
                f"跳过无效的INSERT实体 '{insert_entity.dxf.name}' "
                f"（句柄 {insert_entity.dxf.handle}），因为其块定义不存在。"
            )
            return final_entities

        # 2. 处理来自块定义的几何实体
        # virtual_entities()是一种健壮的方式，可以像分解实体一样获取实体，
        # 正确处理变换和匿名块。
        try:
            # 使用list()来立即计算生成器，以便进行递归处理
            virtuals = list(insert_entity.virtual_entities())
            for entity in virtuals:
                if entity.dxf.dxftype == 'INSERT':
                    # 递归调用以处理嵌套块
                    final_entities.extend(self._handle_insert_entity(entity))
                else:
                    # ATTDEF实体是块定义的一部分，不应包含在分解结果中
                    if entity.dxf.dxftype != 'ATTDEF':
                        final_entities.append(entity)
        except Exception as e:
            logger.error(
                f"处理块 '{insert_entity.dxf.name}' （句柄 {insert_entity.dxf.handle}）的虚拟实体时发生意外错误: {e}"
            )

        # 3. 处理附着的ATTRIB实体
        # 它们与块的几何形状是分开的，并且已经处于正确的全局坐标中。
        # 应直接收集它们。
        if insert_entity.attribs:
            final_entities.extend(insert_entity.attribs)

        return final_entities

    def _analyze_scale_and_unit(self, entities: List[DXFEntity]) -> Dict[str, Any]:
        """
        分析实体以确定绘图比例和单位。

        该方法执行以下操作：
        1. 过滤可靠的DIMENSION实体（未被手动覆盖）。
        2. 计算每个可靠标注的比例因子（物理长度/几何长度）。
        3. 使用DBSCAN对比例因子进行聚类，以找到主导比例。
        4. 根据主要比例的物理值，启发式地推断单位（mm或m）。

        Args:
            entities: 从DXF文件中提取的实体列表。

        Returns:
            一个包含比例信息的字典，包括单位、比例映射和可靠的标注。
        """
        dims = [e for e in entities if e.dxf.dxftype == 'DIMENSION']
        if not dims:
            logger.warning("在图纸中未找到DIMENSION实体。无法确定比例。假定比例为1.0，单位为'mm'。")
            return {"unit": "mm", "scales": {1.0: len(entities)}, "reliable_dims": []}

        reliable_dims = []
        scale_factors = []

        for dim in dims:
            # 过滤掉“非真实”的标注（用户手动覆盖了数值）
            # `dim.dxf.text`为空或"<>"表示该标注跟随几何图形
            if dim.dxf.text == "" or dim.dxf.text == "<>":
                measurement = dim.get_measurement()
                geom_length = self._get_dimension_geom_length(dim)

                if geom_length is not None and measurement > 1e-6 and geom_length > 1e-6:
                    scale_factor = measurement / geom_length
                    scale_factors.append(scale_factor)
                    reliable_dims.append(dim)

        if not scale_factors:
            logger.warning("未找到可靠的DIMENSION实体。无法确定比例。假定比例为1.0，单位为'mm'。")
            return {"unit": "mm", "scales": {1.0: len(entities)}, "reliable_dims": []}

        # 使用DBSCAN寻找主导比例
        X = np.array(scale_factors).reshape(-1, 1)
        # eps: 两个样本被视为邻居的最大距离。
        # 对于比例因子，一个小的绝对公差（如0.5）是合适的。
        # min_samples: 一个点被视为核心点所需的邻居数。
        # 2对于寻找成对的相似比例是有效的。
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        labels = db.labels_

        scales = defaultdict(int)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1) # 移除噪声点

        if not unique_labels:
             # 如果所有点都是噪声，则将它们视为一个单独的簇
            main_scale = np.median(X)
            scales[main_scale] = len(X)
        else:
            for label in unique_labels:
                class_members = X[labels == label]
                # 使用中位数作为簇的代表性比例，以抵抗异常值
                median_scale = np.median(class_members)
                scales[median_scale] = len(class_members)

        # 单位推断逻辑
        # 假设主导比例（最常见的比例）下的物理尺寸能揭示单位。
        # 如果尺寸通常大于1000，则单位可能是mm。否则，可能是m，我们在内部统一为mm。
        main_scale_value = max(scales, key=scales.get)
        main_scale_dims = [
            d for d in reliable_dims
            if main_scale_value - 0.5 <= (d.get_measurement() / self._get_dimension_geom_length(d)) <= main_scale_value + 0.5
        ]

        avg_measurement = np.mean([d.get_measurement() for d in main_scale_dims])
        unit = "mm" if avg_measurement > 1000 else "m"
        logger.info(f"推断单位为: {unit}. 主要比例: {dict(scales)}")

        return {"unit": unit, "scales": dict(scales), "reliable_dims": reliable_dims}

    def _get_dimension_geom_length(self, dim: Dimension) -> Optional[float]:
        """计算DIMENSION实体的几何长度。"""
        try:
            # `get_measurement`返回物理长度，我们需要的是图纸上的几何长度
            # 我们通过使用其定义点来手动计算它
            p1 = dim.dxf.defpoint # 通常是尺寸界线的起点
            p2 = dim.dxf.defpoint2
            return p1.distance(p2)
        except Exception:
            # 某些DIMENSION类型可能没有defpoint/defpoint2
            return None

    def _get_entity_bbox(self, entity: DXFEntity, transformed: bool = False) -> Optional[BoundingBox]:
        """安全地获取实体的边界框。"""
        # Note: This is a simplified bbox implementation.
        # For full accuracy, ezdxf.render.Extents or a more detailed geometric analysis is needed.
        if hasattr(entity, 'transformed_bounds') and transformed:
            return entity.transformed_bounds

        if entity.dxf.dxftype in {'LINE', 'LWPOLYLINE', 'POLYLINE'}:
            try:
                return BoundingBox(entity.points())
            except Exception:
                return None
        elif entity.dxf.dxftype in {'CIRCLE', 'ARC'}:
            try:
                center = entity.dxf.center
                radius = entity.dxf.radius
                return BoundingBox([center - radius, center + radius])
            except AttributeError:
                return None
        elif hasattr(entity, 'dxf.insert'):
            return BoundingBox([entity.dxf.insert, entity.dxf.insert])
        return None

    def _partition_regions(self, entities: List[DXFEntity], scale_info: Dict[str, Any]) -> Dict[str, float]:
        """
        根据空间邻近度，将实体划分到不同的比例区域。

        此方法使用可靠的DIMENSION实体作为“种子”，并根据“最近邻”原则
        将每个几何实体分配到一个比例区域。

        Args:
            entities: 所有已提取的实体。
            scale_info: 来自_analyze_scale_and_unit方法的结果。

        Returns:
            一个将实体句柄映射到其分配的比例因子的字典。
        """
        entity_to_scale: Dict[str, float] = {}
        reliable_dims = scale_info.get("reliable_dims", [])

        if not reliable_dims:
            # 如果没有可靠的标注，则将所有实体分配给最常见的比例，或默认为1.0
            scales = scale_info.get("scales", {})
            main_scale = max(scales, key=scales.get) if scales else 1.0
            logger.info(f"没有可靠的标注作为种子。将所有实体分配给主比例: {main_scale}")
            for entity in entities:
                entity_to_scale[entity.dxf.handle] = main_scale
            return entity_to_scale

        # 1. 创建种子点（可靠标注的中心）及其比例
        dim_scales = {}
        for dim in reliable_dims:
            measurement = dim.get_measurement()
            geom_length = self._get_dimension_geom_length(dim)
            if geom_length is not None and geom_length > 1e-6:
                dim_scales[dim.dxf.handle] = measurement / geom_length

        seed_points = np.array([self._get_entity_bbox(dim).center for dim in reliable_dims if self._get_entity_bbox(dim)])
        seed_scales = np.array([dim_scales[dim.dxf.handle] for dim in reliable_dims if self._get_entity_bbox(dim)])

        # 2. 为每个其他实体找到最近的种子并分配其比例
        for entity in entities:
            if entity.dxf.handle in dim_scales:
                entity_to_scale[entity.dxf.handle] = dim_scales[entity.dxf.handle]
                continue

            bbox = self._get_entity_bbox(entity)
            if bbox is None:
                continue

            entity_point = np.array(bbox.center)
            distances = np.linalg.norm(seed_points - entity_point, axis=1)

            if distances.size > 0:
                closest_seed_index = np.argmin(distances)
                assigned_scale = seed_scales[closest_seed_index]
                entity_to_scale[entity.dxf.handle] = assigned_scale

        logger.info(f"成功将 {len(entity_to_scale)} 个实体划分到比例区域。")
        return entity_to_scale

    def _calculate_transform_params(
        self, entities: List[DXFEntity], stage: int, entity_to_scale: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        根据阶段和比例信息计算变换参数。

        - Stage 1: 计算将所有实体归一化到[0,1]^2空间所需的平移和缩放。
        - Stage 2: 首先应用局部逆向缩放（“野路子”情况），然后计算将所有实体中心化到原点所需的平移。

        Args:
            entities: 实体列表。
            stage: 处理阶段 (1 or 2)。
            entity_to_scale: 将实体句柄映射到其比例因子的字典。

        Returns:
            包含变换参数（平移、缩放）的字典。
        """
        if not entities:
            return {'translate': (0, 0), 'scale': 1.0}

        # 对于Stage 2，首先应用逆向缩放以获得“真实世界”坐标
        if stage == 2:
            temp_entities = []
            for entity in entities:
                scale = entity_to_scale.get(entity.dxf.handle, 1.0)
                if scale != 1.0:
                    # 创建一个副本进行变换，以免修改原始实体
                    e_copy = entity.copy()
                    # 修正：应该乘以比例因子，而不是除以它
                    e_copy.transform(ezdxf.math.Matrix44.scale(scale))
                    temp_entities.append(e_copy)
                else:
                    temp_entities.append(entity)
            entities_to_measure = temp_entities
        else:
            entities_to_measure = entities

        # 计算所有（可能已缩放的）实体的全局边界框
        global_bbox = BoundingBox()
        for entity in entities_to_measure:
            bbox = self._get_entity_bbox(entity)
            if bbox:
                global_bbox.extend(bbox)

        if not global_bbox.has_data:
            return {'translate': (0, 0), 'scale': 1.0}

        if stage == 1:
            # Stage 1: 归一化到 [0,1]^2
            size = global_bbox.size
            max_size = max(size.x, size.y)
            scale = 1.0 / max_size if max_size > 1e-6 else 1.0
            translate = -global_bbox.extmin
            return {'translate': translate, 'scale': scale}
        else: # Stage 2
            # Stage 2: 仅中心化
            center = global_bbox.center
            translate = -center
            return {'translate': translate, 'scale': 1.0}

    def _apply_transformations(
        self, entities: List[DXFEntity], transform_params: Dict[str, Any]
    ) -> List[DXFEntity]:
        """
        将计算出的变换应用到每个实体上。

        Args:
            entities: 原始实体列表。
            transform_params: 包含平移和缩放信息的字典。

        Returns:
            一个包含已变换实体的新列表。
        """
        translate = transform_params.get('translate', (0, 0, 0))
        scale = transform_params.get('scale', 1.0)

        # ezdxf的Matrix44在这里非常有用
        transform_matrix = ezdxf.math.Matrix44.chain(
            ezdxf.math.Matrix44.scale(scale, scale, scale),
            ezdxf.math.Matrix44.translate(translate.x, translate.y, translate.z),
        )

        transformed_entities = []
        for entity in entities:
            try:
                # 复制实体以避免修改原始实体列表
                transformed_entity = entity.copy()
                transformed_entity.transform(transform_matrix)

                # 存储变换后的边界框以供将来使用
                bbox = self._get_entity_bbox(transformed_entity)
                if bbox:
                    transformed_entity.transformed_bounds = bbox
                transformed_entities.append(transformed_entity)

            except Exception as e:
                logger.warning(f"无法变换实体 {entity.dxf.dxftype} (句柄: {entity.dxf.handle}): {e}")

        return transformed_entities

    def _build_hetero_graph(self, entities: List[DXFEntity]) -> HeteroData:
        """
        将实体列表构建成一个PyTorch Geometric的HeteroData对象。

        - 节点: 每种实体类型对应一种节点类型。
        - 节点特征: 编码几何属性（坐标、半径等）。
        - 边: 基于空间邻近度（K-最近邻）构建。

        Args:
            entities: 经过变换的实体列表。

        Returns:
            一个代表DXF图纸的HeteroData对象。
        """
        data = HeteroData()
        node_features = defaultdict(list)
        node_positions = defaultdict(list)
        node_indices = {} # 映射实体句柄到其在类型列表中的索引

        # 1. 节点创建和特征编码
        for i, entity in enumerate(entities):
            etype = entity.dxf.dxftype
            bbox = self._get_entity_bbox(entity, transformed=True)
            if not bbox:
                continue

            pos = torch.tensor(bbox.center, dtype=torch.float32)
            node_positions[etype].append(pos)
            node_indices[entity.dxf.handle] = len(node_positions[etype]) - 1

            # 简化的特征编码
            if etype == 'LINE':
                start = torch.tensor(entity.dxf.start, dtype=torch.float32)
                end = torch.tensor(entity.dxf.end, dtype=torch.float32)
                features = torch.cat([start, end])
            elif etype == 'CIRCLE':
                radius = torch.tensor([entity.dxf.radius], dtype=torch.float32)
                center = torch.tensor(entity.dxf.center, dtype=torch.float32)
                features = torch.cat([center, radius])
            else: # 对其他类型使用通用位置特征
                features = pos

            node_features[etype].append(features)

        for etype, features_list in node_features.items():
            # 需要确保所有特征向量长度相同
            max_len = max(f.shape[0] for f in features_list)
            padded_features = [torch.nn.functional.pad(f, (0, max_len - f.shape[0])) for f in features_list]
            data[etype].x = torch.stack(padded_features)
            data[etype].pos = torch.stack(node_positions[etype])

        # 2. 边构建 (K-NN) - 修正和改进
        all_pos_list = []
        # 创建一个从全局索引到(类型, 局部索引)的健壮映射
        global_idx_to_type_map = []

        sorted_etypes = sorted(data.node_types)
        for etype in sorted_etypes:
            pos_tensor = data[etype].pos
            all_pos_list.append(pos_tensor)
            for local_idx in range(len(pos_tensor)):
                global_idx_to_type_map.append((etype, local_idx))

        if not all_pos_list:
            return data

        all_pos = torch.cat(all_pos_list, dim=0)

        # 使用PyTorch进行高效的距离计算
        dist_matrix = torch.cdist(all_pos, all_pos)

        # K-最近邻
        k = 5
        # 避免请求比图中节点总数还多的邻居
        k = min(k, all_pos.shape[0])
        knn = dist_matrix.topk(k, largest=False)

        edge_indices = defaultdict(list)

        for i in range(all_pos.shape[0]):
            src_type, src_local_idx = global_idx_to_type_map[i]

            for j_idx in knn.indices[i]:
                j = j_idx.item()
                if i == j: continue

                dst_type, dst_local_idx = global_idx_to_type_map[j]

                # 按字母顺序定义边类型以保持一致性
                edge_type_key = tuple(sorted((src_type, dst_type)))
                edge_type = (edge_type_key[0], f'nearby_{k}', edge_type_key[1])

                # 确保边的方向与排序后的类型一致
                if src_type == edge_type_key[0]:
                    edge_indices[edge_type].append([src_local_idx, dst_local_idx])
                else:
                    edge_indices[edge_type].append([dst_local_idx, src_local_idx])


        for edge_type, indices in edge_indices.items():
            # 移除重复的边
            unique_indices = torch.tensor(indices, dtype=torch.long).unique(dim=0)
            data[edge_type].edge_index = unique_indices.t().contiguous()

        logger.info(f"成功构建异构图，包含 {len(data.node_types)} 种节点类型和 {len(data.edge_types)} 种边类型。")
        return data

    def save_exploded_dxf(self, output_path: str) -> None:
        """
        处理DXF文件，将所有块实体分解为基础图元，并将结果保存到一个新的DXF文件中。
        这个方法主要用于调试和验证块分解的正确性。

        Args:
            output_path: 输出DXF文件的路径。
        """
        logger.info("开始执行块分解并保存为新的DXF文件...")

        # 使用stage=2来确保所有类型的实体（几何和注解）都被处理
        exploded_entities = self._extract_and_explode_entities(stage=2)

        # 创建一个新的DXF文档
        new_doc = ezdxf.new()
        new_msp = new_doc.modelspace()

        # 将所有分解后的实体添加到新文档的模型空间
        for entity in exploded_entities:
            try:
                # ATTRIB实体比较特殊，需要作为TEXT添加
                if entity.dxf.dxftype == 'ATTRIB':
                    new_msp.add_text(
                        text=entity.dxf.text,
                        dxfattribs={
                            'insert': entity.dxf.insert,
                            'height': entity.dxf.height,
                            'rotation': entity.dxf.rotation,
                            'style': entity.dxf.style,
                        }
                    )
                else:
                    new_msp.add_entity(entity.copy())
            except Exception as e:
                logger.warning(f"无法添加实体 {entity.dxf.dxftype} 到新文档中: {e}")

        # 保存新文档
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            new_doc.saveas(output_path)
            logger.info(f"成功将分解后的DXF文件保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存新的DXF文件失败: {e}")

    def save_verification_files(self, output_dir: str) -> None:
        """
        生成并保存一系列用于可视化验证的DXF文件。
        - verification_regions.dxf: 按比例区域为实体着色。
        - verification_transformed_stage1.dxf: Stage 1变换结果（缩小1/10）。
        - verification_transformed_stage2.dxf: Stage 2变换结果（中心化）。
        """
        logger.info("正在生成可视化验证文件...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        entities = self._extract_and_explode_entities(stage=2)
        scale_info = self._analyze_scale_and_unit(entities)
        entity_to_scale = self._partition_regions(entities, scale_info)

        # 1. 区域着色验证
        doc_regions = ezdxf.new()
        msp_regions = doc_regions.modelspace()
        unique_scales = sorted(list(set(entity_to_scale.values())))
        colors = [i + 1 for i in range(len(unique_scales))] # 1=Red, 2=Yellow, etc.
        scale_to_color = {scale: color for scale, color in zip(unique_scales, colors)}

        for entity in entities:
            scale = entity_to_scale.get(entity.dxf.handle)
            if scale:
                e_copy = entity.copy()
                e_copy.dxf.color = scale_to_color.get(scale, 0) # 0=ByBlock
                msp_regions.add_entity(e_copy)
        doc_regions.saveas(output_dir / "verification_regions.dxf")
        logger.info(f"已保存区域着色文件到: {output_dir / 'verification_regions.dxf'}")

        # 2. Stage 1 变换验证
        params_s1 = self._calculate_transform_params(entities, 1, entity_to_scale)
        # 额外缩小10倍以便查看
        params_s1['scale'] *= 0.1
        transformed_s1 = self._apply_transformations(entities, params_s1)
        doc_s1 = ezdxf.new()
        msp_s1 = doc_s1.modelspace()
        for entity in transformed_s1:
            msp_s1.add_entity(entity)
        doc_s1.saveas(output_dir / "verification_transformed_stage1.dxf")
        logger.info(f"已保存Stage 1变换验证文件到: {output_dir / 'verification_transformed_stage1.dxf'}")

        # 3. Stage 2 变换验证
        params_s2 = self._calculate_transform_params(entities, 2, entity_to_scale)
        transformed_s2 = self._apply_transformations(entities, params_s2)
        doc_s2 = ezdxf.new()
        msp_s2 = doc_s2.modelspace()
        for entity in transformed_s2:
            msp_s2.add_entity(entity)
        doc_s2.saveas(output_dir / "verification_transformed_stage2.dxf")
        logger.info(f"已保存Stage 2变换验证文件到: {output_dir / 'verification_transformed_stage2.dxf'}")


if __name__ == '__main__':
    # ==============================================================================
    # 使用示例:
    # 1. 将您的DXF测试文件路径替换下面的 'path/to/your/input.dxf'
    # 2. 定义一个输出目录 'output/'
    # 3. 在终端中直接运行此脚本: python DxfParser.py
    # ==============================================================================

    # 请在这里修改输入文件路径
    INPUT_DXF_PATH = 'path/to/your/input.dxf'

    # 定义所有输出文件的目标目录
    OUTPUT_DIR = 'output'

    try:
        # 检查输入文件是否存在
        input_path = Path(INPUT_DXF_PATH)
        if not input_path.is_file():
            logger.error("="*80)
            logger.error(f"测试失败: 输入文件未找到 '{INPUT_DXF_PATH}'")
            logger.error("请在脚本的 if __name__ == '__main__': 部分修改 `INPUT_DXF_PATH` 为您的测试文件路径。")
            logger.error("="*80)
        else:
            logger.info(f"开始处理DXF文件: {INPUT_DXF_PATH}")
            parser = DxfParser(dxf_path=INPUT_DXF_PATH)

            # 1. 生成并保存验证文件
            parser.save_verification_files(output_dir=OUTPUT_DIR)

            # 2. (可选) 运行完整的Stage 2处理流程并保存最终的图数据
            logger.info("="*80)
            logger.info("正在运行完整的Stage 2处理流程...")
            graph, metadata = parser.process(stage=2)

            # 保存图对象
            output_graph_path = Path(OUTPUT_DIR) / f"{input_path.stem}_graph.pt"
            torch.save(graph, output_graph_path)
            logger.info(f"已将最终的图数据保存到: {output_graph_path}")

            # 保存元数据
            import json
            output_meta_path = Path(OUTPUT_DIR) / f"{input_path.stem}_metadata.json"
            # 自定义JSON序列化程序以处理ezdxf和numpy对象
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.ndarray, np.generic)):
                        return obj.tolist()
                    if hasattr(obj, 'dxf'): # ezdxf 实体
                        return f"DXFEntity({obj.dxf.dxftype})"
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    return str(obj)

            with open(output_meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4, cls=CustomEncoder)
            logger.info(f"已将元数据保存到: {output_meta_path}")

            logger.info("="*80)
            logger.info("所有处理和验证流程已完成。")

    except Exception as main_exc:
        logger.error(f"在主执行流程中发生严重错误: {main_exc}", exc_info=True)
