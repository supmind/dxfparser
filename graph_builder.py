# graph_builder.py

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Set, Iterable

from collections import defaultdict
import numpy as np

import ezdxf
import torch
from ezdxf.document import Drawing
from ezdxf import DXFError
from ezdxf.math import Vec3, BoundingBox
from ezdxf.fonts.font_measurements import FontMeasurements
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalPreprocessor:
    """
    全局预处理器，负责扫描所有DXF文件，为整个数据集提取和准备全局信息。

    该类的核心功能是：
    1. 搜集数据集中所有唯一的图层名（layer names）和线型名（linetype names）。
    2. 使用预训练的Sentence Transformer模型为每个唯一的图层名生成高质量的文本嵌入。
    3. 为每个唯一的线型名创建一个从名称到整数索引的映射。
    4. 提供保存和加载这些全局映射的功能，以便在后续处理中复用，避免重复计算。

    Attributes:
        sentence_transformer_model (SentenceTransformer): 用于生成文本嵌入的预训练模型。
        layer_embedding_map (Dict[str, torch.Tensor]): 存储图层名到其对应嵌入向量的字典。
        linetype_to_idx (Dict[str, int]): 存储线型名到其对应整数索引的字典。
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化GlobalPreprocessor。

        Args:
            model_name (str): 要使用的Sentence Transformer模型的名称。
                默认为 'paraphrase-multilingual-MiniLM-L12-v2'，这是一个强大的多语言模型，
                能有效处理包含中文在内的图层名。
        """
        self.sentence_transformer_model = SentenceTransformer(model_name)
        self.layer_embedding_map: Dict[str, torch.Tensor] = {}
        self.linetype_to_idx: Dict[str, int] = {}
        # 获取嵌入向量的维度
        self.embedding_dim = self.sentence_transformer_model.get_sentence_embedding_dimension()

    def fit(self, dxf_file_paths: Iterable[str]) -> None:
        """
        遍历所有给定的DXF文件路径，提取全局信息并生成映射。

        该方法会执行以下操作：
        1. 初始化空的集合来存储唯一的图层名和线型名。
        2. 遍历每个文件路径：
           - 尝试使用 ezdxf 加载文件。
           - 如果文件损坏或无法解析，记录一条警告并跳过该文件。
           - 成功加载后，提取文件中所有的图层名和线型名，并添加到集合中。
        3. 在遍历完所有文件后，对收集到的唯一图层名进行嵌入。
        4. 为唯一的线型名创建整数索引。

        Args:
            dxf_file_paths (Iterable[str]): 一个包含所有DXF文件路径的可迭代对象。
        """
        logging.info(f"开始从 {len(list(dxf_file_paths))} 个DXF文件中提取全局信息...")
        unique_layer_names: Set[str] = set()
        unique_linetype_names: Set[str] = set()

        for file_path in dxf_file_paths:
            try:
                doc: Drawing = ezdxf.readfile(file_path)
                # 提取图层名
                unique_layer_names.update(layer.dxf.name for layer in doc.layers)
                # 提取线型名
                unique_linetype_names.update(linetype.dxf.name for linetype in doc.linetypes)
                logging.info(f"成功处理文件: {file_path}")
            except (IOError, OSError, DXFError) as e:
                logging.warning(f"无法读取或解析文件 {file_path}，已跳过。错误: {e}")
            except Exception as e:
                logging.error(f"处理文件 {file_path} 时发生未知错误，已跳过。错误: {e}")

        logging.info("所有文件扫描完毕。开始生成图层嵌入和线型索引...")

        # 1. 为图层名创建文本嵌入
        # 确保默认图层 '0' 也被包含
        unique_layer_names.add('0')
        layer_names_list = sorted(list(unique_layer_names))
        layer_embeddings = self.sentence_transformer_model.encode(
            layer_names_list,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        self.layer_embedding_map = {name: emb for name, emb in zip(layer_names_list, layer_embeddings)}

        # 2. 为线型名创建整数索引
        # 包含DXF标准默认线型
        unique_linetype_names.update(['BYLAYER', 'BYBLOCK', 'CONTINUOUS'])
        linetype_names_list = sorted(list(unique_linetype_names))
        self.linetype_to_idx = {name: i for i, name in enumerate(linetype_names_list)}

        logging.info("全局预处理完成。")
        logging.info(f"发现 {len(self.layer_embedding_map)} 个唯一图层。")
        logging.info(f"发现 {len(self.linetype_to_idx)} 个唯一线型。")

    def save(self, directory: str) -> None:
        """
        将处理好的图层嵌入映射和线型索引映射保存到磁盘。

        - 图层嵌入映射将以 pickle 格式保存在 'layer_embedding_map.pkl' 文件中。
        - 线型索引映射将以 JSON 格式保存在 'linetype_to_idx.json' 文件中。

        Args:
            directory (str): 要保存文件的目标目录路径。如果目录不存在，将自动创建。
        """
        dir_path = Path(directory)
        # 确保目标目录存在
        dir_path.mkdir(parents=True, exist_ok=True)

        # 保存图层嵌入
        layer_map_path = dir_path / "layer_embedding_map.pkl"
        with open(layer_map_path, 'wb') as f:
            pickle.dump(self.layer_embedding_map, f)
        logging.info(f"图层嵌入映射已保存至: {layer_map_path}")

        # 保存线型索引
        linetype_map_path = dir_path / "linetype_to_idx.json"
        with open(linetype_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.linetype_to_idx, f, ensure_ascii=False, indent=4)
        logging.info(f"线型索引映射已保存至: {linetype_map_path}")

    def load(self, directory: str) -> None:
        """
        从磁盘加载之前保存的图层嵌入映射和线型索引映射。

        Args:
            directory (str): 存储映射文件的目录路径。

        Raises:
            FileNotFoundError: 如果在指定目录中找不到必要的 .pkl 或 .json 文件。
        """
        dir_path = Path(directory)
        layer_map_path = dir_path / "layer_embedding_map.pkl"
        linetype_map_path = dir_path / "linetype_to_idx.json"

        if not layer_map_path.exists() or not linetype_map_path.exists():
            raise FileNotFoundError(f"在目录 '{directory}' 中找不到预处理文件。请先运行 'fit' 和 'save' 方法。")

        # 加载图层嵌入
        with open(layer_map_path, 'rb') as f:
            self.layer_embedding_map = pickle.load(f)
        logging.info(f"从 {layer_map_path} 加载了 {len(self.layer_embedding_map)} 个图层嵌入。")

        # 加载线型索引
        with open(linetype_map_path, 'r', encoding='utf-8') as f:
            self.linetype_to_idx = json.load(f)
        logging.info(f"从 {linetype_map_path} 加载了 {len(self.linetype_to_idx)} 个线型索引。")

        if self.layer_embedding_map:
            # 推断嵌入维度
            self.embedding_dim = next(iter(self.layer_embedding_map.values())).shape[0]


# --- 演示和测试 ---
def _create_dummy_dxf_files(temp_dir: Path):
    """在指定目录中创建用于测试的虚拟DXF文件。"""
    # 文件1：包含标准图层和线型
    doc1 = ezdxf.new()
    doc1.layers.add("Walls")
    doc1.layers.add("Windows")
    doc1.linetypes.add("DASHED", [0.5, -0.25])
    doc1.saveas(temp_dir / "drawing1.dxf")
    logging.info(f"创建了虚拟文件: {temp_dir / 'drawing1.dxf'}")

    # 文件2：包含中文图层和特殊线型
    doc2 = ezdxf.new()
    doc2.layers.add("结构墙")
    doc2.layers.add("门窗")
    doc2.linetypes.add("点线", [0.0, -0.1, 0.1, -0.1])
    doc2.saveas(temp_dir / "drawing2.dxf")
    logging.info(f"创建了虚拟文件: {temp_dir / 'drawing2.dxf'}")

    # 文件3：损坏的文件，用于测试错误处理
    (temp_dir / "corrupt.dxf").touch()
    logging.info(f"创建了损坏文件: {temp_dir / 'corrupt.dxf'}")

    return [str(temp_dir / "drawing1.dxf"), str(temp_dir / "drawing2.dxf"), str(temp_dir / "corrupt.dxf")]

def _demonstrate_preprocessor():
    """演示GlobalPreprocessor的完整流程。"""
    import tempfile
    import shutil

    # 创建一个临时目录来存放虚拟文件和输出
    temp_dir = tempfile.mkdtemp()
    logging.info(f"--- GlobalPreprocessor 演示开始 ---")
    logging.info(f"临时工作目录: {temp_dir}")

    try:
        # 1. 创建虚拟DXF文件
        dxf_files = _create_dummy_dxf_files(Path(temp_dir))

        # 2. 初始化并运行 'fit'
        preprocessor = GlobalPreprocessor()
        preprocessor.fit(dxf_files)

        # 验证fit的结果
        assert "结构墙" in preprocessor.layer_embedding_map
        assert "DASHED" in preprocessor.linetype_to_idx
        assert "BYLAYER" in preprocessor.linetype_to_idx # 检查是否包含默认线型
        logging.info("'fit' 方法执行成功，并找到了预期的图层和线型。")

        # 3. 保存结果
        output_dir = Path(temp_dir) / "preprocessed_output"
        preprocessor.save(str(output_dir))

        # 4. 创建新实例并加载结果
        new_preprocessor = GlobalPreprocessor()
        new_preprocessor.load(str(output_dir))

        # 5. 验证加载的数据是否正确
        assert len(new_preprocessor.layer_embedding_map) == len(preprocessor.layer_embedding_map)
        assert len(new_preprocessor.linetype_to_idx) == len(preprocessor.linetype_to_idx)
        assert torch.equal(
            new_preprocessor.layer_embedding_map["Walls"],
            preprocessor.layer_embedding_map["Walls"]
        )
        assert new_preprocessor.linetype_to_idx["点线"] == preprocessor.linetype_to_idx["点线"]
        logging.info("'load' 方法执行成功，加载的数据与原始数据一致。")
        logging.info(f"加载的图层嵌入维度: {new_preprocessor.embedding_dim}")

    except Exception as e:
        logging.error(f"演示过程中发生错误: {e}")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        logging.info(f"清理并移除了临时目录: {temp_dir}")
        logging.info(f"--- GlobalPreprocessor 演示结束 ---")


class SingleDrawingProcessor:
    """
    处理单张DXF图纸，将其转换为用于图构建的中间数据结构。

    该类的核心职责是：
    1. 使用 `ezdxf` 的 `virtual_entities()` 方法安全、只读地“炸开”所有复杂实体
       （如块引用、尺寸标注），自动处理嵌套变换和属性继承（BYLAYER/BYBLOCK）。
    2. 为图纸中的所有有效几何图元创建本地的 handle -> int 索引。
    3. 计算图纸内容的边界框，并对所有坐标进行归一化处理，同时保留长宽比。
    4. 利用 `GlobalPreprocessor` 提供的全局映射，为每个图元组装一个丰富的特征向量，
       包含几何信息、图层嵌入和线型索引。
    5. 构建图元之间的空间关系边（例如 'nearby'）。
    6. 将所有提取的信息打包成一个标准的Python字典，作为输出。
    """
    def __init__(self, layer_embedding_map: Dict[str, torch.Tensor], linetype_to_idx: Dict[str, int]):
        """
        初始化 SingleDrawingProcessor。

        Args:
            layer_embedding_map (Dict[str, torch.Tensor]): 从 GlobalPreprocessor 加载的
                图层名到嵌入向量的映射。
            linetype_to_idx (Dict[str, int]): 从 GlobalPreprocessor 加载的
                线型名到整数索引的映射。
        """
        self.layer_embedding_map = layer_embedding_map
        self.linetype_to_idx = linetype_to_idx
        # 定义我们关心和支持的基础图元类型
        self.supported_entity_types = {'LINE', 'CIRCLE', 'ARC', 'ELLIPSE', 'LWPOLYLINE', 'SPLINE', 'TEXT', 'MTEXT'}

    def _recursively_flatten(self, entities: Iterable) -> List:
        """递归地扁平化实体列表，处理嵌套的INSERT实体。"""
        flat_entities = []
        for entity in entities:
            if entity.dxf.dxftype == 'INSERT':
                try:
                    # 对块引用的虚拟实体进行递归扁平化
                    flat_entities.extend(self._recursively_flatten(entity.virtual_entities()))
                except RuntimeError as e:
                    logging.warning(f"处理嵌套INSERT {entity.dxf.handle} 时出错: {e}")
            elif entity.dxf.dxftype in self.supported_entity_types:
                flat_entities.append(entity)
        return flat_entities

    def process(self, dxf_file_path: str) -> Dict:
        """
        处理单个DXF文件并提取其结构信息。

        Args:
            dxf_file_path (str): 要处理的DXF文件的路径。

        Returns:
            Dict: 一个包含节点特征、边索引和其他图信息的中间数据字典。
                  返回None如果图纸中没有找到有效图元。
        """
        logging.info(f"开始处理单个图纸: {dxf_file_path}")
        try:
            doc = ezdxf.readfile(dxf_file_path)
            msp = doc.modelspace()
        except (IOError, OSError, DXFError) as e:
            logging.error(f"无法加载或处理DXF文件 {dxf_file_path}: {e}")
            return None

        # --- 步骤 2: 图元提取与扁平化 (递归方案) ---
        entities = self._recursively_flatten(msp)

        if not entities:
            logging.warning(f"在 {dxf_file_path} 中没有找到支持的图元。")
            return None

        logging.info(f"从 {dxf_file_path} 中提取了 {len(entities)} 个扁平化图元。")

        # --- 步骤 3: 坐标归一化与特征组装 ---
        # 3.1 计算整张图纸的边界框 (稳健方法)
        bbox = BoundingBox()
        for entity in entities:
            try:
                if entity.dxf.dxftype == 'LINE':
                    bbox.extend([entity.dxf.start, entity.dxf.end])
                elif entity.dxf.dxftype == 'CIRCLE':
                    center = Vec3(entity.dxf.center)
                    radius = entity.dxf.radius
                    bbox.extend([center - Vec3(radius, radius, 0), center + Vec3(radius, radius, 0)])
                elif entity.dxf.dxftype == 'ARC':
                    # 这是一个简化的处理，只考虑圆弧所在的整个圆
                    center = Vec3(entity.dxf.center)
                    radius = entity.dxf.radius
                    bbox.extend([center - Vec3(radius, radius, 0), center + Vec3(radius, radius, 0)])
                elif hasattr(entity, 'vertices'):
                     bbox.extend(entity.vertices)
            except Exception as e:
                logging.debug(f"无法为图元 {entity.dxf.dxftype} 计算边界框: {e}")
                pass

        if not bbox.has_data:
            logging.warning(f"在 {dxf_file_path} 中无法确定有效的边界框。")
            return None

        # 3.2 计算归一化参数
        size = bbox.size
        max_dim = max(size.x, size.y, size.z)
        if max_dim < 1e-6: # 避免除以零
            max_dim = 1.0

        scale_factor = 1.0 / max_dim
        offset = -bbox.center

        # 3.3 本地句柄映射和特征组装
        node_features_by_type = defaultdict(list)
        entity_to_local_idx = {entity.dxf.handle: i for i, entity in enumerate(entities)}

        for entity in entities:
            entity_type = entity.dxf.dxftype

            # 获取图层嵌入和线型索引
            layer_name = entity.dxf.layer
            layer_embedding = self.layer_embedding_map.get(layer_name, self.layer_embedding_map.get('0')) # 找不到则用默认'0'层

            linetype_name = entity.dxf.linetype
            linetype_idx = self.linetype_to_idx.get(linetype_name, self.linetype_to_idx.get('CONTINUOUS')) # 找不到则用默认'CONTINUOUS'

            # 根据图元类型调用相应的特征提取方法
            feature_extractor = getattr(self, f"_extract_{entity_type.lower()}_features", self._extract_default_features)
            features = feature_extractor(entity, scale_factor, offset, layer_embedding, linetype_idx, float(max_dim))

            if features:
                node_features_by_type[entity_type].append(features)

        # 转换为Tensor
        for entity_type, features_list in node_features_by_type.items():
            continuous_feats = torch.tensor(np.array([f['continuous'] for f in features_list]), dtype=torch.float)
            discrete_feats = torch.tensor(np.array([f['discrete'] for f in features_list]), dtype=torch.long)
            node_features_by_type[entity_type] = {'continuous': continuous_feats, 'discrete': discrete_feats}

        # --- 步骤 4: 边构建 ---
        # 为了构建异构图，我们需要一个从全局索引到(类型, 类型内索引)的映射
        # 注意：这里的entity_to_local_idx是所有图元的全局索引
        local_idx_to_type_map = {}
        type_specific_indices = defaultdict(int)
        for i, entity in enumerate(entities):
            entity_type = entity.dxf.dxftype
            local_idx_to_type_map[i] = (entity_type, type_specific_indices[entity_type])
            type_specific_indices[entity_type] += 1

        edges_dict = self._build_spatial_edges(entities, k_neighbors=5)

        # --- 步骤 5: 返回中间数据结构 (将在后续步骤实现) ---
        # ...

        # 临时的返回，以便于结构验证
        return {
            "file_path": dxf_file_path,
            "nodes": node_features_by_type,
            "edges": edges_dict,
            "local_idx_to_type_map": local_idx_to_type_map,
            "num_nodes": len(entities)
        }

    # --- 特征提取辅助函数 ---

    def _normalize_coords(self, point: Vec3, scale: float, offset: Vec3) -> list:
        """对坐标点进行归一化。"""
        return ((point + offset) * scale).xyz

    def _extract_default_features(self, entity, *args) -> Dict:
        """处理未知或不支持的图元类型的默认函数。"""
        logging.warning(f"图元类型 {entity.dxf.dxftype} handle={entity.dxf.handle} 没有专门的特征提取器，已跳过。")
        return None

    def _extract_line_features(self, entity, scale, offset, layer_emb, linetype_idx, real_scale) -> Dict:
        """提取LINE图元的特征。"""
        start = self._normalize_coords(entity.dxf.start, scale, offset)
        end = self._normalize_coords(entity.dxf.end, scale, offset)
        length = entity.dxf.start.distance(entity.dxf.end)

        continuous = np.concatenate([start, end, [length], [real_scale], layer_emb.cpu().numpy()])
        discrete = [linetype_idx]
        return {"continuous": continuous, "discrete": discrete}

    def _extract_circle_features(self, entity, scale, offset, layer_emb, linetype_idx, real_scale) -> Dict:
        """提取CIRCLE图元的特征。"""
        center = self._normalize_coords(entity.dxf.center, scale, offset)
        radius = entity.dxf.radius

        continuous = np.concatenate([center, [0,0,0], [radius], [real_scale], layer_emb.cpu().numpy()]) # 用[0,0,0]填充end
        discrete = [linetype_idx]
        return {"continuous": continuous, "discrete": discrete}

    def _extract_arc_features(self, entity, scale, offset, layer_emb, linetype_idx, real_scale) -> Dict:
        """提取ARC图元的特征。"""
        center = self._normalize_coords(entity.dxf.center, scale, offset)
        radius = entity.dxf.radius
        start_angle = np.deg2rad(entity.dxf.start_angle)
        end_angle = np.deg2rad(entity.dxf.end_angle)

        continuous = np.concatenate([center, [radius, start_angle, end_angle], [0], [real_scale], layer_emb.cpu().numpy()]) # 用0填充length
        discrete = [linetype_idx]
        return {"continuous": continuous, "discrete": discrete}

    def _extract_text_features(self, entity, scale, offset, layer_emb, linetype_idx, real_scale) -> Dict:
        """提取TEXT/MTEXT图元的特征。"""
        # 对于TEXT和MTEXT，我们主要关心其位置和大小
        # ezdxf的virtual_entities通常会将MTEXT分解为TEXT，但我们以防万一
        if entity.dxf.dxftype == 'TEXT':
            pos = self._normalize_coords(entity.dxf.insert, scale, offset)
            height = entity.dxf.height
        elif entity.dxf.dxftype == 'MTEXT':
            pos = self._normalize_coords(entity.dxf.insert, scale, offset)
            height = entity.dxf.char_height
        else:
            return None # 不应发生

        # 使用 SentenceTransformer 编码文本内容
        text_content = entity.dxf.text
        # text_embedding = self.sentence_transformer_model.encode(text_content, convert_to_tensor=True)

        continuous = np.concatenate([pos, [0,0,0], [height], [real_scale], layer_emb.cpu().numpy()]) # 填充end和length
        discrete = [linetype_idx]
        return {"continuous": continuous, "discrete": discrete}

    _extract_mtext_features = _extract_text_features

    # --- 边构建辅助函数 ---

    def _sample_entity_points(self, entity, num_points=10) -> List[Vec3]:
        """为单个图元采样一组代表其几何形状的点。"""
        points = []
        try:
            # 对于有明确起点和终点的图元
            if entity.dxf.hasattr('start') and entity.dxf.hasattr('end'):
                start = Vec3(entity.dxf.start)
                end = Vec3(entity.dxf.end)
                for i in range(num_points):
                    points.append(start.lerp(end, i / (num_points - 1)))
            # 对于圆或圆弧
            elif entity.dxf.hasattr('center') and entity.dxf.hasattr('radius'):
                center = Vec3(entity.dxf.center)
                radius = entity.dxf.radius
                if entity.dxf.dxftype == 'CIRCLE':
                    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                else: # ARC
                    start_angle = np.deg2rad(entity.dxf.start_angle)
                    end_angle = np.deg2rad(entity.dxf.end_angle)
                    if end_angle < start_angle:
                        end_angle += 2 * np.pi
                    angles = np.linspace(start_angle, end_angle, num_points)
                for angle in angles:
                    points.append(center + Vec3.from_angle(angle, radius))
            # 对于LWPOLYLINE
            elif entity.dxf.dxftype == 'LWPOLYLINE':
                # 只采样顶点
                return list(entity.vertices)
            # 对于TEXT
            elif entity.dxf.dxftype in ['TEXT', 'MTEXT']:
                points.append(Vec3(entity.dxf.insert))
            else:
                 # 降级策略：使用边界框中心
                bbox = BoundingBox([v for v in entity.vertices])
                if bbox.has_data:
                    points.append(bbox.center)
        except Exception:
             # 最后的降级策略
            if hasattr(entity, 'dxf') and hasattr(entity.dxf, 'insert'):
                 points.append(Vec3(entity.dxf.insert))
            elif hasattr(entity, 'dxf') and hasattr(entity.dxf, 'center'):
                points.append(Vec3(entity.dxf.center))

        return points

    def _build_spatial_edges(self, entities: List, k_neighbors: int) -> Dict[str, torch.Tensor]:
        """使用K-D树构建空间邻近边。"""
        if len(entities) < 2:
            return {}

        all_points = []
        point_to_entity_idx = []

        # 1. 为所有图元采样点
        for i, entity in enumerate(entities):
            sampled_points = self._sample_entity_points(entity, num_points=5)
            if sampled_points:
                all_points.extend(sampled_points)
                point_to_entity_idx.extend([i] * len(sampled_points))

        if not all_points:
            return {}

        # 2. 构建K-D树
        kdtree = KDTree([p.xyz for p in all_points])

        # 3. 查询最近邻
        # 我们查询k+1个邻居，因为第一个总是点本身
        distances, indices = kdtree.query(all_points, k=k_neighbors + 1)

        # 4. 构建边列表
        edge_set = set()
        for i, neighbor_indices in enumerate(indices):
            source_entity_idx = point_to_entity_idx[i]
            for neighbor_idx in neighbor_indices[1:]: # 跳过第一个（自身）
                target_entity_idx = point_to_entity_idx[neighbor_idx]
                if source_entity_idx != target_entity_idx:
                    # 确保边的方向唯一性 (v, u) if v > u
                    edge = tuple(sorted((source_entity_idx, target_entity_idx)))
                    edge_set.add(edge)

        if not edge_set:
            return {}

        # 5. 格式化为edge_index
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

        # 返回一个字典，键是描述边的元组，值是edge_index
        # 在异构图中，这个键将是 ('entity_type_A', 'nearby', 'entity_type_B')
        # 在这个阶段，我们先用一个通用的键
        return {("entity", "nearby", "entity"): edge_index}


def _create_complex_dxf_for_single_processor_test(temp_dir: Path) -> str:
    """创建一个包含嵌套块和BYBLOCK/BYLAYER属性的复杂DXF文件用于测试。"""
    doc = ezdxf.new()
    doc.layers.add("RED_LAYER", color=1)  # 1 = red
    doc.layers.add("BLUE_LAYER", color=5) # 5 = blue

    # 块B：基础块
    block_b = doc.blocks.new(name="BLOCK_B")
    block_b.add_circle(center=(0, 0), radius=0.5, dxfattribs={"color": 256}) # 256 = BYBLOCK

    # 块A：包含一个LINE和一个对BlockB的引用
    block_a = doc.blocks.new(name="BLOCK_A")
    block_a.add_line(start=(-1, 0), end=(1, 0), dxfattribs={"color": 256}) # BYBLOCK
    block_a.add_blockref(name="BLOCK_B", insert=(0, 1), dxfattribs={"color": 256}) # BYBLOCK

    # 在模型空间中实例化块A两次
    msp = doc.modelspace()
    # 实例1：在红色图层，默认大小
    msp.add_blockref(name="BLOCK_A", insert=(0, 0), dxfattribs={"layer": "RED_LAYER"})
    # 实例2：在蓝色图层，放大2倍
    msp.add_blockref(name="BLOCK_A", insert=(5, 0), dxfattribs={
        "layer": "BLUE_LAYER",
        "xscale": 2.0,
        "yscale": 2.0,
    })

    file_path = temp_dir / "complex_drawing.dxf"
    doc.saveas(file_path)
    logging.info(f"创建了复杂的测试文件: {file_path}")
    return str(file_path)


def _demonstrate_single_processor():
    """演示SingleDrawingProcessor的完整流程。"""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    logging.info(f"\n--- SingleDrawingProcessor 演示开始 ---")
    logging.info(f"临时工作目录: {temp_dir}")

    try:
        # 1. 创建复杂的DXF文件
        complex_dxf_path = _create_complex_dxf_for_single_processor_test(Path(temp_dir))

        # 2. 运行全局预处理器
        preprocessor = GlobalPreprocessor()
        preprocessor.fit([complex_dxf_path])

        # 3. 初始化并运行单图纸处理器
        single_processor = SingleDrawingProcessor(preprocessor.layer_embedding_map, preprocessor.linetype_to_idx)
        intermediate_data = single_processor.process(complex_dxf_path)

        # 4. 验证输出
        assert intermediate_data is not None
        # 预期图元数量:
        # 实例1: 1 LINE + 1 CIRCLE = 2
        # 实例2: 1 LINE + 1 CIRCLE = 2
        # 总共 = 4
        assert intermediate_data["num_nodes"] == 4, f"Expected 4 nodes, but got {intermediate_data['num_nodes']}"
        logging.info(f"成功处理了复杂文件，提取了 {intermediate_data['num_nodes']} 个节点。")

        assert "LINE" in intermediate_data["nodes"]
        assert "CIRCLE" in intermediate_data["nodes"]
        assert intermediate_data["nodes"]["LINE"]["continuous"].shape[0] == 2
        assert intermediate_data["nodes"]["CIRCLE"]["continuous"].shape[0] == 2

        # 验证边的数量
        num_edges = next(iter(intermediate_data["edges"].values())).shape[1]
        logging.info(f"构建了 {num_edges} 条 'nearby' 边。")
        assert num_edges > 0

        logging.info("SingleDrawingProcessor 演示验证成功！")

    except Exception as e:
        logging.error(f"SingleDrawingProcessor演示过程中发生错误: {e}", exc_info=True)
    finally:
        shutil.rmtree(temp_dir)
        logging.info(f"清理并移除了临时目录: {temp_dir}")
        logging.info(f"--- SingleDrawingProcessor 演示结束 ---")


if __name__ == '__main__':
    _demonstrate_preprocessor()
    _demonstrate_single_processor()
