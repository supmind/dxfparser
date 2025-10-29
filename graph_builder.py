# graph_builder.py

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Set, Iterable

import ezdxf
import torch
from ezdxf.document import Drawing
from ezdxf import DXFError
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


if __name__ == '__main__':
    _demonstrate_preprocessor()
