# Standard library imports
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Third-party imports
import ezdxf
import numpy as np
import torch
from ezdxf.document import Drawing
from ezdxf.layouts import Modelspace
from ezdxf.entities import DXFEntity
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

    def process(self, stage: int) -> Tuple[Optional[HeteroData], Optional[Dict[str, Any]]]:
        """
        Main public method to run the entire processing pipeline.

        Args:
            stage: An integer (1 or 2) to determine the processing stage.
                   Stage 1 focuses on geometry for global normalization.
                   Stage 2 processes all entities for final graph construction.

        Returns:
            A tuple containing the graph data and metadata, or (None, None) if processing fails.
        """
        # 1. Extract and filter entities based on stage
        entities = self._extract_and_explode_entities(stage)

        # 2. Calculate transformation parameters based on stage
        # transform_params = self._calculate_transform_params(entities, stage)

        # 3. Build the heterogeneous graph
        # graph_data = self._build_graph(entities, transform_params, stage)

        # 4. Prepare metadata
        # meta_data = {
        #     'dxf_path': str(self.dxf_path),
        #     'stage': stage,
        #     'transform_params': transform_params
        # }

        # 5. Return graph_data and meta_data
        logger.info(f"Processing for stage {stage} is outlined. Implementation pending.")

        return (None, None)

    def _extract_and_explode_entities(self, stage: int) -> List[DXFEntity]:
        """
        此方法根据`stage`筛选实体类型。
        核心功能: 遍历初始实体列表，当遇到'INSERT'类型时，调用 `_handle_insert_entity` 方法来处理，而不是简单分解。
        """
        if stage == 1:
            allowed_types = self.SUPPORTED_GEOMETRIES
        elif stage == 2:
            allowed_types = self.SUPPORTED_GEOMETRIES + self.SUPPORTED_ANNOTATIONS
        else:
            logger.warning(f"提供了无效的阶段 '{stage}'。将不会提取任何实体。")
            return []

        logger.info(f"阶段 {stage}: 正在提取以下类型的实体: {allowed_types}")
        initial_entities = [e for e in self.modelspace if e.dxf.dxftype() in allowed_types]
        logger.info(f"在模型空间中找到 {len(initial_entities)} 个初始实体。")

        final_entities: List[DXFEntity] = []
        for entity in initial_entities:
            if entity.dxf.dxftype() == 'INSERT':
                final_entities.extend(self._handle_insert_entity(entity))
            else:
                final_entities.append(entity)

        # Important: For stage 1, filter out any non-geometric entities that might
        # have been extracted from blocks (like ATTRIB).
        if stage == 1:
            final_entities = [e for e in final_entities if e.dxf.dxftype() in self.SUPPORTED_GEOMETRIES]

        logger.info(f"块分解后的实体总数: {len(final_entities)}")
        return final_entities

    def _handle_insert_entity(self, insert_entity: DXFEntity) -> List[DXFEntity]:
        """
        这是一个关键的辅助方法，用于正确处理带属性的块。
        第一步: 分解块定义中的几何与静态文本实体，并应用块实例的变换矩阵。此过程应忽略块定义中的'ATTDEF'。
        第二步: 遍历块实例的属性 (`insert_entity.attribs`)，将每个属性实体(`ATTRIB`)作为独立的、已变换好的实体加入到返回列表中。
        支持递归处理嵌套块。
        """
        final_entities: List[DXFEntity] = []
        try:
            block_def = self.doc.blocks.get(insert_entity.dxf.name)
        except KeyError:
            logger.warning(f"找不到块定义 '{insert_entity.dxf.name}'，已跳过。")
            return final_entities

        # Step 1: Handle geometry and static text, ignoring ATTDEFs
        for entity in block_def:
            if entity.dxf.dxftype() == 'ATTDEF':
                continue

            new_entity = entity.copy()
            new_entity.transform(insert_entity.matrix44)

            if new_entity.dxf.dxftype() == 'INSERT':
                final_entities.extend(self._handle_insert_entity(new_entity))
            else:
                final_entities.append(new_entity)

        # Step 2: Handle attribute entities
        if insert_entity.has_attribs:
            for attrib in insert_entity.attribs:
                final_entities.append(attrib)

        return final_entities

    def _calculate_transform_params(self, entities: List[DXFEntity], stage: int) -> Dict[str, Any]:
        """
        Calculates normalization parameters.
        Stage 1: Global normalization (translate+scale to fit in [0,1]^2).
        Stage 2: Centering only (translate to origin, no scaling).

        Args:
            entities: A list of DXF entities.
            stage: The processing stage (1 or 2).

        Returns:
            A dictionary containing transformation parameters.
        """
        logger.info(f"Stage {stage}: Calculating transformation parameters. Implementation pending.")
        return {}

    def _build_graph(self, entities: List[DXFEntity], transform_params: Dict[str, Any], stage: int) -> HeteroData:
        """
        Creates nodes, encodes features, and builds edges for the HeteroData object.

        Args:
            entities: A list of DXF entities.
            transform_params: A dictionary of transformation parameters.
            stage: The processing stage (1 or 2).

        Returns:
            A HeteroData object representing the graph.
        """
        logger.info(f"Stage {stage}: Building heterogeneous graph. Implementation pending.")
        return HeteroData()

    @staticmethod
    def _save_graph(graph_data: HeteroData, meta_data: Dict[str, Any], output_path: str) -> None:
        """
        Saves the graph data and metadata to the specified output path.

        Args:
            graph_data: The HeteroData object to save.
            meta_data: A dictionary of metadata to save as a JSON file.
            output_path: The path to save the output files (without extension).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save graph data
        torch.save(graph_data, f"{output_path}.pt")
        logger.info(f"Graph data saved to {output_path}.pt")

        # Save metadata
        import json
        with open(f"{output_path}.json", 'w') as f:
            json.dump(meta_data, f, indent=4)
        logger.info(f"Metadata saved to {output_path}.json")
