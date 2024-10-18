#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 功能描述：
#   使用智谱AI的嵌入模型作为Chroma向量数据库的嵌入方法，作为参考，你也可以定义和使用其他嵌入模型。
# 注意事项：
#   无
# 安装依赖：
#   pip intsall zhipuai
#   pip intsall chromadb
# 运行示例：
#   详见"__main__"
#
# 作者: Airmomo（https://github.com/Airmomo）
#
# 创建日期: 2024.08.30
# 最后修改日期: 2024.10.18

import logging
from typing import Optional, cast
from zhipuai import ZhipuAI
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb import Collection
from chromadb import PersistentClient as PersistentChroma

logger = logging.getLogger(__name__)


class ZhiPuAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://open.bigmodel.cn/api/paas/v4/embeddings",
        model_name: str = "embedding-3"
    ):
        try:
            import zhipuai
        except ImportError:
            raise ValueError(
                "The zhipuai python package is not installed. Please install it with `pip install zhipuai`"
            )

        # If the api key is still not set, raise an error
        if api_key is None:
            raise ValueError(
                "Please provide an ZhipuAI API key."
            )

        self._client = ZhipuAI(api_key=api_key, base_url=api_base).embeddings
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate the embeddings for the given `input`.

        Args:
            input (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: The embeddings for the given input sorted by index
        """
        # replace newlines, which can negatively affect performance.
        input = [t.replace("\n", " ") for t in input]

        # Call the Embedding API
        embeddings = self._client.create(input=input, model=self._model_name).data

        # Sort resulting embeddings by index
        sorted_embeddings = sorted(
            embeddings, key=lambda e: e.index  # type: ignore
        )

        # Return just the embeddings
        return cast(
            Embeddings, [result.embedding for result in sorted_embeddings]
        )

if __name__ == "__main__":
    # 先设置智谱AI的API_KEY
    ZHIPUAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
    # Chroma默认使用的是all-MiniLM-L6-v2模型来进行 embeddings
    # 这里使用智谱AI的嵌入模型API作为嵌入方法
    embedding_function = ZhiPuAIEmbeddingFunction(api_key=ZHIPUAI_API_KEY)
    # 使用 Chroma 作为本地持久化的向量数据库
    chroma_client = PersistentChroma(path=f"test/chromadb/zhipuai_embedding")
    collection = chroma_client.create_collection(name="test_collection", get_or_create=True, embedding_function=embedding_function)