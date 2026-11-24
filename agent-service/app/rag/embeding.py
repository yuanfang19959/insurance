from __future__ import annotations

from typing import Any

class BGEM3Embedding:
    """加载一次 BGE-M3，同时生成稠密向量和稀疏向量。"""

    def __init__(
        self,
        *,
        device: str | None = None,
    ) -> None:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        # 模型权重较大，因此保存在对象中，由整个应用复用。
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3",
            device=device,
            normalize_embeddings=True,
            use_fp16=False,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def dim(self) -> int:
        """稠密向量有多长。建向量库表时要先声明这个长度，BGE-M3 一般是 1024。"""
        return self.embedding_function.dim['dense']

    def embed_query(self, text: str) -> dict[str, Any]:
        """把用户的一句搜索词转成向量，用来去知识库里找相似内容。

        同一套模型对「提问」和「文档」的编码入口不同，搜索走 encode_queries。
        这里只传一句，所以结果要取出第一条：
        - dense：一条固定长度的数字数组，表示整句意思，类似给这句话算出的坐标
        - sparse：保留成只有 1 行的结构，不拆散，检索接口可以直接拿去用
        """
        if not text.strip():
            raise ValueError("query must not be empty")
        result = self.embedding_function.encode_queries([text])
        return {'dense': result['dense'][0], 'sparse': result['sparse'][0:1]}

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        """把一批文档片段转成向量，写入知识库时用。

        一次可以传多段文字，返回的 dense / sparse 和入参按顺序一一对应，不拆成单条。
        入库走 encode_documents，和搜索时的 encode_queries 是同一模型的两个入口。
        """
        if not texts:
            raise ValueError("texts must not be empty")
        return self.embedding_function.encode_documents(texts)