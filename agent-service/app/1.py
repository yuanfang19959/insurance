from pymilvus import MilvusClient, DataType

# 1.初始化客户端
client = MilvusClient(
    uri="http://127.0.0.1:19530"
)
# 2.创建collection

# 2.1.schema
schema = client.create_schema(
    auto_id=False, # 不自动生成id
    enable_dynamic_field=False, # 是否运行动态字段
)

dim = embedding.dim()

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=4096)
# 稠密向量字段
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
# 稀疏向量字段
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
# 普通元数据字段，但是是数组类型
schema.add_field(
    field_name="section_path",
    datatype=DataType.ARRAY,
    element_type=DataType.VARCHAR,
    max_length=512,
    max_capacity=6,
)

# 2.2.索引
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="id",
    index_type="AUTOINDEX"
)
index_params.add_index(
    field_name="section_path",
    index_type="AUTOINDEX",
)
 # 稠密向量索引
index_params.add_index(
    field_name="dense_vector", # 字段名
    index_type="AUTOINDEX", # 稠密向量索引方式
    metric_type="COSINE" # 相似度计算方式，COSINE、IP
)
# 稀疏向量索引
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX", # 稀疏向量，采用倒排索引
    metric_type="IP", # 相似度计算方式，支持BM25、IP
)

# 2.3.创建collection
client.create_collection(
    collection_name="my_collection_1",
    schema=schema,
    index_params=index_params
)

res = client.get_load_state(
    collection_name="my_collection_1"
)

print(res)


# 查看Collection
res = client.list_collections()

print(res)# ['my_collection_1']

# 删除Collection
# client.drop_collection(
#     collection_name="my_collection_1"
# )