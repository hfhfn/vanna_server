import os
import shutil
from typing import List, Optional, Dict, Union

import pandas as pd
import pymysql
from chromadb import EmbeddingFunction, Documents, Embeddings
from sqlalchemy import text, create_engine
from vanna.exceptions import ImproperlyConfigured, ValidationError

from rewrite_ask import ask
from siliconflow_api import SiliconflowEmbedding
from custom_chat import CustomChat
# from vanna.chromadb import ChromaDB_VectorStore
from rewrite_chromadb_class import AutoCleanChromaDB
from dotenv import load_dotenv
import plotly.io as pio

load_dotenv()
# 设置显示后端为浏览器
pio.renderers.default = 'browser'


class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    A embeddingFunction that to generate embeddings which can use in chromadb.
    """

    def __init__(self, config=None):
        if config is None or "api_key" not in config:
            raise ValueError("Missing 'api_key' in config")

        self.base_url = config.get("base_url", "https://api.siliconflow.cn/v1")
        self.api_key = config["api_key"]
        self.model = config.get("model", "BAAI/bge-m3")

        try:
            self.client = config["embedding_client"](base_url=self.base_url, api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Error initializing client: {e}")

    def __call__(self, input: Documents) -> Embeddings:
        # Replace newlines, which can negatively affect performance.
        input = [t.replace("\n", " ") for t in input]
        all_embeddings = []
        print(f"Generating embeddings for {len(input)} documents")

        # Iterating over each document for individual API calls
        for document in input:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=document
                )
                # print(response)
                embedding = response.data[0].embedding
                all_embeddings.append(embedding)
                # print(f"Cost required: {response.usage.total_tokens}")
            except Exception as e:
                raise ValueError(f"Error generating embedding for document: {e}")

        return all_embeddings


class VannaServer:
    def __init__(self, config):
        self.config = config
        self.db_type = config.get("db_type", "mysql")  # 默认为 mysql
        self.db_name = config.get("db_name", os.getenv("DB_NAME", "dify_data"))

        self.chromadb_rebuild = False
        self.engines = self._create_engines()  # 暂时只支持 mysql
        self.engine = self.engines.get(self.db_name)
        self.vn = self._initialize_vn()
        if self.chromadb_rebuild:
            self.schema_train()


    def _create_engines(self):
        config = self.config
        host_str = config.get("host", os.getenv("DB_HOST", "localhost"))
        hosts = host_str.split(",")
        # dbname = config.get("db_name", os.getenv("DB_NAME", "dify_data"))
        dbname = self.db_name
        user = config.get("user", os.getenv("DB_USER", "root"))
        password = config.get("password", os.getenv("DB_PASSWORD", "mysql"))
        port = int(config.get("port", os.getenv("DB_PORT", 3306)))

        # # 连接数据库并捕获异常(单个连接)
        # try:
        #     vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)
        #     print("Database connection successful.")
        # except Exception as e:
        #     print(f"Failed to connect to database: {e}")
        #     raise

        # 创建 SQLAlchemy 引擎（动态连接池）

        engines = {}
        for host in hosts:
            try:
                engine = create_engine(
                    f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}",
                    pool_size=5,  # 连接池大小
                    max_overflow=20,  # 最大溢出连接数
                    pool_timeout=30,  # 连接超时时间
                    pool_recycle=1800  # 连接回收时间,默认值为0，表示不回收
                )
                print("Database engine created successfully.")
                engines[dbname] = engine
                break
            except Exception as e:
                print(f"Unexpected error creating database engine: {e}")
                raise
        return engines

    def _initialize_vn(self):
        config = self.config
        supplier = config.get("supplier", os.getenv("SUPPLIER", "GITEE"))
        embedding_supplier = config.get("embedding_supplier", os.getenv("EMBEDDING_SUPPLIER", "SiliconFlow"))
        vector_db_path = config.get("vector_db_path", os.getenv("VECTOR_DB_PATH", "../storage/chromadb"))
        EmbeddingClass = config.get("EmbeddingClass", SiliconflowEmbedding)
        ChatClass = config.get("ChatClass", CustomChat)

        os.makedirs(vector_db_path, exist_ok=True)

        # 向量数据库的 embedding_function 参数配置
        embedding_config = {
            "base_url": os.getenv(f"{embedding_supplier}_EMBEDDING_API_BASE"),
            "api_key": os.getenv(f"{embedding_supplier}_EMBEDDING_API_KEY"),
            "model": os.getenv(f"{embedding_supplier}_EMBEDDING_MODEL"),
            "embedding_client": EmbeddingClass
        }

        # 向量数据库的配置
        vector_config = {
            "collection_metadata": {
                "_type": "CollectionConfiguration",  # 必须字段
                "hnsw:space": "cosine",  # 向量空间类型
                "description": "Vanna AI 向量存储"  # 可选描述
            },
            "path": vector_db_path,
            "embedding_function": CustomEmbeddingFunction(embedding_config)
        }

        # CustomChatLLM配置
        LLM_config = {
            "api_key": os.getenv(f"{supplier}_API_KEY"),
            "model": os.getenv(f"{supplier}_CHAT_MODEL"),
            "api_base": os.getenv(f"{supplier}_API_BASE")
        }

        # vanna配置
        config = {}
        config.update(vector_config)
        config.update(LLM_config)

        try:
            MyVanna = make_vanna_class(ChatClass=ChatClass)
            vn = MyVanna(config, self.engine)
        except:
            # 如果既往数据出现干扰，在启动时清空 ChromaDB 存储目录中的文件
            chroma_path = os.path.join(os.getcwd(), vector_db_path)
            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)
                os.makedirs(chroma_path, exist_ok=True)
                print("ChromaDB storage directory 被清空，并重新创建")

            try:
                MyVanna = make_vanna_class(ChatClass=ChatClass)
                vn = MyVanna(config, self.engine)
                self.chromadb_rebuild = True
            except Exception as e:
                print(f"Failed to re-initialize Vanna: {e}")
                raise

        self._copy_fig_html()

        return vn

    def _copy_fig_html(self):
        source_path = 'fig.html'
        target_dir = '../output/html'
        target_path = os.path.join(target_dir, 'vanna_fig.html')

        # 检查目标文件是否存在
        if os.path.exists(target_path):
            print(f"Target file {target_path} already exists. Skipping copy.")
            return

        # 确保源文件存在
        if not os.path.exists(source_path):
            print(f"Source file {source_path} does not exist.")
            return

        # 创建目标目录（如果不存在）
        os.makedirs(target_dir, exist_ok=True)

        # 复制文件
        try:
            shutil.copy(source_path, target_path)
            print(f"Successfully copied {source_path} to {target_path}")
        except Exception as e:
            print(f"Failed to copy {source_path} to {target_path}: {e}")

    def get_table_metadata(self, table_type='BASE TABLE'):
        """专用于获取表元数据的方法"""
        if self.db_type == 'mysql':
            schema_function = "DATABASE()"
            comment_column = "COLUMN_COMMENT"
        elif self.db_type == 'postgresql':
            schema_function = "CURRENT_SCHEMA()"
            comment_column = "col_description(pg_class.oid, ordinal_position)"
        elif self.db_type == 'sqlserver':
            schema_function = "SCHEMA_NAME()"
            comment_column = "NULL"  # SQL Server 没有直接的 COLUMN_COMMENT，可扩展实现
        elif self.db_type == 'oracle':
            schema_function = "USER"
            comment_column = "COMMENTS"
        elif self.db_type == 'snowflake':
            schema_function = "CURRENT_DATABASE()"
            comment_column = "COMMENT"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        try:
            # 下方这个注释是忽略SQL语句检查
            # noinspection SqlNoDataSourceInspection
            sql = f"""
            SELECT 
                TABLE_CATALOG,
                TABLE_SCHEMA,
                TABLE_NAME,
                COLUMN_NAME,
                DATA_TYPE,
                CHARACTER_MAXIMUM_LENGTH,
                NUMERIC_PRECISION,
                NUMERIC_SCALE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                {comment_column} AS COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ({schema_function})
              AND TABLE_NAME IN (
                  SELECT TABLE_NAME 
                  FROM INFORMATION_SCHEMA.TABLES 
                  WHERE TABLE_SCHEMA = ({schema_function})
                    AND TABLE_TYPE = '{table_type}'
              )
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
            return self.vn.run_sql(sql)
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return None

    def schema_train(self):
        # The information schema query may need some tweaking depending on your database. This is a good starting point.
        # df_information_schema = self.vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        df_information_schema = self.get_table_metadata(table_type='BASE TABLE')

        # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
        if df_information_schema is not None:
            plan = self.vn.get_training_plan_generic(df_information_schema)
            # print(plan)

            # If you like the plan, then uncomment this and run it to train
            self.vn.train(plan=plan)
        else:
            print("No table metadata found.")

    def vn_train(self, qa: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                 sql: Optional[Union[str, List[str]]] = None,
                 documentation: Optional[Union[str, List[str]]] = None,
                 ddl: Optional[Union[str, List[str]]] = None,
                 df: Optional[Union[pd.DataFrame, List[pd.DataFrame], str, List[str], dict, List[dict]]] = None):
        if qa:
            # 训练问答对
            if isinstance(qa, list):
                for qa_ in qa:
                    question = qa_['question']
                    sql = qa_['sql']
                    # 训练问答对
                    self.vn.train(
                        question=question,
                        sql=sql
                    )
            else:
                question = qa['question']
                sql = qa['sql']
                self.vn.train(
                    question=question,
                    sql=sql
                )
        elif sql:
            # You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
            if isinstance(sql, list):
                for sql_ in sql:
                    self.vn.train(sql=sql_)
            else:
                self.vn.train(sql=sql)

        if documentation:
            # Sometimes you may want to add documentation about your business terminology or definitions.
            if isinstance(documentation, list):
                for doc in documentation:
                    self.vn.train(documentation=doc)
            else:
                self.vn.train(documentation=documentation)

        if ddl:
            # You can also add DDL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
            if isinstance(ddl, list):
                for ddl_ in ddl:
                    self.vn.train(ddl=ddl_)
            else:
                self.vn.train(ddl=ddl)

        if df is not None:
            if isinstance(df, list):
                for item in df:
                    if isinstance(item, pd.DataFrame) and not item.empty:
                        str_df = item.to_string(index=False)
                        self.vn.train(documentation=str_df)
                    elif isinstance(item, str):
                        self.vn.train(documentation=item)
                    elif isinstance(item, dict):
                        str_dict = str(item)
                        self.vn.train(documentation=str_dict)
                    else:
                        print(f"Unsupported type in list: {type(item)}. Skipping.")
            elif isinstance(df, pd.DataFrame) and not df.empty:
                str_df = df.to_string(index=False)
                self.vn.train(documentation=str_df)
            elif isinstance(df, str):
                self.vn.train(documentation=df)
            elif isinstance(df, dict):
                str_dict = str(df)
                self.vn.train(documentation=str_dict)
            else:
                print(f"Unsupported type: {type(df)}. Skipping.")

        print("Training complete.")

    def get_training_data(self):
        training_data = self.vn.get_training_data()
        # print(training_data)
        return training_data

    def ask(self, question, visualize=True, auto_train=True, *args, **kwargs):
        # sql = self.vn.generate_sql(question=question)
        # print("这里是生成的sql语句： ", sql)
        # df = self.vn.run_sql(sql)
        # print("\n这里是查询的数据： ", df)
        # plotly_code = self.vn.generate_plotly_code(question=question, sql=sql, df_metadata=df)
        # print("\n这里是生成的plotly代码： ", plotly_code)
        # figure = self.vn.get_plotly_figure(plotly_code, df=df)
        # # figure.show()
        # sql, df, fig = self.vn.ask(question, visualize=visualize, auto_train=auto_train)
        try:
            sql, df, fig = ask(self.vn, question, visualize=visualize, auto_train=auto_train, *args, **kwargs)
            # fig.show()
            return sql, df, fig
        except Exception as e:
            print(f"Error in ask method: {e}")
            return None, None, None

    def update_supplier(self, new_supplier):
        """更新供应商配置， 重新初始化vanna实例"""
        if self.config["supplier"] == new_supplier:
            print("Supplier is already set to the new value.")
            return

        self.config["supplier"] = new_supplier
        try:
            self.vn = self._initialize_vn()  # 重新初始化 vn
            if self.vn is None:
                print("Failed to initialize Vanna with the new supplier. Please check the configuration.")
            else:
                print("Supplier updated successfully.")
        except Exception as e:
            print(f"Error updating supplier: {e}")

    def update_db_engine(self, new_db_name):
        """更新数据库名称, 更换连接数据库"""
        if self.db_name == new_db_name:
            print("Database name is already set to the new value.")
            return

        self.db_name = new_db_name
        try:
            if self.db_name not in self.engines:
                print(f"Database '{new_db_name}' not found in the list of engines.")
                self.engines.update(self._create_engines())
                print("Database name updated successfully.")
        except Exception as e:
            print(f"Error updating database name: {e}")
        self.engine = self.engines[self.db_name]
        self.vn.connect_to_mysql(engine=self.engine)

def make_vanna_class(ChatClass=CustomChat):
    class MyVanna(AutoCleanChromaDB, ChatClass):
        def __init__(self, config=None, engine=None):
            AutoCleanChromaDB.__init__(self, config=config)
            ChatClass.__init__(self, config=config)
            """
            # 若想使用 super()，需满足以下条件：
            # 1. 所有父类的 __init__ 必须也调用 super().__init__()。
            # 2. 继承链必须形成一致的调用顺序
            # super().__init__(config=config)
            """
            self.engine = engine
            self.current_dbname = config.get("db_name")  # 记录当前数据库名称
            self.connection = None  # 初始化 connection 为 None
            self.run_sql = None  # 初始化 run_sql 为 None

            # 初始化数据库连接
            self.connect_to_mysql()

        def connect_to_mysql(self, **kwargs):
            """初始化数据库连接，不切换数据库上下文"""
            if kwargs.get("engine"):
                self.connection.close()
                self.engine = kwargs.get("engine")
            try:
                # with self.engine.connect() as connection:
                self.connection = self.engine.connect()
                self.run_sql_is_set = True
                self.run_sql = lambda sql: self._run_sql_with_connection(sql)
                print("Database connection successful.")
            except Exception as e:
                print(f"Failed to connect to database: {e}")
                raise ValidationError(e)

        # 这种方式容易造成数据库使用混乱，不同的数据库，创建不同的engine
        # def switch_database(self, dbname: str):
        #     """切换数据库上下文"""
        #     if not dbname:
        #         raise ImproperlyConfigured("Please set your MySQL database")
        #     if self.current_dbname == dbname:
        #         return
        #
        #     try:
        #         # 切换数据库上下文
        #         self.connection.execute(text(f"USE {dbname}"))
        #         self.current_dbname = dbname
        #         print(f"Switched to database: {dbname}")
        #     except Exception as e:
        #         print(f"Failed to switch to database: {e}")
        #         raise ValidationError(e)

        def _run_sql_with_connection(self, sql: str) -> Union[pd.DataFrame, None]:
            try:
                result = self.connection.execute(text(sql))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            except pymysql.Error as e:
                self.connection.rollback()
                raise ValidationError(e)
            except Exception as e:
                self.connection.rollback()
                raise e

        def is_sql_valid(self, sql: str) -> bool:
            # Your implementation here
            return False

        def generate_query_explanation(self, sql: str):
            my_prompt = [
                self.system_message("You are a helpful assistant that will explain a SQL query"),
                self.user_message("Explain this SQL query: " + sql),
            ]

            return self.submit_prompt(prompt=my_prompt)

    return MyVanna


# 使用示例
if __name__ == '__main__':
    config = {"supplier": "SiliconFlow"}
    server = VannaServer(config)
    # server.schema_train()

    qa = [
        {
            "question": "数据库都有哪些用户表",
            "sql": """SELECT * FROM information_schema.tables
                    WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys');"""
        },
        {
            "question": "sale_demo表中有哪些字段",
            "sql": """SELECT * FROM information_schema.columns
                    WHERE table_name = 'sale_demo';"""
        }
    ]
    server.vn_train(qa=qa)

    # server.ask("数据库都有哪些用户表不要返回系统表")

    # field = server.ask("sale_demo表中有哪些字段")[1]
    # schema = server.ask("查询sale_demo表的schema")[1]
    # server.vn_train(df=[field, schema])

    server.ask("汇总sale_demo表每个类别的销售量和销售额, 并按照销售量进行降序排列")
