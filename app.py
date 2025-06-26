# import base64
import os
import sys

# 获取项目根目录路径
project_root = os.path.abspath(os.path.dirname(__file__))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)

from flask import Flask, request, jsonify
from siliconflow_api import SiliconflowEmbedding
from vanna_text2sql import VannaServer
import plotly.io as pio

from functools import lru_cache
from werkzeug.exceptions import BadRequest
import logging
from dotenv import load_dotenv

load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 创建Flask应用
app = Flask(__name__)


# 集中配置管理类
class Config:
    def __init__(self):
        self.embedding_supplier = os.getenv("EMBEDDING_SUPPLIER", "SiliconFlow")
        self.EmbeddingClass = SiliconflowEmbedding
        self.vector_db_path = os.getenv("VECTOR_DB_PATH")
        self.supplier = os.getenv("SUPPLIER", "GITEE")
        self.mysql_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "db_name": os.getenv("DB_NAME", "dify_data"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", "mysql"),
            "port": int(os.getenv("DB_PORT", 3306))
        }

# 创建全局 VannaServer实例
config = Config()
combined_config = {**config.__dict__, **config.mysql_config}
server = VannaServer(combined_config)


def get_vn_instance(supplier="", db_name=""):
    """获取或创建VannaServer实例"""
    if supplier:
        server.update_supplier(supplier)
    if db_name:
        server.update_db_engine(db_name)

    return server


def validate_input(data, required_fields):
    """输入验证函数"""
    for field in required_fields:
        if field not in data or not data[field]:
            raise BadRequest(f"Missing required field: {field}")


@app.route('/vn_train', methods=['POST'])
def vn_train_route():
    """训练接口"""
    data = request.json
    # required_fields = ['question', 'sql']
    # validate_input(data, required_fields)

    supplier = data.get('supplier', "")
    db_name = data.get('db_name', "")
    qa = data.get('qa', None)
    sql = data.get('sql', None)
    documentation = data.get('documentation', None)
    ddl = data.get('ddl', None)
    schema = data.get('schema', False)

    # 验证至少有一个参数不为空
    if not any([qa, sql, documentation, ddl, schema]):
        return jsonify(
        {'error': 'At least one of the parameters (qa, sql, documentation, ddl, schema) must be provided'}), 400

    server = get_vn_instance(supplier, db_name)
    server.vn_train(qa=qa, sql=sql, documentation=documentation, ddl=ddl)
    if schema:
        try:
            server.schema_train()
        except Exception as e:
            logging.info(f"Error initializing vector store: {e}")

    logging.info("Training completed successfully")
    return jsonify({'status': 'success'}), 200


@app.route('/get_training_data', methods=['GET'])
def get_training_data_route():
    """获取训练数据接口"""
    supplier = request.args.get('supplier', "")
    db_name = request.args.get('db_name', "")
    server = get_vn_instance(supplier, db_name)

    @lru_cache(maxsize=128)  # 添加缓存机制
    def cached_get_training_data():
        return server.get_training_data()

    training_data = cached_get_training_data()
    logging.info("Fetched training data successfully")

    return jsonify(training_data), 200


@app.route('/ask', methods=['POST'])
def ask_route():
    """提问接口"""
    data = request.json
    question = data.get('question', '')
    visualize = data.get('visualize', True)
    auto_train = data.get('auto_train', True)
    supplier = data.get('supplier', "")  # GITEE, ZHIPU, SiliconFlow
    db_name = data.get('db_name', "")

    if not question:
        raise BadRequest("Question is required")

    server = get_vn_instance(supplier, db_name)
    try:
        sql, df, fig = server.ask(question=question, visualize=visualize, auto_train=auto_train)

        # Convert DataFrame to JSON
        df_json = df.to_json(orient='records', force_ascii=False)

        # Convert Plotly figure to JSON
        # Convert fig to base64 encoded string
        ## 不知道为什么，这个方法会卡住，暂时不用了
        # img_bytes = pio.to_image(fig, format="png", scale=2)
        # img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        """
        <img id="plotly-image" src="data:image/png;base64,{{ img_base64 }}" alt="Plotly Image">
        """

        fig_js_path = '../output/html/vanna_fig.js'
        fig_html_path = 'http://localhost:8000/html/vanna_fig.html'
        figure_json = pio.to_json(fig)
        with open(fig_js_path, 'w', encoding='utf-8') as f:
            f.write(figure_json)
        """
          <div id="plotly-div"></div>
          <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          <script>
              var fig_json = {{ fig_json }};
              Plotly.newPlot('plotly-div', fig_json.data, fig_json.layout);
          </script>
        """

        logging.info("Query processed successfully")
        return jsonify({
            'sql': sql,
            'data': df_json,
            # 'img_base64': img_base64,
            'plotly_figure': fig_html_path
        }), 200
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
