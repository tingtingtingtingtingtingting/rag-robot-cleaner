"""
yaml
k: v
- `config/rag.yml`：RAG 相关（切片、召回数量、embedding 模型）
- `config/chroma.yml`：向量库 Chroma 的参数（持久化路径、距离方式）
- `config/prompts.yml`：系统提示词、问答模板
- `config/agent.yml`：Agent 的模型、工具参数、思考策略

1. YAML 只是纯数据，不能执行代码，更安全；config.py 是可执行 Python 代码。
2. YAML 跨语言通用，别的语言也能读取；config.py 只能 Python 用。
3. YAML 适合给非开发人员改，写注释方便，适合大批量调参；config.py 容易写出语法 bug。
4. 方便拆分多份配置（rag /chroma/agent），结构清晰。
"""
import yaml
from path_tool import get_abs_path


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
