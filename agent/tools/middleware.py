# **LangGraph Agent 的三层中间件（切面钩子）**，不是业务工具，
#  是插在 Agent 运行流程里的拦截器，用来：监控工具、打日志、动态换系统提示词。

from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

# 工具执行的监控
# - `request: ToolCallRequest`：框架封装好的工具请求包，里面包含：要调用哪个工具、传了什么参数、runtime 运行时上下文
# - `handler`：真正去跑工具的函数！`handler(request)` = 执行 `rag_summarize / fetch_external_data` 那些你写的工具
# - 逻辑顺序：
#   1. 打印：准备执行哪个工具、参数是什么
#   2. `handler(request)` 执行工具
#   3. 如果执行成功：打印成功日志；**重点判断：如果这个工具名字是 `fill_context_for_report` → 在全局运行上下文打上标记 `report = True`**
#   4. 如果工具里面抛异常：捕获，记录错误日志，然后继续往上抛出，不让异常被悄悄吞掉
#   5. 返回工具结果，交给 Agent
@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

# 模型执行前输出日志
# - `state: AgentState`：Agent 的状态，最核心就是 `messages` 消息列表，包含历史对话、工具返回结果、用户提问
# - `runtime: Runtime`：整个对话会话的运行时容器，和上面 `request.runtime.context` 是一套上下文
# - 作用：**每次准备发给大模型之前打日志**
#   - 打印：现在消息总共有多少条
#   - 打印最后一条消息是什么（是用户提问？还是工具返回的 ToolMessage？）
# - `return None`：不修改状态，只做 “偷看 + 打印”，放行
@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
