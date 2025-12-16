"""
您可能需要将仅在运行时才知道的值绑定到工具。例如，工具逻辑可能需要使用发出请求的用户的 ID。
大多数情况下，此类值不应由 LLM 控制。事实上，允许 LLM 控制用户 ID 可能会导致安全风险。
相反，LLM 应该只控制本应由 LLM 控制的工具参数，而其他参数（如用户 ID）应由应用程序逻辑固定。
本操作指南向您展示了如何防止模型生成某些工具参数并在运行时直接注入它们。
"""
from langchain_core.tools import InjectedToolArg, tool
from langchain_ollama import ChatOllama
from typing import List,Annotated,Optional,Type
from pydantic import BaseModel,Field
from langchain.tools import BaseTool
from langchain_core.messages import ToolMessage
llm = ChatOllama(model="qwen3:4b",temperature=0.1,verbose=True)

class UpdateFavoritePetsSchema(BaseModel):
    """添加或者更新最喜爱的宠物列表。"""
    pets:List[str] = Field(...,description="最喜爱的宠物列表")
    user_id:Annotated[str,InjectedToolArg] = Field(...,description="用户ID") 


user_to_pets = {}

@tool(args_schema=UpdateFavoritePetsSchema)
def update_favorite_pets(pets,user_id):
    user_to_pets[user_id] = pets

#========== second tool ================
# class UpdateFavoritePets(BaseTool):
#     name: str = "update_favorite_pets"
#     description: str = "添加或者更新最喜爱的宠物列表"
#     args_schema: Optional[Type[BaseModel]] = UpdateFavoritePetsSchema

#     def _run(self, pets, user_id):
#         user_to_pets[user_id] = pets

# update_favorite_pets = UpdateFavoritePets()

@tool(parse_docstring=True)
def delete_favorite_pets(user_id:Annotated[str,InjectedToolArg])->None:
    """Delete the list of favorite pets.

    Args:
        user_id: User's ID.
    """
    print(f'delete_favorite_pets is called:{user_id}')
    if user_id in user_to_pets:
        del user_to_pets[user_id]

@tool(parse_docstring=True)
def list_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """List favorite pets if any.

    Args:
        user_id: User's ID.
    """
    print(f'list_favorite_pets is called:{user_id}')
    return user_to_pets.get(user_id, [])

# If we look at the input schemas for these tools,which is what is passed python,  we'll see that user_id is still listed:
print(f'get_input_schema:{update_favorite_pets.get_input_schema().model_json_schema()}')

# But if we look at the tool call schema, which is what is passed to the model for tool-calling, user_id has been removed:
print(f'tool_call_schema:{update_favorite_pets.tool_call_schema.model_json_schema()}')

tools = [
    update_favorite_pets,
    delete_favorite_pets,
    list_favorite_pets,
]
llm_with_tools = llm.bind_tools(tools)

query = "my favorite animals are cats and parrots"
print("---1、调用LLM，将请求转化为json结构---")

ai_msg = llm_with_tools.invoke(query)
print(f'result:{ai_msg.tool_calls}')

from langchain_core.runnables import chain
from copy import deepcopy
user_id = "123"

@chain
def inject_user_id(ai_msg):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        tool_call_copy["args"]["user_id"] = user_id
        tool_calls.append(tool_call_copy)
    return tool_calls


@chain
def tool_executor_with_injection(ai_msg):
    tool_messages = []
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"] # 使用工具调用的 ID
        if tool_name not in tool_map:
            print(f"Error: Tool {tool_name} not found.")
            continue
        tool = tool_map[tool_name]
        tool_args["user_id"] = user_id
        tool_output = tool.invoke(tool_args)
        tool_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_id,
                name=tool_name,
            )
        )
    return tool_messages
        

new_args = inject_user_id.invoke(ai_msg)
print(f'inject_user_id:{new_args}')

tool_map = {tool.name: tool for tool in tools}

@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]

# chain = llm_with_tools | inject_user_id | tool_router.map()
chain = llm_with_tools | tool_executor_with_injection

print("--通过chain实现：2、注入新参数user_id；3、直接调用tool生成结果；4、调用LLM，生成流畅的答案---")

result = chain.invoke(query)
print(f'chain.invoke:{result}')
print(f'now user_to_pets :{user_to_pets}')





