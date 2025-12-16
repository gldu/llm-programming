import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict


class HelloLLMAgent:

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
    
        if not all([self.model,apiKey,baseUrl]):
            raise ValueError("模型ID和服务地址必须被提供或在.env文件中定义。")
        
        self.clent = OpenAI(api_key=apiKey,base_url=baseUrl,timeout=timeout)
    
    def thinking(self,messages:List[dict[str,str]],temperature:float=0)->str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.clent.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "\n".join(collected_content)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

if __name__ == '__main__':
    try:
        llmClient = HelloLLMAgent(
            model="qwen3:4b",
            baseUrl="http://localhost:11434/v1",
            apiKey="ollama"
            )
        exampleMessages = [
                {"role": "system", "content": "You are a helpful assistant that writes Python code."},
                {"role": "user", "content": "写一个快速排序算法"}
            ]
        print("--- 调用LLM ---")
        responseText = llmClient.thinking(exampleMessages)
        if responseText:
                print("\n\n--- 完整模型响应 ---")
                print(responseText)
    except ValueError as e:
        print(e)

