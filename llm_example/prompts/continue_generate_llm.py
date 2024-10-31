#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 功能描述：
#   该程序的目的是处理模型输出长文本续写的情况。具体地说，代码通过循环的方式，确保生成的文本即使在中途被截断，也能继续生成剩余的部分，并在最后将所有的片段拼接在一起。它还处理了在生成文本末尾添加特定符号（<EOF>）来判断是否生成完毕。
# 注意事项：
#   输入的字符串数组不能为空，且每个字符串应为有效的UTF-8编码。
# 安装依赖：
#   pip install openai
# 运行示例：
#   详见 "__main__" 部分
#
# 作者: Airmomo（https://github.com/Airmomo）
#
# 创建日期: 2024.10.31
# 最后修改日期: 2024.10.31

from openai import OpenAI
from typing import List, Optional

client = OpenAI(api_key="", base_url="")

ENDING_SYMBOL = "<EOF>" # 表示结束生成的特定符号
SYSTEM_PROMPT = f"If you have finished your response, please add the identifier {ENDING_SYMBOL} at the end."
CONTINUE_GENERATE_PROMPT_TEMPLATE = 'Your response got cut off, because you only have limited response space. Continue writing exactly where you left off. Do not repeat yourself. Start your response with: "{continue_location_text}", though use backticks where appropriate.'
OVERLAY_LENGTH = 50  # 最大重叠长度
MAX_RETRIES = 3  # 最大重试次数（同限制最大续写次数：用于限制模型的最大生成长度，计算公式是 `MAX_GENERATE_TOKENS = MODEL_MAX_OUTPUT_TOKENS * MAX_RETRIES`）


class ContinueGenerateLLM():

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str
    ):
        self.client = client
        self.model = model
        self.system_prompt = SYSTEM_PROMPT
        self.answer_history: List[str] = []

    def set_system_prompt(self, prompt: str):
        self.system_prompt = "\n".join([prompt, self.system_prompt])

    def get_answers(self, user_input: str, chat_history: Optional[list] = None) -> List[str]:
        if chat_history is None:
            chat_history = [
                {"role": "system", "content": f"{self.system_prompt}"}
            ]
        
        retries = 0
        while retries <= MAX_RETRIES:
            try:
                user_input_message = {"role": "user", "content": f"{user_input}"}
                chat_history.append(user_input_message)
                
                # 将聊天完成请求发送给模型
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=1.0,
                    messages=chat_history
                )
                
                ai_answer = chat_completion.choices[0].message.content
                ai_message = {"role": "assistant", "content": f"{ai_answer}"}
                ai_message_overlap_part = ai_message["content"][-OVERLAY_LENGTH:]

                # 检查答案是否已完成
                finished = ENDING_SYMBOL in ai_answer
                ai_answer = ai_answer.removesuffix(ENDING_SYMBOL)
                self.answer_history.append(ai_answer)

                # 无论是否完成，都将当前的ai_message加入chat_history
                chat_history.append(ai_message)

                if not finished:
                    # 添加最后的重叠内容以提示继续生成
                    user_input = CONTINUE_GENERATE_PROMPT_TEMPLATE.format(continue_location_text=ai_message_overlap_part)
                    retries += 1  # 增加重试次数
                else:
                    break

            except Exception as e:
                if retries >= MAX_RETRIES:
                    raise Exception(f"最大重试次数已达到，上次发生的错误: {str(e)}")
                print(f"发生了错误 {str(e)}. 准备重试... ({retries}/{MAX_RETRIES})")
                retries += 1
        
        # 增加日志打印输出 chat_history，在返回结果前
        print("Final Chat History:", chat_history)
        
        # 返回多次生成的答案数组
        results = self.answer_history
        self.answer_history = []
        return results
    
    def get_answer(self, user_input: str, chat_history: Optional[list] = None) -> str:
        results = self.get_answers(user_input, chat_history)
        # 使用连接重叠字符串数组的方法
        from edit.overlap_join_array import overlap_join_array
        return overlap_join_array(results)


if __name__ == "__main__":
    # 使用自定义基本URL和API密钥初始化OpenAI客户端
    client = OpenAI(base_url="https://api.********.com/v1",
                    api_key="sk-************************")

    # 使用客户端和模型创建ContinueGenerateLLM实例
    llm = ContinueGenerateLLM(client=client, model="model_name")

    # 从文件中读取长文本内容
    with open("llm_example/test/long_content.md", "r") as f:
        content = f.read()
    
    # 请求从模型中获取答案（翻译长文本）
    answer = llm.get_answer(content + "\n translate to chinese")

    # 将answer保存到本地
    output_file = "llm_example/test/output.md"
    with open(output_file, "w") as f:
        f.write(answer)

    print(f"翻译后的内容已保存到 {output_file} 文件中")