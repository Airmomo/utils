# utils

记录一些自己平时在开发过程中编写的小工具，每个工具都提供了一个可运行的示例，详见对应工具的代码文件。

# 目录

## 转换类型

- [解析 Markdown 文件内容转换为 Jupyter Notebook 格式文件](./convert/md_to_ipynb.py)

## 编辑类型

- [连接指定目录下所有的 Markdown（.md）文件](./edit/concatenate_md_files.py)
- [连接两个文本文件中的字符串（或两个字符串），在它们的最大重叠部分进行连接](./edit/text_merge.py)
- [连接一组字符串，从俩俩之间的最大重叠部分开始连接。如果字符串之间没有重叠部分，则直接连接](./edit/overlap_join_array.py)

## 大模型应用开发示例

- [自定义 Chroma 向量数据库的嵌入方法：以替换智谱 AI 的嵌入模型 embeding-3 为例](./llm_example/embeding_functions/zhipu_embeding_function.py)
