#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 功能描述：
#   解析Markdown内容并转换为Jupyter Notebook格式。
# 注意事项：
#   这个脚本假设Markdown文件中的代码块是正确格式化的，即代码块内的缩进是合理的。如果代码块本身就有不正确的缩进，那么这个脚本可能无法正确处理。
# 安装依赖：
#   pip intsall nbformat
# 运行示例：
#   详见 "__main__" 部分;
#   python md_to_ipynb.py [markdown_file.md] [notebook.ipynb]
#
# 作者: Airmomo（https://github.com/Airmomo）
#
# 创建日期: 2024.10.14
# 最后修改日期: 2024.10.18


import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def parse_markdown_to_notebook(md_content):
    """
    解析Markdown内容并转换为Jupyter Notebook格式。
    """
    notebook = new_notebook()
    lines = md_content.split('\n')
    in_code_block = False
    code_block = []
    markdown_block = []

    for line in lines:
        if line.startswith('```'):
            if in_code_block:
                # 结束代码块
                notebook.cells.append(new_code_cell('\n'.join(code_block)))
                code_block = []
                in_code_block = False
            else:
                # 开始新的代码块
                in_code_block = True
                if markdown_block:
                    notebook.cells.append(new_markdown_cell('\n'.join(markdown_block)))
                    markdown_block = []
        elif in_code_block:
            code_block.append(line)
        else:
            markdown_block.append(line)

    # 处理最后的Markdown块
    if markdown_block:
        notebook.cells.append(new_markdown_cell('\n'.join(markdown_block)))

    return notebook

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python md_to_ipynb.py <input_md_file> <output_ipynb_file>")
    else:
        input_md_file = sys.argv[1]
        output_ipynb_file = sys.argv[2]
        # 读取Markdown文件
        with open(input_md_file, 'r', encoding='utf-8') as file:
            md_content = file.read()
        # 解析并转换为Notebook
        notebook = parse_markdown_to_notebook(md_content)
        # 保存为Jupyter Notebook文件
        with open(output_ipynb_file, 'w', encoding='utf-8') as file:
            nbformat.write(notebook, file)