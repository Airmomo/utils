#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 功能描述：
#   该程序用于合并指定目录下所有的Markdown（.md）文件，并将它们的内容按顺序连接在一起，生成一个新的文件。支持自定义输入目录和输出文件路径。
# 注意事项：
#   确保指定目录下的.md文件编码为UTF-8，并且有足够的权限进行读写操作。
# 安装依赖：
#   无
# 运行示例：
#   运行该脚本时，可以指定输入目录和输出文件路径，例如：concatenate_md_files('path/to/directory', 'path/to/output.md')
#
# 作者: Airmomo（https://github.com/Airmomo）
#
# 创建日期: 2024.10.26
# 最后修改日期: 2024.10.26

import os

def concatenate_md_files(input_directory='.', output_file='output.md'):
    # 获取指定目录下所有的.md文件
    md_files = [f for f in os.listdir(input_directory) if f.endswith('.md')]
    
    # 初始化一个空字符串用于存储所有文件的内容
    concatenated_content = ''
    
    # 遍历文件列表，读取内容并添加到concatenated_content中
    for file in md_files:
        file_path = os.path.join(input_directory, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            concatenated_content += f.read() + '\n'
    
    # 将合并后的内容写入到指定的输出文件中
    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.write(concatenated_content)

# 调用示例
concatenate_md_files('edit/test', 'edit/test/merged_output.md')
