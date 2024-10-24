#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 功能描述：
#   该程序用于合并两个文本文件中的字符串（或两个字符串），它会在两者的重叠部分进行连接。
# 注意事项：
#   确保输入文件的编码为UTF-8。
# 安装依赖：
#   无
# 运行示例：
#   参见 "__main__" 部分
#
# 作者: Airmomo（https://github.com/Airmomo）
#
# 创建日期: 2024.10.24
# 最后修改日期: 2024.10.24


def merge_from_strings(str1, str2):
    """
    合并两个字符串，并在重叠部分进行连接。

    参数:
    str1 (str): 第一个字符串
    str2 (str): 第二个字符串

    返回:
    str: 合并后的字符串

    示例:
    merge_from_strings("abc", "cde") 将返回 "abcde"
    merge_from_strings("abc", "bcd") 将返回 "abcd"
    """
    # 重叠合并逻辑
    intersection = ""
    for i in range(min(len(str1), len(str2)), 0, -1):
        if str1[-i:] == str2[:i]:
            intersection = str1[-i:]
            break

    if intersection:
        str1 = str1[:-len(intersection)]
    return str1 + str2

def read_strings_from_file(file_path1, file_path2):
    """
    从两个不同的文件中分别读取整个文件的内容。

    参数:
    file_path1 (str): 第一个文件的路径
    file_path2 (str): 第二个文件的路径

    返回:
    tuple: 包含两个字符串的元组
    """
    with open(file_path1, 'r', encoding='utf-8') as file1:
        str1 = file1.read().strip()
    
    with open(file_path2, 'r', encoding='utf-8') as file2:
        str2 = file2.read().strip()
    
    return str1, str2

# 新增函数，整合读取文件和合并操作
def merge_from_files(file_path1, file_path2):
    """
    从两个文件读取内容并进行合并，返回合并后的字符串。

    参数:
    file_path1 (str): 第一个文件的路径
    file_path2 (str): 第二个文件的路径

    返回:
    str: 合并后的字符串
    """
    str1, str2 = read_strings_from_file(file_path1, file_path2)
    merged_string = merge_from_strings(str1, str2)
    return merged_string

if __name__ == "__main__":
    # 从两个文件读取字符串并合并
    file_path1 = 'edit/test/input1.txt'
    file_path2 = 'edit/test/input2.txt'
    
    file_str1, file_str2 = read_strings_from_file(file_path1, file_path2)
    merged_result = merge_from_files(file_path1, file_path2)

    # 简单的测试用例
    test_cases = [
        ("abc", "bcd", "abcd"),
        ("abc", "def", "abcdef"),
        (file_str1, file_str2, merged_result),  # 使用从文件读取并合并的结果
    ]

    for str1, str2, expected in test_cases:
        result = merge_from_strings(str1, str2)
        print(f"merge_from_strings('{str1}', '{str2}') => '{result}' (Expected: '{expected}')")
        assert result == expected, "Test case failed!"

        # 将结果保存到 merge_result.txt 文件
        with open('edit/test/text_merge_result.txt', 'w', encoding='utf-8') as result_file:
            result_file.write(result)

