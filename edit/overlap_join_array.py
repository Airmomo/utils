#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 功能描述：
#   该程序用于将一组字符串通过最大重叠部分进行连接。如果字符串之间没有重叠部分，则直接连接。
# 注意事项：
#   输入的字符串数组不能为空，且每个字符串应为有效的UTF-8编码。
# 安装依赖：
#   无
# 运行示例：
#   详见 "__main__" 部分
#
# 作者: Airmomo（https://github.com/Airmomo）
#
# 创建日期: 2024.10.30
# 最后修改日期: 2024.10.31

def find_max_overlap(s1, s2):
    """找到两个字符串的最大重叠长度，返回重叠的长度"""
    max_overlap = 0
    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i:] == s2[:i]:
            max_overlap = i
    return max_overlap

def overlap_join_array(strings):
    """将数组中的字符串通过最大重叠部分连接起来，如果没有重叠则直接连接"""
    if not strings:  # 如果输入数组为空，返回空字符串
        return ""
    
    result = strings[0]  # 初始化结果为数组的第一个字符串
    for i in range(1, len(strings)):
        overlap_length = find_max_overlap(result, strings[i])  # 计算当前结果与下一个字符串的重叠长度
        if overlap_length > 0:  # 如果有重叠，按重叠部分连接
            result += strings[i][overlap_length:]
        else:  # 如果没有重叠，直接连接
            result += strings[i]
    
    return result

if __name__ == "__main__":
    # 示例使用
    strings = ["hello", "loworld", "worldwide", "wideweb"]
    result = overlap_join_array(strings)
    print(result)  # 输出: "helloworldwideweb"

    # 另一个示例，没有重叠部分的情况
    strings = ["hello", "abc", "def", "wideweb"]
    result = overlap_join_array(strings)
    print(result)  # 输出: "helloabcdefwideweb"
