import os
import math
import random


def load_folder_specifyFile(folder_path, file_type):
    """
    遍历指定文件夹及其所有子文件夹，获取所有指定后缀的文件路径。

    Args:
        folder_path (str): 目标文件夹的路径。
        file_type (str): 指定的文件后缀（如 '.ply', '.npy' 等）。

    Returns:
        list: 包含所有指定后缀文件路径的列表。

    Raises:
        ValueError: 如果文件夹路径不存在或输入参数格式不正确。
    """
    # 参数校验
    if not isinstance(folder_path, str) or not isinstance(file_type, str):
        raise ValueError("folder_path 和 file_type 必须是字符串类型")

    if not os.path.exists(folder_path):
        raise ValueError(f"文件夹路径 {folder_path} 不存在！")

    if not file_type.startswith('.'):
        raise ValueError("file_type 必须以 '.' 开头，例如 '.ply' 或 '.npy'")

    # 标准化文件后缀为小写
    file_type = file_type.lower()

    specify_files_list = []

    # 使用 os.walk() 递归遍历所有子文件夹
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            # 获取文件扩展名并标准化为小写
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            # 仅处理指定后缀的文件
            if ext == file_type:
                file_path = os.path.join(dirpath, filename)  # 获取完整文件路径
                specify_files_list.append(file_path)

    # 如果未找到符合条件的文件，打印警告信息
    if not specify_files_list:
        print(f"警告：文件夹 {folder_path} 中没有找到 {file_type} 文件！")

    return specify_files_list


def specifyFile_culling(files_list, exclude_substr=None, retain_substr=None, file_nums=None, seed=None):
    """
    文件筛选函数：根据排除条件和数量限制对文件列表进行筛选。

    Args:
        files_list (list of str): 输入的文件路径列表。
        exclude_substr (list of str, optional): 需要排除的子字符串列表。默认为None（不排除任何文件）。
        retain_substr (list of str, optional): 需要保留的子字符串列表。默认为None（保留所有排除后的文件）。exclude_substr的优先级更高。
        file_nums (int, optional): 筛选后保留的文件数量。默认为None（保留所有符合条件的文件）。
        seed (int, optional): 随机数种子，用于结果可复现。默认为None（不固定随机性）。

    Returns:
        list of str: 筛选后的文件路径列表。

    Raises:
        ValueError: 如果输入参数类型不正确或逻辑错误。
    """
    # 参数校验
    if not isinstance(files_list, list) or not all(isinstance(f, str) for f in files_list):
        raise ValueError("files_list 必须是一个字符串列表")

    # 处理exclude_substr
    if exclude_substr is None:
        exclude_substr = []
    elif not isinstance(exclude_substr, list) or not all(isinstance(s, str) for s in exclude_substr):
        raise ValueError("exclude_substr 必须是一个字符串列表")

    # 处理retain_substr
    if retain_substr is not None:
        if not isinstance(retain_substr, list) or not all(isinstance(s, str) for s in retain_substr):
            raise ValueError("retain_substr 必须是一个字符串列表")
        if len(retain_substr) == 0:
            raise ValueError("retain_substr 不能是空列表")

    if file_nums is not None and (not isinstance(file_nums, int) or file_nums <= 0):
        raise ValueError("file_nums 必须是正整数")

    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed 必须是整数或None")

    # 排除包含指定子字符串的文件
    filtered_files = [
        file for file in files_list
        if not any(substring in file for substring in exclude_substr)
    ]

    # 应用保留条件
    if retain_substr is not None:
        filtered_files = [
            file for file in filtered_files
            if any(substring in file for substring in retain_substr)
        ]

    # 如果需要随机采样
    if file_nums is not None:
        if len(filtered_files) < file_nums:
            raise ValueError(f"筛选后的文件数量({len(filtered_files)})小于请求的数量({file_nums})")

        # 设置随机种子以确保结果可复现
        if seed is not None:
            random.seed(seed)

        selected_files = random.sample(filtered_files, k=file_nums)
    else:
        selected_files = filtered_files

    return selected_files


def paths_load_and_divide(divide_folder_path: str, percent_need: float, file_type: str, seed: int = None):
    """
    从指定文件夹中加载特定类型文件，并按比例随机分割为两个列表，可设置随机种子以保证划分可复现。

    Args:
        divide_folder_path (str): 需要处理的文件夹路径。
        percent_need (float): 需要提取的文件占比（0 < percent_need ≤ 1）。
        file_type (str): 文件后缀名（例如 ".jpg"）。
        seed (int, 可选): 随机种子，若指定则划分结果固定。

    Returns:
        tuple[list, list]:
            - files_need: 随机选取的指定比例的文件路径列表（向上取整确保至少1个文件）。
            - files_other: 剩余未被选中的文件路径列表。

    Notes:
        1. 使用 math.ceil 确保即使比例不足1个文件时也会返回至少1个文件。
        2. 若设置了 seed，则划分结果可复现。
        3. 通过随机索引实现无放回抽样，保证两个列表无重复文件。
        4. 若文件夹中无匹配文件，返回两个空列表。
    """
    # 检查比例参数是否合理
    if percent_need <= 0 or percent_need > 1:
        raise ValueError(f"percent_need 必须在 (0, 1] 范围内，当前值: {percent_need}")

    if seed is not None:
        random.seed(seed)  # 设置随机种子

    # 固定文件顺序以保证跨平台一致
    paths_list = sorted(
        [os.path.join(divide_folder_path, f) for f in os.listdir(divide_folder_path) if f.endswith(file_type)]
    )
    # 若文件夹中无匹配文件，返回空列表
    if not paths_list:
        return [], []
    # 计算需要选取的文件数，向上取整保证至少一个文件
    size_need = math.ceil(percent_need * len(paths_list))
    # 随机选择指定数量的索引，无放回抽样
    random_indices = random.sample(range(len(paths_list)), size_need)
    # 生成选中与未选中文件列表
    files_need = [paths_list[i] for i in random_indices]
    files_other = [paths_list[i] for i in range(len(paths_list)) if i not in random_indices]

    # 返回划分结果
    return files_need, files_other
