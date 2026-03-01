import os
import math
import random


def load_folder_specifyFile(folder_path, file_type):
    """
    Recursively traverse the specified folder and all subfolders to retrieve 
    paths of all files with the specified suffix.

    Args:
        folder_path (str): Path to the target folder.
        file_type (str): Specified file suffix (e.g., '.ply', '.npy', etc.).

    Returns:
        list: List containing paths of all files with the specified suffix.

    Raises:
        ValueError: If the folder path does not exist or input parameter formats are incorrect.
    """
    if not isinstance(folder_path, str) or not isinstance(file_type, str):
        raise ValueError("folder_path and file_type must be of string type")

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path {folder_path} does not exist!")

    if not file_type.startswith('.'):
        raise ValueError("file_type must start with '.', e.g., '.ply' or '.npy'")

    file_type = file_type.lower()
    specify_files_list = []

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext == file_type:
                file_path = os.path.join(dirpath, filename)  
                specify_files_list.append(file_path)

    if not specify_files_list:
        print(f"Warning: No {file_type} files found in folder {folder_path}!")

    return specify_files_list


def specifyFile_culling(files_list, exclude_substr=None, retain_substr=None, file_nums=None, seed=None):
    """
    File filtering function: Filters a file list based on exclusion criteria and quantity limits.

    Args:
        files_list (list of str): Input list of file paths.
        exclude_substr (list of str, optional): List of substrings to exclude. 
            Default is None (exclude no files).
        retain_substr (list of str, optional): List of substrings to retain. 
            Default is None (retain all files after exclusion). 
            exclude_substr has higher priority.
        file_nums (int, optional): Number of files to retain after filtering. 
            Default is None (retain all qualifying files).
        seed (int, optional): Random seed for reproducibility. 
            Default is None (no fixed randomness).

    Returns:
        list of str: Filtered list of file paths.

    Raises:
        ValueError: If input parameter types are incorrect or logical errors occur.
    """
    if not isinstance(files_list, list) or not all(isinstance(f, str) for f in files_list):
        raise ValueError("files_list must be a list of strings")

    if exclude_substr is None:
        exclude_substr = []
    elif not isinstance(exclude_substr, list) or not all(isinstance(s, str) for s in exclude_substr):
        raise ValueError("exclude_substr must be a list of strings")

    if retain_substr is not None:
        if not isinstance(retain_substr, list) or not all(isinstance(s, str) for s in retain_substr):
            raise ValueError("retain_substr must be a list of strings")
        if len(retain_substr) == 0:
            raise ValueError("retain_substr cannot be an empty list")

    if file_nums is not None and (not isinstance(file_nums, int) or file_nums <= 0):
        raise ValueError("file_nums must be a positive integer")

    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer or None")

    filtered_files = [
        file for file in files_list
        if not any(substring in file for substring in exclude_substr)
    ]

    if retain_substr is not None:
        filtered_files = [
            file for file in filtered_files
            if any(substring in file for substring in retain_substr)
        ]

    if file_nums is not None:
        if len(filtered_files) < file_nums:
            raise ValueError(f"Number of filtered files ({len(filtered_files)}) is less than requested quantity ({file_nums})")

        if seed is not None:
            random.seed(seed)

        selected_files = random.sample(filtered_files, k=file_nums)
    else:
        selected_files = filtered_files

    return selected_files


def paths_load_and_divide(divide_folder_path: str, percent_need: float, file_type: str, seed: int = None):
    """
    Load specific type files from a specified folder and randomly split them into 
    two lists by proportion. Random seed can be set to ensure reproducible splitting.

    Args:
        divide_folder_path (str): Path to the folder to be processed.
        percent_need (float): Proportion of files to extract (0 < percent_need ≤ 1).
        file_type (str): File suffix (e.g., ".jpg").
        seed (int, optional): Random seed. If specified, the splitting result is fixed.

    Returns:
        tuple[list, list]:
            - files_need: List of file paths randomly selected based on the specified proportion 
              (rounded up to ensure at least 1 file).
            - files_other: List of remaining unselected file paths.

    Notes:
        1. Use math.ceil to ensure at least 1 file is returned even if the proportion is less than 1.
        2. If seed is set, the splitting result is reproducible.
        3. Implement sampling without replacement via random indices to ensure no duplicate files across the two lists.
        4. Return two empty lists if no matching files are found in the folder.
    """
    if percent_need <= 0 or percent_need > 1:
        raise ValueError(f"percent_need must be in the range (0, 1], current value: {percent_need}")

    if seed is not None:
        random.seed(seed)  

    paths_list = sorted(
        [os.path.join(divide_folder_path, f) for f in os.listdir(divide_folder_path) if f.endswith(file_type)]
    )

    if not paths_list:
        return [], []
    size_need = math.ceil(percent_need * len(paths_list))
    random_indices = random.sample(range(len(paths_list)), size_need)
    files_need = [paths_list[i] for i in random_indices]
    files_other = [paths_list[i] for i in range(len(paths_list)) if i not in random_indices]

    return files_need, files_other
