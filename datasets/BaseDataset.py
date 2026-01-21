import warnings
import numpy as np  # 导入NumPy库，进行数组操作
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from utils.BaseAugmentations import CoordsAugmentations as CoordsAugs


class PointCloudMethods:
    # 本身没有任何实际作用，单纯用来存储针对点云的方法
    # It has no practical effect in itself, and is simply used to store methods for point clouds.

    @staticmethod  # 定义为静态方法
    def coords_initialize(coords, method_init):
        if method_init == -1:
            coords_center = coords.min(0, keepdims=True)
        elif method_init == 0:  # 3D质心(baseline默认)
            coords_center = coords.mean(0, keepdims=True)
        elif method_init == 1:  # 2D质心
            coords_center = coords.mean(0, keepdims=True)
            coords_center[0, 2] = 0  # 将中心的Z坐标设为0,只改变平面坐标
        elif method_init == 2:  # 3D形心
            coords_center = (coords.max(0, keepdims=True) + coords.min(0, keepdims=True)) / 2
        elif method_init == 3:  # 2D形心
            coords_center = (coords.max(0, keepdims=True) + coords.min(0, keepdims=True)) / 2
            coords_center[0, 2] = 0  # 将中心的Z坐标设为0,只改变平面坐标
        else:
            raise NotImplementedError
        return coords - coords_center  # 计算标准化坐标

    @staticmethod  # 定义为静态方法
    def feature_initialize(*arrays):
        """
        将多个 NumPy 数组按列拼接，支持一维和二维数组。

        Args:
            *arrays: 任意数量的 NumPy 数组，可以是 (n,) 或 (n, d)
        Returns:
            拼接后的 NumPy 数组，形状为 (n, sum(d_i))
        Raises:
            ValueError: 如果输入为空、非 NumPy 数组、行数不一致或包含零维数组
        """
        if not arrays:
            raise ValueError("至少需要一个输入数组")

        # 检查输入是否为 NumPy 数组
        for arr in arrays:
            if not isinstance(arr, np.ndarray):
                raise ValueError("所有输入必须是 NumPy 数组")

        processed_arrays = []
        for arr in arrays:
            if len(arr.shape) == 1:

                processed_arrays.append(arr.reshape(-1, 1))  # 一维数组转为二维 (n, 1)
            elif len(arr.shape) == 2:

                processed_arrays.append(arr)  # 二维数组保留原样
            else:
                raise ValueError(f"输入数组必须是 1D 或 2D 的，但得到的形状为 {arr.shape}")

        # 检查所有数组的行数是否一致
        row_counts = [arr.shape[0] for arr in processed_arrays]
        if not all(count == row_counts[0] for count in row_counts):
            raise ValueError(f"所有输入数组的行数必须一致，但得到的行数为: {row_counts}")

        return np.concatenate(processed_arrays, axis=1)  # 按列拼接

    @staticmethod  # 定义为静态方法
    def label_initialize(label, label2name_dict, ignore_label, output_class_dim, fill_value=-1):
        """转换标签以进行损失计算.

        Args:
            label: Array-like, original label data.
            label2name_dict: Dict mapping labels to names.
            ignore_label: Int, list, or tuple of labels to ignore.
            output_class_dim: Int, number of output classes.
            fill_value: Int, value to fill ignored labels with (default: -1).

        Returns:
            Tuple containing:
            - converted_label: numpy.ndarray, converted label array.
            - keep_indices: tuple, indices of kept positions.
            - origin2updated: dict, mapping from original to new labels.

        Raises:
            ValueError: If labels are not defined in dict or dimension mismatch.
        """
        # 获取所有有效的标签类
        valid_label_classes = [key for key in label2name_dict.keys()]
        # 获取输入标签中的唯一值
        unique_labels_in_data = np.unique(label)

        # 验证数据中的标签是否都在字典中定义
        if not set(unique_labels_in_data).issubset(set(valid_label_classes)):
            invalid_labels = set(unique_labels_in_data) - set(valid_label_classes)
            raise ValueError(f"发现未在label2name_dict中定义的标签: {invalid_labels}")

        # 确保ignore_label是列表格式
        if not isinstance(ignore_label, (list, tuple)):
            ignore_label = list(ignore_label)

        # 计算有效标签（即不包含忽略标签的标签）
        effective_labels = list(set(valid_label_classes) - set(ignore_label))
        effective_labels.sort()  # 排序以确保一致性

        # 检查有效标签数量是否等于输出类别维度
        if len(effective_labels) == output_class_dim:
            # 创建标签映射字典
            origin2updated = {effective_labels[i]: i for i in range(len(effective_labels))}
            # 转换标签
            label_array = np.array(label)

            # 初始化转换后的标签数组，使用fill_value填充
            converted_label = np.full_like(label_array, fill_value)

            # 只对有效标签进行转换
            for old_label, new_label in origin2updated.items():
                converted_label[label_array == old_label] = new_label

            return converted_label, origin2updated
        else:
            raise ValueError(f"有效标签数量 {len(effective_labels)} ({effective_labels}) "
                             f"不等于输出类别维度 {output_class_dim}。")

    @staticmethod  # 定义为静态方法
    def augs_xyz(coords, from_paper='CosMix'):
        if from_paper == 'Custom':
            coords = CoordsAugs.translate(coords, max_translation=5)
            coords = CoordsAugs.rotate(coords, rotation_bounds=((0, 0), (0, 0), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        elif from_paper == 'Mix3D':
            # https://github.com/kumuji/mix3d/blob/9015e138cf64ece5aa392be173b8fb9d5afe9573/mix3d/conf/augmentation/volumentations_aug.yaml
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (-np.pi / 24, np.pi / 24), (-np.pi / 24, np.pi / 24), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.90, 1.10))
        elif from_paper == 'GrowSP':
            coords = CoordsAugs.translate(coords, max_translation=50)
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.90, 1.10))
        elif from_paper == 'CosMix':
            # https://github.com/saltoricristiano/cosmix-uda/blob/07bb2a64d5341f32e41e1e81343887b3c134430c/utils/datasets/dataset.py#L51
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20)))
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        elif from_paper == 'PolarMix':
            # https://github.com/xiaoaoran/polarmix/blob/main/core/datasets/semantic_kitti_polarmix.py#L217
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (0, 0), (0, 0), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        elif from_paper == 'DGLSS':
            coords = CoordsAugs.rotate(coords, rotation_bounds=((0, 0), (0, 0), (-np.pi, np.pi)))
            coords = CoordsAugs.flip_xy(coords, flip_prob=0.7)
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        else:
            warnings.warn('Basic Data Augment开启，但未设置参数')
        return coords  # 返回增强后的坐标

    @staticmethod  # 定义为静态方法
    def mix_scene(data_ori, data_mix, mask_ori=None, mask_mix=None):
        """
        按可选掩码混合两个点云数据字典，并进行安全检查与深拷贝（支持字段差异，只取交集字段）。

        Args:
            data_ori (dict): 原始点云数据字典 {字段名: ndarray}
            data_mix (dict): 混合点云数据字典 {字段名: ndarray}
            mask_ori (ndarray[bool], optional): 原始点云掩码
            mask_mix (ndarray[bool], optional): 混合点云掩码

        Returns:
            dict: 混合后的点云数据字典（仅保留两者共同字段）
        """
        # === 安全检查 1: 字段交集与差异字段提示 ===
        common_keys = set(data_ori.keys()) & set(data_mix.keys())  # 交集字段
        diff_keys_ori = set(data_ori.keys()) - common_keys  # 仅在 data_ori 中的字段
        diff_keys_mix = set(data_mix.keys()) - common_keys  # 仅在 data_mix 中的字段

        if diff_keys_ori:
            warnings.warn(f"以下字段仅存在于 data_ori，将被忽略: {diff_keys_ori}", category=UserWarning)
        if diff_keys_mix:
            warnings.warn(f"以下字段仅存在于 data_mix，将被忽略: {diff_keys_mix}", category=UserWarning)

        if not common_keys:
            raise ValueError("data_ori 与 data_mix 没有共同字段，无法混合")

        # === 安全检查 2: 掩码长度一致性 ===
        ori_len = next(iter(data_ori.values())).shape[0]
        mix_len = next(iter(data_mix.values())).shape[0]
        if mask_ori is not None and mask_ori.shape[0] != ori_len:
            raise ValueError(f"mask_ori 长度 {mask_ori.shape[0]} 与 data_ori 点数 {ori_len} 不一致")
        if mask_mix is not None and mask_mix.shape[0] != mix_len:
            raise ValueError(f"mask_mix 长度 {mask_mix.shape[0]} 与 data_mix 点数 {mix_len} 不一致")

        # === 过滤 + 深拷贝（只处理交集字段） ===
        if mask_ori is None:
            selected_ori = {k: data_ori[k].copy() for k in common_keys}
        else:
            selected_ori = {k: data_ori[k][mask_ori].copy() for k in common_keys}

        if mask_mix is None:
            selected_mix = {k: data_mix[k].copy() for k in common_keys}
        else:
            selected_mix = {k: data_mix[k][mask_mix].copy() for k in common_keys}

        # === 拼接 ===
        result_mix = {
            k: np.concatenate((selected_ori[k], selected_mix[k]), axis=0)
            for k in common_keys
        }

        return result_mix

    @staticmethod  # 定义为静态方法
    def voxelize(coords, features, voxel_size):
        """Minkowski voxelization"""

        # 添加断言确保 voxel_size > 0
        assert voxel_size > 0, "体素大小必须为正"

        # 计算缩放因子：将坐标缩放到体素坐标系
        # scale = 1 / voxel_size，用于将真实坐标转换为体素坐标
        scale = 1 / voxel_size

        # === 坐标量化阶段 ===
        # 将连续坐标按体素大小进行量化
        # coords * scale: 将坐标缩放到体素坐标系
        # np.floor(): 向下取整，将坐标量化到最近的体素中心
        coords_int = np.floor(coords * scale)

        # 提取原始特征，体素化过程中特征会被聚合
        coordsV, featsV, unique_map, inverse_map = ME.utils.sparse_quantize(np.ascontiguousarray(coords_int),
                                                                            features=features,
                                                                            ignore_label=-1,
                                                                            return_index=True,
                                                                            return_inverse=True)
        # 全部体素化并保留逆向索引
        return coordsV, featsV, unique_map, inverse_map


class GeneralDataset(Dataset):
    def __init__(self, dataset_path: list, mode, transforms=None):
        self.mode = mode
        self.transforms = transforms

        self.execute_file = dataset_path

    def __len__(self):
        return len(self.execute_file)  # 返回文件数量

    def __getitem__(self, idx):
        # 需要在子类中重构
        raise NotImplementedError
