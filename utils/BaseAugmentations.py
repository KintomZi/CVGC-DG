import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm


class PointAugmentations:
    @staticmethod
    def shuffle_indices():
        pass


class CoordsAugmentations:
    """
    三维坐标数据增强类
    提供平移、旋转、缩放、翻转、覆盖等几何变换方法，用于点云数据预处理与增强。
    """

    @staticmethod
    def _rotation_matrix(axis_vector, rotation_angle):
        """
        根据旋转轴和旋转角度生成旋转矩阵（Rodrigues公式）。

        Args:
            axis_vector (ndarray): 旋转轴向量，形状为 (3,)
            rotation_angle (float): 旋转角度（弧度）

        Returns:
            ndarray: 旋转矩阵 (3x3)
        """
        unit_axis = axis_vector / norm(axis_vector)  # 单位化旋转轴
        return expm(np.cross(np.eye(3), unit_axis * rotation_angle))  # 旋转矩阵

    @staticmethod
    def translate(coords, max_translation):
        """
        对三维坐标进行随机平移变换。

        Args:
            coords (ndarray): 输入坐标，形状为 (N, 3)
            max_translation (ndarray): 各轴最大平移量 (3,)

        Returns:
            ndarray: 平移后的坐标
        """
        translation_vector = np.random.uniform(0, 1, 3) * max_translation
        return coords + translation_vector

    @staticmethod
    def rotate(coords, rotation_bounds=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))):
        """
        对三维坐标进行随机旋转。

        Args:
            coords (ndarray): 输入坐标，形状为 (N, 3)
            rotation_bounds (tuple[tuple[float, float]]): 每个轴的旋转角度范围 (弧度)

        Returns:
            ndarray: 旋转后的坐标
        """
        rotation_matrices = []
        for axis_idx, angle_range in enumerate(rotation_bounds):
            axis = np.zeros(3, dtype=np.float64)
            axis[axis_idx] = 1.0  # 设置当前旋转轴
            angle = 0.0
            if angle_range is not None:
                angle = np.random.uniform(*angle_range)  # 随机旋转角
            rotation_matrices.append(CoordsAugmentations._rotation_matrix(axis, angle))

        np.random.shuffle(rotation_matrices)  # 打乱旋转顺序
        composite_rotation = rotation_matrices[0] @ rotation_matrices[1] @ rotation_matrices[2]
        return coords @ composite_rotation

    @staticmethod
    def scale(coords, scale_range=(0.8, 1.25), anisotropic=False):
        """
        对三维坐标进行缩放，可选择等比缩放或各向异性缩放。

        Args:
            coords (ndarray): 输入坐标，形状为 (N, 3)
            scale_range (tuple[float, float]): 缩放系数范围
            anisotropic (bool): 各向异性开关
                - False: 三轴统一缩放
                - True: 各轴独立缩放

        Returns:
            ndarray: 缩放后的坐标
        """
        if anisotropic:
            # 各轴独立缩放，分别采样缩放系数
            scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=3)  # (3,)
        else:
            # 三轴统一缩放，采样一个缩放系数
            scale_factors = np.random.uniform(scale_range[0], scale_range[1])  # 标量

        return coords * scale_factors

    @staticmethod
    def flip_xy(coords, flip_prob=0.5):
        """
        随机翻转(镜像)点云的 X 轴或 Y 轴。

        Args:
            coords (ndarray): 输入坐标，形状为 (N, 3)
            flip_prob (float): 执行翻转的概率

        Returns:
            ndarray: 翻转后的坐标
        """
        if np.random.rand() < flip_prob:
            if np.random.rand() > 0.5:
                coords[:, 0] = -coords[:, 0]  # 翻转 X 轴
            else:
                coords[:, 1] = -coords[:, 1]  # 翻转 Y 轴
        return coords

    @staticmethod
    def cover(coords, cover_ratio=0.5):
        """
        随机将一部分点覆盖（置零），模拟遮挡或缺失。

        Args:
            coords (ndarray): 输入坐标，形状为 (N, 3)
            cover_ratio (float): 需要覆盖的比例

        Returns:
            tuple:
                - ndarray: 覆盖后的坐标
                - ndarray: 未被覆盖点的索引
        """
        num_cover_points = int(coords.shape[0] * cover_ratio)
        covered_indices = np.random.choice(coords.shape[0], size=num_cover_points, replace=False)
        coords[covered_indices] = 0  # 将选中的点置为 (0, 0, 0)
        remaining_indices = np.setdiff1d(np.arange(coords.shape[0]), covered_indices)
        return coords, remaining_indices
