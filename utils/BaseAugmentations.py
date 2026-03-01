import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm


class PointAugmentations:
    @staticmethod
    def shuffle_indices():
        pass


class CoordsAugmentations:
    """
    3D Coordinate Data Augmentation Class.
    Provides geometric transformation methods such as translation, rotation, 
    scaling, flipping, and covering, used for point cloud data preprocessing and augmentation.
    """

    @staticmethod
    def _rotation_matrix(axis_vector, rotation_angle):
        """
        Generate a rotation matrix based on the rotation axis and angle (Rodrigues' formula).

        Args:
            axis_vector (ndarray): Rotation axis vector, shape (3,).
            rotation_angle (float): Rotation angle (radians).

        Returns:
            ndarray: Rotation matrix (3x3).
        """
        unit_axis = axis_vector / norm(axis_vector)  # Normalize rotation axis
        return expm(np.cross(np.eye(3), unit_axis * rotation_angle))  # Rotation matrix

    @staticmethod
    def translate(coords, max_translation):
        """
        Apply random translation transformation to 3D coordinates.

        Args:
            coords (ndarray): Input coordinates, shape (N, 3).
            max_translation (ndarray): Maximum translation amount per axis (3,).

        Returns:
            ndarray: Translated coordinates.
        """
        translation_vector = np.random.uniform(0, 1, 3) * max_translation
        return coords + translation_vector

    @staticmethod
    def rotate(coords, rotation_bounds=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))):
        """
        Apply random rotation to 3D coordinates.

        Args:
            coords (ndarray): Input coordinates, shape (N, 3).
            rotation_bounds (tuple[tuple[float, float]]): Rotation angle range for each axis (radians).

        Returns:
            ndarray: Rotated coordinates.
        """
        rotation_matrices = []
        for axis_idx, angle_range in enumerate(rotation_bounds):
            axis = np.zeros(3, dtype=np.float64)
            axis[axis_idx] = 1.0  # Set current rotation axis
            angle = 0.0
            if angle_range is not None:
                angle = np.random.uniform(*angle_range)  # Random rotation angle
            rotation_matrices.append(CoordsAugmentations._rotation_matrix(axis, angle))

        np.random.shuffle(rotation_matrices)  # Shuffle rotation order
        composite_rotation = rotation_matrices[0] @ rotation_matrices[1] @ rotation_matrices[2]
        return coords @ composite_rotation

    @staticmethod
    def scale(coords, scale_range=(0.8, 1.25), anisotropic=False):
        """
        Scale 3D coordinates, with option for isotropic or anisotropic scaling.

        Args:
            coords (ndarray): Input coordinates, shape (N, 3).
            scale_range (tuple[float, float]): Scaling factor range.
            anisotropic (bool): Anisotropic switch.
                - False: Uniform scaling across three axes.
                - True: Independent scaling per axis.

        Returns:
            ndarray: Scaled coordinates.
        """
        if anisotropic:
            # Independent scaling per axis, sample scaling factors separately
            scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=3)  # (3,)
        else:
            # Uniform scaling across three axes, sample a single scaling factor
            scale_factors = np.random.uniform(scale_range[0], scale_range[1])  # Scalar

        return coords * scale_factors

    @staticmethod
    def flip_xy(coords, flip_prob=0.5):
        """
        Randomly flip (mirror) the point cloud along the X or Y axis.

        Args:
            coords (ndarray): Input coordinates, shape (N, 3).
            flip_prob (float): Probability of performing flip.

        Returns:
            ndarray: Flipped coordinates.
        """
        if np.random.rand() < flip_prob:
            if np.random.rand() > 0.5:
                coords[:, 0] = -coords[:, 0]  # Flip X axis
            else:
                coords[:, 1] = -coords[:, 1]  # Flip Y axis
        return coords

    @staticmethod
    def cover(coords, cover_ratio=0.5):
        """
        Randomly cover (zero out) a portion of points to simulate occlusion or missing data.

        Args:
            coords (ndarray): Input coordinates, shape (N, 3).
            cover_ratio (float): Proportion to cover.

        Returns:
            tuple:
                - ndarray: Coordinates after covering.
                - ndarray: Indices of uncovered points.
        """
        num_cover_points = int(coords.shape[0] * cover_ratio)
        covered_indices = np.random.choice(coords.shape[0], size=num_cover_points, replace=False)
        coords[covered_indices] = 0  # Set selected points to (0, 0, 0)
        remaining_indices = np.setdiff1d(np.arange(coords.shape[0]), covered_indices)
        return coords, remaining_indices
