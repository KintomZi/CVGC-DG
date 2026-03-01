import warnings
import numpy as np 
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from utils.BaseAugmentations import CoordsAugmentations as CoordsAugs


class PointCloudMethods:
    # It has no practical effect in itself, and is simply used to store methods for point clouds.

    @staticmethod
    def coords_initialize(coords, method_init):
        """
        Normalize point cloud coordinates by subtracting a reference center point.

        Args:
            coords (ndarray): Input coordinates with shape (N, 3).
            method_init (int): Method for selecting the reference center:
                -1: Minimum value of 3D coordinates (min-x, min-y, min-z)
                 0: Centroid of 3D coordinates (mean of x, y, z) [Default baseline]
                 1: Centroid of XY plane only; Z-axis remains unchanged
                 2: 3D bounding box center ((max + min) / 2)
                 3: 2D bounding box center in XY plane; Z-axis remains unchanged

        Returns:
            ndarray: Normalized coordinates with reference center subtracted.

        Raises:
            NotImplementedError: If method_init is not supported.
        """
        if method_init == -1:
            coords_center = coords.min(0, keepdims=True)
        elif method_init == 0:  # 3D centroid (baseline default)
            coords_center = coords.mean(0, keepdims=True)
        elif method_init == 1:  # 2D centroid (XY plane only)
            coords_center = coords.mean(0, keepdims=True)
            coords_center[0, 2] = 0  # Set Z-coordinate of center to 0; only modify planar coordinates
        elif method_init == 2:  # 3D bounding box center
            coords_center = (coords.max(0, keepdims=True) + coords.min(0, keepdims=True)) / 2
        elif method_init == 3:  # 2D bounding box center (XY plane only)
            coords_center = (coords.max(0, keepdims=True) + coords.min(0, keepdims=True)) / 2
            coords_center[0, 2] = 0  # Set Z-coordinate of center to 0; only modify planar coordinates
        else:
            raise NotImplementedError
        return coords - coords_center  # Return normalized coordinates

    @staticmethod 
    def feature_initialize(*arrays):
        """
        Concatenate multiple NumPy arrays column-wise, supporting both 1D and 2D arrays.

        Args:
            *arrays: Variable number of NumPy arrays, each with shape (n,) or (n, d).

        Returns:
            ndarray: Concatenated array with shape (n, sum(d_i)).

        Raises:
            ValueError: If input is empty, contains non-NumPy arrays, has inconsistent 
                       row counts, or includes arrays with unsupported dimensions.
        """
        if not arrays:
            raise ValueError("At least one input array is required")

        # Validate that all inputs are NumPy arrays
        for arr in arrays:
            if not isinstance(arr, np.ndarray):
                raise ValueError("All inputs must be NumPy arrays")

        processed_arrays = []
        for arr in arrays:
            if len(arr.shape) == 1:
                processed_arrays.append(arr.reshape(-1, 1))  # Reshape 1D array to 2D (n, 1)
            elif len(arr.shape) == 2:
                processed_arrays.append(arr)  # Keep 2D arrays as-is
            else:
                raise ValueError(f"Input arrays must be 1D or 2D, but got shape {arr.shape}")

        # Validate consistent row counts across all arrays
        row_counts = [arr.shape[0] for arr in processed_arrays]
        if not all(count == row_counts[0] for count in row_counts):
            raise ValueError(f"All input arrays must have consistent row counts, but got: {row_counts}")

        return np.concatenate(processed_arrays, axis=1)  # Concatenate along column axis

    @staticmethod 
        def label_initialize(label, label2name_dict, ignore_label, output_class_dim, fill_value=-1):
        """
        Convert labels for loss computation by remapping to consecutive indices.

        Args:
            label (array-like): Original label data.
            label2name_dict (dict): Dictionary mapping label IDs to class names.
            ignore_label (int | list | tuple): Label(s) to be ignored during training.
            output_class_dim (int): Number of output classes (excluding ignored labels).
            fill_value (int, optional): Value to fill ignored positions with. Default: -1.

        Returns:
            tuple:
                - converted_label (ndarray): Remapped label array with consecutive indices.
                - origin2updated (dict): Mapping from original label IDs to new indices.

        Raises:
            ValueError: If labels are undefined in dict or dimension mismatch occurs.
        """
        # Get all valid label classes from dictionary
        valid_label_classes = list(label2name_dict.keys())
        # Get unique labels present in input data
        unique_labels_in_data = np.unique(label)

        # Validate that all data labels are defined in the dictionary
        if not set(unique_labels_in_data).issubset(set(valid_label_classes)):
            invalid_labels = set(unique_labels_in_data) - set(valid_label_classes)
            raise ValueError(f"Labels not defined in label2name_dict found: {invalid_labels}")

        # Ensure ignore_label is in list format
        if not isinstance(ignore_label, (list, tuple)):
            ignore_label = [ignore_label]

        # Compute effective labels (excluding ignored labels)
        effective_labels = list(set(valid_label_classes) - set(ignore_label))
        effective_labels.sort()  # Sort for consistency

        # Check if effective label count matches output class dimension
        if len(effective_labels) == output_class_dim:
            # Create label mapping dictionary: original -> new index
            origin2updated = {effective_labels[i]: i for i in range(len(effective_labels))}
            # Convert labels
            label_array = np.array(label)

            # Initialize converted label array, filled with fill_value
            converted_label = np.full_like(label_array, fill_value)

            # Remap only effective labels
            for old_label, new_label in origin2updated.items():
                converted_label[label_array == old_label] = new_label

            return converted_label, origin2updated
        else:
            raise ValueError(f"Number of effective labels {len(effective_labels)} ({effective_labels}) "
                             f"does not match output class dimension {output_class_dim}.")

    @staticmethod
    def augs_xyz(coords, from_paper='CosMix'):
        """
        Apply coordinate-level data augmentation based on strategies from different papers.

        Args:
            coords (ndarray): Input coordinates with shape (N, 3).
            from_paper (str): Augmentation strategy name. Options:
                'Custom', 'Mix3D', 'GrowSP', 'CosMix', 'PolarMix', 'DGLSS'

        Returns:
            ndarray: Augmented coordinates.
        """
        if from_paper == 'Custom':
            coords = CoordsAugs.translate(coords, max_translation=5)
            coords = CoordsAugs.rotate(coords, rotation_bounds=((0, 0), (0, 0), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        elif from_paper == 'Mix3D':
            # Reference: https://github.com/kumuji/mix3d/blob/9015e138cf64ece5aa392be173b8fb9d5afe9573/mix3d/conf/augmentation/volumentations_aug.yaml
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (-np.pi / 24, np.pi / 24), (-np.pi / 24, np.pi / 24), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.90, 1.10))
        elif from_paper == 'GrowSP':
            coords = CoordsAugs.translate(coords, max_translation=50)
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.90, 1.10))
        elif from_paper == 'CosMix':
            # Reference: https://github.com/saltoricristiano/cosmix-uda/blob/07bb2a64d5341f32e41e1e81343887b3c134430c/utils/datasets/dataset.py#L51
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20)))
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        elif from_paper == 'PolarMix':
            # Reference: https://github.com/xiaoaoran/polarmix/blob/main/core/datasets/semantic_kitti_polarmix.py#L217
            coords = CoordsAugs.rotate(coords, rotation_bounds=(
                (0, 0), (0, 0), (-np.pi, np.pi)))
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        elif from_paper == 'DGLSS':
            coords = CoordsAugs.rotate(coords, rotation_bounds=((0, 0), (0, 0), (-np.pi, np.pi)))
            coords = CoordsAugs.flip_xy(coords, flip_prob=0.7)
            coords = CoordsAugs.scale(coords, scale_range=(0.95, 1.05))
        else:
            warnings.warn('Basic data augmentation enabled, but no valid parameter specified')
        return coords  # Return augmented coordinates

    @staticmethod
def mix_scene(data_ori, data_mix, mask_ori=None, mask_mix=None):
        """
        Mix two point cloud data dictionaries based on optional boolean masks, 
        with safety checks and deep copying (supports field differences; only intersects common fields).

        Args:
            data_ori (dict): Original point cloud data dictionary {field_name: ndarray}.
            data_mix (dict): Mixed point cloud data dictionary {field_name: ndarray}.
            mask_ori (ndarray[bool], optional): Boolean mask for original point cloud.
            mask_mix (ndarray[bool], optional): Boolean mask for mixed point cloud.

        Returns:
            dict: Mixed point cloud data dictionary (only common fields retained).

        Raises:
            ValueError: If mask lengths are inconsistent with data or no common fields exist.
        """
        # === Safety Check 1: Field intersection and difference warnings ===
        common_keys = set(data_ori.keys()) & set(data_mix.keys())  # Common fields
        diff_keys_ori = set(data_ori.keys()) - common_keys  # Fields only in data_ori
        diff_keys_mix = set(data_mix.keys()) - common_keys  # Fields only in data_mix

        if diff_keys_ori:
            warnings.warn(f"The following fields exist only in data_ori and will be ignored: {diff_keys_ori}", 
                         category=UserWarning)
        if diff_keys_mix:
            warnings.warn(f"The following fields exist only in data_mix and will be ignored: {diff_keys_mix}", 
                         category=UserWarning)

        if not common_keys:
            raise ValueError("data_ori and data_mix have no common fields; cannot mix")

        # === Safety Check 2: Mask length consistency ===
        ori_len = next(iter(data_ori.values())).shape[0]
        mix_len = next(iter(data_mix.values())).shape[0]
        if mask_ori is not None and mask_ori.shape[0] != ori_len:
            raise ValueError(f"mask_ori length {mask_ori.shape[0]} inconsistent with data_ori point count {ori_len}")
        if mask_mix is not None and mask_mix.shape[0] != mix_len:
            raise ValueError(f"mask_mix length {mask_mix.shape[0]} inconsistent with data_mix point count {mix_len}")

        # === Filtering + Deep copy (process only common fields) ===
        if mask_ori is None:
            selected_ori = {k: data_ori[k].copy() for k in common_keys}
        else:
            selected_ori = {k: data_ori[k][mask_ori].copy() for k in common_keys}

        if mask_mix is None:
            selected_mix = {k: data_mix[k].copy() for k in common_keys}
        else:
            selected_mix = {k: data_mix[k][mask_mix].copy() for k in common_keys}

        # === Concatenation ===
        result_mix = {
            k: np.concatenate((selected_ori[k], selected_mix[k]), axis=0)
            for k in common_keys
        }

        return result_mix

    @staticmethod  # 定义为静态方法
    def voxelize(coords, features, voxel_size):
        """
        Perform MinkowskiEngine-compatible sparse voxelization.

        Args:
            coords (ndarray): Input coordinates with shape (N, 3).
            features (ndarray): Input features with shape (N, D).
            voxel_size (float): Voxel resolution in meters (must be > 0).

        Returns:
            tuple:
                - coordsV (ndarray): Quantized voxel coordinates.
                - featsV (ndarray): Aggregated features per voxel.
                - unique_map (ndarray): Indices mapping voxels to original points.
                - inverse_map (ndarray): Indices mapping original points to voxels.
        """
        # Assert voxel_size is positive
        assert voxel_size > 0, "Voxel size must be positive"

        # Compute scaling factor: convert real-world coordinates to voxel coordinate system
        # scale = 1 / voxel_size
        scale = 1 / voxel_size

        # === Coordinate quantization stage ===
        # Quantize continuous coordinates to voxel grid
        # coords * scale: Scale coordinates to voxel coordinate system
        # np.floor(): Round down to nearest voxel center
        coords_int = np.floor(coords * scale)

        # Perform sparse quantization using MinkowskiEngine utilities
        # Features are aggregated during voxelization; ignore_label=-1 excludes invalid points
        coordsV, featsV, unique_map, inverse_map = ME.utils.sparse_quantize(
            np.ascontiguousarray(coords_int),
            features=features,
            ignore_label=-1,
            return_index=True,
            return_inverse=True
        )
        # Return voxelized data with inverse mapping for dense output restoration
        return coordsV, featsV, unique_map, inverse_map


class GeneralDataset(Dataset):
    def __init__(self, dataset_path: list, mode, transforms=None):
        self.mode = mode
        self.transforms = transforms

        self.execute_file = dataset_path

    def __len__(self):
        return len(self.execute_file) 

    def __getitem__(self, idx):
        raise NotImplementedError
