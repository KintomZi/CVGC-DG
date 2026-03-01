import torch
import numpy as np
from datasets.BaseDataset import GeneralDataset
from datasets.BaseDataset import PointCloudMethods as pcMethods


class MinkowskiDataset(GeneralDataset):
    def __init__(self, config, dataset_path: list, mode, transforms=None):
        super().__init__(dataset_path=dataset_path,
                         mode=mode,
                         transforms=transforms)
        self.config = config

    def data_load(self, filePath: str, center_type: int = 2):
        try:
            npy_data = np.load(filePath)
            # Note: Evaluate whether post-loading processing logic is time-consuming
        except Exception as e:
            print(f"Failed to load file: {filePath}, Error: {e}")
            raise
        coords = np.vstack((npy_data['x'], npy_data['y'], npy_data['z'])).T.astype(np.float32)
        point_id = np.arange(coords.shape[0])
        #########################################################
        # Coordinate processing
        try:
            coords_init = pcMethods.coords_initialize(coords, center_type)
        except Exception as e:
            print(filePath)
            raise e

        #########################################################
        # Feature processing: decide whether to load coords before voxelization
        # colors = np.vstack((npy_data['red'], npy_data['green'], npy_data['blue'])).T
        # features = pcMethods.feature_initialize(colors)
        features = np.array([])  # Placeholder: features to be initialized later

        #########################################################
        # Ground truth label processing
        ground_truth = np.array(npy_data['label'])
        # if np.any(ground_truth == 0):
        #     tempH = coords_init[ground_truth == 0][:, 2].mean(axis=0)
        #     coords_init[:, 2] = coords_init[:, 2] - tempH

        # ground_truth, origin2updated = pcMethods.label_initialize(npy_data['label'],
        #                                                      self.config.datasets.label_to_names,
        #                                                      self.config.datasets.ignore_label,
        #                                                      self.config.model.output_class_dim)

        #########################################################
        # Other fields

        return {
            'point_id': point_id,
            'coords_raw': coords,
            'coords': coords_init,
            'features': features,
            'ground_truth': ground_truth,
        }

    def voxelize_data(self, coords, features, voxel_size, data_dict):
        """Minkowski voxelization"""
        if features.shape[0] == 0:
            # If no features provided, use coordinates as default features
            features = pcMethods.feature_initialize(coords)
        else:
            # Concatenate coordinates with provided features
            features = pcMethods.feature_initialize(coords, features)

        coordsV, featsV, unique_map, inverse_map = pcMethods.voxelize(coords, features, voxel_size)
        data_dict['coordsV'] = coordsV
        data_dict['featsV'] = featsV
        data_dict['unique_map'] = unique_map
        data_dict['inverse_map'] = inverse_map
        return data_dict

    def __len__(self):
        return len(self.execute_file)

    def mode_train(self, index):
        args_dataset = self.config.dataset
        data_result = {}

        file_path = self.execute_file[index]
        data_ori = self.data_load(file_path, center_type=args_dataset.center_type)
        # data_ori['mix_id'] = np.full((data_ori['coords'].shape[0],), 0, dtype=int)

        if self.transforms:
            # Standard augmentation: rotation, translation, scaling
            data_ori['coordsTrans'] = pcMethods.augs_xyz(data_ori['coords'], from_paper=args_dataset.aug_parameter)
        else:
            data_ori['coordsTrans'] = data_ori['coords']

        # Voxelization processing
        data_ori = self.voxelize_data(data_ori['coordsTrans'], data_ori['features'], args_dataset.voxel_size, data_ori)

        data_result['origin'] = {
            'point_id': data_ori['point_id'],
            'coords': data_ori['coordsV'],
            'features': data_ori['featsV'],
            'ground_truth': data_ori['ground_truth'],
            'unique_map': data_ori['unique_map'],
            'inverse_map': data_ori['inverse_map'],
        }

        return data_result

    def mode_eval(self, index):
        args_dataset = self.config.dataset
        data_result = {}

        file_path = self.execute_file[index]
        data_ori = self.data_load(file_path, center_type=args_dataset.center_type)
        # data_ori['mix_id'] = np.full((data_ori['coords'].shape[0],), 0, dtype=int)

        if self.transforms:
            data_ori['coordsTrans'] = pcMethods.augs_xyz(data_ori['coords'], from_paper=args_dataset.aug_parameter)
        else:
            data_ori['coordsTrans'] = data_ori['coords']

        data_ori = self.voxelize_data(data_ori['coordsTrans'], data_ori['features'], args_dataset.voxel_size, data_ori)

        data_result['eval'] = {
            'point_id': data_ori['point_id'],
            'coords': data_ori['coordsV'],
            'features': data_ori['featsV'],
            'ground_truth': data_ori['ground_truth'],
            'unique_map': data_ori['unique_map'],
            'inverse_map': data_ori['inverse_map'],
        }

        return data_result

    def __getitem__(self, index: int):
        if self.mode == 'train':
            data_result = self.mode_train(index)
        elif self.mode == 'val' or self.mode == 'test':
            data_result = self.mode_eval(index)
        else:
            raise ValueError(f"Invalid mode: {self.mode}, expected 'train', 'val', or 'test'.")

        return data_result, index


class collate_fn_mink:
    """
    Custom collate function for batching MinkowskiEngine sparse tensors.
    
    Handles concatenation of sparse coordinates, feature aggregation, and index offsetting
    to support batched sparse convolution operations.
    """
    def __call__(self, data_list):
        """
        Collate a list of samples into a batch dictionary compatible with MinkowskiEngine.

        Args:
            data_list (list): List of tuples (data_dict, index) from dataset.__getitem__.

        Returns:
            tuple:
                - merged_dict (dict): Batched data dictionary with concatenated tensors.
                - indices (list): List of original sample indices.
        """
        # Unpack data list to separate data dictionaries and indices
        data_all_list, indices = zip(*data_list)

        # Collect all existing keys (e.g., 'origin', 'mix', etc.)
        type_key_all = data_all_list[0].keys()
        # all_keys = set(key for data in data_all_list for key in data.keys())

        # Initialize batch storage structure
        batch_dict = {
            type_key: {property_key: []
                       for property_key in data_all_list[0][type_key].keys()}
            for type_key in type_key_all
        }
        for type_key in type_key_all:
            batch_dict[type_key]['batch_id'] = []

        # Compute offsets for index remapping across batched samples
        inverse_offset = {}  # Offset for original point indices
        unique_offsets = {}  # Offset for unique voxel indices
        for type_key in type_key_all:
            # Compute cumulative offset for original point counts per sample
            inverse_counts = [len(data_all[type_key]['unique_map']) for data_all in data_all_list]
            inverse_offset[type_key] = np.cumsum([0] + inverse_counts[:-1])

            # Compute cumulative offset for unique voxel counts per sample
            unique_counts = [len(data_all[type_key]['inverse_map']) for data_all in data_all_list]
            unique_offsets[type_key] = np.cumsum([0] + unique_counts[:-1])

        # Iterate through each sample in the batch
        for batch_id, data_all in enumerate(data_all_list):
            # Iterate through each data variant (e.g., 'origin', 'mix')
            for type_key in data_all.keys():
                item = data_all[type_key]

                # Add batch dimension to coordinates for MinkowskiEngine compatibility
                batch_coords = np.hstack([
                    np.full((item['coords'].shape[0], 1), batch_id),
                    item['coords']
                ])

                # Collect data into corresponding batch slots
                temp_batch_id = np.full((item['ground_truth'].shape[0],), batch_id)
                batch_dict[type_key]['batch_id'].append(torch.from_numpy(temp_batch_id).long())
                batch_dict[type_key]['point_id'].append(torch.from_numpy(item['point_id']).long())
                batch_dict[type_key]['coords'].append(torch.from_numpy(batch_coords).int())
                batch_dict[type_key]['features'].append(torch.from_numpy(item['features']).float())
                batch_dict[type_key]['ground_truth'].append(torch.from_numpy(item['ground_truth']).long())
                # Adjust unique_map indices with cumulative offset for batch concatenation
                batch_dict[type_key]['unique_map'].append(
                    item['unique_map'].long() + unique_offsets[type_key][batch_id])
                # Adjust inverse_map indices with cumulative offset for batch concatenation
                batch_dict[type_key]['inverse_map'].append(
                    item['inverse_map'].long() + inverse_offset[type_key][batch_id])
                ############################
                # Optional fields (conditionally added)
                if 'mix_id' in batch_dict[type_key].keys():
                    # Used for mixed-scene augmentation strategies (e.g., 'Mix3D', 'PolarMix')
                    batch_dict[type_key]['mix_id'].append(torch.from_numpy(item['mix_id']).long())
                if 'same2ori' in batch_dict[type_key].keys():
                    same2ori = np.vstack([np.full((item['same2ori'].shape[0],), batch_id), item['same2ori']]).T
                    batch_dict[type_key]['same2ori'].append(torch.from_numpy(same2ori).long())
                if 'same2aug' in batch_dict[type_key].keys():
                    same2aug = np.vstack([np.full((item['same2aug'].shape[0],), batch_id), item['same2aug']]).T
                    batch_dict[type_key]['same2aug'].append(torch.from_numpy(same2aug).long())

                if 'coords_raw' in batch_dict[type_key].keys():
                    batch_dict[type_key]['coords_raw'].append(torch.from_numpy(item['coords_raw']))

                if 'coords_Occ' in batch_dict[type_key].keys():
                    id_coords_Occ = np.hstack([np.full((item['coords_Occ'].shape[0], 1), batch_id), item['coords_Occ']])
                    batch_dict[type_key]['coords_Occ'].append(torch.from_numpy(id_coords_Occ).int())
                if 'label_Occ' in batch_dict[type_key].keys():
                    batch_dict[type_key]['label_Occ'].append(torch.from_numpy(item['label_Occ']).long())
                if 'index_Occ' in batch_dict[type_key].keys():
                    batch_dict[type_key]['index_Occ'].append(torch.from_numpy(item['index_Occ']).int())
                if 'weight_Occ' in batch_dict[type_key].keys():
                    batch_dict[type_key]['weight_Occ'].append(torch.from_numpy(item['weight_Occ']).float())

        # Concatenate tensors across all samples for each data variant
        merged_dict = {}
        for type_key in type_key_all:
            merged_dict[type_key] = {
                'batch_id': torch.cat(batch_dict[type_key]['batch_id'], dim=0),
                'point_id': torch.cat(batch_dict[type_key]['point_id'], dim=0),
                'coords': torch.cat(batch_dict[type_key]['coords'], dim=0).float(),
                'features': torch.cat(batch_dict[type_key]['features'], dim=0),
                'ground_truth': torch.cat(batch_dict[type_key]['ground_truth'], dim=0),
                'unique_map': torch.cat(batch_dict[type_key]['unique_map'], dim=0),
                'inverse_map': torch.cat(batch_dict[type_key]['inverse_map'], dim=0),
            }
            ############################
            # Optional fields (conditionally merged)
            if 'mix_id' in batch_dict[type_key].keys():
                # Used for mixed-scene augmentation strategies (e.g., 'Mix3D', 'PolarMix')
                merged_dict[type_key]['mix_id'] = torch.cat(batch_dict[type_key]['mix_id'], dim=0)
            if 'same2aug' in batch_dict[type_key].keys():
                merged_dict[type_key]['same2aug'] = torch.cat(batch_dict[type_key]['same2aug'], dim=0)
            if 'same2ori' in batch_dict[type_key].keys():
                merged_dict[type_key]['same2ori'] = torch.cat(batch_dict[type_key]['same2ori'], dim=0)

            if 'coords_raw' in batch_dict[type_key].keys():
                merged_dict[type_key]['coords_raw'] = torch.cat(batch_dict[type_key]['coords_raw'], dim=0)

            if 'coords_Occ' in batch_dict[type_key].keys():
                merged_dict[type_key]['coords_Occ'] = torch.cat(batch_dict[type_key]['coords_Occ'], dim=0).int()
            if 'label_Occ' in batch_dict[type_key].keys():
                merged_dict[type_key]['label_Occ'] = torch.cat(batch_dict[type_key]['label_Occ'], dim=0)
            if 'index_Occ' in batch_dict[type_key].keys():
                merged_dict[type_key]['index_Occ'] = torch.cat(batch_dict[type_key]['index_Occ'], dim=0)
            if 'weight_Occ' in batch_dict[type_key].keys():
                merged_dict[type_key]['weight_Occ'] = torch.cat(batch_dict[type_key]['weight_Occ'], dim=0)

        return merged_dict, list(indices)
