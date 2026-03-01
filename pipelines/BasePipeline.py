import os
import logging
import random
import warnings

import torch
import numpy as np
from omegaconf import ListConfig
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int):
    random.seed(seed)  # 1. Set seed for Python's built-in random module
    np.random.seed(seed)  # 2. Set seed for NumPy random number generator
    torch.manual_seed(seed)  # 3. Set seed for PyTorch random number generator
    # 4. Set seed for CUDA random number generator (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        # 5. Configure CUDA backend for deterministic behavior
        torch.backends.cudnn.deterministic = True  # Ensure deterministic convolution algorithms
        torch.backends.cudnn.benchmark = False  # Disable cudnn's non-deterministic optimizations
    os.environ['PYTHONHASHSEED'] = str(seed)  # 6. Set Python hash seed for reproducibility


class PredictionProcessor:
    @staticmethod
    def confidence(pred_logits, conf_threshold, ignore_label=-1):
        """
        Generate pseudo-labels and validity mask based on prediction logits and confidence threshold.

        Args:
            pred_logits (Tensor[N, C]): Model output class logits, shape [N, C].
            conf_threshold (float): Confidence threshold (0, 1]; predictions with confidence 
                less than or equal to this value will be ignored.
            ignore_label (int): Label value assigned to low-confidence positions (typically -1).

        Returns:
            pseudo_labels (Tensor[N]): Pseudo-label array; low-confidence points are set to ignore_label.
            valid_mask (Tensor[N], bool): Validity mask for points; True indicates confidence exceeds threshold.
        """
        # Convert logits to probability distribution [N, C]
        class_probs = torch.nn.functional.softmax(pred_logits, dim=-1)

        # Find maximum class probability and its class index for each point
        max_probs, max_classes = class_probs.max(dim=-1)  # [N], [N]

        # Filter by threshold
        if 0 < conf_threshold <= 1:
            valid_mask = max_probs > conf_threshold  # Validity mask for points
            # Initialize with ignore_label
            pseudo_labels = torch.full_like(max_classes, fill_value=ignore_label)
            # Retain only high-confidence labels
            pseudo_labels[valid_mask] = max_classes[valid_mask]
        else:
            # If threshold is invalid, retain all predictions
            valid_mask = torch.ones_like(max_classes, dtype=torch.bool)
            pseudo_labels = max_classes

        return pseudo_labels.detach(), valid_mask

    @staticmethod
    def remap_gt_labels(gt_labels, label2name, ignore_labels, output_class_dim, fill_value=-1):
        """
        Remap original ground truth labels to consecutive indices starting from 0, 
        while ignoring specified labels.

        Args:
            gt_labels (array-like): Original label array.
            label2name (dict): {original_label_id: class_name}.
            ignore_labels (int | list | tuple): Labels to be ignored.
            output_class_dim (int): Number of output classes.
            fill_value (int): Fill value for ignored positions.

        Returns:
            converted_labels (ndarray): Remapped labels.
            keep_indices (ndarray): Indices of retained positions.
            label2name_updated (dict): {new_label_id: class_name}.
        """
        # Ensure ignore_labels is a list
        if isinstance(ignore_labels, (int, np.integer)):  # Convert single int to list
            ignore_labels = [ignore_labels]
        elif isinstance(ignore_labels, ListConfig):  # Convert ListConfig to standard list
            ignore_labels = list(ignore_labels)
        elif not isinstance(ignore_labels, (list, tuple)):  # Raise error for unsupported types
            raise TypeError(f"ignore_labels type not supported: {type(ignore_labels)}")

        # Effective labels = labels in label2name excluding ignore_labels
        all_labels = list(label2name.keys())
        effective_labels = sorted(set(all_labels) - set(ignore_labels))

        # Check if number of effective labels matches model output dimension
        if len(effective_labels) != output_class_dim:
            raise ValueError(
                f"Number of effective labels {len(effective_labels)} ({effective_labels}) "
                f"does not match output class dimension {output_class_dim}."
            )

        # Validate input label legality
        unique_labels = np.unique(gt_labels)
        if not set(unique_labels).issubset(all_labels):
            invalids = set(unique_labels) - set(all_labels)
            raise ValueError(f"Labels not defined in label2name found: {invalids}")

        # Create mapping: old_id → new_id
        old_to_new = {old: new for new, old in enumerate(effective_labels)}

        # Create mapping: new_id → class name (for evaluation phase)
        label2name_updated = {new: label2name[old] for old, new in old_to_new.items()}

        # Convert labels
        gt_array = np.array(gt_labels)
        keep_mask = np.isin(gt_array, effective_labels)
        keep_indices = np.where(keep_mask)[0]
        converted_labels = np.full_like(gt_array, fill_value)
        for old, new in old_to_new.items():
            converted_labels[gt_array == old] = new

        return converted_labels, keep_indices, label2name_updated

       @staticmethod
    def compute_metrics(gt, preds, label2name=None, ignore_labels=(-1,)):
        """
        Compute common metrics for semantic segmentation/classification tasks (ignoring specified labels).

        Args:
            gt (ndarray): Ground truth labels, shape (N,).
            preds (ndarray): Predicted labels, shape (N,).
            label2name (dict | None): {class_id: class_name}, optional.
            ignore_labels (int | list | tuple): Class IDs to ignore.

        Returns:
            metrics (dict): Dictionary containing OA, mAcc, mIoU, mF1, confusion matrix, etc.
        """
        # Parameter preprocessing
        # Ensure ignore_labels is converted to list
        if isinstance(ignore_labels, (int, np.integer)):  # Single int
            ignore_labels = [ignore_labels]
        elif isinstance(ignore_labels, ListConfig):  # ListConfig → list
            ignore_labels = list(ignore_labels)
        elif not isinstance(ignore_labels, (list, tuple)):
            raise TypeError(f"ignore_label_ids type not supported: {type(ignore_labels)}")

        # Convert to numpy and flatten
        gt = np.asarray(gt).flatten()
        preds = np.asarray(preds).flatten()

        if gt.shape != preds.shape:
            raise ValueError("Shape mismatch between GT and predictions")

        # ---------------- Label validation and name mapping ----------------
        if label2name is None:
            # Default class names as string-formatted class IDs
            label2name = {i: str(i) for i in np.unique(gt) if i not in ignore_labels}
        else:
            # Ensure GT and prediction classes are in known classes or ignore labels
            known_labels = set(label2name.keys()) | set(ignore_labels)
            if not set(gt).issubset(known_labels):
                invalid_gt = set(gt) - known_labels
                raise ValueError(f"Unknown classes in GT: {invalid_gt}")
            if not set(preds).issubset(known_labels):
                invalid_preds = set(preds) - known_labels
                raise ValueError(f"Unknown classes in predictions: {invalid_preds}")

        # Get full class set (excluding ignore_labels)
        all_classes = sorted(set(label2name.keys()) - set(ignore_labels))

        # Filter out ignore_labels
        mask = ~np.isin(gt, ignore_labels)
        gt = gt[mask]
        preds = preds[mask]

        # Construct confusion matrix
        num_classes = len(all_classes)
        label_to_index = {label: idx for idx, label in enumerate(all_classes)}
        gt_idx = np.array([label_to_index[l] for l in gt])
        pred_idx = np.array([label_to_index[l] for l in preds])
        cm = np.bincount(
            num_classes * gt_idx + pred_idx,
            minlength=num_classes**2
        ).reshape(num_classes, num_classes)

        # Compute metrics
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        total = cm.sum()

        oa = tp.sum() / (total + 1e-8)
        class_acc = tp / (cm.sum(axis=1) + 1e-8)
        mean_acc = np.mean(class_acc)
        iou = tp / (tp + fp + fn + 1e-8)
        mean_iou = np.mean(iou)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        mean_f1 = np.mean(f1)

        # Per-class detailed information
        class_info = {
            label2name[cls]: {
                'id_confusion': idx,
                'acc': float(class_acc[idx]),
                'iou': float(iou[idx]),
                'f1': float(f1[idx]),
            }
            for idx, cls in enumerate(all_classes)
        }

        return {
            'oa': float(oa),
            'mean_acc': float(mean_acc),
            'mean_iou': float(mean_iou),
            'mean_f1': float(mean_f1),
            'confusion_matrix': cm,
            'class_info': class_info
        }
        
    @staticmethod
    def metrics2text(metrics_dict):
        """
        Format the results from compute_metrics into readable text.

        Args:
            metrics_dict (dict): Return value from compute_metrics.

        Returns:
            str: Formatted metric text.
        """
        # Overall metrics
        oa = metrics_dict['oa'] * 100  # Overall Accuracy
        mean_acc = metrics_dict['mean_acc'] * 100  # Mean Accuracy
        mean_iou = metrics_dict['mean_iou'] * 100  # Mean IoU
        mean_f1 = metrics_dict['mean_f1'] * 100  # Macro F1

        # Extract per-class metrics
        class_info = metrics_dict['class_info']
        class_names = list(class_info.keys())
        ious = [class_info[n]['iou'] * 100 for n in class_names]
        f1s = [class_info[n]['f1'] * 100 for n in class_names]

        # Construct text output
        metric_text = (
                f"Overall Accuracy: {oa:.2f} || Mean Accuracy: {mean_acc:.2f}\n"
                f"Mean IoU: {mean_iou:.2f} | Per-class IoU:  " +
                "  ".join([f"{n}:{v:.2f}" for n, v in zip(class_names, ious)]) + "\n" +
                f"Macro F1: {mean_f1:.2f} | Per-class  F1:  " +
                "  ".join([f"{n}:{v:.2f}" for n, v in zip(class_names, f1s)])
        )
        return metric_text


#########################################################################################


class BaseRecorder:
    def __init__(self, result_folder_path, timestamp_start):
        # Initialize timestamp
        self.timestamp_start = timestamp_start

        # Validate and set result path
        self.result_path = result_folder_path

        # Initialize subdirectories and utilities
        self.ckpt_folder = self._ckpt_folder_init()  # Create checkpoint subfolder
        self.logger = self._logger_init(self.ckpt_folder)  # Initialize logger
        self.tensorboard = self._tensorboard_init(self.ckpt_folder)  # Initialize TensorBoard writer

    def _ckpt_folder_init(self):
        """
        Create checkpoint folder for saving models; automatically append suffix if directory with same name exists.

        Returns:
            ckpt_dir: str, final created checkpoint path.
        """
        ckpt_dir_name = f"checkpoint_{self.timestamp_start.strftime('%m.%d_%H-%M')}"
        ckpt_dir = os.path.join(self.result_path, ckpt_dir_name)
        counter = 1
        while os.path.exists(ckpt_dir):
            ckpt_dir = os.path.join(self.result_path, f"{ckpt_dir_name}_{counter:02d}")
            counter += 1
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @staticmethod  # Defined as static method
    def _logger_init(checkpoint_folder):
        """
        Configure logger to output to both file and console.

        Returns:
            logger: logging.Logger instance.
        """
        # Configure logger
        logger = logging.getLogger(os.path.basename(checkpoint_folder))
        logger.setLevel(logging.INFO)  # Set default log level

        # Check if handlers are already configured to avoid duplicates
        if not logger.handlers:
            # File handler
            log_file = os.path.join(checkpoint_folder, f"{os.path.basename(checkpoint_folder)}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Format configuration
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    @staticmethod  # Defined as static method
    def _tensorboard_init(checkpoint_folder):
        """
        Initialize TensorBoard log directory and writer.

        Returns:
            writer: SummaryWriter instance.
        """
        # TensorBoard directory
        tensorboard_dir = os.path.join(checkpoint_folder, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        return writer

    def save_tsne_embedding(self, features, labels, global_step=0, tag="embedding"):
        """
        Save features and labels for TensorBoard t-SNE visualization. 
        TensorBoard embedding visualization is not suitable for >10k samples.

        Args:
            features (np.ndarray or torch.Tensor): Embedding vectors with shape (N, D).
            labels (list or np.ndarray): Label list of length N (strings or convertible to strings).
            global_step (int): Corresponding training step (optional, default 0).
            tag (str): Namespace for embedding (optional, default "embedding").
        """
        try:
            # Convert to numpy array if Tensor
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            # Ensure numpy type
            features = np.asarray(features)
            labels = np.asarray(labels)

            # Check if sample counts match
            assert features.shape[0] == len(labels), \
                f"Feature count ({features.shape[0]}) does not match label count ({len(labels)})!"

            num_samples = features.shape[0]
            unique_labels = np.unique(labels)

            # ---------- Class-balanced sampling ----------
            selected_indices = []
            per_class_info = {}

            for cls in unique_labels:
                cls_indices = np.where(labels == cls)[0]
                cls_count = len(cls_indices)
                if cls_count > 750:
                    # If class has too many samples, sample at fixed intervals (non-random)
                    step = cls_count // 750
                    sampled_idx = cls_indices[::step][:750]
                else:
                    # If class has few samples, retain all
                    sampled_idx = cls_indices

                selected_indices.extend(sampled_idx.tolist())
                per_class_info[str(cls)] = len(sampled_idx)

            # ---------- Control overall upper limit ----------
            if len(selected_indices) > 10000:
                # If still exceeds limit after balanced sampling, downsample globally at fixed intervals
                step = len(selected_indices) // 10000
                selected_indices = selected_indices[::step][:10000]

            # ---------- Extract final data ----------
            features_sub = features[selected_indices]
            labels_sub = labels[selected_indices]

            # ---------- Write to TensorBoard ----------
            self.tensorboard.add_embedding(
                mat=features_sub,
                metadata=[str(l) for l in labels_sub],
                global_step=global_step,
                tag=tag
            )

        except Exception as e:
            self.logger.error(f"Failed to save t-SNE embedding [{tag}]: {e}")

    def metric_tensorboard(self, title: str, epoch: int, metrics):
        """
        Record evaluation results to TensorBoard.

        Args:
            title (str): Title to distinguish different evaluation tasks, e.g., "Train" or "Val".
            epoch (int): Current training epoch.
            metrics (dict): Evaluation result dictionary returned by compute_metrics.
        """
        # Record overall evaluation metrics
        self.tensorboard.add_scalar(f'{title}/Overall Accuracy', metrics['oa'], epoch)
        self.tensorboard.add_scalar(f'{title}/Mean Accuracy', metrics['mean_acc'], epoch)
        self.tensorboard.add_scalar(f'{title}/Mean IoU', metrics['mean_iou'], epoch)
        self.tensorboard.add_scalar(f'{title}/Macro F1', metrics['mean_f1'], epoch)

        # Extract per-class metric information
        class_names = list(metrics['class_info'].keys())  # List of class names or ID strings
        id_confusion = [metrics['class_info'][k]['id_confusion'] for k in class_names]
        ious = [metrics['class_info'][k]['iou'] for k in class_names]  # Per-class IoU
        f1s = [metrics['class_info'][k]['f1'] for k in class_names]  # Per-class F1

        # Merge all class IoU and F1 curves in the same plot
        iou_dict = {k: v for k, v in zip(class_names, ious)}
        f1_dict = {k: v for k, v in zip(class_names, f1s)}
        self.tensorboard.add_scalars(f'{title}/Per-IoU', iou_dict, epoch)
        self.tensorboard.add_scalars(f'{title}/Per-F1', f1_dict, epoch)

        # Predicted label distribution under ground truth labels
        row_sums = metrics['confusion_matrix'].sum(axis=1, keepdims=True)
        gt_perPreds = np.divide(metrics['confusion_matrix'], row_sums, where=row_sums != 0) * 100
        gt_perPreds = np.round(gt_perPreds, 2)

        # Ground truth label distribution under predicted labels
        col_sums = metrics['confusion_matrix'].sum(axis=0, keepdims=True)
        preds_perGT = np.divide(metrics['confusion_matrix'], col_sums, where=col_sums != 0) * 100
        preds_perGT = np.round(preds_perGT, 2)
        
        # Add predicted label distribution under each ground truth class
        for i, class_name in enumerate(class_names):
            gt_pred_dict = {f'pred-{pred_class}': gt_perPreds[i, j]
                            for j, pred_class in enumerate(class_names)}
            self.tensorboard.add_scalars(
                f'{title}_PredDist/gt-{class_name}', gt_pred_dict, epoch)

        # Add ground truth label distribution under each predicted class
        for i, class_name in enumerate(class_names):
            pred_gt_dict = {f'gt-{gt_class}': preds_perGT[j, i]
                            for j, gt_class in enumerate(class_names)}
            self.tensorboard.add_scalars(
                f'{title}_GTDist/pred-{class_name}', pred_gt_dict, epoch)


class PointBasePipeline:
    def __init__(self, config):
        self.config = config
        self.recorder = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dataset_cls = None
        self._model_cls = None

        self._datasets = None
        self._dataloaders = None

        self._optimizer = None
        self._scheduler = None
        self._criterion = None

    @property
    def dataset_cls(self):
        if self._dataset_cls is None:
            raise NotImplementedError('Please implement dataset_init()')
        else:
            return self._dataset_cls

    @property
    def model_cls(self):
        if self._model_cls is None:
            raise NotImplementedError('Please implement model_init()')
        else:
            return self._model_cls

    @property
    def datasets(self):
        if self._datasets is None:
            raise NotImplementedError('Please implement dataset_create()')
        else:
            return self._datasets

    @property
    def dataloaders(self):
        if self._dataloaders is None:
            raise NotImplementedError('Please implement dataloader_create()')
        else:
            return self._dataloaders

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise NotImplementedError('Please implement optimizer_create()')
        else:
            return self._optimizer

    @property
    def scheduler(self):
        if self._scheduler is None:
            warnings.warn('If using scheduler, please implement scheduler_create(); otherwise, this warning can be ignored')
            return None
        else:
            return self._scheduler

    @property
    def criterion(self):
        if self._criterion is None:
            raise NotImplementedError('Please implement criterion_create()')
        return self._criterion

    @staticmethod  # Defined as static method
    def path_verification(path):
        """
        Check if specified path exists; raise exception if not.
        
        Args:
            path: str, folder path.
            
        Returns:
            path: str, validated path.
            
        Raises:
            FileNotFoundError: If path does not exist.
            NotADirectoryError: If path is not a directory.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Specified path does not exist: {path}")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Specified path is not a directory: {path}")
        return path

    def model_init(self, model_cls):
        """
        Model may not be unique, so do not instantiate directly.

        :param model_cls: Model class (e.g., custom neural network class).
        """
        self._model_cls = model_cls

    def dataset_init(self, dataset_cls):
        """
        Initialize dataset class reference.
        
        :param dataset_cls: Dataset class (e.g., torch.utils.data.Dataset subclass).
        """
        self._dataset_cls = dataset_cls

    def optimizer_create(self, optimizer_cls, model_params, **kwargs):
        """
        Initialize optimizer.

        :param optimizer_cls: Optimizer class (e.g., torch.optim.Adam).
        :param model_params: Model parameters (model.parameters()).
        :param kwargs: Optimizer keyword arguments.
        """
        self._optimizer = optimizer_cls(model_params, **kwargs)

    def scheduler_create(self, scheduler_cls, **kwargs):
        """
        Initialize learning rate scheduler.
        
        :param scheduler_cls: Scheduler class (e.g., torch.optim.lr_scheduler.StepLR).
        :param kwargs: Scheduler keyword arguments.
        """
        self._scheduler = scheduler_cls(self.optimizer, **kwargs)

    def criterion_create(self, criterion_cls, **kwargs):
        """
        Initialize loss function.
        
        :param criterion_cls: Loss function class (e.g., torch.nn.CrossEntropyLoss).
        :param kwargs: Loss function keyword arguments.
        """
        self._criterion = criterion_cls(**kwargs).cuda()
