import os
import logging
import random
import warnings

import torch
import numpy as np
from omegaconf import ListConfig
from torch.utils.tensorboard import SummaryWriter


def seed_everything(seed: int):
    random.seed(seed)  # 1. 设置Python内置random模块的种子
    np.random.seed(seed)  # 2. 设置numpy的随机种子
    torch.manual_seed(seed)  # 3. 设置PyTorch的随机种子
    # 4. 设置CUDA的随机种子（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        # 5. 设置CUDA的后端
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # 禁用cudnn的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 6. 设置Python的hash种子


class PredictionProcessor:
    @staticmethod
    def confidence(pred_logits, conf_threshold, ignore_label=-1):
        """
        根据预测 logits 和置信度阈值，生成伪标签和有效掩码。

        Args:
            pred_logits (Tensor[N, C]): 模型输出的类别 logits，形状为 [N, C]
            conf_threshold (float): 置信度阈值 (0, 1]，小于等于该值的预测将被忽略
            ignore_label (int): 对低置信度位置填充的标签值（通常设为 -1）

        Returns:
            pseudo_labels (Tensor[N]): 伪标签数组，低置信度点被设置为 ignore_label
            valid_mask (Tensor[N], bool): 有效点掩码，True 表示置信度大于阈值
        """
        # 将 logits 转为概率分布 [N, C]
        class_probs = torch.nn.functional.softmax(pred_logits, dim=-1)

        # 找到每个点的最大类别概率及其类别索引
        max_probs, max_classes = class_probs.max(dim=-1)  # [N], [N]

        # 按阈值过滤
        if 0 < conf_threshold <= 1:
            valid_mask = max_probs > conf_threshold  # 有效点掩码
            # 初始化为 ignore_label
            pseudo_labels = torch.full_like(max_classes, fill_value=ignore_label)
            # 仅保留高置信度的标签
            pseudo_labels[valid_mask] = max_classes[valid_mask]
        else:
            # 阈值无效时，全部保留
            valid_mask = torch.ones_like(max_classes, dtype=torch.bool)
            pseudo_labels = max_classes

        return pseudo_labels.detach(), valid_mask

    @staticmethod
    def remap_gt_labels(gt_labels, label2name, ignore_labels, output_class_dim, fill_value=-1):
        """
        将原始 GT 标签映射为从 0 开始的连续类别索引，并忽略指定标签。

        Args:
            gt_labels (array-like): 原始标签数组
            label2name (dict): {原始标签ID: 类别名称}
            ignore_labels (int | list | tuple): 要忽略的标签
            output_class_dim (int): 输出类别数量
            fill_value (int): 被忽略位置的填充值

        Returns:
            converted_labels (ndarray): 转换后的标签
            keep_indices (ndarray): 保留的索引位置
            label2name_updated (dict): {新标签ID: 类别名称}
        """
        # 确保 ignore_labels 为列表
        if isinstance(ignore_labels, (int, np.integer)):  # 单个 int 转 list
            ignore_labels = [ignore_labels]
        elif isinstance(ignore_labels, ListConfig):  # ListConfig 转普通 list
            ignore_labels = list(ignore_labels)
        elif not isinstance(ignore_labels, (list, tuple)):  # 不支持的类型报错
            raise TypeError(f"ignore_labels 类型不支持: {type(ignore_labels)}")

        # 有效标签 = label2name 中除 ignore_labels 外的标签
        all_labels = list(label2name.keys())
        effective_labels = sorted(set(all_labels) - set(ignore_labels))

        # 检查有效标签数量是否与模型输出一致
        if len(effective_labels) != output_class_dim:
            raise ValueError(
                f"有效标签数量 {len(effective_labels)} ({effective_labels}) "
                f"不等于输出类别维度 {output_class_dim}。"
            )

        # 验证输入标签合法性
        unique_labels = np.unique(gt_labels)
        if not set(unique_labels).issubset(all_labels):
            invalids = set(unique_labels) - set(all_labels)
            raise ValueError(f"发现未在 label2name 中定义的标签: {invalids}")

        # 创建映射：旧ID → 新ID
        old_to_new = {old: new for new, old in enumerate(effective_labels)}

        # 创建映射：新ID → 类别名称（用于评估阶段）
        label2name_updated = {new: label2name[old] for old, new in old_to_new.items()}

        # 转换标签
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
        计算语义分割/分类任务常用指标（忽略指定标签）

        Args:
            gt (ndarray): 真实标签 (N,)
            preds (ndarray): 预测标签 (N,)
            label2name (dict | None): {类别ID: 类别名称}，可选
            ignore_labels (int | list | tuple): 忽略的类别ID

        Returns:
            metrics (dict): 包含 OA、mAcc、mIoU、mF1、混淆矩阵等
        """
        # 参数预处理
        # 确保 ignore_label_ids 转为 list
        if isinstance(ignore_labels, (int, np.integer)):  # 单个 int
            ignore_labels = [ignore_labels]
        elif isinstance(ignore_labels, ListConfig):  # ListConfig → list
            ignore_labels = list(ignore_labels)
        elif not isinstance(ignore_labels, (list, tuple)):
            raise TypeError(f"ignore_label_ids 类型不支持: {type(ignore_labels)}")

        # 转 numpy 并展平
        gt = np.asarray(gt).flatten()
        preds = np.asarray(preds).flatten()

        if gt.shape != preds.shape:
            raise ValueError("GT 与预测结果的形状不一致")

        # ---------------- 标签检查与名称映射 ----------------
        if label2name is None:
            # 默认类别名称为字符串形式的类别 ID
            label2name = {i: str(i) for i in np.unique(gt) if i not in ignore_labels}
        else:
            # 确保 GT 和预测类别都在已知类别或忽略标签中
            known_labels = set(label2name.keys()) | set(ignore_labels)
            if not set(gt).issubset(known_labels):
                invalid_gt = set(gt) - known_labels
                raise ValueError(f"GT 中存在未知类别: {invalid_gt}")
            if not set(preds).issubset(known_labels):
                invalid_preds = set(preds) - known_labels
                raise ValueError(f"预测结果中存在未知类别: {invalid_preds}")

        # 获取类别全集（除 ignore_labels 外）
        all_classes = sorted(set(label2name.keys()) - set(ignore_labels))

        # 过滤 ignore_labels
        mask = ~np.isin(gt, ignore_labels)
        gt = gt[mask]
        preds = preds[mask]

        # 构造混淆矩阵
        num_classes = len(all_classes)
        label_to_index = {label: idx for idx, label in enumerate(all_classes)}
        gt_idx = np.array([label_to_index[l] for l in gt])
        pred_idx = np.array([label_to_index[l] for l in preds])
        cm = np.bincount(
            num_classes * gt_idx + pred_idx,
            minlength=num_classes**2
        ).reshape(num_classes, num_classes)

        # 计算指标
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

        # 每类详细信息
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
        将 compute_metrics 的结果格式化为可读文本。

        Args:
            metrics_dict (dict): compute_metrics 的返回结果

        Returns:
            str: 格式化的指标文本
        """
        # 整体指标
        oa = metrics_dict['oa'] * 100  # Overall Accuracy
        mean_acc = metrics_dict['mean_acc'] * 100  # Mean Accuracy
        mean_iou = metrics_dict['mean_iou'] * 100  # Mean IoU
        mean_f1 = metrics_dict['mean_f1'] * 100  # Macro F1

        # 提取每类指标
        class_info = metrics_dict['class_info']
        class_names = list(class_info.keys())
        ious = [class_info[n]['iou'] * 100 for n in class_names]
        f1s = [class_info[n]['f1'] * 100 for n in class_names]

        # 构造文本
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
        # 初始化时间戳
        self.timestamp_start = timestamp_start

        # 验证并设置结果路径
        self.result_path = result_folder_path

        # 初始化子目录和工具
        self.ckpt_folder = self._ckpt_folder_init()  # 创建 checkpoint 子文件夹
        self.logger = self._logger_init(self.ckpt_folder)  # 初始化 logger
        self.tensorboard = self._tensorboard_init(self.ckpt_folder)  # 初始化 TensorBoard writer

    def _ckpt_folder_init(self):
        """
        创建用于保存模型的 checkpoint 文件夹，若存在重名目录则自动编号。

        Returns:
            ckpt_dir: str，最终创建的 checkpoint 路径
        """
        ckpt_dir_name = f"checkpoint_{self.timestamp_start.strftime('%m.%d_%H-%M')}"
        ckpt_dir = os.path.join(self.result_path, ckpt_dir_name)
        counter = 1
        while os.path.exists(ckpt_dir):
            ckpt_dir = os.path.join(self.result_path, f"{ckpt_dir_name}_{counter:02d}")
            counter += 1
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @staticmethod  # 定义为静态方法
    def _logger_init(checkpoint_folder):
        """
        配置日志记录器，输出到文件和终端控制台。

        Returns:
            logger: logging.Logger 实例
        """
        # 配置日志记录器
        logger = logging.getLogger(os.path.basename(checkpoint_folder))
        logger.setLevel(logging.INFO)  # 设置默认日志级别

        # 检查是否已配置过 Handler，避免重复添加
        if not logger.handlers:
            # 文件日志
            log_file = os.path.join(checkpoint_folder, f"{os.path.basename(checkpoint_folder)}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # 控制台日志
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式设置
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加 Handler 到 Logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        return logger

    @staticmethod  # 定义为静态方法
    def _tensorboard_init(checkpoint_folder):
        """
        初始化 TensorBoard 日志目录与写入器。

        Returns:
            writer: SummaryWriter 实例
        """
        # TensorBoard目录
        tensorboard_dir = os.path.join(checkpoint_folder, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        return writer

    def save_tsne_embedding(self, features, labels, global_step=0, tag="embedding"):
        """
        保存特征和标签用于 TensorBoard 的 t-SNE 可视化。TensorBoard 嵌入可视化不适合 >10k 样本

        Args:
            features (np.ndarray or torch.Tensor): 形状为 (N, D) 的嵌入向量。
            labels (list or np.ndarray): 长度为 N 的标签列表（字符串或可转为字符串）。
            global_step (int): 对应的训练步数（可选，默认 0）。
            tag (str): 嵌入的命名空间（可选，默认 "embedding"）。
        """
        try:
            # 若为 Tensor，转换为 numpy 数组
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()

            # 确保 numpy 类型
            features = np.asarray(features)
            labels = np.asarray(labels)

            # 检查样本数量是否匹配
            assert features.shape[0] == len(labels), \
                f"特征数 ({features.shape[0]}) 与标签数 ({len(labels)}) 不匹配！"

            num_samples = features.shape[0]
            unique_labels = np.unique(labels)

            # ---------- 类别均衡采样 ----------
            selected_indices = []
            per_class_info = {}

            for cls in unique_labels:
                cls_indices = np.where(labels == cls)[0]
                cls_count = len(cls_indices)
                if cls_count > 750:
                    # 若类别样本过多，按比例间隔采样（非随机）
                    step = cls_count // 750
                    sampled_idx = cls_indices[::step][:750]
                else:
                    # 若类别样本较少，则全部保留
                    sampled_idx = cls_indices

                selected_indices.extend(sampled_idx.tolist())
                per_class_info[str(cls)] = len(sampled_idx)

            # ---------- 控制总体上限 ----------
            if len(selected_indices) > 10000:
                # 若均衡采样后仍超出最大限制，则等间隔全局下采样
                step = len(selected_indices) // 10000
                selected_indices = selected_indices[::step][:10000]

            # ---------- 提取最终数据 ----------
            features_sub = features[selected_indices]
            labels_sub = labels[selected_indices]

            # ---------- 写入 TensorBoard ----------
            self.tensorboard.add_embedding(
                mat=features_sub,
                metadata=[str(l) for l in labels_sub],
                global_step=global_step,
                tag=tag
            )

        except Exception as e:
            self.logger.error(f"保存 t-SNE 嵌入失败 [{tag}]: {e}")

    def metric_tensorboard(self, title: str, epoch: int, metrics):
        """
        将评估结果记录到 TensorBoard。

        Args:
            title (str): 用于区分不同评估任务的标题，例如"Train"或"Val"。
            epoch (int): 当前训练轮次。
            metrics (dict): 由 compute_metrics_values 返回的评估结果字典。
        """
        # 记录整体评估指标
        self.tensorboard.add_scalar(f'{title}/Overall Accuracy', metrics['oa'], epoch)
        self.tensorboard.add_scalar(f'{title}/Mean Accuracy', metrics['mean_acc'], epoch)
        self.tensorboard.add_scalar(f'{title}/Mean IoU', metrics['mean_iou'], epoch)
        self.tensorboard.add_scalar(f'{title}/Macro F1', metrics['mean_f1'], epoch)

        # 提取每类指标信息
        class_names = list(metrics['class_info'].keys())  # 类别名或ID字符串列表
        id_confusion = [metrics['class_info'][k]['id_confusion'] for k in class_names]
        ious = [metrics['class_info'][k]['iou'] for k in class_names]  # 每类IoU
        f1s = [metrics['class_info'][k]['f1'] for k in class_names]  # 每类F1

        # 合并所有类别的IoU和F1曲线在同一图中
        iou_dict = {k: v for k, v in zip(class_names, ious)}
        f1_dict = {k: v for k, v in zip(class_names, f1s)}
        self.tensorboard.add_scalars(f'{title}/Per-IoU', iou_dict, epoch)
        self.tensorboard.add_scalars(f'{title}/Per-F1', f1_dict, epoch)

        # 真值标签下的预测标签分布
        row_sums = metrics['confusion_matrix'].sum(axis=1, keepdims=True)
        gt_perPreds = np.divide(metrics['confusion_matrix'], row_sums, where=row_sums != 0) * 100
        gt_perPreds = np.round(gt_perPreds, 2)

        # 预测标签下的真值标签分布
        col_sums = metrics['confusion_matrix'].sum(axis=0, keepdims=True)
        preds_perGT = np.divide(metrics['confusion_matrix'], col_sums, where=col_sums != 0) * 100
        preds_perGT = np.round(preds_perGT, 2)

        # 为每个类别添加真值标签下的预测标签分布
        for i, class_name in enumerate(class_names):
            gt_pred_dict = {f'pred-{pred_class}': gt_perPreds[i, j]
                            for j, pred_class in enumerate(class_names)}
            self.tensorboard.add_scalars(
                f'{title}_PredDist/gt-{class_name}', gt_pred_dict, epoch)

        # 为每个类别添加预测标签下的真值标签分布
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
            raise NotImplementedError('请进行 dataset_init()')
        else:
            return self._dataset_cls

    @property
    def model_cls(self):
        if self._model_cls is None:
            raise NotImplementedError('请进行 model_init()')
        else:
            return self._model_cls

    @property
    def datasets(self):
        if self._datasets is None:
            raise NotImplementedError('请进行 dataset_create()')
        else:
            return self._datasets

    @property
    def dataloaders(self):
        if self._dataloaders is None:
            raise NotImplementedError('请进行 dataloader_create()')
        else:
            return self._dataloaders

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise NotImplementedError('请进行 optimizer_create()')
        else:
            return self._optimizer

    @property
    def scheduler(self):
        if self._scheduler is None:
            warnings.warn('若使用scheduler，需进行 scheduler_create()，否则忽略此警告')
            return None
        else:
            return self._scheduler

    @property
    def criterion(self):
        if self._criterion is None:
            raise NotImplementedError('请进行 criterion_create()')
        return self._criterion

    @staticmethod  # 定义为静态方法
    def path_verification(path):
        """
        检查指定路径是否存在，如不存在则抛出异常。
        Args:
            path: str，文件夹路径
        Returns:
            path: str，合法路径
        Raises:
            FileNotFoundError: 如果路径不存在
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"指定的路径不存在: {path}")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"指定的路径不是目录: {path}")
        return path

    def model_init(self, model_cls):
        """
        模型不一定只有一个，所以不直接实例化

        :param model_cls: 优化器类 (如 torch.optim.Adam)
        """
        self._model_cls = model_cls

    def dataset_init(self, dataset_cls):
        self._dataset_cls = dataset_cls

    def optimizer_create(self, optimizer_cls, model_params, **kwargs):
        """
        初始化优化器

        :param optimizer_cls: 优化器类 (如 torch.optim.Adam)
        :param model_params: 模型参数 (model.parameters())
        :param kwargs: 优化器关键字参数
        """
        self._optimizer = optimizer_cls(model_params, **kwargs)

    def scheduler_create(self, scheduler_cls, **kwargs):
        self._scheduler = scheduler_cls(self.optimizer, **kwargs)

    def criterion_create(self, criterion_cls, **kwargs):
        self._criterion = criterion_cls(**kwargs).cuda()
