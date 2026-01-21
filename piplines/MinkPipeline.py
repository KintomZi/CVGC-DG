import os
import gc
import time
import json
import torch
import random
import numpy as np
import MinkowskiEngine as ME
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from utils.file import paths_load_and_divide
from datasets.MinkDataset import collate_fn_mink
from pipelines.BasePipeline import PointBasePipeline, seed_everything, BaseRecorder
from pipelines.BasePipeline import PredictionProcessor as PredsProcessor


class MinkowskiPipeline(PointBasePipeline):
    def __init__(self, config):
        self.global_seed = config.env.seed  # 从配置中获取全局种子点
        seed_everything(self.global_seed)  # 设置全局种子点
        super().__init__(config)
        self.model = None

    def datasets_create(self):
        args_file = self.config.file
        # 将数据集准备逻辑封装成独立函数，提高可读性
        if self.dataset_cls is None:
            raise ValueError('')
        folder_train = self.path_verification(args_file.folder_train)
        folder_test = self.path_verification(args_file.folder_test)

        list_train, list_val = paths_load_and_divide(folder_train, 1, '.npy')

        list_test, _ = paths_load_and_divide(folder_test, 1, '.npy')

        self._datasets = {
            'train': self.dataset_cls(self.config, dataset_path=list_train, mode='train', transforms=True),

            'validation': self.dataset_cls(self.config, dataset_path=list_test, mode='val', transforms=False),

            'test': self.dataset_cls(self.config, dataset_path=list_test, mode='test', transforms=False),
        }

    def dataloaders_create(self):
        args_env = self.config.env
        args_model = self.config.model

        self._dataloaders = {
            name: DataLoader(
                dataset,
                batch_size={
                    'train': args_model.train_batch_size,
                    'validation': args_model.val_batch_size,
                    'test': args_model.test_batch_size,
                }[name],
                shuffle=(name == 'train'),  # 仅训练集打乱数据
                collate_fn=collate_fn_mink(),  # 确保 collate_fn 兼容所有数据集
                num_workers=args_env.num_workers,
                pin_memory=False
            )
            for name, dataset in self.datasets.items()
        }

    def model_create(self):
        args_model = self.config.model
        self.model = self.model_cls(args_model.input_features_dim, args_model.output_class_dim).to(self.device)

    def _record_config(self):
        conf_dict = OmegaConf.to_container(self.config, resolve=True)
        self.recorder.logger.info(json.dumps(conf_dict, indent=2))
        self.recorder.logger.info(f"随机种子设置为{self.global_seed}，初始化数据集...")
        self.recorder.logger.info(f'模型加载...\n{self.model}\n')
        self.recorder.logger.info(f'优化器加载...\n{self.optimizer}\n')

        if self.scheduler is None:
            self.recorder.logger.warning(f'调度器未设置参数\n')
        else:
            scheduler_config = {k: str(v) for k, v in self.scheduler.__dict__.items()}
            self.recorder.logger.info(f'调度器加载...\n{json.dumps(scheduler_config, indent=2)}\n')

        loss_config = {k: str(v) for k, v in self.criterion.__dict__.items()}
        self.recorder.logger.info(f'损失函数加载...\n{json.dumps(loss_config, indent=2)}\n')
        self.recorder.logger.info(
            f'\ntensorboard --logdir={self.recorder.ckpt_folder}/tensorboard --bind_all --port 6007\n')

    def _load_weights(self, weights_pretrain, weights_eval=True):
        args_dataset = self.config.dataset
        self.recorder.logger.info('加载已有训练权重...')

        if not os.path.exists(weights_pretrain):
            raise ValueError('已有训练权重 不存在，请重新导入')
        else:
            self.model.load_state_dict(torch.load(weights_pretrain))
            metrics_pre, metrics_text_pre = None, None
            # 是否进行权重评估
            if weights_eval:
                _, preds_pre, gt_pre = self.eval_epoch(self.model, 'test')  # 使用临时模型进行验证
                metrics_pre = PredsProcessor.compute_metrics(gt=gt_pre,
                                                             preds=preds_pre,
                                                             ignore_labels=args_dataset.ignore_label,
                                                             label2name=args_dataset.label_to_names)
                metrics_text_pre = PredsProcessor.metrics2text(metrics_pre)
        return metrics_pre, metrics_text_pre

    def training_module(self, weights_pretrain=None, eval_pretrain=True, trainingVal=True, trainingTest=True):
        seed_everything(self.global_seed)  # 确保训练开始时重置种子点
        train_start_timestamp = datetime.now()

        args_file = self.config.file
        args_model = self.config.model
        args_dataset = self.config.dataset

        # 初始化日志文件
        self.recorder = BaseRecorder(self.path_verification(args_file.result_folder), train_start_timestamp)
        self._record_config()

        # 基于已有权重进行训练
        if weights_pretrain is not None:
            metrics_pre, metrics_text_pre = self._load_weights(weights_pretrain, weights_eval=eval_pretrain)
            # 是否进行已有权重的评估
            if eval_pretrain:
                self.recorder.logger.info(f'{metrics_text_pre}')
                self.recorder.metric_tensorboard('Test', 0, metrics_pre)

        ####################################################################################################################
        self.recorder.logger.info('开始训练...')
        iter_counter = 0  # 迭代次数计数器
        best_mIoU = 0
        best_metric_text = ''
        for epoch in range(1, args_model.train_epoch + 1):  # 训练过程进行多个epoch

            train_start = time.time()
            loss_train, iter_counter = self.training_epoch(epoch, iter_counter)
            train_end = time.time()

            # 每个epoch记录学习率
            self.recorder.tensorboard.add_scalar('Learning Rate/Epoch', self.optimizer.param_groups[0]['lr'], epoch)

            epoch_info = (f'Epoch: {epoch:4}/{args_model.train_epoch}  '
                          + f'lr: {self.optimizer.param_groups[0]["lr"]:.3e}  '
                          + f"TrainLoss({train_end - train_start:.2f}s): {loss_train:.10f}  ")

            # 是否进行训练时验证
            if trainingVal:
                val_start = time.time()
                loss_val, preds_val, gt_val = self.eval_epoch(self.model, 'val')
                val_end = time.time()

                metrics_val = PredsProcessor.compute_metrics(gt=gt_val,
                                                             preds=preds_val,
                                                             ignore_labels=args_dataset.ignore_label,
                                                             label2name=args_dataset.label_to_names)

                self.recorder.metric_tensorboard('Val', epoch, metrics_val)

                epoch_info += f"ValLoss({val_end - val_start:.2f}s): {loss_val:.10f}  "
                # 记录训练集和验证集的损失
                self.recorder.tensorboard.add_scalars('Loss/Epoch', {"Train": loss_train, "Val": loss_val},
                                                      epoch)
            else:
                # 记录训练集和验证集的损失
                self.recorder.tensorboard.add_scalar('Loss/Epoch/Train', loss_train, epoch)

            self.recorder.logger.info(epoch_info)

            # 权重保存与测试
            if epoch % args_model.test_interval == 0 or epoch == args_model.train_epoch:
                pth_save_path = os.path.join(self.recorder.ckpt_folder, f"trained_{epoch}.pth")
                torch.save(self.model.state_dict(), pth_save_path)  # 保存模型权重
                # 权重测试
                if trainingTest:
                    loss_test, preds_test, gt_test = self.eval_epoch(self.model, 'test')

                    metrics_test = PredsProcessor.compute_metrics(gt=gt_test,
                                                                  preds=preds_test,
                                                                  ignore_labels=args_dataset.ignore_label,
                                                                  label2name=args_dataset.label_to_names)
                    metrics_text_test = PredsProcessor.metrics2text(metrics_test)

                    self.recorder.metric_tensorboard('Test', epoch, metrics_test)

                    if metrics_test['mean_iou'] > best_mIoU:
                        best_mIoU = metrics_test['mean_iou']
                        best_metric_text = f'Test Epoch: {epoch}\n' + metrics_text_test

                    self.recorder.logger.info(f'Test Epoch: {epoch}\n{metrics_text_test}')
                torch.cuda.empty_cache()  # 释放显存

        train_end_timestamp = datetime.now()

        self.recorder.tensorboard.close()
        self.recorder.logger.info(f'训练结束 总耗时：{train_end_timestamp - train_start_timestamp}')
        self.recorder.logger.info(f'\n>>>Best MeanIOU<<<')
        self.recorder.logger.info(best_metric_text)

    def test_module(self, weight_train, test_times=1):
        args_model = self.config.model
        args_dataset = self.config.dataset
        model = self.model_cls(args_model.input_features_dim, args_model.output_class_dim)
        if not os.path.exists(weight_train):
            raise ValueError('请导入训练模型权重以作测试')
        else:
            model.load_state_dict(torch.load(weight_train))

        seeds = [self.global_seed]  # 始终包含全局种子点
        if test_times > 1:
            seeds.extend([random.randint(0, 10000) for _ in range(test_times - 1)])
        elif test_times < 1:
            raise ValueError('test_times 必须为正整数')

        for seed in seeds:
            seed_everything(seed)  # 为每次测试设置种子点
            loss_test, preds_test, gt_test = self.eval_epoch(model, 'test')

            metrics_test = PredsProcessor.compute_metrics(gt=gt_test,
                                                          preds=preds_test,
                                                          ignore_labels=args_dataset.ignore_label,
                                                          label2name=args_dataset.label_to_names)
            metrics_text_test = PredsProcessor.metrics2text(metrics_test)
            print(f'Seed: {seed}')
            print(f'{metrics_text_test}')

    def training_epoch(self, epoch, iter_num):
        args_model = self.config.model

        self.model.train()  # 设置模型为训练模式

        accum_iter_loss = 0  # 初始化 累积损失
        iter_sub_counter = iter_num  # 初始化 累积迭代(iteration)次数
        # tqdm 包装 train_loader 并显示进度
        with tqdm(self.dataloaders['train'], desc=f'Training {epoch}/{args_model.train_epoch}',
                  dynamic_ncols=True,  # 自适应终端宽度
                  leave=True,  # 训练结束后 是否 保留进度条
                  ) as tepoch:
            for batch_idx, data_zip in enumerate(tepoch):
                self.optimizer.zero_grad()  # 清零梯度
                data_all, indices = data_zip

                loss_ori, _other = self.training_step(data_all['origin'], self.model)

                loss_total = loss_ori['sem']

                loss_total.backward()  # 反向传播
                self.optimizer.step()  # 更新模型参数

                # 累加损失和迭代次数
                accum_iter_loss += loss_total.item()  # 累计当前iter损失
                iter_sub_counter += 1  # 总批次数记录

                # 每个 迭代 记录学习率和损失
                self.recorder.tensorboard.add_scalar('Learning Rate/Iteration', self.optimizer.param_groups[0]['lr'],
                                                     iter_sub_counter)
                self.recorder.tensorboard.add_scalars('Loss/Train/Iteration',
                                                      {"loss_sup": loss_ori['sem'].item(),
                                                       "loss_total": loss_total.item()}, iter_sub_counter)
                # 更新进度条显示
                tepoch.set_postfix(loss=loss_total.item(), lr=self.optimizer.param_groups[0]["lr"])
                # self.scheduler.step()  # 每个迭代更新学习率
        # self.scheduler.step()  # 每个epoch更新学习率

        avg_loss = accum_iter_loss / (iter_sub_counter - iter_num)
        return avg_loss, iter_sub_counter

    def eval_epoch(self, model, mode):
        args_model = self.config.model
        torch.cuda.empty_cache()  # 清空前轮显存缓存
        torch.cuda.synchronize()  # 确保前一轮训练所有GPU任务完成
        temp_model = self.model_cls(args_model.input_features_dim, args_model.output_class_dim).to(self.device)
        temp_model.load_state_dict(model.state_dict())
        temp_model.eval()  # 切换到评估模式

        if mode == 'test':
            eval_loader = self.dataloaders['test']
        elif mode == 'val':
            eval_loader = self.dataloaders['validation']
        else:
            raise ValueError

        val_loss = 0.0
        all_preds, all_gt = [], []

        # 禁用梯度计算以节省内存和计算资源
        with torch.no_grad():
            for datas in eval_loader:
                gt, pred, loss = self.eval_step(datas, temp_model)
                pred = pred.argmax(dim=1)
                all_preds.append(pred)
                all_gt.append(gt)
                val_loss += loss.item()

        # 处理空的验证集
        if not all_preds or not all_gt:
            raise ValueError('标签数组里面没有值')

        # 合并所有预测结果和标签，移到CPU并转换为numpy数组
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_gt = torch.cat(all_gt).cpu().numpy()
        # 计算平均验证损失
        avg_val_loss = val_loss / len(eval_loader)

        del temp_model
        gc.collect()
        torch.cuda.empty_cache()  # 清除当前评估残留缓存
        torch.cuda.synchronize()  # 确保全部释放同步完成

        return avg_val_loss, all_preds, all_gt

    def training_step(self, datas_train, model):
        point_id = datas_train['point_id'].to(self.device)
        coords = datas_train['coords'].to(self.device)
        features = datas_train['features'].to(self.device)
        # type_inds = datas_train['mix_id'].to(self.device)
        ground_truth = datas_train['ground_truth'].to(self.device)
        unique_map = datas_train['unique_map']
        inverse_map = datas_train['inverse_map']

        tensor_field = ME.TensorField(features=features, coordinates=coords)
        sparse_tensor = tensor_field.sparse()  # 稀疏张量输入

        preds_logits, _, _ = model(sparse_tensor)  # 模型前向传播
        out_preds = preds_logits.slice(tensor_field).F  # 输出恢复为稠密点特征

        del tensor_field, sparse_tensor, preds_logits

        gt_downSample = ground_truth[unique_map]

        # 过滤被忽略标签
        mask_complete = ~np.isin(ground_truth.cpu().numpy(), list(self.config.dataset.ignore_label))
        mask_complete = torch.from_numpy(mask_complete).to(self.device)
        mask_downSample = mask_complete[unique_map]

        loss_dict = {
            'sem': self.criterion(out_preds[mask_downSample], gt_downSample[mask_downSample]),
        }
        other_dict = {}

        # 逆射到完整点云的真值(筛去忽视点)，特征(筛去忽视点)，其他元素
        return loss_dict, other_dict

    def eval_step(self, datas_eval, model):

        data_dict, index = datas_eval
        single_data = data_dict['eval']

        coords = single_data['coords'].to(self.device)
        features = single_data['features'].to(self.device)
        ground_truth = single_data['ground_truth'].to(self.device)
        unique_map = single_data['unique_map'].to(self.device)
        inverse_map = single_data['inverse_map'].to(self.device)

        # 构造 Minkowski Engine 的 TensorField（稀疏输入）
        tensor_field = ME.TensorField(features=features, coordinates=coords)
        sparse_tensor = tensor_field.sparse()  # 稀疏张量输入

        preds_logits, _, _ = model(sparse_tensor)  # 模型前向传播
        out_preds = preds_logits.slice(tensor_field).F  # 输出恢复为稠密点特征

        del tensor_field, sparse_tensor, preds_logits

        # 以概率最大的为预测标签
        probs = F.softmax(out_preds, dim=1)
        probs_C = probs[inverse_map]

        gt_downSample = ground_truth[unique_map]

        # 过滤被忽略标签
        mask_complete = ~np.isin(ground_truth.cpu().numpy(), list(self.config.dataset.ignore_label))

        mask_complete = torch.from_numpy(mask_complete).to(self.device)
        mask_downSample = mask_complete[unique_map]

        # 计算损失
        loss = self.criterion(out_preds[mask_downSample], gt_downSample[mask_downSample])

        # 逆射到完整点云的真值(筛去忽视点)，最大概率标签(筛去忽视点)，损失
        return ground_truth[mask_complete], probs_C[mask_complete], loss
