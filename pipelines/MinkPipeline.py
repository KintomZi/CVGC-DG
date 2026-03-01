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
        self.global_seed = config.env.seed  # Retrieve global seed from configuration
        seed_everything(self.global_seed)  # Set global seed for reproducibility
        super().__init__(config)
        self.model = None

    def datasets_create(self):
        args_file = self.config.file
        # Encapsulate dataset preparation logic into an independent function for better readability
        if self.dataset_cls is None:
            raise ValueError('')
        folder_train = self.path_verification(args_file.folder_train)
        folder_test = self.path_verification(args_file.folder_test)

        list_train, list_val = paths_load_and_divide(folder_train, 0.85, '.npy')

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
                shuffle=(name == 'train'),  # Shuffle data only for training set
                collate_fn=collate_fn_mink(),  # Ensure collate_fn is compatible with all datasets
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
        self.recorder.logger.info(f"Random seed set to {self.global_seed}, initializing datasets...")
        self.recorder.logger.info(f'Model loaded...\n{self.model}\n')
        self.recorder.logger.info(f'Optimizer loaded...\n{self.optimizer}\n')

        if self.scheduler is None:
            self.recorder.logger.warning(f'Scheduler parameters not set\n')
        else:
            scheduler_config = {k: str(v) for k, v in self.scheduler.__dict__.items()}
            self.recorder.logger.info(f'Scheduler loaded...\n{json.dumps(scheduler_config, indent=2)}\n')

        loss_config = {k: str(v) for k, v in self.criterion.__dict__.items()}
        self.recorder.logger.info(f'Loss function loaded...\n{json.dumps(loss_config, indent=2)}\n')
        self.recorder.logger.info(
            f'\ntensorboard --logdir={self.recorder.ckpt_folder}/tensorboard --bind_all --port 6007\n')

    def _load_weights(self, weights_pretrain, weights_eval=True):
        args_dataset = self.config.dataset
        self.recorder.logger.info('Loading pre-trained weights...')

        if not os.path.exists(weights_pretrain):
            raise ValueError('Pre-trained weights do not exist, please re-import')
        else:
            self.model.load_state_dict(torch.load(weights_pretrain))
            metrics_pre, metrics_text_pre = None, None
            # Whether to evaluate the loaded weights
            if weights_eval:
                _, preds_pre, gt_pre = self.eval_epoch(self.model, 'test')  # Validate using temporary model
                metrics_pre = PredsProcessor.compute_metrics(gt=gt_pre,
                                                             preds=preds_pre,
                                                             ignore_labels=args_dataset.ignore_label,
                                                             label2name=args_dataset.label_to_names)
                metrics_text_pre = PredsProcessor.metrics2text(metrics_pre)
        return metrics_pre, metrics_text_pre

    def training_module(self, weights_pretrain=None, eval_pretrain=True, trainingVal=True, trainingTest=True):
        seed_everything(self.global_seed)  # Ensure seed is reset at training start
        train_start_timestamp = datetime.now()

        args_file = self.config.file
        args_model = self.config.model
        args_dataset = self.config.dataset

        # Initialize log files
        self.recorder = BaseRecorder(self.path_verification(args_file.result_folder), train_start_timestamp)
        self._record_config()

        # Train based on pre-trained weights
        if weights_pretrain is not None:
            metrics_pre, metrics_text_pre = self._load_weights(weights_pretrain, weights_eval=eval_pretrain)
            # Whether to evaluate pre-trained weights
            if eval_pretrain:
                self.recorder.logger.info(f'{metrics_text_pre}')
                self.recorder.metric_tensorboard('Test', 0, metrics_pre)

        ####################################################################################################################
        self.recorder.logger.info('Starting training...')
        iter_counter = 0  # Iteration counter
        best_mIoU = 0
        best_metric_text = ''
        for epoch in range(1, args_model.train_epoch + 1):  # Training process spans multiple epochs

            train_start = time.time()
            loss_train, iter_counter = self.training_epoch(epoch, iter_counter)
            train_end = time.time()

            # Record learning rate at each epoch
            self.recorder.tensorboard.add_scalar('Learning Rate/Epoch', self.optimizer.param_groups[0]['lr'], epoch)

            epoch_info = (f'Epoch: {epoch:4}/{args_model.train_epoch}  '
                          + f'lr: {self.optimizer.param_groups[0]["lr"]:.3e}  '
                          + f"TrainLoss({train_end - train_start:.2f}s): {loss_train:.10f}  ")

            # Whether to perform validation during training
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
                # Record losses for training and validation sets
                self.recorder.tensorboard.add_scalars('Loss/Epoch', {"Train": loss_train, "Val": loss_val},
                                                      epoch)
            else:
                # Record training set loss
                self.recorder.tensorboard.add_scalar('Loss/Epoch/Train', loss_train, epoch)

            self.recorder.logger.info(epoch_info)

            # Weight saving and testing
            if epoch % args_model.test_interval == 0 or epoch == args_model.train_epoch:
                pth_save_path = os.path.join(self.recorder.ckpt_folder, f"trained_{epoch}.pth")
                torch.save(self.model.state_dict(), pth_save_path)  # Save model weights
                # Weight testing
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
                torch.cuda.empty_cache()  # Release GPU memory

        train_end_timestamp = datetime.now()

        self.recorder.tensorboard.close()
        self.recorder.logger.info(f'Training finished. Total elapsed time: {train_end_timestamp - train_start_timestamp}')
        self.recorder.logger.info(f'\n>>>Best MeanIOU<<<')
        self.recorder.logger.info(best_metric_text)

   def test_module(self, weight_train, test_times=1):
        args_model = self.config.model
        args_dataset = self.config.dataset
        model = self.model_cls(args_model.input_features_dim, args_model.output_class_dim)
        if not os.path.exists(weight_train):
            raise ValueError('Please import trained model weights for testing')
        else:
            model.load_state_dict(torch.load(weight_train))

        seeds = [self.global_seed]  # Always include the global seed
        if test_times > 1:
            seeds.extend([random.randint(0, 10000) for _ in range(test_times - 1)])
        elif test_times < 1:
            raise ValueError('test_times must be a positive integer')

        for seed in seeds:
            seed_everything(seed)  # Set seed for each test run
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

        self.model.train()  # Set model to training mode

        accum_iter_loss = 0  # Initialize accumulated loss
        iter_sub_counter = iter_num  # Initialize accumulated iteration counter
        # Wrap train_loader with tqdm and display progress
        with tqdm(self.dataloaders['train'], desc=f'Training {epoch}/{args_model.train_epoch}',
                  dynamic_ncols=True,  # Adapt to terminal width
                  leave=True,  # Whether to keep progress bar after training completes
                  ) as tepoch:
            for batch_idx, data_zip in enumerate(tepoch):
                self.optimizer.zero_grad()  # Zero out gradients
                data_all, indices = data_zip

                loss_ori, _other = self.training_step(data_all['origin'], self.model)

                loss_total = loss_ori['sem']

                loss_total.backward()  # Backpropagation
                self.optimizer.step()  # Update model parameters

                # Accumulate loss and iteration count
                accum_iter_loss += loss_total.item()  # Accumulate current iteration loss
                iter_sub_counter += 1  # Record total batch count

                # Record learning rate and loss at each iteration
                self.recorder.tensorboard.add_scalar('Learning Rate/Iteration', self.optimizer.param_groups[0]['lr'],
                                                     iter_sub_counter)
                self.recorder.tensorboard.add_scalars('Loss/Train/Iteration',
                                                      {"loss_sup": loss_ori['sem'].item(),
                                                       "loss_total": loss_total.item()}, iter_sub_counter)
                # Update progress bar display
                tepoch.set_postfix(loss=loss_total.item(), lr=self.optimizer.param_groups[0]["lr"])
                # self.scheduler.step()  # Update learning rate at each iteration
        # self.scheduler.step()  # Update learning rate at each epoch

        avg_loss = accum_iter_loss / (iter_sub_counter - iter_num)
        return avg_loss, iter_sub_counter

    def eval_epoch(self, model, mode):
        args_model = self.config.model
        torch.cuda.empty_cache()  # Clear GPU memory cache from previous round
        torch.cuda.synchronize()  # Ensure all GPU tasks from previous round are completed
        temp_model = self.model_cls(args_model.input_features_dim, args_model.output_class_dim).to(self.device)
        temp_model.load_state_dict(model.state_dict())
        temp_model.eval()  # Switch to evaluation mode

        if mode == 'test':
            eval_loader = self.dataloaders['test']
        elif mode == 'val':
            eval_loader = self.dataloaders['validation']
        else:
            raise ValueError

        val_loss = 0.0
        all_preds, all_gt = [], []

        # Disable gradient computation to save memory and computational resources
        with torch.no_grad():
            for datas in eval_loader:
                gt, pred, loss = self.eval_step(datas, temp_model)
                pred = pred.argmax(dim=1)
                all_preds.append(pred)
                all_gt.append(gt)
                val_loss += loss.item()

        # Handle empty validation set
        if not all_preds or not all_gt:
            raise ValueError('No values in label array')

        # Concatenate all predictions and labels, move to CPU and convert to numpy arrays
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_gt = torch.cat(all_gt).cpu().numpy()
        # Calculate average validation loss
        avg_val_loss = val_loss / len(eval_loader)

        del temp_model
        gc.collect()
        torch.cuda.empty_cache()  # Clear residual cache from current evaluation
        torch.cuda.synchronize()  # Ensure all releases are synchronized and completed

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
        sparse_tensor = tensor_field.sparse()  # Sparse tensor input

        preds_logits, _, _ = model(sparse_tensor)  # Model forward pass
        out_preds = preds_logits.slice(tensor_field).F  # Restore output to dense point features

        del tensor_field, sparse_tensor, preds_logits

        gt_downSample = ground_truth[unique_map]

        # Filter out ignored labels
        mask_complete = ~np.isin(ground_truth.cpu().numpy(), list(self.config.dataset.ignore_label))
        mask_complete = torch.from_numpy(mask_complete).to(self.device)
        mask_downSample = mask_complete[unique_map]

        loss_dict = {
            'sem': self.criterion(out_preds[mask_downSample], gt_downSample[mask_downSample]),
        }
        other_dict = {}

        # Map back to complete point cloud: ground truth (excluding ignored points), 
        # features (excluding ignored points), and other elements
        return loss_dict, other_dict

    def eval_step(self, datas_eval, model):

        data_dict, index = datas_eval
        single_data = data_dict['eval']

        coords = single_data['coords'].to(self.device)
        features = single_data['features'].to(self.device)
        ground_truth = single_data['ground_truth'].to(self.device)
        unique_map = single_data['unique_map'].to(self.device)
        inverse_map = single_data['inverse_map'].to(self.device)

        # Construct Minkowski Engine TensorField (sparse input)
        tensor_field = ME.TensorField(features=features, coordinates=coords)
        sparse_tensor = tensor_field.sparse()  # Sparse tensor input

        preds_logits, _, _ = model(sparse_tensor)  # Model forward pass
        out_preds = preds_logits.slice(tensor_field).F  # Restore output to dense point features

        del tensor_field, sparse_tensor, preds_logits

        # Predict label with highest probability
        probs = F.softmax(out_preds, dim=1)
        probs_C = probs[inverse_map]

        gt_downSample = ground_truth[unique_map]

        # Filter out ignored labels
        mask_complete = ~np.isin(ground_truth.cpu().numpy(), list(self.config.dataset.ignore_label))

        mask_complete = torch.from_numpy(mask_complete).to(self.device)
        mask_downSample = mask_complete[unique_map]

        # Calculate loss
        loss = self.criterion(out_preds[mask_downSample], gt_downSample[mask_downSample])

        # Map back to complete point cloud: ground truth (excluding ignored points), 
        # predicted labels with highest probability (excluding ignored points), and loss
        return ground_truth[mask_complete], probs_C[mask_complete], loss
