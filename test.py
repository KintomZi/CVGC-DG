# @Note    : MinkowskiNet的基础训练的模型/The basic training model of MinkowskiNet
import os

import torch
from omegaconf import OmegaConf
from datasets.MinkDataset import MinkowskiDataset
from models import MinkUNet34
from pipelines.MinkPipeline import MinkowskiPipeline

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')

class BasicPipeline(MinkowskiPipeline):
    def __init__(self, config):
        super().__init__(config)
        self.model_init(MinkUNet34)
        self.dataset_init(MinkowskiDataset)
        self.model_create()
        self.datasets_create()
        self.dataloaders_create()

        self.optimizer_create(torch.optim.Adam, self.model.parameters(), lr=config.model.learning_rate,)
        # self.scheduler_create(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.95)
        self.criterion_create(torch.nn.CrossEntropyLoss, ignore_index=-1)


if __name__ == '__main__':
    config_path = os.path.join(CONFIG_DIR, 'H3D.yaml')
    args = OmegaConf.load(config_path)

    mink = BasicPipeline(args)
  
    # mink.training_module(weights_pretrain=None, eval_pretrain=True, trainingTest=True, trainingVal=True)
  
    test_pth = f'/home/huiwei/Desktop/Code/result_temp/_valid_/H3D_AS/trained_300.pth'
    mink.test_module(test_pth)
