from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log


class Pipeline:


    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict
        )

        self.config['device'], self.config['use_ddp'] = init_device() 

        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )


        use_fp16 = self.config.get('use_fp16', False)
        accelerator_kwargs = dict(log_with='tensorboard', project_dir=self.project_dir)
        if use_fp16:
            accelerator_kwargs['mixed_precision'] = 'fp16'
        self.accelerator = Accelerator(**accelerator_kwargs)
        self.config['accelerator'] = self.accelerator


        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'Device: {self.config["device"]}')

        self.raw_dataset = get_dataset(dataset_name)(self.config)

        self.log('231')
        self.log(self.raw_dataset)





        self.split_datasets = self.raw_dataset.split()

        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)

        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)



        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)


        self.log(self.model)
        self.log(self.model.n_parameters)

        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:

            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        # DataLoader


        train_dataloader = DataLoader(
            self.tokenized_datasets['train'],
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            collate_fn=self.tokenizer.collate_fn['train']
        )
        val_dataloader = DataLoader(
            self.tokenized_datasets['val'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['val']
        )
        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )




        if self.config.get('mode') == 'test':
            # test-only：跳过训练，直接加载指定 ckpt
            ckpt_path = self.config.get('ckpt_path')
            if not ckpt_path:
                # auto-find: latest ckpt matching run_id in ckpt_dir
                import glob
                run_id = self.config.get('run_id', '')
                ckpt_dir = self.config.get('ckpt_dir', 'ckpt/')
                pattern = os.path.join(ckpt_dir, f'{run_id}-*.pth')
                matches = sorted(glob.glob(pattern))
                if matches:
                    ckpt_path = matches[-1]
                else:
                    ckpt_path = self.trainer.saved_model_ckpt
            self.log(f'[Test-only] Loading checkpoint from {ckpt_path}')
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.config['device']))
            self.model = self.model.to(self.config['device'])

            # DDBC evaluation
            if self.trainer.ddbc_enabled:
                from genrec.evaluate_ddbc import evaluate_ddbc_tiger
                _, test_recall = evaluate_ddbc_tiger(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.config['device'],
                    predict_nums=self.trainer.ddbc_predict_nums,
                    multipliers=self.trainer.ddbc_multipliers,
                    seed=self.trainer.ddbc_seed,
                    split='test',
                    config=self.config,
                    predict_mode=self.trainer.ddbc_predict_mode,
                )
                self.log(f'[Test-only] DDBC test recall={test_recall:.4f}')
            else:
                self.model, test_dataloader = self.accelerator.prepare(self.model, test_dataloader)
                test_results = self.trainer.evaluate(test_dataloader)
                self.log(f'Test Results: {test_results}')
            self.trainer.end()
            return
        else:
            self.trainer.fit(train_dataloader, val_dataloader)
            self.accelerator.wait_for_everyone()
            self.model = self.accelerator.unwrap_model(self.model)
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))

        self.model, test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader
        )
        if self.accelerator.is_main_process:
            self.log(f'Loaded best model checkpoint')

        test_results = self.trainer.evaluate(test_dataloader)

        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')

        self.trainer.end()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
