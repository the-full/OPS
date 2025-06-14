import torch

from .basic_evaluator import BasicEvaluator


class TransferEvaluator(BasicEvaluator):
    def __init__(
        self,
        meta_info,
        surrogate_model,
        attacker,
        defenser,
        victim_models_dict,
        evaluate_dataset,
        **kwargs
    ):
        super(TransferEvaluator, self).__init__(
            meta_info = meta_info,
            attacker  = attacker, 
            defenser  = defenser, 
            model     = surrogate_model, 
            dataset   = evaluate_dataset, 
            **kwargs
        )
        self.victim_models = victim_models_dict

    @staticmethod
    def make_dataloader(data_dict, batch_size):
        from torch.utils.data import Dataset, DataLoader
        class TempDataset(Dataset):
            def __init__(self, data_dict):
                self.data_dict = data_dict

            def __len__(self):
                return self.data_dict['xyz'].shape[0]

            def __getitem__(self, idx):
                data = {}
                for k, v in self.data_dict.items():
                    data.update({k: v[idx]})
                return data

        dataloader = DataLoader(
            TempDataset(data_dict), 
            batch_size=batch_size,
            shuffle=False, 
            drop_last=False
        )
        return dataloader

    def record_atk_result(self, data_dict):
        metric_list = self.evaluate_transferbility(data_dict)
        self.update_metrics(metric_list)
        return data_dict 

    def evaluate_transferbility(self, data_dict):
        metric_list = []
        with torch.no_grad():
            for model_name, (victim_model, batch_size) in self.victim_models.items():
                dataloader = self.make_dataloader(data_dict, batch_size=batch_size)
                for data_dict in dataloader:
                    labels = data_dict['category'].view(-1)
                    logits = victim_model.__predict__(data_dict)
                    atk_success = self.get_atk_success(logits, labels, self.target)
                    metric_list.extend(self.metrics_helper(f'{model_name}_TASR', atk_success, ['mean']))

        return metric_list
