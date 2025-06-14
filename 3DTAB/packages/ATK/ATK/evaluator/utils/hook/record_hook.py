import os.path as osp

from ATK.utils.metric import (
    chamfer_distance,
    hausdorff_distance,
    knn_outlier_distance,
    curvature_std_distance,
)
from ATK.utils.ops import estimate_normal_pytorch3d

from .basic_hook import BasicHook


class RecordHook(BasicHook):
    def __init__(
        self,
        victim_models: list[str],
        record_chamfer   = dict(open=False, reduce=['mean']),
        record_hausdorff = dict(open=False, reduce=['mean']),
        record_knn       = dict(open=False, reduce=['mean'], k=4, alpha=1.05),
        record_csd       = dict(open=False, reduce=['mean'], k=4),
        record_budget    = dict(open=True, reduce=['mean']),
        mode             = 'attach',
        file_path        = './result.txt',
    ):
        self.victim_models = victim_models
        self.record_cha    = record_chamfer
        self.record_hau    = record_hausdorff
        self.record_knn    = record_knn
        self.record_csd    = record_csd
        self.record_bgt    = record_budget
        assert mode in ['attach', 'cover']
        self.mode = mode
        self.file_path  = file_path


    @staticmethod
    def get_test_case_info(evaluator):
        meta_info     = evaluator.meta_info.copy()
        model_name    = meta_info['model_name']
        attacker_name = meta_info['attacker_name']
        defenser_name = meta_info['defenser_name']
        return model_name + '@' + attacker_name + '@' + defenser_name
        

    def after_attack(self, evaluator, data_dict):
        delta = data_dict['delta']
        adv_pcs = data_dict['xyz']
        ori_pcs = adv_pcs - delta

        if self.record_cha['open']:
            chamfer_dis = chamfer_distance(ori_pcs, adv_pcs, False, 'max')
            metric_list = evaluator.metrics_helper(
                'chamfer', chamfer_dis, self.record_cha['reduce']
            )
            evaluator.update_metrics(metric_list)

        if self.record_hau['open']:
            hausdorff_dis = hausdorff_distance(ori_pcs, adv_pcs, False, 'max')
            metric_list = evaluator.metrics_helper(
                'hausdorff', hausdorff_dis, self.record_hau['reduce']
            )
            evaluator.update_metrics(metric_list)

        if self.record_knn['open']:
            k, alpha = self.record_knn['k'], self.record_knn['alpha']
            knn_dis = knn_outlier_distance(adv_pcs, k=k, alpha=alpha)
            metric_list = evaluator.metrics_helper(
                'knn', knn_dis, self.record_knn['reduce']
            )
            evaluator.update_metrics(metric_list)

        if self.record_csd['open']:
            ori_normals = data_dict.get('normal', None)
            if ori_normals is None:
                ori_normals = estimate_normal_pytorch3d(ori_pcs)
            curvature_dis = curvature_std_distance(
                ori_pcs, adv_pcs, ori_normals, k=self.record_csd['k']
            )
            metric_list = evaluator.metrics_helper(
                'csd', curvature_dis, self.record_csd['reduce']
            )
            evaluator.update_metrics(metric_list)

        if self.record_bgt['open']:
            budget_type = evaluator.budget_type
            if budget_type == 'point_linfty':
                delta_norm = delta.norm(dim=-1, p=2)
                used_budget = delta_norm.abs().max(dim=-1)[0]
            elif budget_type == 'linfty':
                used_budget = delta.abs().max(dim=-1)[0]
            else: 
                raise NotImplemented

            metric_list = evaluator.metrics_helper(
                'budget', used_budget, self.record_csd['reduce']
            )
            evaluator.update_metrics(metric_list)



    def on_test_end(self, evaluator):
        case_info   = self.get_test_case_info(evaluator)
        header_line = ['case']
        data_line   = [f'{case_info:<30}']
        
        final_metrics = evaluator.metrics

        # NOTE: parse TASR
        tasr_result = [' 0.00'] * len(self.victim_models)
        for metric in final_metrics.values():
            if metric.metric_name.endswith('_TASR_mean'):
                victim_model = metric.metric_name.split('_TASR_mean')[0]
                try:
                    idx = self.victim_models.index(victim_model)
                except ValueError:
                    continue
                tasr_result[idx] = get_aligned_asr_text(metric.result)

        header_line += self.victim_models
        data_line   += tasr_result

        # NOTE: parse chamfer distance metrics
        dist_result = ['0.0000e+00'] * len(self.record_cha['reduce'])
        for metric in final_metrics.values():
            if 'chamfer' in metric.metric_name:
                reduce_method = metric.metric_name.split('_')[-1]
                try:
                    idx = self.record_cha['reduce'].index(reduce_method)
                except ValueError:
                    continue
                dist_result[idx] = f'{metric.result:.4e}'

        header_line += ['chamfer_' + reduce for reduce in self.record_cha['reduce']]
        data_line   += dist_result

        # NOTE: parse hausdorff distance metrics
        dist_result = ['0.0000e+00'] * len(self.record_hau['reduce'])
        for metric in final_metrics.values():
            if 'hausdorff' in metric.metric_name:
                reduce_method = metric.metric_name.split('_')[-1]
                try:
                    idx = self.record_hau['reduce'].index(reduce_method)
                except ValueError:
                    continue
                dist_result[idx] = f'{metric.result:.4e}'

        header_line += ['hausdorff_' + reduce for reduce in self.record_hau['reduce']]
        data_line   += dist_result

        # NOTE: parse knn distance metrics
        dist_result = ['0.0000e+00'] * len(self.record_knn['reduce'])
        for metric in final_metrics.values():
            if 'knn' in metric.metric_name:
                reduce_method = metric.metric_name.split('_')[-1]
                try:
                    idx = self.record_knn['reduce'].index(reduce_method)
                except ValueError:
                    continue
                dist_result[idx] = f'{metric.result:.4e}'

        header_line += ['knn_' + reduce for reduce in self.record_knn['reduce']]
        data_line   += dist_result

        # NOTE: parse csd distance metrics
        dist_result = ['0.0000'] * len(self.record_csd['reduce'])
        for metric in final_metrics.values():
            if 'csd' in metric.metric_name:
                reduce_method = metric.metric_name.split('_')[-1]
                try:
                    idx = self.record_csd['reduce'].index(reduce_method)
                except ValueError:
                    continue
                dist_result[idx] = f'{metric.result:.4f}'

        header_line += ['csd_' + reduce for reduce in self.record_csd['reduce']]
        data_line   += dist_result

        # NOTE: parse used budget metrics
        dist_result = ['0.000'] * len(self.record_bgt['reduce'])
        for metric in final_metrics.values():
            if 'budget' in metric.metric_name:
                reduce_method = metric.metric_name.split('_')[-1]
                try:
                    idx = self.record_bgt['reduce'].index(reduce_method)
                except ValueError:
                    continue
                dist_result[idx] = f'{metric.result:.3f}'

        header_line += ['budget_' + reduce for reduce in self.record_bgt['reduce']]
        data_line   += dist_result

        self.record_result(header_line, data_line)


    def record_result(self, header_line, data_line, **kwargs):
        file_exists  = osp.exists(self.file_path)
        is_no_header = False

        if file_exists:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Read the last line to check if it's blank
                lines = f.readlines()
                if lines:
                    is_no_header = not lines[-1].strip()

        data_str = ' | '.join(data_line)
        data_str = f'{data_str} |'

        with open(self.file_path, 'a', newline='') as f:
            if is_no_header:
                header_str = ' | '.join(header_line)
                header_str = f'| {header_str} |'
                f.write(header_str + '\n')
            f.write(data_str + '\n')


def get_aligned_asr_text(asr):
    if asr == 1.0:
        return '100.0'
    elif asr >= 0.1:
        return f'{asr*100:.2f}'
    else:
        return f' {asr*100:.2f}'
