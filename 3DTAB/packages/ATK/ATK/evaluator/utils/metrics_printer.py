from .metric_recorder import MetricRecorder

class MetricsPrinter:
    def __init__(self, rules):
        self.rules = rules

    def __call__(self, logger, metric: MetricRecorder):
        metric_name   = metric.metric_name
        metric_result = metric.result
        for condition, print_func in self.rules.items():
            if condition(metric_name):
                print_func(metric_name, metric_result, logger)
                break


def print_as_percentage(metric_name, metric_value, logger):
    logger.info(f'{metric_name}: {metric_value * 100:.4f}%')

def print_in_scientific_notation(metric_name, metric_value, logger):
    logger.info(f'{metric_name}: {metric_value:.4e}')

def is_asr_metric(metric_name):
    return 'ASR' in metric_name

rules = {
    is_asr_metric: print_as_percentage,
    (lambda metric_name: True) : print_in_scientific_notation
}

metrics_printer = MetricsPrinter(rules)
