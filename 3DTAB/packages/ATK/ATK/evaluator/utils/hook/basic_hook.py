class BasicHook(object):
    def __init__(self, *args, **kwargs):
        pass

    def on_test_begin(self, evaluator):
        ...

    def on_dataloader_loop_begin(self, evaluator):
        ...

    def before_attack(self, evaluator, data_dict):
        ...

    def after_attack(self, evaluator, data_dict):
        ...

    def before_defense(self, evaluator, data_dict):
        ...

    def after_defense(self, evaluator, data_dict):
        ...

    def on_dataloader_loop_end(self, evaluator):
        ...

    def on_test_end(self, evaluator):
        ...
