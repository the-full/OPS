import ipdb

from .basic_hook import BasicHook


class DebugHook(BasicHook):
    def __init__(
        self,
        debug_on_dataloader_loop_begin=False,
        debug_before_attack = False,
        debug_after_attack = False,
        debug_before_defense = False,
        debug_after_defense = False,
        debug_on_dataloader_loop_end=False
    ):
        self.debug_on_dataloader_loop_begin = debug_on_dataloader_loop_begin
        self.debug_before_attack = debug_before_attack
        self.debug_after_attack = debug_after_attack
        self.debug_before_defense = debug_before_defense
        self.debug_after_defense = debug_after_defense
        self.debug_on_dataloader_loop_end = debug_on_dataloader_loop_end

    def on_dataloader_loop_begin(self, evaluator):
        ipdb.set_trace(cond=self.debug_on_dataloader_loop_begin)

    def before_attack(self, evaluator, data_dict):
        ipdb.set_trace(cond=self.debug_before_attack)

    def after_attack(self, evaluator, data_dict):
        ipdb.set_trace(cond=self.debug_after_attack)

    def before_defense(self, evaluator, data_dict):
        ipdb.set_trace(cond=self.debug_before_defense)

    def after_defense(self, evaluator, data_dict):
        ipdb.set_trace(cond=self.debug_after_defense)

    def on_dataloader_loop_end(self, evaluator):
        ipdb.set_trace(cond=self.debug_on_dataloader_loop_end)
