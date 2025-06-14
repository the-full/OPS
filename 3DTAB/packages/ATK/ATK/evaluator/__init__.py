from .basic_evaluator import BasicEvaluator
from .transfer_evaluator import TransferEvaluator

evaluators = {
    'BasicEvaluator': BasicEvaluator,
    'TransferEvaluator': TransferEvaluator,
}

from .utils.hook import hooks
