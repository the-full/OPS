from .basic_hook import BasicHook
from .save_hook import SaveHook
from .debug_hook import DebugHook
from .record_hook import RecordHook

hooks = {
    'SaveHook': SaveHook,
    'DebugHook': DebugHook,
    'RecordHook': RecordHook,
    'none': BasicHook
}
