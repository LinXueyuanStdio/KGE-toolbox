import json
from typing import Optional, List, Dict, Union

import dataclasses


@dataclasses.dataclass
class TrainerState:
    """
    A class containing the :class:`Trainer` inner state that will be saved along the model and optimizer
    when checkpointing.

    .. note::

        In all this class, one step is to be understood as one update step. When using gradient accumulation, one
        update step may require several forward and backward passes: if you use :obj:`gradient_accumulation_steps=n`,
        then one update step requires going throuch `n` batches.

    Args:
        epoch (:obj:`float`, `optional`):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        init_step (:obj:`int`, `optional`, defaults to 0):
            init_step <= current_step <= max_steps
        current_step (:obj:`int`, `optional`, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (:obj:`int`, `optional`, defaults to 0):
            The number of update steps to do during the current training.
        total_flos (:obj:`float`, `optional`, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (:obj:`List[Dict[str, float]]`, `optional`):
            The list of logs done since the beginning of training.
        best_metric (:obj:`float`, `optional`):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (:obj:`str`, `optional`):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be :obj:`True` for one process).
        is_hyper_param_search (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way data will be logged in TensorBoard.
    """

    epoch: Optional[float] = None
    init_step: int = 0
    current_step: int = 0
    max_steps: int = 500000
    num_train_epochs: int = 0
    total_flos: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def save_to_json(self, json_path: str):
        """ Save the content of this instance in JSON format inside :obj:`json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """ Create an instance from the content of :obj:`json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))
