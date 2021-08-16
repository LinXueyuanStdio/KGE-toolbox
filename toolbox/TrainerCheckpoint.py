import dataclasses
import torch
from torch import nn
from torch.optim import optimizer

from toolbox.TrainerState import TrainerState

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_TRAINER_STATE_DICT = "trainer_state_dict"


def load_trainer_checkpoint(model: nn.Module,
                            optim: optimizer.Optimizer,
                            checkpoint_path="./result/fr_en/checkpoint.tar") -> TrainerState:
    """Loads training checkpoint.

    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return: tuple of starting epoch id, starting step id, best checkpoint score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    state_dict = checkpoint[_TRAINER_STATE_DICT]
    state = TrainerState(state_dict)
    state.init_step = state.current_step
    return state


def save_trainer_checkpoint(model: nn.Module,
                            optim: optimizer.Optimizer,
                            state: TrainerState,
                            save_path="./result/fr_en/checkpoint.tar"):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _TRAINER_STATE_DICT: dataclasses.asdict(state),
    }, save_path)
