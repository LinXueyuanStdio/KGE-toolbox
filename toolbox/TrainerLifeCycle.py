from enum import Enum
from typing import Dict, List, Union

from dataclasses import dataclass

from toolbox.KGArgs import TrainingArguments
from toolbox.TrainerState import TrainerState


@dataclass
class TrainerControl:
    """
    A class that handles the :class:`~transformers.Trainer` control flow. This class is used by the
    :class:`~transformers.TrainerCallback` to activate some switches in the training loop.

    Args:
        should_training_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the training should be interrupted.

            If :obj:`True`, this variable will not be set back to :obj:`False`. The training will just stop.
        should_epoch_stop (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the current epoch should be interrupted.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next epoch.
        should_save (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be saved at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should be evaluated at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
        should_log (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the logs should be reported at this step.

            If :obj:`True`, this variable will be set back to :obj:`False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """ Internal method that resets the variable for a new training. """
        self.should_training_stop = False

    def _new_epoch(self):
        """ Internal method that resets the variable for a new epoch. """
        self.should_epoch_stop = False

    def _new_step(self):
        """ Internal method that resets the variable for a new step. """
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerLifeCycleObserver:
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args (:class:`~transformers.TrainingArguments`):
            The training arguments used to instantiate the :class:`~transformers.Trainer`.
        state (:class:`~transformers.TrainerState`):
            The current state of the :class:`~transformers.Trainer`.
        control (:class:`~transformers.TrainerControl`):
            The object that is returned to the :class:`~transformers.Trainer` and can be used to make some decisions.
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`):
            The model being trained.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for encoding the data.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (:obj:`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        eval_dataloader (:obj:`torch.utils.data.dataloader.DataLoader`, `optional`):
            The current dataloader used for training.
        metrics (:obj:`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event :obj:`on_evaluate`.
        logs  (:obj:`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event :obj:`on_log`.

    The :obj:`control` object is the only one that can be changed by the callback, in which case the event that changes
    it should return the modified version.

    The argument :obj:`args`, :obj:`state` and :obj:`control` are positionals for all events, all the others are
    grouped in :obj:`kwargs`. You can unpack the ones you need in the signature of the event using them. As an example,
    see the code of the simple :class:`~transformer.PrinterCallback`.

    Example::

        class PrinterCallback(TrainerCallback):

            def on_log(self, args, state, control, logs=None, **kwargs):
                _ = logs.pop("total_flos", None)
                if state.is_local_process_zero:
                    print(logs)
    """

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the :class:`~transformers.Trainer`.
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


class CustomEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class Event(CustomEnum):
    ON_PRE_CREATE = "onPreCreate"
    ON_CREATE = "onCreate"
    ON_POST_CREATE = "onPostCreate"
    ON_PREPARE_DATALOADER = "onPrepareDataloader"
    ON_COMPUTE_LOSS = "onComputeLoss"
    ON_RESUME = "onResume"
    ON_START = "onStart"
    ON_STOP = "onStop"
    ON_DESTROY = "onDestroy"


class State(CustomEnum):
    DESTROYED = -1
    INITIALIZED = 0
    CREATED = 1
    STARTED = 2
    RESUMED = 3


def getStateAfter(event: Event) -> State:
    if event is Event.ON_STOP:
        return State.RESUMED


def downEvent(state: State) -> Event:
    pass


def upEvent(state: State) -> Event:
    pass


class LifecycleObserver:
    def onStateChange(self):
        pass


class LifeCycle:
    def addObserver(self, observer: LifecycleObserver):
        pass

    def removeObserver(self, observer: LifecycleObserver):
        pass

    def getCurrentState(self):
        pass


class LifecycleOwner:
    def getLifeCycle(self) -> LifeCycle:
        pass


def minState(state1: State, state2: State) -> State:
    return state2 if state2 is not None and state2 < state1 else state1


@dataclass
class ObserverWithState:
    mState: State
    mLifecycleObserver: LifecycleObserver

    def dispatchEvent(self, event: Event):
        new_state = getStateAfter(event)
        self.mState = minState(self.mState, new_state)
        self.mLifecycleObserver.onStateChange()
        self.mState = new_state


def eldest(observer_map: Dict[LifecycleObserver, ObserverWithState]) -> Union[None, State]:
    return None


def newest(observer_map: Dict[LifecycleObserver, ObserverWithState]) -> Union[None, State]:
    return None


class LifecycleRegistry(LifeCycle):
    def __init__(self, provider: LifecycleOwner):
        self.mLifecycleOwner = provider
        self.mState: State = State.INITIALIZED
        self.mObserverMap: Dict[LifecycleObserver, ObserverWithState] = {}

        self.mNewEventOccurred = False
        self.mHandlingEvent = False
        self.mAddingObserverCounter = 0
        self.mParentStates: List[State] = []

    def getObserverCount(self) -> int:
        return len(self.mObserverMap)

    def getCurrentState(self) -> State:
        return self.mState

    def moveToState(self, next_state: State):
        if self.mState == next_state:
            return
        self.mState = next_state
        if self.mHandlingEvent or self.mAddingObserverCounter != 0:
            self.mNewEventOccurred = True
            return
        self.mHandlingEvent = True
        self.sync()
        self.mHandlingEvent = False

    def isSynced(self) -> bool:
        if self.getObserverCount() == 0:
            return True

    def sync(self):
        if self.mLifecycleOwner is None:
            return
        while not self.isSynced():
            self.mNewEventOccurred = False

    def markState(self, state: State):
        self.moveToState(state)

    def handleLifecycleEvent(self, event: Event):
        next_state = getStateAfter(event)
        self.markState(next_state)

    def addObserver(self, observer: LifecycleObserver):
        pass

    def removeObserver(self, observer: LifecycleObserver):
        pass


class TrainerLifeCycle:
    """ Internal class that just calls the list of callbacks in order. """

    def __init__(self, callbacks):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control
