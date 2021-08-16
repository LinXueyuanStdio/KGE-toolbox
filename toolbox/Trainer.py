import time
from typing import List

import torch
from dataclasses import dataclass
from torch.nn import Module

from toolbox.KGArgs import TrainingArguments
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds
from toolbox.TrainerCheckpoint import load_trainer_checkpoint
from toolbox.TrainerState import TrainerState


@dataclass
class Trainer:
    """Trainer interface"""
    def on_prepare_dataloader(self):
        pass

    def on_compute_loss(self, model: Module, state: TrainerState, args: TrainingArguments) -> torch.Tensor:
        pass

    def skip_current_step(self, state: TrainerState, args: TrainingArguments) -> bool:
        return False


class ComposableTrainer:
    def __init__(self,
                 trainers: List[Trainer],
                 output: OutputSchema,
                 model: Module,
                 args: TrainingArguments,
                 ):
        self.trainers = trainers
        self.args = args
        self.output = output
        self.out = output.pathSchema
        self.log = output.logger.info
        set_seeds(self.args.seed)
        # force device and distributed setup init explicitly
        args._setup_devices

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full fp16 eval - since the model needs to be half'ed first
        # 4. Sharded DDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
                self.is_model_parallel
                or (args.deepspeed and args.do_train)
                or (args.fp16_full_eval and not args.do_train)
        ):
            self.place_model_on_device = False

        if self.place_model_on_device:
            model = model.to(args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.learning_rate
        )

        self.state = TrainerState()
        self.on_prepare_dataloader()

    def on_prepare_dataloader(self):
        for i in self.trainers:
            i.on_prepare_dataloader()

    def run_train(self):
        if self.args.resume_training:
            self.state = load_trainer_checkpoint(self.model, self.optim, self.out.checkpoint_path())
            self.log("恢复模型后，查看一下模型状态")
            self.state.best_metric = self.run_test()
        self.log("start training")

        progbar = Progbar(max_step=self.state.max_steps)
        start_time = time.time()
        for step in range(self.state.init_step, self.state.max_steps):
            self.state.current_step = step
            self.model.train()
            self.optim.zero_grad()

            loss_list = []
            loss = None
            for i in self.trainers:
                if i.skip_current_step(self.state, self.args):
                    continue
                loss_i = i.on_compute_loss(self.model, self.state, self.args)
                loss_list.append(loss_i)
                if loss is None:
                    loss = loss_i
                else:
                    loss = loss + loss_i
            if loss is None:
                continue
            loss = loss / len(loss_list)
            loss.backward()
            self.optim.step()

            progbar.update(step + 1, [("loss", loss.item())]
                           + [("loss-%d" % trainer_i, loss_i.item()) for trainer_i, loss_i in enumerate(loss_list)]
                           + [("cost", round((time.time() - start_time)))])
            if step > self.state.current_step and step % self.args.eval_steps == 0:
                score = self.run_test()
                last_score = self.state.best_metric
                self.log("loss = %s, " % str(loss.item())[:8] + ", ".join([
                    "loss-%d=%s" % (trainer_i, str(loss_i.item())[:8]) for trainer_i, loss_i in enumerate(loss_list)
                ]))
                self.save_extra(score, last_score)
                if score > last_score:
                    self.log("保存 (+%.2f%%) (%.2f%% > %.2f%%)" %
                             ((score - last_score) * 100, score * 100, last_score * 100))
                    self.state.best_metric = score

    def run_test(self):
        """
        compute your metric here
        return metric. it will be saved if higher than before
        """
        return 0

    def save_extra(self, score, last_score):
        pass
