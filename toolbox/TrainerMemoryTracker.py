import gc
import inspect
import threading

from toolbox.Framework import is_torch_cuda_available, is_psutil_available


class TrainerMemoryTracker:
    """
    A helper class that tracks cpu and gpu memory.

    This class will silently skip unless ``psutil`` is available. Install with ``pip install psutil``.

    When a stage completes, it can pass metrics dict to update with the memory metrics gathered during this stage.

    Example ::

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()
        code ...
        metrics = {"train_runtime": 10.5}
        self._memory_tracker.stop_and_update_metrics(metrics)

    At the moment GPU tracking is only for ``pytorch``, but can be extended to support ``tensorflow``.

    To understand this class' intricacies please read the documentation of :meth:`~transformers.Trainer.log_metrics`.

    """

    # map trainer methods to metrics prefix
    stages = {
        "__init__": "init",
        "train": "train",
        "evaluate": "eval",
        "predict": "test",
    }

    def __init__(self, skip_memory_metrics=False):

        self.skip_memory_metrics = skip_memory_metrics

        if not is_psutil_available():
            # soft dependency on psutil
            self.skip_memory_metrics = True

        if self.skip_memory_metrics:
            return

        import psutil  # noqa

        if is_torch_cuda_available():
            import torch

            self.torch = torch
            self.gpu = {}
        else:
            self.torch = None

        self.process = psutil.Process()

        self.cur_stage = None
        self.cpu = {}
        self.init_reported = False

    def derive_stage(self):
        """ derives the stage/caller name automatically """
        caller = inspect.currentframe().f_back.f_back.f_code.co_name
        if caller in self.stages:
            return self.stages[caller]
        else:
            raise ValueError(
                f"was called from {caller}, but only expect to be called from one of {self.stages.keys()}"
            )

    def cpu_mem_used(self):
        """ get resident set size memory for the current process """
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_mem_used_peak = -1

        while True:
            self.cpu_mem_used_peak = max(self.cpu_mem_used(), self.cpu_mem_used_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def start(self):
        """ start tracking for the caller's stage """
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        self.cur_stage = stage

        gc.collect()

        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()
            self.torch.cuda.empty_cache()

        # gpu
        if self.torch is not None:
            self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()

        # cpu
        self.cpu_mem_used_at_start = self.cpu_mem_used()

        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def stop(self, stage):
        """ stop tracking for the passed stage """

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # this sends a signal to peak_monitor_func to complete its loop
        self.peak_monitoring = False

        # first ensure all objects get collected and their memory is freed
        gc.collect()

        if self.torch is not None:
            self.torch.cuda.empty_cache()

        # concepts:
        # - alloc_delta:  the difference of allocated memory between the end and the start
        # - peaked_delta: the difference between the peak memory and the current memory
        # in order to know how much memory the measured code consumed one needs to sum these two

        # gpu
        if self.torch is not None:
            self.gpu_mem_used_now = self.torch.cuda.memory_allocated()
            self.gpu_mem_used_peak = self.torch.cuda.max_memory_allocated()
            self.gpu[self.cur_stage] = dict(
                alloc=(self.gpu_mem_used_now - self.gpu_mem_used_at_start),
                peaked=max(0, self.gpu_mem_used_peak - self.gpu_mem_used_now),
            )

        # cpu
        self.cpu_mem_used_now = self.cpu_mem_used()
        self.cpu[self.cur_stage] = dict(
            alloc=(self.cpu_mem_used_now - self.cpu_mem_used_at_start),
            peaked=max(0, self.cpu_mem_used_peak - self.cpu_mem_used_now),
        )

        # reset - cycle finished
        self.cur_stage = None

    def update_metrics(self, stage, metrics):
        """ stop tracking for the passed stage """
        if self.skip_memory_metrics:
            return

        # deal with nested calls of eval during train - simply ignore those
        if self.cur_stage is not None and self.cur_stage != stage:
            return

        # since we don't have a way to return init metrics, we push them into the first of train/val/predict
        stages = [stage]
        if not self.init_reported:
            stages.insert(0, "init")
            self.init_reported = True

        for stage in stages:
            for t in ["alloc", "peaked"]:
                if stage in self.cpu and t in self.cpu[stage]:
                    metrics[f"{stage}_mem_cpu_{t}_delta"] = self.cpu[stage][t]
                if self.torch is not None and stage in self.gpu and t in self.gpu[stage]:
                    metrics[f"{stage}_mem_gpu_{t}_delta"] = self.gpu[stage][t]

    def stop_and_update_metrics(self, metrics=None):
        """ combine stop + update in one call for simpler code """
        if self.skip_memory_metrics:
            return

        stage = self.derive_stage()
        self.stop(stage)

        # init doesn't have metrics to update so we just save that data for later stages to retrieve
        if metrics is not None:
            self.update_metrics(stage, metrics)
