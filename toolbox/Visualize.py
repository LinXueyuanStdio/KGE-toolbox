# 可视化
from torch.utils.tensorboard import SummaryWriter


def get_writer(log_dir: str, comments=""):
    return SummaryWriter(log_dir, comments)


def add_scalar(writer, name: str, value, step_num: int):
    writer.add_scalar(name, value, step_num)


def add_result(writer, result, step_num: int):
    left2right = result["left2right"]
    right2left = result["right2left"]
    using_time = result["time"]
    sorted(left2right)
    sorted(right2left)
    for i in left2right:
        add_scalar(writer, i, left2right[i], step_num)
    for i in right2left:
        add_scalar(writer, i, right2left[i], step_num)
    add_scalar(writer, "using time (s)", using_time, step_num)
