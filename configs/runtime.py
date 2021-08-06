from dataclasses import dataclass
from typing import Optional

from .utils import PythonConfig


@dataclass(init=False)
class RuntimeConfig(PythonConfig):
    seed: int
    output_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    tb_log_dir: Optional[str] = None
    log_dir: Optional[str] = None
    # Will create a run id and keep checkpoints and tensorboard in a separate folder
    label_this_run: bool = False
    resume_from: Optional[str] = None
    evaluate_only: bool = False
    debug: bool = False
