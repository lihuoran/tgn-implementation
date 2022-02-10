from dataclasses import dataclass
from logging import Logger


@dataclass
class WorkflowContext:
    logger: Logger
    dry_run: bool = False

    @property
    def dry_run_iter_limit(self) -> int:
        return 5

    @property
    def dry_run_data_limit(self) -> int:
        return 2000
