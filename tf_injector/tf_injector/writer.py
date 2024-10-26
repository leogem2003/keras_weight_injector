import csv
import os
from datetime import datetime

from tf_injector.utils import REPORT_HEADER, DEFAULT_REPORT_DIR


class CampaignWriter:
    """
    class CampaignWriter
    writes the results of an injection campaign to a csv file
    result path: file_dir/dataset/network/
    """

    def __init__(self, dataset: str, network: str, file_dir: str = DEFAULT_REPORT_DIR):
        target_dir = os.path.join(file_dir, dataset, network)
        os.makedirs(target_dir, exist_ok=True)
        self.filepath = os.path.join(target_dir, self.get_filename(dataset, network))

    def __enter__(self) -> "CampaignWriter":
        write_header = not os.path.exists(self.filepath)
        self.file = open(self.filepath, "a")
        self.writer = csv.writer(self.file)

        if write_header:
            self.writer.writerow(REPORT_HEADER)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    @staticmethod
    def get_filename(dataset: str, network: str) -> str:
        return f"{dataset}_{network}_{datetime.now().strftime('%y%m%d_%H%M')}.csv"

    def write_gold(self, gold_row: tuple[int, int]):
        padding = [None]
        row = ("GOLDEN", *(padding * 3), *gold_row, *(padding * 4))
        self.writer.writerow(row)

    def write_fault(
        self,
        fault_id: int,
        num_injections: int,
        fault: tuple[int, ...],
        fault_metrics: tuple[int, ...],
    ):
        row = (fault_id, *fault, num_injections, *fault_metrics)
        self.writer.writerow(row)
