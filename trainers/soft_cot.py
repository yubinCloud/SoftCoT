from transformers import Trainer

from typing import Optional


class SoftCoTTrainer(Trainer):
    """
    Trainer for SoftCoT.
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
            self.model.save_pretrained(output_dir)
