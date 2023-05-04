import torch.nn as nn
import torch
from transformers import Wav2Vec2Model

from typing import Optional


class PretrainedWav2Vec2ForTransducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("./wav2vec_pretrained")


    def forward(self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None):


        outputs = self.wav2vec2(
            x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state
        lengths = [sample.shape[0] for sample in last_hidden_state]

        return last_hidden_state, lengths






    

