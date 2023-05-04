import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoConfig, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
import logging

import torch
import torchaudio
import torchaudio.functional as F

import random
from random import choices
import numpy as np

import sentencepiece as spm
from train import get_transducer_model
from icefall.utils import AttributeDict
import k2


def get_effects(sample_rate):
    all_effects = [
        {'effect':[["treble", f"{random.randint(-100,100)}"]], 'prob': 0.3},
        {'effect':choices([["tempo", f"{random.randint(50,150)/100}"], ["speed", f"{random.randint(50,150)/100}"]], k=1), 'prob': 0.7},
        {'effect':[["reverb", "-w", "1"],["channels", "1"]], 'prob': 0.3},
        {'effect':[["pitch", f"{random.randint(-700,700)}"]], 'prob': 0.7},
        {'effect':[["lowpass", "-1", f"{random.uniform(0.01,500)}"]], 'prob': 0.5},
        {'effect':[["highpass", "-1", f"{random.randint(500,1500)}"]], 'prob': 0.5},
        {'effect':[["rate", f"{sample_rate}"]], 'prob': 1}
    ]
    probed_effects = []
    for foo in all_effects:
        eff = foo['effect']
        prob = foo['prob']
        if random.randint(0,100)/100 < prob:
            probed_effects.extend(eff)
    # print(probed_effects)  #если нужно проверить, какие эффекты мы наложили на аудио
    return probed_effects


def aug_example(SAMPLE_WAV):
    # Load the data
    waveform1 = torch.Tensor([SAMPLE_WAV.get('array')])
    sample_rate1 = SAMPLE_WAV.get('sampling_rate')
    
    # Define effects
    effects = get_effects(sample_rate1)

    # Apply effects
    waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(waveform1, sample_rate1, effects)

    print(waveform1.shape, sample_rate1)
    print(waveform2.shape, sample_rate2)

    plot_waveform(waveform1, sample_rate1, title="Original", xlim=(-0.1, 3.2))
    plot_specgram(waveform1, sample_rate1, title="Original", xlim=(0, 3.04))

    plot_waveform(waveform2, sample_rate2, title="Augment", xlim=(-0.1, 3.2))
    plot_specgram(waveform2, sample_rate2, title="Augment", xlim=(0, 3.04))
    augment_audio = {'path': None, 'array': waveform2.tolist(), 'sampling_rate': sample_rate2}
    return augment_audio


def aug_sample(sample):
    sampling_rate = 16000
    sample = torch.Tensor([sample])
    effects = get_effects(sampling_rate)    
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(sample, sampling_rate, effects)
    wav = wav[0].tolist()
    return wav



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    device: torch.device
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        
        input_features = [{"input_values": aug_sample(feature["input_values"])} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        batch["x_lens"] = torch.tensor([feature["x_lens"] for feature in features])

#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 return_tensors="pt",
#             )

        # replace padding with -100 to ignore loss correctly
        #labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        batch["labels"] = k2.RaggedTensor([feature["labels"] for feature in features])
        

        return batch
    



def run_training():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    
    print("loading processor and bpe")
    processor = Wav2Vec2Processor.from_pretrained("bond005/wav2vec2-large-ru-golos")
    sp = spm.SentencePieceProcessor()
    sp.load('./spiece.model')
    
    
    print("loading model")
    params = AttributeDict({'vocab_size':sp.get_piece_size(),'encoder_dim':512,  'decoder_dim':512,'joiner_dim':512,
                        'blank_id':sp.piece_to_id("<blk>"), 'context_size':2 })
    model = get_transducer_model(params)
    
    
    print("loading dataset")
    ds = load_dataset("MegaKosT/RuDevSberDS")
    
    print("processing dataset")
    
    def prepare_dataset(batch):
        #print(batch)
        audio = batch["audio"]

        # batched output is "un-batched"
        
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["x_lens"] = len(batch["input_values"])
        batch["labels"] = sp.encode(batch["transcription"])
        return batch

    ds_train = ds['train'].select(range(1000))
    ds_test = ds['test'].select(range(500))
    
    ds_train = ds_train.filter(
    lambda it1: (it1["transcription"] is not None) and (len(it1["transcription"].strip()) > 0)
    )
    ds_test = ds_test.filter(
        lambda it1: (it1["transcription"] is not None) and (len(it1["transcription"].strip()) > 0)
    )
    
    ds_train = ds_train.map(prepare_dataset, remove_columns=ds_train.column_names)
    ds_test = ds_test.map(prepare_dataset, remove_columns=ds_test.column_names)
    
    print("start of training")
    data_collator = DataCollatorCTCWithPadding(processor=processor,device = device, padding=True)

    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        group_by_length=True,
        per_device_train_batch_size= 16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=15,
        #gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )
    
    wer_metric = load_metric("wer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=processor.feature_extractor)

    trainer.train()
    
run_training()