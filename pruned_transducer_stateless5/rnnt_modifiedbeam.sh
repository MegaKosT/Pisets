#!/bin/bash
python pretrained.py --bpe-model ./spiece.model --method modified_beam_search --sound_files ./test_audio_16000.wav ./test_audio_16000.wav
