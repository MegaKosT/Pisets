from pandas import DataFrame, read_parquet
from os import walk
import json
from librosa import load
import pyarrow.parquet as pq
import pyarrow as pa
from IPython.display import Audio
import wave



def make_RuDev_ds():
    wav_list = []
    text_list = []

    for root, _, files in walk("./RuDevices"):
        if len(files):
            for file in files:
                audio_file = file[:-4] + '.wav'
                text_file = file[:-4] + '.txt'

                with wave.open(root + '/' + audio_file, 'rb') as bytes_audio_f:
                    data = bytes_audio_f.readframes(-1)
                bts = {'bytes':data, 'path':None}
                wav_list.append(bts)

                with open(root + '/' + text_file, 'r') as trans_f:
                    trans = trans_f.read()
                text_list.append(trans)

                files.remove(audio_file)
                files.remove(text_file)
            
    ds = DataFrame({'audio':wav_list, 'transcription':text_list})
    ds_len = ds.shape[0]
    train_top_ind = round(ds_len*0.7)
    val_top_ind = round(ds_len*0.9)

    train = ds.iloc[:train_top_ind, :]
    val = ds.iloc[train_top_ind:val_top_ind, :]
    test = ds.iloc[val_top_ind:, :]
    return train, val, test

    
def read_sber():
    wav_list=[]
    text_list=[]
    with open('./test/crowd/manifest.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for i in range(9000):
        json_str = json_list[i]
        result = json.loads(json_str)
        file_name = result['id']+'.wav'
        trans = result['text']

        with wave.open('./test/crowd/files/'+file_name, 'rb') as bytes_audio_f:
            data = bytes_audio_f.readframes(-1)
        wav_list.append(data)
        text_list.append(trans)

    sber_ds = DataFrame({'audio':wav_list, 'transcription':text_list})
    val_sber = sber_ds.iloc[:6000, :]
    test_sber = sber_ds.iloc[6000:, :]

    return val_sber, test_sber

def create_ds():
    train, val, test = make_RuDev_ds()
    print('RuDev readed')
    val_sber, test_sber = read_sber()
    print('sber readed')

    val.append(val_sber)
    test.append(test_sber)
    print('ds completed')

    train_table = pa.Table.from_pandas(train)
    val_table = pa.Table.from_pandas(val)
    test_table =  pa.Table.from_pandas(test)
    print('tables completed')

    pq.write_table(train_table, 'train.parquet')
    pq.write_table(test_table, 'test.parquet')
    pq.write_table(val_table, 'val.parquet')

create_ds()
#my_parq = read_parquet('/home/konstantin/py_projects/diplom/test.parquet')