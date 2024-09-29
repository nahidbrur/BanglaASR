# --- Building Manifest Files --- #
import os
import json
import librosa
import pandas as pd


"""
Data format
{
    "audio_filepath": "path/to/audio.wav", 
    "duration": 3.45, 
    "text": "this is a nemo tutorial"
}

"""

def read_data(data_path, data_chunk = 0):
    if data_chunk==0:
        data = pd.read_csv(data_path, sep='\t')
    else:
        data = pd.read_csv(data_path, sep='\t')[:data_chunk]
    data.columns = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"]
    data = data.drop(["client_id","up_votes", "down_votes", "age", "gender", "accents", "locale", "segment"], axis=1)
    return data

def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

# Function to build a manifest
def build_manifest(train_data, input_path):

    """
    
    
    """
    file_name, sentences = train_data["path"].tolist(), train_data["sentence"].tolist()
    data = []
    for file, sentence in zip(file_name, sentences):
        print(file, sentence)
        audio_path = os.path.join(input_path, file)
        duration = librosa.core.get_duration(filename=audio_path)
        print(duration)
        metadata = {
            "audio_filepath": audio_path,
            "duration": duration,
            "text": sentence
            }
        data.append(metadata)
    return data
                

if __name__ == "__main__":

    mp3_path = "/media/sayan/hdd/ARS/whisper/dataset/clips"
    # input_tsv = "/media/sayan/hdd/ARS/whisper/dataset/validated.tsv"
    input_tsv = "/media/sayan/hdd/ARS/whisper/dataset/dev.tsv"

    output_path = "/media/sayan/hdd/ARS/whisper/dataset/val.json"

    train_data = read_data(input_tsv)
    data = build_manifest(train_data, mp3_path)

    save_jsonl(data, output_path)

    # print(train_data)

    # train_transcripts = data_dir + '/an4/etc/an4_train.transcription'
    # train_manifest = data_dir + '/an4/train_manifest.json'
    # if not os.path.isfile(train_manifest):
    #     build_manifest(train_transcripts, train_manifest, 'an4/wav/an4_clstk')
    #     print("Training manifest created.")

    # test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
    # test_manifest = data_dir + '/an4/test_manifest.json'
    # if not os.path.isfile(test_manifest):
    #     build_manifest(test_transcripts, test_manifest, 'an4/wav/an4test_clstk')
    #     print("Test manifest created.")
    # print("***Done***")