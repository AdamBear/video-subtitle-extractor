from spleeter.separator import Separator
import shutil
from pyvad import split
import numpy as np
import IPython.display as ipd
import soundfile as sf
import soundfile
import numpy as np
import os
import pathlib
import time

import torch
import librosa

from bytesep.models.lightning_modules import get_model_class
from bytesep.separator import Separator
from dotenv import load_dotenv


srate = 16000
separator = None

load_dotenv()
not_local_test = os.getenv('LOCAL_TEST') == "false"
is_autodl = os.getenv('AUTO-DL') == "true"

if is_autodl:
    model_path = "/python/models/"
else:
    if not_local_test:
        model_path = "/mnt/models/"
    else:
        model_path = "d:/apps/nlp/models/"

sep_vocal, sep_accom = None, None

def get_sep(is_vocals=True):
    input_channels = 2
    output_channels = 2
    target_sources_num = 1
    segment_samples = int(44100 * 30.)
    batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_type = "ResUNet143_Subbandtime"

    if model_type == "ResUNet143_Subbandtime":
        if is_vocals:
            checkpoint_path = os.path.join(model_path,
                                           "resunet143_subbtandtime_vocals_8.7dB_500k_steps_v2.pth")
        else:
            checkpoint_path = os.path.join(model_path,
                                           "resunet143_subbtandtime_accompaniment_16.4dB_500k_steps_v2.pth")

    # Get model class.
    Model = get_model_class(model_type)

    # Create model.
    model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])

    # Move model to device.
    model.to(device)

    # Create separator.
    separator = Separator(
        model=model,
        segment_samples=segment_samples,
        batch_size=batch_size,
        device=device,
    )

    return separator


def separate_wav(audio_path, need_clean=True, vocal=True):
    global sep_vocal, sep_accom
    sample_rate = 44100

    path, file = os.path.split(audio_path)
    name, ext = os.path.splitext(file)

    # Load audio.
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=False)

    # audio = audio[None, :]

    input_dict = {'waveform': audio}

    # Separate
    separate_time = time.time()

    if vocal:
        if not sep_vocal:
            sep_vocal = get_sep(True)
        sep_wav = sep_vocal.separate(input_dict)
    else:
        if not sep_accom:
            sep_accom = get_sep(False)
        sep_wav = sep_accom.separate(input_dict)

    print('Separate time: {:.3f} s'.format(time.time() - separate_time))

    # Write out separated audio.
    if vocal:
        t_vocals_file = os.path.join(path, name + "_vocals.wav")
        soundfile.write(file=t_vocals_file, data=sep_wav.T, samplerate=sample_rate)

        if need_clean:
            t_vocals_file = get_wav(t_vocals_file)

        return t_vocals_file
    else:
        t_bgm_file = os.path.join(path, name + "_bgm.wav")
        soundfile.write(file=t_bgm_file, data=sep_wav.T, samplerate=sample_rate)
        return t_bgm_file


def to_timecode(milliseconds):
    """
    将视频帧转换成时间
    :param frame_no: 视频的帧号，i.e. 第几帧视频帧
    :returns: SMPTE格式时间戳 as string, 如'01:02:12:32' 或者 '01:02:12;32'
    """
    seconds = milliseconds // 1000
    milliseconds = int(milliseconds % 1000)
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
    if minutes >= 60:
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
    smpte_token = ','
    # cap.release()
    return "%02d:%02d:%02d%s%02d" % (hours, minutes, seconds, smpte_token, milliseconds)


def split_wav(input_file, hop_length=10):
    data, fs = librosa.load(input_file)
    data = librosa.resample(data, fs, srate)
    data *= 0.95 / np.abs(data).max()

    edges = split(data, srate, fs_vad=srate, hop_length=hop_length, vad_mode=2)
    return edges


def export_split_wavs(input_file, hop_length=10):
    path, name = os.path.split(input_file)
    name = "".join(name.split(".")[:-1])

    path = os.path.join(path, name)
    os.makedirs(path, exist_ok=True)
    data, fs = librosa.load(input_file)
    data = librosa.resample(data, fs, srate)
    data *= 0.95 / np.abs(data).max()

    edges = split(data, srate, fs_vad=srate, hop_length=hop_length, vad_mode=2)
    index = 0
    for edge in edges:
        index += 1
        seg = data[edge[0]:edge[1]]
        sf.write(os.path.join(path, name + "_" + str(index) + ".wav"), seg, 16000, subtype='PCM_16')
    return path


def get_wav(input_file):
    output_file = input_file + "_mono_16k.wav"
    cmd = f"ffmpeg -y -i {input_file} -acodec pcm_s16le -ac 1 -ar 16000 {output_file}"
    ret = os.system(cmd)
    if ret == 0:
        return output_file
    else:
        return ""


def get_full_wav(input_file):
    output_file = input_file + "_full.wav"
    cmd = f"ffmpeg -y -i {input_file} {output_file}"
    ret = os.system(cmd)
    if ret == 0:
        return output_file
    else:
        return ""



#use spleeter
def separate_wav2(wav, need_clean=True):
    global separator
    if not separator:
        separator = Separator('spleeter.json')

    path, file = os.path.split(wav)
    name, ext = os.path.splitext(file)

    separator.separate_to_file(wav, path)

    bgm_file = os.path.join(path, name, "accompaniment.wav")
    vocals_file = os.path.join(path, name, "vocals.wav")

    if need_clean:
        cleaned_vocals_file = get_wav(vocals_file)
    else:
        cleaned_vocals_file = vocals_file

    t_bgm_file = os.path.join(path, name + "_bgm.wav")
    t_vocals_file = os.path.join(path, name + "_vocals.wav")
    # move files here
    if os.path.isfile(bgm_file) and os.path.isfile(cleaned_vocals_file):
        shutil.move(bgm_file, t_bgm_file)
        shutil.move(cleaned_vocals_file, t_vocals_file)
        shutil.rmtree(os.path.join(path, name))
        return t_vocals_file, t_bgm_file
    else:
        return "", ""


def extract_and_split(input_mp4, job_id, separate, hop=20, cb_url=None, retry=0, only_split_seconds=False):
    if separate:
        wav_file = get_full_wav(input_mp4)
        bgm_file = separate_wav(wav_file, vocal=False)
        vocals_file = separate_wav(wav_file, vocal=True)
    else:
        wav_file = get_full_wav(input_mp4)
        if hop > 0:
            # vocals_file, _ = separate_wav(wav_file, False)
            vocals_file = separate_wav(wav_file, vocal=True)
        else:
            vocals_file = ""
        bgm_file = ""

    if hop > 0:
        splits = split_wav(vocals_file, hop)
        if only_split_seconds:
            splits_times = [str(t[0] / 16000) for t in splits]
        else:
            splits_times = [[to_timecode(t[0] / 16), to_timecode(t[1] / 16)] for t in splits]
    else:
        splits_times = []

    return {"vocals": vocals_file, "bgm": bgm_file, "wav": wav_file, "splits": splits_times}
