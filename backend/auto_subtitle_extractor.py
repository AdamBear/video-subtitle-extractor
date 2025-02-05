import sys

from dotenv import load_dotenv
import os

load_dotenv()
not_local_test = os.getenv('LOCAL_TEST') == "false"
is_autodl = os.getenv('AUTO-DL') == "true"

if is_autodl:
    image_ai_demos_path = "/data/image_ai_demos/"
    sys.path.insert(-1, image_ai_demos_path)
else:
    if not not_local_test:
        image_ai_demos_path = "D:/apps/nlp/prompt/image_ai_demos"
        sys.path.insert(-1, image_ai_demos_path)

#os.environ['AUTO-DL'] = "true"

import paddle
import hashlib
import json
import os
import shutil
from collections import Counter
import string
import config
import cv2
import requests
import unicodedata
from Levenshtein import ratio
from config import interface_config

from scenedetect.detectors import ContentDetector
# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
from PIL import Image, ImageChops

import librosa

from paddlespeech.cli.asr import ASRExecutor

from pyvad import split
import copy
from paddlenlp import Taskflow
import paddlehub as hub

import time
import numpy as np
from split_wav import extract_and_split

import azure.cognitiveservices.speech as speechsdk

from moviepy import editor
from moviepy.config import change_settings

if is_autodl:
    change_settings({"IMAGEMAGICK_BINARY": r"/usr/bin/convert"})
else:
    change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe"})

from api_keys import subscription_key, endpoint, speech_key

from image_fill import TestOptions, process_image

fill_model = None


# to multiple excute at once, encapulate to API
def to_english(q):
    import os, requests, uuid, json

    path = '/translate'
    params = {
        'api-version': '3.0',
        'from': 'zh',
        'to': ['en']
    }
    constructed_url = endpoint + path

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': 'global',
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    is_str = True
    if isinstance(q, str):
        body = [{
            'text': q
        }]
    else:
        is_str = False
        body = [{
            'text': t
        } for t in q]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    json.dumps(response, sort_keys=True, indent=4, separators=(',', ': '))
    if is_str:
        return response[0]["translations"][0]["text"]
    else:
        return [r["translations"][0]["text"] for r in response]


def get_english_dubbed(input_video, ocr_result, fps, voice_name="en-US-AriaNeural", no_voice=False):
    base_name = ".".join(input_video.split(".")[:-1])
    segs = merge_ocr_subtitles(ocr_result, fps)
    englishs = to_english([t["text"] for t in segs])
    for s, e in zip(segs, englishs):
        s["english"] = e

    if voice_name:
        for i, s in enumerate(segs):
            tts_file = base_name + "_" + str(i) + "_eng_tts.wav"
            say_english(s["english"], tts_file, voice_name=voice_name)
            s["eng_tts_file"] = tts_file

    # fix the end tag to next start tag
    for i in range(len(segs)):
        if i < len(segs) - 1:
            segs[i]["end"] = segs[i + 1]["start"]

    return segs


def say_english(text, output_file, voice_name="en-US-AriaNeural", service_region="eastasia"):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Set the voice name, refer to https://aka.ms/speech/voices/neural for full list.
    # speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    speech_config.speech_synthesis_voice_name = voice_name
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    result = speech_synthesizer.speak_text_async(text).get()

    # Checks result.
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(output_file)
        return True

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")
    return False


# 直接通过ocr_subtitle进行合理的合并，单段的秒数设置最大长度，停顿间隔可以计算出来
def merge_ocr_subtitles(ocr_subtitle, fps):
    segments = []
    for s in ocr_subtitle:
        start = s[2] / fps
        end = s[3] / fps
        text = s[4]
        segments.append({"start": start, "end": end, "text": text})

    results = []
    i = 0
    while i < len(segments):
        s = segments[i]
        for j in range(i + 1, len(segments)):
            if segments[j]['start'] < s['end'] + 0.4:
                s['end'] = segments[j]['end']
                s['text'] += "," + segments[j]['text']
                i = j

                # 重新计算长度，超过5秒后就截断，防止音画过于不同步
                length = s["end"] - s["start"]
                if length > 5:
                    break
            else:
                break
        i += 1
        results.append({"start": s["start"], "end": s["end"], "text": s["text"], "len": s["end"] - s["start"]})

    return results


def gen_subtitled_video(input_video, segs, rect=None, color="white", fontsize=32, font="Keep-Calm-Medium-bold",
                        no_voice=False):

    base_name = ".".join(input_video.split(".")[:-1])

    if not no_voice:
        ret = extract_and_split(input_video, job_id=None, separate=True, hop=0)
        bgm = editor.AudioFileClip(ret["bgm"])
    else:
        bgm = None

    video = editor.VideoFileClip(input_video)
    clips = [video.subclip(0, segs[0]['start'])]

    for s in segs:
        if s["start"] > video.duration:
            break
        if s["end"] > video.duration:
            s["end"] = video.duration

        video_s = video.subclip(s['start'], s['end'])

        if not no_voice:
            audio = editor.AudioFileClip(s["eng_tts_file"])
            print("eng_tts_file", s["eng_tts_file"])
            print(video_s.duration, audio.duration)

            # change the video speed to fit the audio duration
            video_s = video_s.without_audio().fx(editor.vfx.speedx, video_s.duration / audio.duration)
            print("adaptered", video_s.duration, audio.duration)

            video_s = video_s.without_audio().set_audio(audio).set_duration(audio.duration)
            del s["eng_tts_file"]

        # 字体颜色改为"白色"， 大小可以再细调，自动字幕的位置和宽度如何定
        # 原始的背景音乐还配得上吗？ 因为视频已经拉长了。
        # font=font,
        text_clip = editor.TextClip(txt=s["english"], color=color, fontsize=fontsize, stroke_width=2,
                                    kerning=-2, interline=-1, size=(rect[2], None), method='caption').set_duration(
            video_s.duration).set_position([rect[0], rect[1]])
        video_s = editor.CompositeVideoClip([video_s, text_clip])

        clips.append(video_s)

    final_clip = editor.concatenate_videoclips(clips, method="compose")

    if not no_voice:
        video_audio_clip = final_clip.audio.volumex(2)

        # 设置背景音乐循环，时间与视频时间一致
        bgm_audio_clip = bgm.audio_loop(duration=final_clip.duration).volumex(0.8)

        # 视频声音和背景音乐，音频叠加
        audio_clip_add = editor.CompositeAudioClip([video_audio_clip, bgm_audio_clip])

        # 视频写入背景音
        final_video = final_clip.without_audio().set_audio(audio_clip_add)
    else:
        final_video = final_clip

    video_file = base_name + "_english.mp4"
    final_video.write_videofile(video_file, audio_codec='aac')

    return video_file


# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)

"""
use pyvad to split the audio and get the time-stampped ASR result
see https://github.com/F-Tag/python-vad
need install pyvad first:
$ pip install pyvad
"""

splitted_asr_executor, text_correct_model, punc_model = None, None, None


def get_models():
    # asr_model = hub.Module(name='u2_conformer_aishell')
    splitted_asr_executor = SplitASRExecutor()
    text_correct_model = Taskflow("text_correction")
    punc_model = hub.Module(name='auto_punc')
    return splitted_asr_executor, text_correct_model, punc_model


def get_wav(input_file, ar=48000):
    output_file = input_file + "_mono_{}k.wav".format(int(ar / 1000))
    cmd = f"ffmpeg -y -loglevel quiet -i {input_file} -acodec pcm_s16le -ac 1 -ar {ar} {output_file}"
    ret = os.system(cmd)
    if ret == 0:
        return output_file
    else:
        return ""


def extract_audio(source: str, srate=16000):
    return get_wav(source, srate)


def speech_recognize(file):
    global splitted_asr_executor, text_correct_model, punc_model
    if not splitted_asr_executor:
        splitted_asr_executor, text_correct_model, punc_model = get_models()

    # text = asr_model.speech_recognize(file, device='cpu')
    splitted_timelines, ret = splitted_asr_executor(model='conformer_wenetspeech',
                                                    lang='zh',
                                                    sample_rate=16000,
                                                    config=None,
                                                    # Set config and ckpt_path to None to use pretrained model.
                                                    ckpt_path=None,
                                                    decode_method='attention_rescoring',
                                                    audio_file=file,
                                                    force_yes=False,
                                                    device=paddle.get_device(),
                                                    verbose=False)

    punc_ret = []
    for text, t in zip(ret, splitted_timelines):
        if len(text) > 0:
            text_correction = text_correct_model(text)[0]
            cor_text, errors = text_correction['target'], text_correction['errors']
            print(f'[Text Correction] errors: {errors}')
            punc_text = punc_model.add_puncs(cor_text, device='cpu')[0]
        else:
            punc_text = ""

        punc_ret.append([t[0], t[1], punc_text.replace("、", "").replace("：", "").replace(":", "")])

    return punc_ret


def vad_split(audio_file, srate=16000):
    data, fs = librosa.load(audio_file)
    data = librosa.resample(data, fs, srate)
    data *= 0.95 / np.abs(data).max()
    edges = split(data, srate, fs_vad=srate, hop_length=20, vad_mode=3)
    return data, edges


def get_splitted_timelines(data, edges, srate):
    # fix the begin and end vad frames pos
    edges = copy.deepcopy(edges)
    total_len = len(data)
    edges[len(edges) - 1][1] = total_len - 1
    edges[0][0] = 0

    if len(edges) == 1:
        time_line_pos = [edges[0][0], edges[0][1], edges[0][0], edges[0][1], edges[0][1] - edges[0][0]]
        return [[pos / srate for pos in time_line_pos]]

    # merge too short vad frames(less than 1 seconds)
    new_edges = [edges[0]]
    for i in range(1, len(edges)):
        if edges[i][1] - edges[i][0] < srate:
            new_edges[len(new_edges) - 1][1] = edges[i][1]
        else:
            new_edges.append(edges[i])

    edges = new_edges

    # split at the middle of tow vad edge frames
    time_lines = []
    for i in range(len(edges)):
        if i == 0:
            time_line_pos = [edges[i][0], edges[i][1] + (edges[i + 1][0] - edges[i][1]) / 2,
                             edges[i][0], edges[i][1], edges[i][1] - edges[i][0] + (edges[i + 1][0] - edges[i][1]) / 2]
        elif 0 < i < len(edges) - 1:
            time_line_pos = [edges[i][0] - (edges[i][0] - edges[i - 1][1]) / 2,
                             edges[i][1] + (edges[i + 1][0] - edges[i][1]) / 2,
                             edges[i][0], edges[i][1],
                             edges[i][1] - edges[i][0] + (edges[i][0] - edges[i - 1][1]) / 2 + (
                                     edges[i + 1][0] - edges[i][1]) / 2]
        else:
            time_line_pos = [edges[i][0] - (edges[i][0] - edges[i - 1][1]) / 2, edges[i][1],
                             edges[i][0], edges[i][1], edges[i][1] - edges[i][0] + (edges[i][0] - edges[i - 1][1]) / 2]

        time_line = [pos / srate for pos in time_line_pos]
        time_line.append(time_line_pos[-1] / total_len)
        time_lines.append(time_line)

    return time_lines


def split_audio(inputs, splitted_timelines):
    full_audio = inputs["full_audio"]
    full_audio_len = inputs["full_audio_len"]

    cur_pos = 0
    for t in splitted_timelines:
        cur_len = int(full_audio_len * t[-1])
        audio = full_audio[:, cur_pos:cur_pos + cur_len, :]
        cur_pos += cur_len
        yield audio, paddle.to_tensor(cur_len)


class SplitASRExecutor(ASRExecutor):
    def __init__(self):
        super().__init__()
        self.change_format = False

    def postprocess(self):
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._outputs["result"]

    def __call__(self,
                 audio_file: os.PathLike,
                 model: str = 'conformer_wenetspeech',
                 lang: str = 'zh',
                 sample_rate: int = 16000,
                 config: os.PathLike = None,
                 ckpt_path: os.PathLike = None,
                 decode_method: str = 'attention_rescoring',
                 num_decoding_left_chunks: int = -1,
                 force_yes: bool = False,
                 rtf: bool = False,
                 device=paddle.get_device(),
                 verbose=False):

        if verbose:
            self.disable_task_loggers()

        """
        Python API to call an executor.
        """
        # self._check(audio_file, sample_rate, force_yes)
        paddle.set_device(device)
        self._init_from_path(model, lang, sample_rate, config, decode_method,
                             num_decoding_left_chunks, ckpt_path)

        audio_file = os.path.abspath(audio_file)
        self.preprocess(model, audio_file)

        self._inputs["full_audio"] = self._inputs["audio"]
        self._inputs["full_audio_len"] = self._inputs["audio_len"]

        data, edeges = vad_split(audio_file, sample_rate)
        self.splitted_timelines = get_splitted_timelines(data, edeges, sample_rate)
        self.audio_data = data

        if verbose:
            self.disable_task_loggers()

        ret = []
        if len(self.splitted_timelines) > 1:
            for audio, audio_len in split_audio(self._inputs, self.splitted_timelines):
                self._inputs["audio"] = audio
                self._inputs["audio_len"] = audio_len
                self.infer(model_type="conformer_wenetspeech")
                ret.append(self._outputs["result"])
        else:
            self.infer(model_type="conformer_wenetspeech")
            ret.append(self._outputs["result"])

        return self.splitted_timelines, ret


def kill_pid_ocr():
    with open("/python/video-subtitle-extractor/backend/pid_ocr.txt", "r") as f:
        pid_ocr = int(f.read())
        print(f"kill {pid_ocr}")
        os.system("kill -9 {}".format(pid_ocr))


def post_to_recognize(image_file_list):
    retry_times = 2
    wait_time = 60

    url = "http://127.0.0.1:8866/predict/ch_pp-ocrv3"

    # mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3)

    headers = {"Content-type": "application/json"}
    total_time = 0

    r = None
    while retry_times > 0:
        try:
            starttime = time.time()
            data = {'images': [], 'paths': image_file_list}
            r = requests.post(url=url, headers=headers, data=json.dumps(data))
            elapse = time.time() - starttime
            total_time += elapse
        except Exception as e:
            print(e)
            retry_times -= 1
            kill_pid_ocr()
            while wait_time > 0:
                time.sleep(1)
                wait_time -= 1
            wait_time = 60
        if r:
            ret = r.json()
            if "results" in ret:
                if len(ret["results"]) == 0:
                    if "msg" in ret:
                        print(ret["msg"])

                    retry_times -= 1
                    kill_pid_ocr()
                    while wait_time > 0:
                        time.sleep(1)
                        wait_time -= 1
                    wait_time = 60
                else:
                    if "msg" in ret:
                        print(ret["msg"])
                        break
            else:
                break

    # new_mem_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3)
    if r:
        return r.json()
    else:
        return {"msg": 'failed', "result": []}

    # if r:
    #     if new_mem_used - mem_used > 13:
    #         kill_pid_ocr()
    #         while wait_time > 0:
    #             time.sleep(1)
    #             wait_time -= 1
    #     return r.json()
    # else:
    #     return {"msg": 'failed', "result": []}


def get_hash(url):
    # encoding the string using encode()
    en = url.encode()
    # passing the encoded string to MD5
    hex_result = hashlib.md5(en)
    # printing the equivalent hexadecimal value
    return str(hex_result.hexdigest())


def is_chinese_char(uchar):
    if (uchar >= u'\u4e00' and uchar <= u'\u9fff'):
        return True
    else:
        return False


def is_all_chinese_char(sentence):
    for s in sentence:
        if not is_chinese_char(s):
            return False
    return True


def is_all_english_char(sentence):
    for s in sentence:
        if is_chinese_char(s):
            return False
    return True


def get_rec_area(rec):
    ymin = min(rec[2], rec[3])
    ymax = max(rec[3], rec[2])
    xmin = min(rec[0], rec[1])
    xmax = max(rec[1], rec[0])
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def add_mask(maskimg, area):
    for i in range(area[3]):
        for j in range(area[2]):
            maskimg[area[1] + i][area[0] + j] = 255
    return maskimg


def get_douyin_rec(rec, w, h):
    return (max(rec[0] - 70, 0), min(rec[1] + 60, w), max(rec[2] - 18, 0), min(rec[3] + 18, h))


def get_douyin_hao_rec(rec, w, h):
    return (max(rec[0] - 18, 0), min(rec[1] + 30, w), max(rec[2] - 18, 0), min(rec[3] + 18, h))


def grow_rec(rec, w, h, pad=10):
    return (max(rec[0] - 3 * pad, 0), min(rec[1] + 3 * pad, w), max(rec[2] - 2*pad, 0), min(rec[3] + 2*pad, h))


def is_float(test_string):
    try:
        float(test_string)
        return True
    except:
        return False


def get_split_spans(splits, fps, frame_count):
    split_spans = []
    # 分段分割出视频
    if "," in splits:
        last_end_frame = 0
        spans = splits.split(",")
        for i in range(len(spans)):
            end_frame = float(spans[i]) * fps
            if end_frame < last_end_frame:
                return None

            if end_frame >= frame_count:
                split_spans.append(frame_count)
                return split_spans
            if end_frame - last_end_frame < fps:
                continue
            last_end_frame = end_frame
            split_spans.append(end_frame)

        if last_end_frame < frame_count:
            split_spans.append(frame_count)

        if len(split_spans) > 0 and int(split_spans[0]) == 0:
            return split_spans[1:]

        return split_spans

    elif is_float(splits.strip()):
        span = float(splits.strip())
        i = 1
        while i * span * fps < frame_count and span > 0:
            split_spans.append(i * span * fps)
            i += 1

        split_spans.append(frame_count)
        return split_spans
    else:
        return None


def cut_video(input_mp4, output_mp4, start, end, debug=False):
    cmd = f'ffmpeg -loglevel quiet -y -ss {start} -to {end} -i {input_mp4}  {output_mp4}'
    r = os.system(cmd)
    if debug:
        print(cmd)
    if r == 0:
        return output_mp4
    else:
        return ""


def export_cover(input_mp4, output_jpeg, debug=False):
    cmd = f'ffmpeg -loglevel quiet -y -i {input_mp4} -f image2 -frames 1 {output_jpeg}'
    r = os.system(cmd)
    if debug:
        print(cmd)
    if r == 0:
        return output_jpeg
    else:
        return ""


def fix_video_fps_to_25(input1_mp4, output_file, debug=False):
    cmd = f"ffmpeg -y -i {input1_mp4} -r 25 {output_file}"
    r = os.system(cmd)
    if debug:
        print(cmd)
    if r == 0:
        return output_file
    else:
        return ""


def get_fill_model():
    global fill_model
    if fill_model:
        return fill_model

    import sys
    import image_models as models

    sys.argv.append("--port")
    sys.argv.append("8897")
    sys.argv.append("--image_dir")
    sys.argv.append("./datasets/places2sample1k_val/places2samples1k_crop256")
    sys.argv.append("--mask_dir")
    sys.argv.append("./datasets/places2sample1k_val/places2samples1k_256_mask_square128")
    sys.argv.append("--output_dir")
    sys.argv.append("./results")
    sys.argv.append("--checkpoints_dir")
    sys.argv.append(os.path.join(image_ai_demos_path, "checkpoints"))

    opt = TestOptions().parse()

    model = models.create_model(opt)
    model.eval()
    fill_model = model
    return model


class AutoSubtitleExtractor():
    """
    视频字幕提取类
    """

    def __init__(self, vd_path, export_key_frames, start_ms=-1, end_ms=-1):
        self.sub_area = None
        self.export_key_frames = export_key_frames
        self.export_cut_video = True
        self.debug = False
        self.remove_too_common = True
        self.detect_subtitle = True
        self.detect_scene = False
        self.no_cut = True
        self.splits = ""
        self.split_spans = []
        self.scenes = []
        self.asr_only = False
        self.start_ms = start_ms
        self.end_ms = end_ms
        # 字幕区域位置
        self.subtitle_area = config.SubtitleArea.LOWER_PART

        # 临时存储文件夹
        # self.temp_output_dir = os.path.join(os.path.dirname(config.BASE_DIR), 'output')
        self.temp_output_dir = os.path.join(config.TEMP_OUTPUT_DIR, get_hash(vd_path))

        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频秒数
        self.video_length = float(self.frame_count / self.fps)

        # 视频尺寸
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.h = self.frame_height
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.w = self.frame_width
        self.size = (self.frame_width, self.frame_height)

        self.start_frame = -1
        self.end_frame = -1

        # 提取的视频帧储存目录
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # 提取的字幕文件存储目录
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # 定义vsf的字幕输出路径
        self.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        # 不存在则创建文件夹
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        # 提取的原始字幕文本存储路径
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')

        self.num_frame = 0
        # 处理进度
        self.progress = 0

        self.ms_per_frame = 1000 / self.fps
        self.mask_cache = {}
        self.fill_model = None
        self.rec_result = None
        self.skip_watermark = True
        self.ocr_result = None
        self.refilled_video = None
        self.max_rect = None
        self.segments = None

    def run(self):

        if self.asr_only:
            self.export_key_frames = False
            self.detect_scene = False
            return self.generate_asr_subtitle()

        # 指定分割点时不输出关键帧，不检查场景
        if len(self.splits) > 0:
            self.export_key_frames = False
            self.detect_scene = False

        """
        运行整个提取视频的步骤
        """
        print(interface_config['Main']['StartProcessFrame'])
        self.extract_frame_by_fps()

        print(interface_config['Main']['FinishProcessFrame'])

        print(interface_config['Main']['StartFindSub'])
        # 重置进度条
        self.progress = 0
        self.extract_subtitles()
        print(interface_config['Main']['FinishFindSub'])

        print(interface_config['Main']['StartGenerateSub'])

        # 判断是否开启精准模式
        result = self.generate_subtitle_file()

        # if len(result) == 0:
        #     kill_pid_ocr()

        # if len(result) == 0:
        #     self.export_key_frames = False
        #     self.detect_scene = False
        #     self.export_cut_video = False
        #     result = self.generate_asr_subtitle()

        # 清理临时文件
        # self._delete_frame_cache()

        # 如果识别的字幕语言包含英文，则将英文分词
        # if config.REC_CHAR_TYPE in ('ch', 'EN', 'en', 'ch_tra'):
        #     reformat(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'))

        print(interface_config['Main']['FinishGenerateSub'])
        self.progress = 100
        self.ocr_result = result
        return result

    # todo: 继续优化，为了兼容去台标处理，可以考虑不切分，而使用全OCR识别，字幕区域可以使用算法去除
    def extract_frame_by_fps(self):
        """
        根据帧率，定时提取视频帧，容易丢字幕，但速度快
        """
        # 删除缓存
        self.delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0

        self.frame_key_frame = {}
        self.next_key_frame = {}
        last_key_frame = -1
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            else:
                frame_no += 1

                # 处理指定的提取范围
                if self.start_ms > 0:
                    if self.ms_per_frame * frame_no < self.start_ms:
                        continue

                if self.end_ms > 0:
                    if self.ms_per_frame * frame_no > self.end_ms:
                        break

                # 读取视频帧成功
                if self.h < 0:
                    self.h, self.w = frame.shape[0], frame.shape[1]

                if self.start_frame < 0:
                    self.start_frame = frame_no

                # 记录当前帧所对应的处理关键帧
                if last_key_frame > 0:
                    self.next_key_frame[last_key_frame] = frame_no

                last_key_frame = frame_no
                self.frame_key_frame[frame_no] = last_key_frame
                self.num_frame += 1

                filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.jpg')

                if self.export_key_frames:
                    # 保存原始视频帧，保存为jpg，临时使用png文件名
                    org_filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.org.png')
                    cv2.imwrite(filename, frame)
                    os.rename(filename, org_filename)

                if not self.no_cut:
                    frame = self._frame_preprocess(frame)

                # 帧名往前补零，后续用于排序与时间戳转换，补足8位
                # 一部10h电影，fps120帧最多也才1*60*60*120=432000 6位，所以8位足够

                # 保存截取的原视频帧
                cv2.imwrite(filename, frame)

                # 跳过剩下的帧
                for i in range(int(self.fps // config.EXTRACT_FREQUENCY) - 1):
                    ret, _ = self.video_cap.read()
                    if ret:
                        frame_no += 1
                        self.num_frame += 1
                        self.frame_key_frame[frame_no] = last_key_frame

                        # 更新进度条
                        self.progress = (frame_no / self.frame_count) * 100

        self.video_cap.release()

    def extract_subtitles(self):
        """
        提取视频帧中的字幕信息，生成一个txt文件
        """
        global image_file_list

        # 删除缓存
        if os.path.exists(self.raw_subtitle_path):
            os.remove(self.raw_subtitle_path)
        # 新建文件
        with open(self.raw_subtitle_path, mode='w+', encoding='utf-8') as f:
            # 视频帧列表
            frame_list = [i for i in sorted(os.listdir(self.frame_output_dir)) if i.endswith('.jpg')]
            image_file_list = [os.path.join(self.frame_output_dir, i).replace("\\", "/") for i in frame_list]

            rec_result = post_to_recognize(image_file_list)

            self.rec_result = rec_result

            for i, (frame, rec_ret) in enumerate(zip(frame_list, rec_result["results"])):
                if "data" in rec_ret:
                    rec_ret = rec_ret["data"]

                if len(rec_ret) == 0:
                    continue

                if "text_region" in rec_ret[0]:
                    dt_box = [r["text_region"] for r in rec_ret]
                elif "text_box_position" in rec_ret[0]:
                    dt_box = [r["text_box_position"] for r in rec_ret]
                else:
                    print(i, rec_ret)
                    print("no text region detected!")

                rec_res = [(r["text"], r["confidence"]) for r in rec_ret]
                coordinates = self._get_coordinates(dt_box)

                # 写入返回结果
                for content, coordinate in zip(rec_res, coordinates):
                    if content[1] > config.DROP_SCORE:
                        f.write(f'{os.path.splitext(frame)[0]}\t'
                                f'{coordinate}\t'
                                f'{content[0]}\n')
                        # 关闭文件
            f.close()

            shutil.copyfile(self.raw_subtitle_path, self.raw_subtitle_path + ".raw.txt")

    def find_scenes(self, threshold=30.0):
        # Create our video & scene managers, then add the detector.
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=threshold))

        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        # Each returned scene is a tuple of the (start, end) timecode.
        return scene_manager.get_scene_list()

    def generate_subtitle_file(self):
        """
        生成srt格式的字幕文件
        """
        if self.detect_scene:
            scenes_codes = self.find_scenes()
            self.scenes = [[] for i in range(len(scenes_codes))]

        # 先检测并删除可能的台标或者背景，然后才能，所以需要考虑
        if self.remove_too_common:
            self._remove_too_common()

        if self.detect_subtitle:
            self.filter_scene_text()

        subtitle_content = self._remove_duplicate_subtitle()

        if len(subtitle_content) == 0:
            shutil.copyfile(self.raw_subtitle_path + ".raw.txt", self.raw_subtitle_path)
            if self.remove_too_common:
                self._remove_too_common()
            subtitle_content = self._remove_duplicate_subtitle()

        self.coordinates_list, self.frame_contents, self.cord_frame_list = self._get_content_list()

        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        processed_subtitle = []

        print(f"splits:{self.splits}, fps:{self.fps}, frame_count:{self.frame_count}")

        # todo: auto_split的需要调用asr的声音分割处理
        split_spans = []
        if len(self.splits) > 0:
            cur_split_lines = ""
            last_span_no = 0
            last_span_frame = 1
            split_spans = get_split_spans(self.splits, self.fps, self.frame_count)
            self.split_spans = split_spans
            self.scenes = [[] for _ in range(len(split_spans))]

        with open(srt_filename, mode='w', encoding='utf-8') as f:
            for index, content in enumerate(subtitle_content):
                line_code = index + 1
                frame_no_start = int(content[0])
                frame_start = self._frame_to_timecode(frame_no_start)
                # 比较起始帧号与结束帧号， 如果字幕持续时间不足1秒，则将显示时间设为1s
                if abs(int(content[1]) - int(content[0])) < self.fps:
                    frame_no_end = int(int(content[0]) + self.fps)
                    frame_end = self._frame_to_timecode(frame_no_end)
                else:
                    frame_no_end = int(content[1])
                    frame_end = self._frame_to_timecode(frame_no_end)
                frame_content = content[2]
                processed_subtitle.append([frame_start, frame_end, frame_no_start, frame_no_end, frame_content])
                subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                f.write(subtitle_line)

                # 简单算法，按场景将字幕时间轴分组出来
                if self.detect_scene:
                    last_scene_no = 0
                    for i in range(last_scene_no, len(scenes_codes)):
                        s = scenes_codes[i]
                        if frame_no_start >= int(s[0]) and frame_no_end <= int(s[1]):
                            last_scene_no = i
                            if len(self.scenes[i]) > 0:
                                self.scenes[i][1] = frame_end
                                self.scenes[i][2].append(index)
                            else:
                                self.scenes[i] = [frame_start, frame_end, [index]]

                # 按指定分割来分场景出来
                if len(split_spans) > 0:
                    # 需要在拼接前去除重复
                    phrases = set()
                    if len(cur_split_lines) > 0:
                        for line in cur_split_lines.split("\n"):
                            for part in line.split(" "):
                                phrases.add(part.strip())

                    for part in frame_content.split(" "):
                        if not part.replace("\n", "") in phrases:
                            cur_split_lines += " " + part

                    if last_span_no < len(split_spans) and frame_no_end >= split_spans[last_span_no]:
                        # 此处有需要跳过部分无字幕的部分场景分段，使last_span_no增大跨度
                        while last_span_no + 1 < len(split_spans) and frame_no_end >= split_spans[last_span_no + 1]:
                            last_span_no += 1
                            if last_span_frame < int(split_spans[last_span_no - 1]):
                                last_span_frame = int(split_spans[last_span_no - 1])

                        split_vd_filename = os.path.join(self.temp_output_dir,
                                                         os.path.split(self.video_path)[1] + "_" + str(
                                                             last_span_frame) + "_" + str(frame_no_end) + ".mp4")
                        split_cover_filename = split_vd_filename[:-4] + ".jpg"

                        print(f"cut video {last_span_frame} to {frame_no_end}")

                        if self.export_cut_video:
                            if not os.path.isfile(split_vd_filename):
                                cut_video(self.video_path, split_vd_filename, last_span_frame / self.fps,
                                          frame_no_end / self.fps)
                            if not os.path.isfile(split_cover_filename):
                                export_cover(split_vd_filename, split_cover_filename)

                        self.scenes[last_span_no] = [self._frame_to_timecode(last_span_frame), frame_end,
                                                     cur_split_lines, split_vd_filename, split_cover_filename,
                                                     (frame_no_end - last_span_frame) / self.fps, frame_no_end]
                        cur_split_lines = ""
                        last_span_no += 1
                        last_span_frame = frame_no_end

        # 补上空白视频位置和最后的尾部字幕
        if len(split_spans) > 0:
            for i in range(len(self.scenes)):
                if len(self.scenes[i]) == 0:
                    if i == 0:
                        last_span_frame = 1
                    else:
                        last_span_frame = int(split_spans[i - 1])
                        last_scenes_end_frame_no = int(self.scenes[i - 1][6])
                        if last_scenes_end_frame_no > last_span_frame:
                            last_span_frame = last_scenes_end_frame_no

                    frame_no_end = int(split_spans[i])
                    split_vd_filename = os.path.join(self.temp_output_dir,
                                                     os.path.split(self.video_path)[1] + "_" + str(
                                                         last_span_frame) + "_" + str(frame_no_end) + ".mp4")
                    split_cover_filename = split_vd_filename[:-4] + ".jpg"

                    print(f"cut video {last_span_frame} to {frame_no_end}")

                    if not os.path.isfile(split_vd_filename):
                        cut_video(self.video_path, split_vd_filename, last_span_frame / self.fps,
                                  frame_no_end / self.fps)
                    if not os.path.isfile(split_cover_filename):
                        export_cover(split_vd_filename, split_cover_filename)

                    subtitles = ""
                    if i == len(self.scenes) - 1 and len(cur_split_lines) > 0:
                        subtitles = cur_split_lines

                    self.scenes[i] = [self._frame_to_timecode(last_span_frame), self._frame_to_timecode(frame_no_end),
                                      subtitles,
                                      split_vd_filename,
                                      split_cover_filename,
                                      (frame_no_end - last_span_frame) / self.fps, frame_no_end]

        print(f"{interface_config['Main']['SubLocation']} {srt_filename}")

        # 保存原始关键帧
        # 取得保存路径，拼到kfs目录下
        if self.export_key_frames:
            for f in processed_subtitle:
                org_filename = os.path.join(self.frame_output_dir, str(f[2]).zfill(8) + '.org.png')
                f.append(org_filename)
                f.append(str(f[2]).zfill(8) + ".jpg")

        # self.split_spans = split_spans
        return processed_subtitle

    def _frame_preprocess(self, frame):
        """
        将视频帧进行裁剪
        """
        # 对于分辨率大于1920*1080的视频，将其视频帧进行等比缩放至1280*720进行识别
        # paddlepaddle会将图像压缩为640*640
        # if self.frame_width > 1280:
        #     scale_rate = round(float(1280 / self.frame_width), 2)
        #     frames = cv2.resize(frames, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_AREA)
        cropped = int(frame.shape[0] // 2) - config.SUBTITLE_AREA_DEVIATION_PIXEL

        # 如果字幕出现的区域在下部分
        if self.subtitle_area == config.SubtitleArea.LOWER_PART:
            # 将视频帧切割为下半部分
            frame = frame[cropped:]
        # 如果字幕出现的区域在上半部分
        elif self.subtitle_area == config.SubtitleArea.UPPER_PART:
            # 将视频帧切割为下半部分
            frame = frame[:cropped]
        return frame

    def _frame_to_timecode(self, frame_no, smpte_token=','):
        """
        将视频帧转换成时间
        :param frame_no: 视频的帧号，i.e. 第几帧视频帧
        :returns: SMPTE格式时间戳 as string, 如'01:02:12:32' 或者 '01:02:12;32'
        """
        # 设置当前帧号
        # cap = cv2.VideoCapture(self.video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        # cap.read()
        # 获取当前帧号对应的时间戳
        # milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        milliseconds = 1000 / self.fps * frame_no
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
            # cap.release()
        return "%02d:%02d:%02d%s%02d" % (hours, minutes, seconds, smpte_token, milliseconds)

    def _remove_duplicate_subtitle(self):
        """
        读取原始的raw txt，去除重复行，返回去除了重复后的字幕列表
        """
        self._concat_content_with_same_frameno()

        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        content_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            content = line.split('\t')[2]
            # 只有一个字的一般是误识别，可以忽略
            if len(content) < 3:
                continue

            content_list.append((frame_no, content))

        # 循环遍历每行字幕，记录开始时间与结束时间
        index = 0
        # 去重后的字幕列表
        unique_subtitle_list = []
        for i in content_list:
            # TODO: 时间复杂度非常高，有待优化
            # 定义字幕开始帧帧号
            start_frame = i[0]
            for j in content_list[index:]:
                # 计算当前行与下一行的Levenshtein距离
                distance = ratio(i[1], j[1])
                if distance < config.THRESHOLD_TEXT_SIMILARITY or j == content_list[-1]:
                    # 定义字幕结束帧帧号
                    end_frame = content_list[content_list.index(j) - 1][0]
                    if end_frame == start_frame:
                        end_frame = j[0]
                    # 如果是第一行字幕，直接添加进列表
                    if len(unique_subtitle_list) < 1:
                        unique_subtitle_list.append((start_frame, end_frame, i[1]))
                    else:
                        string_a = unique_subtitle_list[-1][2].replace(' ', '').translate(
                            str.maketrans('', '', string.punctuation))
                        string_b = i[1].replace(' ', '').translate(str.maketrans('', '', string.punctuation))
                        similarity_ratio = ratio(string_a, string_b)
                        # 打印相似度
                        # print(f'{similarity_ratio}: {unique_subtitle_list[-1][2]} vs {i[1]}')
                        # 如果相似度小于阈值，说明该两行字幕不一样
                        if similarity_ratio < config.THRESHOLD_TEXT_SIMILARITY:
                            unique_subtitle_list.append((start_frame, end_frame, i[1]))
                        else:
                            # todo，相似的取出现次数更多的来保留!!!!!，现在的算法是有问题的。取长的是为了防止飞字，但是第一个容易误识别
                            # 如果大于阈值，但又不完全相同，说明两行字幕相似
                            # 可能出现以下情况: "但如何进人并接管上海" vs "但如何进入并接管上海"
                            # OCR识别出现了错误识别
                            if similarity_ratio < 1:
                                if len(string_a) < len(string_b):
                                    unique_subtitle_list[-1] = (start_frame, end_frame, i[1])
                    index += 1
                    break
                else:
                    continue
        return unique_subtitle_list

    def _get_content_list(self):
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
        line = f.readline()  # 以行的形式进行读取文件
        # 坐标点列表
        coordinates_list = []
        frame_contents = {}
        cord_frame_list = []

        last_frame_no = -1
        while line:
            frame_no = int(line.split('\t')[0])
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            content = line.split('\t')[2]

            cord = (int(text_position[0]),
                    int(text_position[1]),
                    int(text_position[2]),
                    int(text_position[3]))

            cord_index = len(coordinates_list)
            coordinates_list.append(cord)

            if frame_no != last_frame_no:
                frame_contents[frame_no] = []
                last_frame_no = frame_no

            frame_contents[frame_no].append([cord, content, cord_index, None])
            cord_frame_list.append(frame_no)

            line = f.readline()
        f.close()
        return coordinates_list, frame_contents, cord_frame_list

    def _is_possible_watermark(self, text):
        return "抖音" in text

    def _remove_too_common(self):
        """
        将raw txt文本中具有相同帧号的字幕行合并
        # todo: 通过重复计数，移除超过次数的文本，可能是台标
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()

        content_list = []
        frame_no_list = []
        contents = []
        for line in lines:
            frame_no = line.split('\t')[0]
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]

            frame_no_list.append(frame_no)
            contents.append(content)
            content_list.append([frame_no, coordinate, content])

        max_common_count = 15

        self.too_commons = set()
        self.counter = Counter(contents).most_common()
        for c in self.counter:
            if c[1] > max_common_count and len(c[0]) < 12:
                if not self.skip_watermark:
                    if self._is_possible_watermark(c[0]):
                        continue
                self.too_commons.add(c[0])
            else:
                break

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in content_list:
                if not content in self.too_commons:
                    content = unicodedata.normalize('NFKC', content)
                    f.write(f'{frame_no}\t{coordinate}\t{content}')

    def _concat_content_with_same_frameno(self):
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()

        content_list = []
        frame_no_list = []
        contents = []
        for line in lines:
            frame_no = line.split('\t')[0]
            frame_no_list.append(frame_no)
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]
            contents.append(content)
            content_list.append([frame_no, coordinate, content])

        # 找出那些不止一行的帧号
        frame_no_list = [i[0] for i in Counter(frame_no_list).most_common() if i[1] > 1]

        # 找出这些帧号出现的位置
        concatenation_list = []
        for frame_no in frame_no_list:
            position = [i for i, x in enumerate(content_list) if x[0] == frame_no]
            concatenation_list.append((frame_no, position))

        for i in concatenation_list:
            content = []
            for j in i[1]:
                txt = content_list[j][2]
                txt = txt.replace(" ", "").replace('\n', '')
                # 全部是英文的不要处理，过短的不要处理，一般都是识别错误
                if not is_all_english_char(txt) and len(txt) > 2:
                    if len(content) < 9:
                        pos = content_list[j][1].split('(')[1].split(')')[0].split(', ')
                        pos_center = int(pos[0]) + int((int(pos[1]) - int(pos[0])) / 2)
                        if pos_center > self.frame_width / 2 - 150 and pos_center < self.frame_width / 2 + 150:
                            content.append(txt)
                    else:
                        content.append(txt)

            content = ' '.join(content) + '\n'
            for k in i[1]:
                content_list[k][2] = content

        # 将多余的字幕行删除
        to_delete = []
        for i in concatenation_list:
            for j in i[1][1:]:
                to_delete.append(content_list[j])

        # 移出空白行
        for i in content_list:
            if len(i[2].replace("\n", "").strip()) == 0:
                to_delete.append(i)

        for i in to_delete:
            if i in content_list:
                content_list.remove(i)

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in content_list:
                skip = False
                if len(content) < 9:
                    # coordinate = content_list[j][1]
                    pos = coordinate.split('(')[1].split(')')[0].split(', ')
                    pos_center = int(pos[0]) + int((int(pos[1]) - int(pos[0])) / 2)
                    if pos_center > self.frame_width / 2 - 150 and pos_center < self.frame_width / 2 + 150:
                        content = unicodedata.normalize('NFKC', content)
                    else:
                        skip = True
                else:
                    content = unicodedata.normalize('NFKC', content)
                if not skip:
                    f.write(f'{frame_no}\t{coordinate}\t{content}')

    def _detect_watermark_area(self):
        """
        根据识别出来的raw txt文件中的坐标点信息，查找水印区域
        假定：水印区域（台标）的坐标在水平和垂直方向都是固定的，也就是具有(xmin, xmax, ymin, ymax)相对固定
        根据坐标点信息，进行统计，将一直具有固定坐标的文本区域选出
        :return 返回最有可能的水印区域
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as f:  # 打开txt文件，以‘utf-8’编码读取
            line = f.readline()  # 以行的形式进行读取文件
            # 坐标点列表
            coordinates_list = []
            # 帧列表
            frame_no_list = []
            # 内容列表
            content_list = []
            while line:
                frame_no = line.split('\t')[0]
                text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
                content = line.split('\t')[2]
                frame_no_list.append(frame_no)
                coordinates_list.append((int(text_position[0]),
                                         int(text_position[1]),
                                         int(text_position[2]),
                                         int(text_position[3])))
                content_list.append(content)
                line = f.readline()
            f.close()
            # 将坐标列表的相似值统一
            coordinates_list = self._unite_coordinates(coordinates_list)

            # 将原txt文件的坐标更新为归一后的坐标
            with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
                for frame_no, coordinate, content in zip(frame_no_list, coordinates_list, content_list):
                    f.write(f'{frame_no}\t{coordinate}\t{content}')

            counter = Counter(coordinates_list).most_common()
            self.most_areas = counter
            self.watermark_areas = set()

            if len(counter) > config.WATERMARK_AREA_NUM:
                # 读取配置文件，返回可能为水印区域的坐标列表
                for c in counter:
                    if c[1] >= 9:
                        self.watermark_areas.add(c[0])
                    else:
                        break
                return Counter(coordinates_list).most_common(config.WATERMARK_AREA_NUM)
            else:
                # 不够则有几个返回几个
                return Counter(coordinates_list).most_common()

    def _detect_subtitle_area(self):
        """
        读取过滤水印区域后的raw txt文件，根据坐标信息，查找字幕区域
        假定：字幕区域在y轴上有一个相对固定的坐标范围，相对于场景文本，这个范围出现频率更高
        :return 返回字幕的区域位置
        """
        # 打开去水印区域处理过的raw txt
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as f:  # 打开txt文件，以‘utf-8’编码读取
            line = f.readline()  # 以行的形式进行读取文件
            # y坐标点列表
            y_coordinates_list = []
            while line:
                text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
                y_coordinates_list.append((int(text_position[2]), int(text_position[3])))
                line = f.readline()
            f.close()
            return Counter(y_coordinates_list).most_common(5)

    def filter_scene_text(self):
        # todo: 需要可选的保留抖音的水印

        # 检查水印区域，并处理区域合并
        # 此处需要重构，最多的一个未尝是正确的
        self._detect_watermark_area()

        # 统计比较前5个，取字符数量最多的前两个加入到最终的字幕中
        subtile_areas = self._detect_subtitle_area()

        print("subtitle area detected:" + str(len(subtile_areas)))

        with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
            content = f.readlines()

            area_subtitles = []
            for i in range(len(subtile_areas)):
                subtitle_area = subtile_areas[i][0]

                # 为了防止有双行字幕，根据容忍度，将字幕区域y范围加高
                ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
                ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL

                area_subtitles.append(((ymin, ymax), set()))

                # todo: 这里可以再加一个判断，对于整体长度和位置的
                for i in content:
                    i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                    i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                    if ymin <= i_ymin and i_ymax <= ymax:
                        area_subtitles[-1][1].update(set(i))

            self.areas = []
            for i, sa in enumerate(area_subtitles):
                if len(sa[1]) > 50 and len(self.areas) < 3:
                    self.areas.append(sa[0])

            if len(self.areas) == 0:
                if len(area_subtitles) > 0:
                    self.areas.append(area_subtitles[0][0])

            f.seek(0)
            for i in content:
                i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                if not self.skip_watermark:
                    if self._is_possible_watermark(i):
                        f.write(i)
                        continue

                if len(self.areas) > 0:
                    for (ymin, ymax) in self.areas:
                        if ymin <= i_ymin and i_ymax <= ymax:
                            f.write(i)
                            break
                else:
                    f.write(i)

                f.truncate()
            f.close()

    def _unite_coordinates(self, coordinates_list):
        """
        给定一个坐标列表，将这个列表中相似的坐标统一为一个值
        e.g. 由于检测框检测的结果不是一致的，相同位置文字的坐标可能一次检测为(255,123,456,789)，另一次检测为(253,122,456,799)
        因此要对相似的坐标进行值的统一
        :param coordinates_list 包含坐标点的列表
        :return: 返回一个统一值后的坐标列表
        """
        # 将相似的坐标统一为一个
        index = 0
        for coordinate in coordinates_list:  # TODO：时间复杂度n^2，待优化
            for i in coordinates_list:
                if self._is_coordinate_similar(coordinate, i):
                    coordinates_list[index] = i
            index += 1
        return coordinates_list

    def _get_coordinates(self, dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def _is_coordinate_similar(self, coordinate1, coordinate2):
        """
        计算两个坐标是否相似，如果两个坐标点的xmin,xmax,ymin,ymax的差值都在像素点容忍度内
        则认为这两个坐标点相似
        """
        return abs(coordinate1[0] - coordinate2[0]) < config.PIXEL_TOLERANCE_X and \
               abs(coordinate1[1] - coordinate2[1]) < config.PIXEL_TOLERANCE_X and \
               abs(coordinate1[2] - coordinate2[2]) < config.PIXEL_TOLERANCE_Y and \
               abs(coordinate1[3] - coordinate2[3]) < config.PIXEL_TOLERANCE_Y

    def delete_frame_cache(self):
        if len(os.listdir(self.frame_output_dir)) > 0:
            for i in os.listdir(self.frame_output_dir):
                os.remove(os.path.join(self.frame_output_dir, i))

    def _make_content_mask(self, maskimg, content, w, h, remove_text=True, remove_watermark=True, more_grow=True,
                           max_rect=None):
        pad = 35 if more_grow else 10
        # print(pad)
        for item in content:
            if remove_watermark:
                if "抖音\n" == item[1]:
                    maskimg = add_mask(maskimg, get_rec_area(get_douyin_rec(item[0], w, h)))
                if "抖音号：" == item[1][:4]:
                    maskimg = add_mask(maskimg, get_rec_area(get_douyin_hao_rec(item[0], w, h)))
                # if item[3] in self.watermark_areas:
                #    maskimg = add_mask(maskimg, get_rec_area(grow_rec(item[0], w, h, pad)))

            if remove_text:
                if len(self.areas) > 0:
                    for (ymin, ymax) in self.areas:
                        if item[0][2] > ymin and item[0][3] < ymax:
                            rect_area = get_rec_area(grow_rec(item[0], w, h, pad))
                            maskimg = add_mask(maskimg, rect_area)

                            if max_rect:
                                if max_rect[0] > rect_area[0] and rect_area[0] > 0:
                                    max_rect[0] = rect_area[0]
                                if max_rect[1] > rect_area[1] and rect_area[1] > 0:
                                    max_rect[1] = rect_area[1]
                                if max_rect[2] < rect_area[2]:
                                    max_rect[2] = rect_area[2]
                                if max_rect[3] < rect_area[3]:
                                    max_rect[3] = rect_area[3]

        return maskimg

    # 获取可能水印台标遮罩
    def _get_frame_mask(self, frame_no, remove_text=True, remove_watermark=True, max_rect=None):
        frame_contents = self.frame_contents
        w = self.w
        h = self.h

        pre_k = self.frame_key_frame[frame_no]
        if pre_k in self.next_key_frame:
            found_k = self.next_key_frame[pre_k]
        else:
            found_k = pre_k

        if pre_k in self.mask_cache:
            return self.mask_cache[pre_k]

        a_maskimg = np.zeros((h, w), dtype=np.uint8)

        # 两张全部加入到mask中
        if pre_k in frame_contents:
            pre_content = frame_contents[pre_k]
            a_maskimg = self._make_content_mask(a_maskimg, pre_content, w, h, remove_text, remove_watermark,
                                                more_grow=True, max_rect=max_rect)

        if found_k in frame_contents:
            next_content = frame_contents[found_k]
            a_maskimg = self._make_content_mask(a_maskimg, next_content, w, h, remove_text, remove_watermark,
                                                more_grow=True, max_rect=max_rect)

        self.mask_cache[pre_k] = a_maskimg

        return a_maskimg

    def process_rect_fill(self, img, a_mask, rect_fill_color="black"):
        if rect_fill_color == "black":
            invert_a_mask = ImageChops.invert(a_mask)
        else:
            invert_a_mask = a_mask
        img.paste(invert_a_mask, (0, 0), mask=a_mask)
        return img

    def remove_text_watermark(self, output_file=None, rect=None, remove_text=True, remove_watermark=True, tqdm=None,
                              st_progress_bar=None, rect_fill_color=None):
        if not self.fill_model and not rect_fill_color:
            self.fill_model = get_fill_model()
            if not self.fill_model:
                print("model not initialized!")
                return ""

        if not output_file:
            # name = "fixed"
            # if remove_text:
            #     name += "_detexted"
            # if remove_watermark:
            #     name += "_dewatermark"
            # name += ".mp4"
            output_file = os.path.join(self.temp_output_dir, "fixed_detexted_dewatermark.mp4")

        output_file = ".".join(output_file.split(".")[:-1]) + "_wa.mp4"

        writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h))

        video_cap = cv2.VideoCapture(self.video_path)
        frame_no = 0

        t = None
        if tqdm:
            if st_progress_bar:
                t = tqdm(range(int(self.num_frame)), st_progress_bar=st_progress_bar)
            else:
                t = tqdm(range(int(self.num_frame)))

        if rect:
            a_maskimg = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            add_mask(a_maskimg, rect)
            a_mask = Image.fromarray(np.array(a_maskimg))
            max_rect = None
        else:
            max_rect = [self.w, self.h, 0, 0]

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            else:
                frame_no += 1

                # 处理指定的提取范围
                if self.start_ms > 0:
                    if self.ms_per_frame * frame_no < self.start_ms:
                        continue

                if self.end_ms > 0:
                    if self.ms_per_frame * frame_no > self.end_ms:
                        break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                if not rect:
                    masking = self._get_frame_mask(frame_no, remove_text, remove_watermark, max_rect)
                    a_mask = Image.fromarray(np.array(masking))

                if not rect_fill_color:
                    img = process_image(self.fill_model, img, a_mask)
                else:
                    img = self.process_rect_fill(img, a_mask, rect_fill_color)

                writer.write(cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2RGB))

                if t:
                    t.update()

        writer.release()

        start_frame = 1
        end_frame = frame_no

        start_time = self._frame_to_timecode(start_frame - 1, smpte_token=".")
        end_time = self._frame_to_timecode(end_frame - 1, smpte_token=".")
        sourceVideo = self.video_path
        tempAudioFileName = os.path.join(self.temp_output_dir, "temp.aac")
        mp4_file = output_file
        output_file = mp4_file[:-len("_wa.mp4")] + ".mp4"

        os.system(f'ffmpeg -y -ss {start_time} -to {end_time} -i "{sourceVideo}" -c:a copy -vn {tempAudioFileName}')
        os.system(f'ffmpeg -y -i "{mp4_file}" -i {tempAudioFileName} -c copy "{output_file}"')

        return output_file, max_rect

    def generate_english_dubbed(self, voice_name=None, color="white", fontsize=32,
                                font="Keep-Calm-Medium-bold", rect_fill_color=None):
        # if self.fps != 25:
        #     self.video_path = fix_video_fps_to_25(self.video_path, self.video_path + ".f25.mp4")
        #     self.video_cap = cv2.VideoCapture(self.video_path)
        #     # 视频帧总数
        #     self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #     # 视频帧率
        #     self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        #     # 视频秒数
        #     self.video_length = float(self.frame_count / self.fps)

        if not self.ocr_result:
            self.run()

        if len(self.ocr_result) == 0:
            return None, None, None

        if not self.refilled_video:
            self.refilled_video, self.max_rect = self.remove_text_watermark(rect_fill_color=rect_fill_color)

        if not self.segments:
            self.segments = get_english_dubbed(self.refilled_video, self.ocr_result, self.fps, voice_name)

        # 此处还缺少一步带入参数，有此参数这个接口的功能就可以完成了
        if not color:
            color = "white"
        if rect_fill_color == "white" and color == "white":
            color = "black"

        output_file = gen_subtitled_video(self.refilled_video, self.segments, self.max_rect, color=color, fontsize=fontsize,
                                          font=font, no_voice=(voice_name is None or len(voice_name) == 0))

        return output_file, self.segments, self.max_rect

    def generate_asr_subtitle(self):
        """
        使用ASR方法来生成srt格式的字幕文件
        """
        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        processed_subtitle = []

        print(f"splits:{self.splits}, fps:{self.fps}, frame_count:{self.frame_count}")

        save_path = extract_audio(self.video_path)
        if not save_path:
            return []

        ret = speech_recognize(save_path)
        # [[0.0, 2.86, '对于今年的市场销售，做了一个规划。'], [2.86, 6.7406875, '我个人也是对这次的方案进行了一个全面的策划。']]

        self.split_spans = [[r[0] * self.fps, r[1] * self.fps] for r in ret]
        self.scenes = [[] for _ in range(len(ret))]

        last_span_frame = 1
        last_span_no = 0
        with open(srt_filename, mode='w', encoding='utf-8') as f:
            for index, content in enumerate(ret):
                line_code = index + 1
                frame_no_start = int(content[0] * self.fps)
                frame_start = self._frame_to_timecode(frame_no_start)
                frame_no_end = int(content[1] * self.fps)
                frame_end = self._frame_to_timecode(frame_no_end)
                frame_content = content[2].replace("\n", "") + "\n"
                processed_subtitle.append([frame_start, frame_end, frame_no_start, frame_no_end, frame_content])
                subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                f.write(subtitle_line)

                if self.export_cut_video:
                    split_vd_filename = os.path.join(self.temp_output_dir,
                                                     os.path.split(self.video_path)[1] + "_" + str(
                                                         last_span_frame) + "_" + str(frame_no_end) + ".mp4")
                    split_cover_filename = split_vd_filename[:-4] + ".jpg"

                    print(f"cut video {last_span_frame} to {frame_no_end}")

                    if not os.path.isfile(split_vd_filename):
                        cut_video(self.video_path, split_vd_filename, last_span_frame / self.fps,
                                  frame_no_end / self.fps)
                    if not os.path.isfile(split_cover_filename):
                        export_cover(split_vd_filename, split_cover_filename)

                    self.scenes[last_span_no] = [self._frame_to_timecode(last_span_frame), frame_end,
                                                 frame_content, split_vd_filename, split_cover_filename,
                                                 (frame_no_end - last_span_frame) / self.fps, frame_no_end]

                    last_span_no += 1
                    last_span_frame = frame_no_end

        return processed_subtitle