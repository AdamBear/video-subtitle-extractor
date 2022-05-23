import hashlib
import json
import os
import shutil
import time
from collections import Counter
import string
import config
import cv2
import requests
import unicodedata
from Levenshtein import ratio
from config import interface_config
from tools.reformat_en import reformat
from scenedetect.detectors import ContentDetector
# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
from PIL import Image


def post_to_recognize(image_file_list):
    url = "http://127.0.0.1:8868/predict/ocr_system"

    headers = {"Content-type": "application/json"}
    total_time = 0
    starttime = time.time()
    data = {'images': [], 'paths': image_file_list}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    elapse = time.time() - starttime
    total_time += elapse
    return r.json()


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

            if end_frame > frame_count:
                split_spans.append(frame_count)
                return split_spans
            last_end_frame = end_frame
            split_spans.append(end_frame)

        return split_spans

    elif is_float(splits.strip()):
        span = float(splits.strip())
        i = 1
        while i * span * fps < frame_count:
            split_spans.append(i * span * fps)
            i += 1

        split_spans.append(frame_count)
        return split_spans
    else:
        return None


def cut_video(input_mp4, output_mp4, start, end, debug=False):
    cmd = f'ffmpeg -y -ss {start} -to {end} -i {input_mp4}  {output_mp4}'
    r = os.system(cmd)
    if debug:
        print(cmd)
    if r == 0:
        return output_mp4
    else:
        return ""


class AutoSubtitleExtractor():
    """
    视频字幕提取类
    """

    def __init__(self, vd_path, export_key_frames):
        self.sub_area = None
        self.export_key_frames = export_key_frames
        self.debug = False
        self.remove_too_common = True
        self.detect_subtitle = True
        self.detect_scene = True
        self.no_cut = True
        self.splits = []
        self.scenes = []

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
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.size = (self.frame_width, self.frame_height)

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

        # 处理进度
        self.progress = 0

    def run(self):

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

        # 清理临时文件
        # self._delete_frame_cache()

        # 如果识别的字幕语言包含英文，则将英文分词
        if config.REC_CHAR_TYPE in ('ch', 'EN', 'en', 'ch_tra'):
            reformat(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'))
        print(interface_config['Main']['FinishGenerateSub'])
        self.progress = 100

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

        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            else:
                frame_no += 1

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
        f = open(self.raw_subtitle_path, mode='w+', encoding='utf-8')

        # 视频帧列表
        frame_list = [i for i in sorted(os.listdir(self.frame_output_dir)) if i.endswith('.jpg')]
        image_file_list = [os.path.join(self.frame_output_dir, i).replace("\\", "/") for i in frame_list]

        rec_result = post_to_recognize(image_file_list)

        for i, (frame, rec_ret) in enumerate(zip(frame_list, rec_result["results"])):
            dt_box = [r["text_region"] for r in rec_ret]
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

        if self.debug:
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

        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        processed_subtitle = []

        split_spans = []
        if len(self.splits) > 0:
            cur_split_lines = ""
            last_span_no = 0
            last_span_frame = 1
            split_spans = get_split_spans(self.splits, self.fps, self.frame_count)
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
                    cur_split_lines += frame_content
                    if frame_no_end > split_spans[last_span_no] and last_span_no < len(split_spans):
                        split_vd_filename = os.path.join(self.temp_output_dir,
                                                         os.path.split(self.video_path)[1] + "_" + str(
                                                             last_span_frame) + "_" + str(frame_no_end) + ".mp4")
                        print(f"cut video {last_span_frame} to {frame_no_end}")
                        cut_video(self.video_path, split_vd_filename, last_span_frame / self.fps,
                                  frame_no_end / self.fps)
                        self.scenes[last_span_no] = [self._frame_to_timecode(last_span_frame), frame_end,
                                                     cur_split_lines, split_vd_filename]
                        cur_split_lines = ""
                        last_span_no += 1
                        last_span_frame = frame_no_end

        print(f"{interface_config['Main']['SubLocation']} {srt_filename}")

        # 保存原始关键帧
        # 取得保存路径，拼到kfs目录下
        if self.export_key_frames:
            for f in processed_subtitle:
                org_filename = os.path.join(self.frame_output_dir, str(f[2]).zfill(8) + '.org.png')
                f.append(org_filename)
                f.append(str(f[2]).zfill(8) + ".jpg")

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

    def _frame_to_timecode(self, frame_no):
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
        smpte_token = ','
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

        # 需要返回特别重复的台标或水印位置

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
            frame_no_list.append(frame_no)
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]
            contents.append(content)
            content_list.append([frame_no, coordinate, content])

        max_common_count = 15

        self.too_commons = set()
        self.counter = Counter(contents).most_common()
        for c in self.counter:
            if c[1] > max_common_count:
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
                content = unicodedata.normalize('NFKC', content)
                f.write(f'{frame_no}\t{coordinate}\t{content}')

    def _detect_watermark_area(self):
        """
        根据识别出来的raw txt文件中的坐标点信息，查找水印区域
        假定：水印区域（台标）的坐标在水平和垂直方向都是固定的，也就是具有(xmin, xmax, ymin, ymax)相对固定
        根据坐标点信息，进行统计，将一直具有固定坐标的文本区域选出
        :return 返回最有可能的水印区域
        """
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
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

        if len(Counter(coordinates_list).most_common()) > config.WATERMARK_AREA_NUM:
            # 读取配置文件，返回可能为水印区域的坐标列表
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
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
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
        # 检查水印区域，并处理区域合并
        # 此处需要重构，最多的一个未尝是正确的
        self._detect_watermark_area()

        # 统计比较前5个，取字符数量最多的前两个加入到最终的字幕中
        subtile_areas = self._detect_subtitle_area()

        with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
            content = f.readlines()

            area_subtitles = []
            for i in range(len(subtile_areas)):
                subtitle_area = subtile_areas[i][0]

                # 为了防止有双行字幕，根据容忍度，将字幕区域y范围加高
                ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
                ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL

                area_subtitles.append(((ymin, ymax), set()))

                for i in content:
                    i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                    i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                    if ymin <= i_ymin and i_ymax <= ymax:
                        area_subtitles[-1][1].update(set(i))

            self.areas = []
            for i, sa in enumerate(area_subtitles):
                if len(sa[1]) > 50:
                    self.areas.append(sa[0])

            f.seek(0)
            for i in content:
                i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                for (ymin, ymax) in self.areas:
                    if ymin <= i_ymin and i_ymax <= ymax:
                        f.write(i)
                        break
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