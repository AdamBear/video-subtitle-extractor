import hashlib
import json
import os
import shutil
import time
from collections import Counter

import config
import cv2
import requests
import unicodedata
from Levenshtein import ratio
from config import interface_config
from tools.reformat_en import reformat


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


class AutoSubtitleExtractor():
    """
    视频字幕提取类
    """

    def __init__(self, vd_path, export_key_frames=False):
        self.sub_area = None
        self.export_key_frames = export_key_frames

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
        # 字幕出现区域
        self.subtitle_area = config.SUBTITLE_AREA
        print(
            f"{interface_config['Main']['FrameCount']}：{self.frame_count}，{interface_config['Main']['FrameRate']}：{self.fps}")
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

                if self.export_key_frames:
                    # 保存原始视频帧
                    org_filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.org.png')
                    cv2.imwrite(org_filename, frame)

                frame = self._frame_preprocess(frame)

                # 帧名往前补零，后续用于排序与时间戳转换，补足8位
                # 一部10h电影，fps120帧最多也才1*60*60*120=432000 6位，所以8位足够
                filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.jpg')

                # 保存视频帧
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
        # shutil.copyfile(self.raw_subtitle_path, self.raw_subtitle_path + ".raw.txt")
        f.close()

    def generate_subtitle_file(self):
        """
        生成srt格式的字幕文件
        """
        subtitle_content = self._remove_duplicate_subtitle()
        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        processed_subtitle = []

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
        print(f"{interface_config['Main']['SubLocation']} {srt_filename}")

        # 保存原始关键帧
        # 取得保存路径，拼到kfs目录下
        if self.export_key_frames:
            # kfs_path = os.path.join(os.path.dirname(self.video_path), "kfs")
            # if not os.path.exists(kfs_path):
            #     os.makedirs(kfs_path)
            # for f in processed_subtitle:
            #     org_filename = os.path.join(self.frame_output_dir, str(f[2]).zfill(8) + '.org.png')
            #     exported_filename = os.path.join(kfs_path, str(f[2]).zfill(8) + '.jpg')
            #     f.append(str(f[2]).zfill(8) + '.jpg')
            #     shutil.copyfile(org_filename, exported_filename)
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
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        cap.read()
        # 获取当前帧号对应的时间戳
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
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
        cap.release()
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
                        string_a = unique_subtitle_list[-1][2].replace(' ', '')
                        string_b = i[1].replace(' ', '')
                        similarity_ratio = ratio(string_a, string_b)
                        # 打印相似度
                        # print(f'{similarity_ratio}: {unique_subtitle_list[-1][2]} vs {i[1]}')
                        # 如果相似度小于阈值，说明该两行字幕不一样
                        if similarity_ratio < config.THRESHOLD_TEXT_SIMILARITY:
                            unique_subtitle_list.append((start_frame, end_frame, i[1]))
                        else:
                            # 如果大于阈值，但又不完全相同，说明两行字幕相似
                            # 可能出现以下情况: "但如何进人并接管上海" vs "但如何进入并接管上海"
                            # OCR识别出现了错误识别
                            if similarity_ratio < 1:
                                # TODO:
                                # 1) 取出两行字幕的并集
                                # 2) 纠错
                                # print(f'{round(similarity_ratio, 2)}, 需要手动纠错:\n {string_a} vs\n {string_b}')
                                # 保存较长的
                                if len(string_a) < len(string_b):
                                    unique_subtitle_list[-1] = (start_frame, end_frame, i[1])
                    index += 1
                    break
                else:
                    continue
        return unique_subtitle_list

    def _concat_content_with_same_frameno(self):
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

        too_commons = set()
        for c in Counter(contents).most_common():
            if c[1] > 50:
                too_commons.add(c[0])
            else:
                break

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
                if txt not in too_commons and not is_all_english_char(txt):
                    content.append(txt)

            content = ' '.join(content).replace('\n', ' ') + '\n'
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