import os
import logging
import random

import openai

from ..smp import *
from .base import BaseAPI


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    return image


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    return video


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if base_url is None:
        return None
    normalized = base_url.rstrip('/')
    if normalized.endswith('/chat/completions'):
        normalized = normalized[: -len('/chat/completions')]
    return normalized or None


class KeyeFastAPI(BaseAPI):

    is_api: bool = True
    VIDEO_LLM = True
    allowed_types = ['text', 'image', 'video']

    def __init__(
        self,
        retry: int = 5,
        wait: int = 5,
        key: str = None,
        verbose: bool = False,
        system_prompt: str = None,
        temperature: float = 0,
        top_p: float = 0.001,
        top_k: int = 1,
        presence_penalty: float = 1.0,
        timeout: int = 60,
        api_base: str = None,
        api_base_list: list = None,
        max_tokens: int = 10240,
        img_size: int = -1,
        img_detail: str = 'low',
        no_think: bool = False,
        think: bool = False,
        is_xiaomi: bool = False,
        max_pixels: int = None,
        min_pixels: int = None,
        video_total_pixels: int = None,
        nframes: int = None,
        max_frames: int = None,
        fps: int = None,
        is_vllm: bool = False,
        model: str = None,
        **kwargs,
    ):
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.key = key or 'EMPTY'
        self.img_size = img_size
        self.img_detail = img_detail
        self.timeout = timeout
        self.no_think = no_think
        self.think = think
        self.is_xiaomi = is_xiaomi
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.video_total_pixels = video_total_pixels
        self.nframes = nframes
        self.max_frames = max_frames
        self.fps = fps
        self.is_vllm = is_vllm
        self.model = model

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        self.api_base_list = [normalize_openai_base_url(base) for base in api_base_list] if api_base_list else None
        self.api_base = normalize_openai_base_url(api_base)
        self._client_model_cache = {}
        if self.api_base_list is not None:
            self.clients = [openai.Client(api_key=self.key, base_url=base) for base in self.api_base_list]
            self.client = self.clients[0]
        else:
            self.client = openai.Client(api_key=self.key, base_url=self.api_base)

        self.logger.info(
            f'Base Url: {self.api_base or self.api_base_list} API Key: {self.key}, verbose: {self.verbose}'
        )

    def _image_payload(self, value, dataset=None):
        item = {
            'type': 'image_url',
            'image_url': {
                'url': ensure_image_url(value),
                'detail': self.img_detail,
            },
        }
        if dataset in ['OCRBench', 'Benchmark_V21']:
            item['min_pixels'] = 10 * 10 * 28 * 28
            if self.max_pixels is not None:
                item['max_pixels'] = int(self.max_pixels)
        else:
            env_max_pixels = os.environ.get('MAX_PIXELS')
            if env_max_pixels is not None:
                item['max_pixels'] = int(env_max_pixels)
            if self.max_pixels is not None:
                item['max_pixels'] = int(self.max_pixels)
            if self.min_pixels is not None:
                item['min_pixels'] = int(self.min_pixels)
        return item

    def _video_payload(self, value):
        item = {
            'type': 'video_url',
            'video_url': {
                'url': ensure_video_url(value),
            },
        }
        if self.video_total_pixels is not None:
            item['video_total_pixels'] = int(self.video_total_pixels)
        if self.nframes is not None:
            item['nframes'] = int(self.nframes)
        if self.fps is not None:
            item['fps'] = int(self.fps)
        if self.max_frames is not None:
            item['max_frames'] = int(self.max_frames)
        if self.max_pixels is not None:
            item['max_pixels'] = int(self.max_pixels)
        if self.min_pixels is not None:
            item['min_pixels'] = int(self.min_pixels)
        return item

    def prepare_itlist(self, inputs, dataset: str | None = None):
        assert np.all([isinstance(x, dict) for x in inputs])
        content_list = []
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        has_videos = np.sum([x['type'] == 'video' for x in inputs])

        if not has_images and not has_videos:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            return [dict(type='text', text=text)]

        for msg in inputs:
            if msg['type'] == 'text':
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                content_list.append(self._image_payload(msg['value'], dataset=dataset))
            elif msg['type'] == 'video':
                content_list.append(self._video_payload(msg['value']))
            else:
                raise ValueError(f'Unsupported message type: {msg}')
        return content_list

    def prepare_inputs(self, inputs, dataset: str | None = None):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'], dataset)))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs, dataset)))
        return input_msgs

    def _maybe_append_reasoning_suffix(self, input_msgs, dataset=None):
        if not input_msgs or not input_msgs[-1]['content']:
            return
        text_items = [d for d in input_msgs[-1]['content'] if d['type'] == 'text']
        if not text_items:
            return
        last_text = text_items[-1]
        if self.no_think:
            last_text['text'] += '/no_think'
            return
        if self.think:
            last_text['text'] += '/think'
            return
        if not (self.is_xiaomi and dataset):
            return

        if dataset in {'Video-MME', 'Video-MME_64frame', 'VLMareBlind', 'VisuLogic', 'AI2D_TEST', 'BLINK',
                       'CharXiv_reasoning_val', 'CharXiv_descriptive_val', 'MMMU_Pro_10c_COT', 'MMMU_Pro_V_COT',
                       'VStar', 'ZeroBench_main', 'ZeroBench_sub', 'RealWorldQA'}:
            last_text['text'] += '\nPut your final answer in \\boxed{}.'
        elif dataset in {'ChartQA_TEST', 'DocVQA_VAL', 'InfoVQA_VAL', 'OCRBench'}:
            last_text['text'] += '\nPut your final answer in \\boxed{}. Try to answer using the content in the image as much as possible.'
        elif dataset in {'CountBench'}:
            last_text['text'] += '\nPlease count the objects by grounding. Put your answer in \\boxed{}.'
        elif dataset in {'VideoMMMU_64frame', 'VideoMMMU_1fps'}:
            last_text['text'] += '\nPlease answer the question based on the video content and put your final answer in \\boxed{}.'
        else:
            last_text['text'] += '\nPut your final answer in \\boxed{}.'

    def _resolve_model_name(self, client):
        if self.model is not None:
            return self.model

        cache_key = getattr(client, 'base_url', None) or id(client)
        if cache_key in self._client_model_cache:
            return self._client_model_cache[cache_key]

        model_name = client.models.list().data[0].id
        self._client_model_cache[cache_key] = model_name
        return model_name

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs, dataset=kwargs.get('dataset', None))
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)

        if input_msgs[-1]['content'] and input_msgs[-1]['content'][-1].get('type') != 'text':
            content_video = [d for d in input_msgs[-1]['content'] if d['type'] == 'video_url']
            content_img = [d for d in input_msgs[-1]['content'] if d['type'] == 'image_url']
            content_text = [d for d in input_msgs[-1]['content'] if d['type'] == 'text']
            input_msgs[-1]['content'] = content_video + content_img + content_text

        self._maybe_append_reasoning_suffix(input_msgs, dataset=dataset)

        client = random.choice(self.clients) if self.api_base_list is not None else self.client

        payload = dict(
            model=self._resolve_model_name(client),
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            timeout=max(self.timeout, 3600),
            extra_body={},
        )

        if self.top_p is not None:
            payload['top_p'] = self.top_p
        if self.top_k is not None:
            payload['extra_body']['top_k'] = self.top_k
        if self.presence_penalty is not None:
            payload['presence_penalty'] = self.presence_penalty
        if self.max_pixels is not None:
            payload['extra_body']['mm_processor_kwargs'] = {'longest_edge': self.max_pixels}

        try:
            response = client.chat.completions.create(**payload)
            answer = response.choices[0].message.content
            if self.verbose:
                self.logger.debug(f'KeyeFastAPI answer: {answer}')
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
            return -1, '', str(err)
