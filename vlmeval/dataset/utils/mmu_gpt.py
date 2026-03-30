import json
import time
import uuid


class MMUGPT:
    """Internal MMU-based judge wrapper.

    This wrapper intentionally keeps the public interface close to existing
    judge implementations used by VLMEvalKit datasets:
    - ``model`` attribute for result file naming
    - ``working()`` health check
    - ``generate()`` returning plain text
    """

    def __init__(
        self,
        model=None,
        version='gpt-4-1106-preview',
        retry=5,
        timeout=180,
        max_tokens=2048,
        system_prompt=None,
        img_detail='high',
        verbose=False,
        **kwargs,
    ):
        self.model = model or version
        self.version = version
        self.retry = retry
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.img_detail = img_detail
        self.verbose = verbose
        self.kwargs = kwargs

        self._client = None
        self._client_error = None

    def __str__(self):
        return str(self.model)

    def __repr__(self):
        return f'MMUGPT(model={self.model!r}, version={self.version!r})'

    def _ensure_client(self, silent=False):
        if self._client is not None:
            return True
        if self._client_error is not None:
            if silent:
                return False
            raise RuntimeError(self._client_error)
        try:
            import sys

            internal_path = '/hetu_group/chenjiankang/research'
            if internal_path not in sys.path:
                sys.path.insert(0, internal_path)

            from mmu_chat_gpt_pb2 import MmuChatGptRequest
            from mmu_chat_gpt_pb2_grpc import MmuChatGptServiceStub
            from kess.framework import ClientOption, GrpcClient

            client_option = ClientOption(
                biz_def='mmu',
                grpc_service_name='mmu-chat-gpt-service',
                grpc_stub_class=MmuChatGptServiceStub,
            )
            self._request_cls = MmuChatGptRequest
            self._client = GrpcClient(client_option)
            return True
        except Exception as err:
            self._client_error = (
                'Failed to initialize MMU judge client. '
                'This backend requires internal runtime dependencies '
                '(mmu_chat_gpt_pb2, mmu_chat_gpt_pb2_grpc, kess.framework). '
                f'Original error: {type(err).__name__}: {err}'
            )
            if silent:
                return False
            raise RuntimeError(self._client_error) from err

    def working(self):
        return self._ensure_client(silent=True)

    def _apply_message_format(self, original_messages):
        messages = [dict(role='user', content=[])]
        for item in original_messages:
            if item['type'] == 'text':
                messages[0]['content'].append(
                    dict(type='text', text=item['value'])
                )
            elif item['type'] == 'image':
                messages[0]['content'].append(
                    dict(type='image_url', image_url=dict(url=item['value']))
                )
            elif item['type'] == 'video':
                messages[0]['content'].append(
                    dict(type='video_url', video_url=dict(url=item['value']))
                )
            else:
                raise ValueError(f'Unsupported message type for MMU judge: {item}')
        return messages

    def _normalize_messages(self, prompt=None, **kwargs):
        if self.system_prompt is not None:
            messages = [{'role': 'system', 'content': self.system_prompt}]
        else:
            messages = []

        if isinstance(prompt, str):
            messages.append(
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': prompt}],
                }
            )
            return messages

        if isinstance(prompt, dict):
            return [prompt]

        if isinstance(prompt, list):
            if len(prompt) and 'role' not in prompt[0]:
                return self._apply_message_format(prompt)
            return prompt

        if 'message' in kwargs:
            messages = kwargs['message']
            if isinstance(messages, list) and len(messages) and 'role' not in messages[0]:
                return self._apply_message_format(messages)
            return messages

        raise ValueError('MMU judge expects prompt as str/dict/list or kwargs["message"].')

    def generate(self, prompt=None, temperature=0, max_tokens=None, top_p=None, seed=None, **kwargs):
        self._ensure_client()

        request = self._request_cls(biz=self.version)
        request.session_id = str(uuid.uuid4())
        request.req_id = str(uuid.uuid4())
        request.config['messages'] = 'True'
        request.config['temperature'] = str(temperature).encode()
        request.config['img_detail'] = kwargs.get('img_detail', self.img_detail)
        request.config['max_tokens'] = str(max_tokens) if max_tokens else str(self.max_tokens)

        if top_p is not None:
            request.config['top_p'] = str(top_p)
        if seed is not None:
            request.config['seed'] = str(seed)
        if kwargs.get('paygo_only'):
            request.config['paygo_only'] = str(kwargs.get('paygo_only'))

        messages = self._normalize_messages(prompt, **kwargs)
        request.query = json.dumps(messages)

        start_time = time.time()
        last_error = None
        for retry_idx in range(max(int(self.retry), 1)):
            try:
                resp = self._client.Chat(request, timeout=self.timeout)
                if resp.status.code == 1 and resp.answer != 'UNKNOWN ERROR':
                    payload = json.loads(resp.answer)
                    output = payload['choices'][0]['message']['content']
                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(output)
                        print(f'Tot time: {elapsed}')
                    return output
                if 'invalid_prompt' in str(resp) or 'context_length_exceeded' in str(resp):
                    return 'Z'
                last_error = str(resp)
            except Exception as err:
                last_error = f'{type(err).__name__}: {err}'
            if self.verbose and retry_idx > 0:
                print(f'retry_times: {retry_idx}')

        raise RuntimeError(f'MMU judge request failed after retries: {last_error}')


def mmu_gpt(version, **kwargs):
    return MMUGPT(model=version, version=version, **kwargs)
