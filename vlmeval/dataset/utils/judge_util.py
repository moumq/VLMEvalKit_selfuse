import os

from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)

OPENAI_JUDGE_MODEL_MAP = {
    'gpt-4-turbo': 'gpt-4-1106-preview',
    'gpt-4-0613': 'gpt-4-0613',
    'gpt-4-0125': 'gpt-4-0125-preview',
    'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
    'chatgpt-1106': 'gpt-3.5-turbo-1106',
    'chatgpt-0125': 'gpt-3.5-turbo-0125',
    'gpt-4o': 'gpt-4o-2024-05-13',
    'gpt-4o-0806': 'gpt-4o-2024-08-06',
    'gpt-4o-1120': 'gpt-4o-2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4.1': 'gpt-4.1',
    'gpt-4.1-mini': 'gpt-4.1-mini',
    'gpt-4.1-2025-04-14': 'gpt-4.1-2025-04-14',
}

SILICONFLOW_JUDGE_MODEL_MAP = {
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'deepseek': 'deepseek-ai/DeepSeek-V3',
}

HF_JUDGE_MODEL_MAP = {
    'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
}

MMU_MODEL_MAP = {
    'gpt-4-turbo': 'wenbin_d06b99ea_gpt-4-1106-Preview',
    'chatgpt-0125': 'wenbin_9cd16197_gpt-35-turbo-0125',
    'gpt-4o': 'wenbin_93bc5129_gpt-4o-2024-05-13',
    'gpt-4o-0806': 'wenbin_ae504dfc_gpt-4o-2024-08-06',
    'gpt-4o-mini': 'wenbin_97df206e_gpt-4o-mini-2024-07-18',
    'gpt-4.1': 'lizhenyu03_3481b071_gpt-4.1',
}


def _resolve_model_version(model, model_map, local_override=None):
    if local_override is not None:
        return local_override
    return model_map.get(model, model)



def build_judge(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI, HFChatModel

    model = kwargs.pop('model', None)
    judge_backend = kwargs.pop('judge_backend', os.environ.get('JUDGE_BACKEND', None))
    kwargs.pop('nproc', None)
    load_env()

    if judge_backend == 'mmu':
        from .mmu_gpt import MMUGPT

        model_version = _resolve_model_version(model, MMU_MODEL_MAP)
        return MMUGPT(model=model, version=model_version, **kwargs)

    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)

    if model in SILICONFLOW_JUDGE_MODEL_MAP:
        model_version = _resolve_model_version(model, SILICONFLOW_JUDGE_MODEL_MAP, LOCAL_LLM)
        return SiliconFlowAPI(model_version, **kwargs)
    if model in HF_JUDGE_MODEL_MAP:
        model_version = _resolve_model_version(model, HF_JUDGE_MODEL_MAP, LOCAL_LLM)
        return HFChatModel(model_version, **kwargs)

    model_version = _resolve_model_version(model, OPENAI_JUDGE_MODEL_MAP, LOCAL_LLM)
    return OpenAIWrapper(model_version, **kwargs)


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You can see the specific error if the API call fails.
"""
