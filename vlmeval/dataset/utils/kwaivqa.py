# import os
# import sys
# from collections import defaultdict
# from tqdm import tqdm
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import functools
# import glob
# from ...smp import *
# import re 

# FAIL_MSG = 'Failed to obtain answer via API.'

# def build_KwaiAI_gpt4_prompt(line):
#     prompt = '''给你一个标准答案和一个回答，请根据参考答案判断回答是否正确。标准答案：“{}”。回答：“{}”。请以“回答正确”或“回答错误”的格式返回答案。'''
#     answer = line['answer']
#     prediction = line['prediction']
#     input_prompt = prompt.format(answer, prediction)
#     return input_prompt

# def KwaiAI_eval(model, line):
#     prompt = build_KwaiAI_gpt4_prompt(line)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

# def build_Interest_gpt4_prompt(line):
#     prompt = """你是一个评估助手，任务是判断模型的回答中命中了多少个标准答案中的要点。标准答案是一个列表，其中每一项都涉及不同的要点，请阅读并遵循以下样例：
#     标准答案：['老鹰报恩的温情故事', '野外救助的纪实感', '人与自然和谐共处', '动物救治专业知识', '万物有灵的哲学表达']
#     模型回答：['拯救被困的鹰', '人与自然的温情', '治愈的相遇', '老鹰的感恩']
#     要点计数：2

#     标准答案：['色彩鲜明的创意汤品设计', '幽默文案互动挑战', '清澈汤品的质感呈现', '艺术化的摆盘美学', '隐藏寓意的悬念设置']
#     模型回答：['店员说对半挖就免单', '美食甜品治愈一切不开心', '饭搭子已就位', '美食分享今日美味', '内容启发分享计划', '对半挖旧免单展现实力的时候到了', '哈哈哈我可以免单啦']
#     要点计数：0

#     标准答案： ”{}“
#     模型回答：“{}”
#     要点计数：
# """
#     answer = line['answer']
#     prediction = line['prediction']
#     input_prompt = prompt.format(answer, prediction)
#     return input_prompt

# def Interest_eval(model, line):
#     prompt = build_Interest_gpt4_prompt(line)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

# def build_CommentGen_gpt4_prompt(line):
#     prompt = '''给你一个标准答案和一个回答，其中标准答案是多个合理回答的集合，请判断回答是否与标准答案在整体上风格一致。标准答案：“{}”。回答：“{}”。请以“一致”或“不一致”的格式返回答案。'''
#     answer = line['answer']
#     prediction = line['prediction']
#     input_prompt = prompt.format(answer, prediction)
#     return input_prompt

# def CommentGen_eval(model, line):
#     prompt = build_CommentGen_gpt4_prompt(line)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')


# def IP_eval(model, line):
#     prediction = line['prediction']
#     if str(eval(line['answer'])[0]) in str(prediction):
#         return dict(log='Succeed', res='回答正确')
#     return dict(log=f'output is {prediction}, failed to parse.', res='回答错误')


# def post_yn_check(line):
#     response = line['res']
#     try:
#         if '回答正确' in response:
#             return True
#         else:
#             return False
#     except:
#         print(response)
#         return False

# def post_cmt_check(line):
#     response = line['res']
#     try:
#         if '不一致' in response:
#             return False
#         elif '一致' in response and '不一致' not in response:
#             return True 
#         return False
#     except:
#         print(response)
#         return False

# def post_int_check(line):
#     response = line['res']
#     try:
#         match = re.search(r'\d+(?:\.\d+)?', line['res'])
#         if match:
#             number = match.group()
#             return float(number) / len(eval(line['answer']))
#         return float(line['res'].replace('要点计数：', '')) / len(eval(line['answer']))
#     except:
#         print(response)
#         return 0


# def build_VideoOrder_gpt4_prompt(line):
#     prompt = '''给你一个标准答案和一个回答，请根据标准答案判断回答是否正确，同时忽略回答中的分析过程中，直接对比回答中的结论和标准答案是否一致，标准答案：“{}”。回答：“{}”。请以“回答正确”或“回答错误”的格式返回答案。'''
#     answer = line['answer']
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(answer, prediction)
#     return input_prompt

# def VideoOrder_eval(model, line):
#     prompt = build_VideoOrder_gpt4_prompt(line)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')


# # def VideoOrder_eval_new(model, line):
# #     prompt = '''给你一个标准答案和一个回答，请根据标准答案判断回答是否正确，同时忽略回答中的分析过程中，直接对比回答中的结论和标准答案是否一致，标准答案：“{}”。回答：“{}”。请以“回答正确”或“回答错误”的格式返回答案。'''
# #     answer = line['answer']
# #     prediction = remove_think_tags(line['prediction']).strip()
# #     input_prompt = prompt.format(answer, prediction)
# #     log = ''
# #     retry = 5

# #     for i in range(retry):
# #         prediction = line['prediction']
# #         res = model.generate(input_prompt, temperature=i * 0.5)

# #         if FAIL_MSG in res:
# #             log += f'Try {i}: output is {prediction}, failed to parse.\n'
# #         else:
# #             log += 'Succeed'
# #             return dict(log=log, res=res)
# #     log += 'All 5 retries failed.\n'
# #     return dict(log=log, res='')


# def remove_think_tags(text):
#     # 使用正则表达式匹配并移除 [think] 和 [/think] 包裹的内容
#     cleaned_text = re.sub(r'\<think\>.*?\</think\>', '', text, flags=re.DOTALL)
#     return cleaned_text
    
# def YorN_eval(model, line):
#     prompt = '''这是一道是非判断题，标准答案中会明确表示“是”或者“否”。请根据提供的标准答案判断给定的回答和标准答案是否一致，同时忽略回答中的分析过程，如果是英文则转换为中文理解。标准答案：“{}”。回答：“{}”。请以“回答正确”或“回答错误”的格式返回答案。'''
#     answer = line['answer']
#     prediction = remove_think_tags(line['prediction'])
#     input_prompt = prompt.format(answer, prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

# def Porn_YorN_eval(model, line):
#     prompt = '''这是一段对内容是否为色情评论的判断，其中可能包含了一些分析，请帮我提取出最后的结论。内容：”{}“。请以“是”或“否”的格式返回答案，“是”表示该评论是色情评论，”否“则表示不是。'''
#     answer = line['answer'].strip()
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         if len(prediction) == 1:
#             res = prediction
#         else:
#             res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             if answer in res:
#                 res = "回答正确"
#             else:
#                 res = "回答错误"
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

# def wenjuan_YorN_eval(model, line):
#     prompt = '''这是一段对内容是否令用户满意的判断，其中可能包含了一些分析，请帮我提取出最后的结论。内容：“{}”。请以“是”或“否”的格式返回答案，“是”表示该内容令用户满意，”否“则表示不是。'''
#     answer = line['answer'].strip()
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         if len(prediction) == 1:
#             res = prediction
#         else:
#             res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             if answer in res:
#                 res = "回答正确"
#             else:
#                 res = "回答错误"
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')


# def spu_YorN_eval(model, line):
#     prompt = '''这是一段对两个商品是否属于同款的判断，其中可能包含了一些分析，请帮我提取出最后的结论。内容：“{}”。请以“是”或“否”的格式返回答案，“是”表示两个商品属于同款，”否“则表示不是。'''
#     answer = line['answer'].strip()
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         if len(prediction) == 1:
#             res = prediction
#         else:
#             res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             if answer in res:
#                 res = "回答正确"
#             else:
#                 res = "回答错误"
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')


# def high_YorN_eval(model, line):
    
#     prompt = '''这是一段对视频是否会被大部分用户点赞的判断，其中可能包含了一些分析，请帮我提取出最后的结论。内容：“{}”。请以“是”或“否”的格式返回答案，“是”表示会被大部分用户点赞，”否“则表示不是。'''
#     answer = line['answer'].strip()
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         if len(prediction) == 1:
#             res = prediction
#         else:
#             res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += 'Succeed'
#             if answer in res:
#                 res = "回答正确"
#             else:
#                 res = "回答错误"
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

# def extract_last_bracket_content(text):
#     # 使用正则表达式匹配 [] 括起来的内容
#     pattern = r'\[(.*?)\]'
#     matches = re.finditer(pattern, text)
    
#     # 获取最后一个匹配项
#     last_match = None
#     for match in matches:
#         last_match = match
    
#     # 如果没有匹配项，返回 None
#     if last_match is None:
#         return None
#     return last_match.group(1)

# def is_topic_right(gt, pred):
#     pred = extract_last_bracket_content(pred)
#     idx = 0  # 初始化索引
#     try:
#         for s_1 in gt:
#             if s_1 == "是" or s_1 == "否":
#                 # 确保 pred 中的当前元素也是 "是" 或 "否"
#                 while idx < len(pred) and (pred[idx] != "是" and pred[idx] != "否"):
#                     idx += 1
#                 # 如果 idx 超出范围或不匹配，返回 False
#                 if idx >= len(pred) or pred[idx] != s_1:
#                     return False
#                 idx += 1  # 匹配成功，移动到下一个元素
#         while idx < len(pred):
#             if pred[idx] == "是" or pred[idx] == "否":
#                 return False
#             idx += 1
#         return True
#     except:
#         return False


# def Topic_eval(model, line):
#     prompt = '''这是判断给定的视频列表中的视频是否与第一个视频属于同一个主题的结果，其中可能包含了一些分析，请帮我提取出最后的结论.内容为：”{}“。请将答案直接以列表的形式输出，列表的每一项为“是”或者“否”，表示对应视频是否与第一个视频属于同一个主题,答案形式为[是，否...是]的格式。'''
#     answer = line['answer'].split("_")
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += res
#             if is_topic_right(answer, res):
#                 res = "回答正确"
#             else:
#                 res = "回答错误"
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

# def god_eval(model, line):
#     prompt = '''
#         这是分析解释一下所给的评论为什么可以被审核人员评定成为神评的原因，请评估以下标准答案和模型预测中的分析是否一致。通过比较关键点判断两者分析内容是否一致，标准答案：“{}”。模型预测：“{}”。假设你是一位语文阅卷老师正在批阅阅读理解题，请根据以下标准对模型预测进行评分：完整性：模型预测是否包含了标准答案所有重要的信息点？，请直接给出结论,模型“回答正确”或“回答错误”。'''
#     answer = line['answer']
#     prediction = remove_think_tags(line['prediction']).strip()
#     input_prompt = prompt.format(answer, prediction)
#     log = ''
#     retry = 5

#     for i in range(retry):
#         prediction = line['prediction']
#         res = model.generate(input_prompt, temperature=i * 0.5)

#         if FAIL_MSG in res:
#             log += f'Try {i}: output is {prediction}, failed to parse.\n'
#         else:
#             log += res
#             if "回答正确"  in res:
#                 res = "回答正确"
#             else:
#                 res = "回答错误"
#             return dict(log=log, res=res)
#     log += 'All 5 retries failed.\n'
#     return dict(log=log, res='')

import os
import sys
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import glob
from ...smp import *
import re 
import json
FAIL_MSG = 'Failed to obtain answer via API.'

def remove_think_tags(text):
    cleaned_text = re.sub(r'\<think\>.*?\</think\>', '', text, flags=re.DOTALL)
    return cleaned_text

def post_yn_check(line):
    response = line['res']
    try:
        if '回答正确' in response:
            return True
        else:
            return False
    except:
        print(response)
        return False

def post_cmt_check(line):
    response = line['res']
    try:
        if '不一致' in response:
            return False
        elif '一致' in response and '不一致' not in response:
            return True 
        return False
    except:
        print(response)
        return False



def extract_last_bracket_content(text):
    pattern = r'\[(.*?)\]'
    matches = re.finditer(pattern, text)
    
    last_match = None
    for match in matches:
        last_match = match
    
    if last_match is None:
        return None
    return last_match.group(1)

def is_topic_right(gt, pred):
    pred = extract_last_bracket_content(pred)
    idx = 0  
    try:
        for s_1 in gt:
            if s_1 == "是" or s_1 == "否":
                while idx < len(pred) and (pred[idx] != "是" and pred[idx] != "否"):
                    idx += 1
                if idx >= len(pred) or pred[idx] != s_1:
                    return False
                idx += 1  
        while idx < len(pred):
            if pred[idx] == "是" or pred[idx] == "否":
                return False
            idx += 1
        return True
    except:
        return False


def Topic_eval(model, line):
    prompt = '''这是判断给定的视频列表中的视频是否与第一个视频属于同一个主题的结果，其中可能包含了一些分析，请帮我提取出最后的结论.内容为：”{}“。请将答案直接以列表的形式输出，列表的每一项为“是”或者“否”，表示对应视频是否与第一个视频属于同一个主题,答案形式为[是，否...是]的格式。'''
    answer = line['answer'].split("_")
    prediction = remove_think_tags(line['prediction']).strip()
    input_prompt = prompt.format(prediction)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(input_prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += res
            if is_topic_right(answer, res):
                res = "回答正确"
            else:
                res = "回答错误"
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')


def build_Kwaivqa_gpt4_prompt(line):
    prompt = '''给你一个标准答案和一个回答，请根据标准答案判断回答是否正确，同时忽略回答中的分析过程中，直接对比回答中的结论和标准答案是否一致，标准答案：“{}”。回答：“{}”。请以“回答正确”或“回答错误”的格式返回答案。'''
    answer = line['answer']
    prediction = remove_think_tags(line['prediction']).strip()
    input_prompt = prompt.format(answer, prediction)
    return input_prompt

def Kwaivqa_eval(model, line):
    prompt = build_Kwaivqa_gpt4_prompt(line)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log=log, res=res)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='')
