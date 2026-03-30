# coding:utf-8
import cv2
import time, re
import glob
import base64
from openai import OpenAI
import os, sys, json
import requests
from openai import AzureOpenAI
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from collections import defaultdict

def frames_gpt4_answer_score(line_json):
    prompt = """
    你要作为专业语言裁判来完成下面这个任务，你的任务是根据我提供[参考答案]，对相应的的[问题]和[结果]进行评估。
    通过我提供的[参考答案]，然后评估我提供的[结果]在不同的维度上的表现能力高低；
    确保你的评估是全面和公正的，并且能够帮助我理解[结果]在各个维度上的表现优劣。

    [问题]:
    {}

    [参考答案]:
    {}

    [结果]:
    {}

    打分要求：
    1、评估[结果]时需要分为以下几个方面，正确性、相关性、全面性、流畅度，在不同的维度上需要展开叙述下打分的依据。
    2、[正确性]：考察模型识别信息的正确程度，信息是否完全正确，知识信息是否完全符合事实。注：正确性评估时只关注信息的正确程度，不关注信息的完整程度，描述中若遗漏了部分信息时仅扣全面性分数，不需要扣正确性分数。
    3、[相关性]：考察模型输出内容的相关程度，用户指令是否被完全理解，回复内容是否针对于问题进行回答。
    4、[全面性]：考察模型识别全部信息的能力，在描述生成类任务上输出内容是否能完整概括[参考结果]，是否能充分体现细节信息，而不仅仅是通用点；在问答类任务上，考察模型回答问题的完整性能力，输出内容是否有一些解释说明或推理过程，而不仅仅是简单结果。注：全面性评估时只关注信息的完整程度，不关注信息的正确程度，描述包含错误的信息时仅扣正确性评分，不扣全面性，避免两次惩罚扣分。
    5、[流畅度]：考查模型在生成类任务上输出内容是否语句通顺、含错字、病句及杂质，结构是否有条理、逻辑顺畅、可阅读性高、易于理解。

    [返回结果格式要求]
    1. 返回一个直接可解析的json object
    2. 确保每一个字段都是符合要求的
    3. 返回的结果用中文表达

    格式如下:
    ```
    {{
        "正确性_原因": string // 针对提供的视频、[问题]和[结果],参考[正确性]维度内容，详细的叙述下[结果]在[正确性]维度上的表现能力
        "正确性_分数": string // 请根据上述的正确性叙述原因，对[结果]的正确性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：模型输出内容完全或基本正确无误；4分：核心意图信息正确且衍生信息错误占比不超过20% ；3分：核心意图信息正确，衍生信息错误占比不超过40%；2分：模型输出错误信息超过一半；1分：模型输出存在大量错误信息或基本完全错误；请直接返回所属类别即可。
        "相关性_原因": string // 针对提供的视频、[问题]和[结果],参考[相关性]维度内容，详细的叙述下[结果]在[相关性]维度上的表现能力
        "相关性_分数": string // 请根据上述的相关性叙述原因，对[结果]的相关性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：模型输出内容完美；4分：优秀；3分：核心切题且基本满足[问题]限定要求，少量偏题或延伸相关性差的信息；2分：命中问题，存在部分偏题、未满足[问题]部分限定要求；1分：模型输出和[问题]不相关；请直接返回所属类别即可。
        "全面性_原因": string // 针对提供的视频、[问题]和[结果],参考[全面性]维度内容，详细的叙述下[结果]在[全面性]维度上的表现能力
        "全面性_分数": string // 请根据上述的全面性叙述原因，对[结果]的全面性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：描述类问题模型输出完全或基本完全涵盖[参考答案]，或问答类问题模型输出有非常详细、完整且有逻辑的推理过程、解释说明；4分：描述类问题模型输出遗漏了[参考答案]中不超过20%的明显内容，或问答类问题模型输出提供了较全面的推理过程、解释说明，但还有改进空间；3分：描述类问题模型输出遗漏了[参考答案]中不超过40%的明显内容，或问答类问题模型输出提供了一些推理过程、解释说明，但相对有限；2分：描述类问题模型输出遗漏了[参考答案]中超过50%的明显内容，或问答类问题模型输出较简单，有少量补充说明信息；1分：描述类问题模型输出存在大量信息遗漏或几乎完全遗漏[参考答案]中的明显内容，或问答类问题模型输出简略，仅有最终结果或是否，没有任何解释说明；请直接返回所属类别即可。
        "流畅度_原因": string // 针对提供的视频、[问题]和[结果],参考[流畅度]维度内容，详细的叙述下[结果]在[流畅度]维度上的表现能力
        "流畅度_分数": string // 请根据上述的流畅度叙述原因，对[结果]的流畅度表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：清晰有序使读者能够毫不费力地理解内容，创作类表达优美且具有艺术性，无重复问题；4分：较易理解，文字表达基本连贯且清晰，读者能够较快速地获取所传达的信息。但存在一些整体表达结构组织上不够顺畅的部分，有可优化的空间，或者轻微套模板；3分：可以理解但不完全清晰，需要一些推测和解读才能明确所表达的意思，可以存在个别错别字或语法小问题但不影响理解，个别列表/少量片段重复；2分：有很大的歧义或表达混乱，严重影响读者的理解，需要花费较长时间去猜测所表达的意思。回复语种与提问语种不一致。严重套模板、较多列表/片段重复。；1分：文字表达完全不连贯，多个错误混杂，无法传达任何有效信息。或者大量重复、崩溃性重复。；请直接返回所属类别即可。
        "整体评估原因": string // 根据上述的所有子维度评估叙述，综合评价[结果]的表现能力。特殊情况，针对某些简单的问答类问题，核心意图回答正确，但无解释说明，可酌情不扣分，比如，视频中人物的衣服是什么颜色，模型回答是红色，回答正确，可以附5分，因为此类问题似乎也不需要过多的解释说明。
        "整体评估_分数": string // 根据上述所有叙述的表现能力，对整体进行一个打分评估，分值所属类别和上述子维度一致，分为5分-1分；在问答类任务，正确性和相关性是最重要的；在描述类任务，正确性、全面性是最重要的；任一重要维度有低于3分的情况，原则上综合得分是不能高于3分的。请直接返回所属类别。
    }}
    ```
    """
    
    prompt1 = """
    你要作为专业语言裁判来完成下面这个任务，你的任务是根据我提供[参考答案]，对相应的的[问题]和[结果]进行评估。
    通过我提供的[参考答案]，然后评估我提供的[结果]在不同的维度上的表现能力高低；
    确保你的评估是全面和公正的，并且能够帮助我理解[结果]在各个维度上的表现优劣。

    [问题]:
    {}

    [参考答案]:
    {}

    [结果]:
    {}

    打分要求：
    1、评估[结果]时需要分为以下几个方面，正确性、相关性、全面性、流畅度、创造性，在不同的维度上需要展开叙述下打分的依据。
    2、[正确性]：考察模型识别信息的正确程度，信息是否完全正确，知识信息是否完全符合事实。注：正确性评估时只关注信息的正确程度，不关注信息的完整程度，描述中若遗漏了部分信息时仅扣全面性分数，不需要扣正确性分数。
    3、[相关性]：考察模型输出内容的相关程度，用户指令是否被完全理解，回复内容是否针对于问题进行回答。
    4、[全面性]：考察模型识别全部信息的能力，在描述生成类任务上输出内容是否能完整概括[参考结果]，是否能充分体现细节信息，而不仅仅是通用点；在问答类任务上，考察模型回答问题的完整性能力，输出内容是否有一些解释说明或推理过程，而不仅仅是简单结果。注：全面性评估时只关注信息的完整程度，不关注信息的正确程度，描述包含错误的信息时仅扣正确性评分，不扣全面性，避免两次惩罚扣分。
    5、[流畅度]：考查模型在生成类任务上输出内容是否语句通顺、含错字、病句及杂质，结构是否有条理、逻辑顺畅、可阅读性高、易于理解。
    6、[创造性]：考察模型在创意生成类任务上输出内容是否具有创意、想象力，多样性和丰富度是否足够。

    [返回结果格式要求]
    1. 返回一个直接可解析的json object
    2. 确保每一个字段都是符合要求的
    3. 返回的结果用中文表达

    格式如下:
    ```
    {{
        "正确性_原因": string // 针对提供的视频、[问题]和[结果],参考[正确性]维度内容，详细的叙述下[结果]在[正确性]维度上的表现能力
        "正确性_分数": string // 请根据上述的正确性叙述原因，对[结果]的正确性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：模型输出内容完全或基本正确无误；4分：核心意图信息正确且衍生信息错误占比不超过20% ；3分：核心意图信息正确，衍生信息错误占比不超过40%；2分：模型输出错误信息超过一半；1分：模型输出存在大量错误信息或基本完全错误；请直接返回所属类别即可。
        "相关性_原因": string // 针对提供的视频、[问题]和[结果],参考[相关性]维度内容，详细的叙述下[结果]在[相关性]维度上的表现能力
        "相关性_分数": string // 请根据上述的相关性叙述原因，对[结果]的相关性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：模型输出内容完美；4分：优秀；3分：核心切题且基本满足[问题]限定要求，少量偏题或延伸相关性差的信息；2分：命中问题，存在部分偏题、未满足[问题]部分限定要求；1分：模型输出和[问题]不相关；请直接返回所属类别即可。
        "全面性_原因": string // 针对提供的视频、[问题]和[结果],参考[全面性]维度内容，详细的叙述下[结果]在[全面性]维度上的表现能力
        "全面性_分数": string // 请根据上述的全面性叙述原因，对[结果]的全面性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：描述类问题模型输出完全或基本完全涵盖[参考答案]，或问答类问题模型输出有非常详细、完整且有逻辑的推理过程、解释说明；4分：描述类问题模型输出遗漏了[参考答案]中不超过20%的明显内容，或问答类问题模型输出提供了较全面的推理过程、解释说明，但还有改进空间；3分：描述类问题模型输出遗漏了[参考答案]中不超过40%的明显内容，或问答类问题模型输出提供了一些推理过程、解释说明，但相对有限；2分：描述类问题模型输出遗漏了[参考答案]中超过50%的明显内容，或问答类问题模型输出较简单，有少量补充说明信息；1分：描述类问题模型输出存在大量信息遗漏或几乎完全遗漏[参考答案]中的明显内容，或问答类问题模型输出简略，仅有最终结果或是否，没有任何解释说明；请直接返回所属类别即可。
        "流畅度_原因": string // 针对提供的视频、[问题]和[结果],参考[流畅度]维度内容，详细的叙述下[结果]在[流畅度]维度上的表现能力
        "流畅度_分数": string // 请根据上述的流畅度叙述原因，对[结果]的流畅度表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：清晰有序使读者能够毫不费力地理解内容，创作类表达优美且具有艺术性，无重复问题；4分：较易理解，文字表达基本连贯且清晰，读者能够较快速地获取所传达的信息。但存在一些整体表达结构组织上不够顺畅的部分，有可优化的空间，或者轻微套模板；3分：可以理解但不完全清晰，需要一些推测和解读才能明确所表达的意思，可以存在个别错别字或语法小问题但不影响理解，个别列表/少量片段重复；2分：有很大的歧义或表达混乱，严重影响读者的理解，需要花费较长时间去猜测所表达的意思。回复语种与提问语种不一致。严重套模板、较多列表/片段重复。；1分：文字表达完全不连贯，多个错误混杂，无法传达任何有效信息。或者大量重复、崩溃性重复。；请直接返回所属类别即可。
        "创造性_原因": string // 针对提供的视频、[问题]和[结果],参考[创造性]维度内容，详细的叙述下[结果]在[创造性]维度上的表现能力
        "创造性_分数": string // 请根据上述的创造性叙述原因，对[结果]的创造性表现能力进行打分，打分分为5个类别，分别是5分、4分、3分、2分、1分。5分：回答具有独特的创意和新颖性，能够引人入胜并产生强烈的吸引力；4分：回答具有创意和想象力，经常包含令人印象深刻的新颖元素和独特的表达方式，但可能没有达到非常出色的程度，还有改进的空间 ；3分：回答具有一定的创意和新颖性，能够产生一些令人惊喜的新颖元素，但整体创造性仍不够突出，整体体感一般；2分：模型在某些情况下能够生成稍微不同于常规的回答，但创新性有限，回答中偶尔包含一些新的元素或表述方式，但不够突出；1分：回答完全没有创意和新颖性，可能只是简单的复制或改编现有的文本；请直接返回所属类别即可。
        "整体评估原因": string // 根据上述的所有子维度评估叙述，综合评价[结果]的表现能力。特殊情况，针对某些简单的问答类问题，核心意图回答正确，但无解释说明，可酌情不扣分，比如，视频中人物的衣服是什么颜色，模型回答是红色，回答正确，可以附5分，因为此类问题似乎也不需要过多的解释说明。
        "整体评估_分数": string // 根据上述所有叙述的表现能力，对整体进行一个打分评估，分值所属类别和上述子维度一致，分为5分-1分；相关性和创意性是最重要的。决定了得分是否及格，任一重要维度有低于3分的情况，原则上综合得分是不能高于3分的。请直接返回所属类别。
    }}
    ```
    """
    
    if line_json['question_type'] == '创作能力':
        input_query = prompt1.format(line_json['question'], line_json['answer'], line_json['prediction'])
    else:
        input_query = prompt.format(line_json['question'], line_json['answer'], line_json['prediction'])
    return input_query

def extract_markdown_code_or_text(markdown_text):
    """
    从 Markdown 文本中提取第一个代码块的内容。
    如果没有代码块，则返回原始文本。
    """
    match = re.search(r"```json(?:\w+)?\s*([\s\S]*?)```", markdown_text)
    if match:
        return match.group(1).strip()
    else:
        return markdown_text.strip()

def single_main(model, line_json):
    for attempt in range(5):  # 尝试最多5次
        try:
            prompt = frames_gpt4_answer_score(line_json)
            gen = model.generate(prompt)
            return gen
        except Exception as e:
            # import pdb; pdb.set_trace()
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 4:  # 最后一次尝试仍失败
                print("Final attempt failed. No more retries.")
                time.sleep(1)
                return None

def multi_main(f_model_answer, f_eval_score):
    done_cases = {}
    if os.path.isfile(f_eval_score):
        with open(f_eval_score, 'r') as f:
            for line in f:
                x = json.loads(line.strip())
                done_cases[x['data_id']] = 1

    cases_ = []
    with open(f_model_answer, 'r') as ff:
        for line in ff.readlines():
            if not line:
                continue
            line_json = json.loads(line.strip())
            data_id = line_json['data_id']
            if data_id in done_cases:
                continue
            else:
                cases_.append(line_json)

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(single_main, case_) for case_ in cases_]
        for future in as_completed(futures):
            result = future.result()
            if result:
                with open(f_eval_score, "a") as f:
                    f.write(result)
    
    print("******************评分完成******************")
    count = len(open(f_model_answer, 'r').readlines())
    print('模型推理结果总条数为：', count)
    count = len(open(f_eval_score, 'r').readlines())
    print('LLM-judge评分结果总条数为：', count)

def eval_score_calculate(f_eval_score):
    def process(str_json):
        match = re.search(r'\{.*\}', str_json, re.DOTALL)
        if match:
            json_string = match.group(0)
            data = json.loads(json_string)
            return data
        else:
            return {'正确性_分数':'none','相关性_分数':'none','全面性_分数':'none','流畅度_分数':'none','创造性_分数':'none','整体评估_分数':'none'}

    dimensions = ['正确性','相关性','全面性','流畅度','创造性','整体评估']
    skills = ['视觉元素识别','时序信息理解','推理能力','基于先验知识问答能力','鲁棒性能力','专业领域能力', "描述能力", "创作能力"]
    dimensions_score_lst = defaultdict(list)
    skills_score_lst = defaultdict(list)
    
    # f_eval_score = "/Users/raochongling/Desktop/哥伦布项目/视频多模态/评测版本/VQA/竞品对比/video_qa_prompt_1409_v241219_lst.json_InternVL2_5-8B-MPO_result_.json_gpt4_eval_score.json"
    with open(f_eval_score, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
            line_json = json.loads(line.strip())
            eval_score_dict = process(line_json['eval'])
            # print(eval_score_dict)
            for i in dimensions:
                key = i + "_分数"
                if key in eval_score_dict.keys() and eval_score_dict[key].replace('分', '').isdigit():
                    dimensions_score_lst[i].append(int(eval_score_dict[key].replace('分', '')))

            for j in skills:
                if line_json['考察能力'] == j:
                    key = "整体评估" + "_分数"
                    if key in eval_score_dict.keys() and eval_score_dict[key].replace('分', '').isdigit():
                        skills_score_lst[j].append(int(eval_score_dict[key].replace('分', '')))
    
    print(f_eval_score)
    print("******************按评测维度细分统计******************")
    for k in dimensions:
        v = dimensions_score_lst[k]
        print(f"{k}\t{len(v)}\t{np.sum(v)}\t{round(np.mean(v), 3)}\t{len([x for x in v if x >= 1 and x < 2])}\t{len([x for x in v if x >= 2 and x < 3])}\t{len([x for x in v if x >= 3 and x < 4])}\t{len([x for x in v if x >= 4 and x < 5])}\t{len([x for x in v if x >= 5])}\t{round(len([x for x in v if x >= 3]) / len(v) * 100, 3)}%\t{round(len([x for x in v if x >= 4 and x <= 5]) / len(v) * 100, 3)}%")

    print("******************按评测能力细分统计******************")
    for k in skills:
        v = skills_score_lst[k]
        print(f"{k}\t{len(v)}\t{np.sum(v)}\t{round(np.mean(v), 3)}\t{len([x for x in v if x >= 1 and x < 2])}\t{len([x for x in v if x >= 2 and x < 3])}\t{len([x for x in v if x >= 3 and x < 4])}\t{len([x for x in v if x >= 4 and x < 5])}\t{len([x for x in v if x >= 5])}\t{round(len([x for x in v if x >= 3]) / len(v) * 100, 3)}%\t{round(len([x for x in v if x >= 4 and x <= 5]) / len(v) * 100, 3)}%")

if __name__ == '__main__':
    f_model_answer = '/Users/raochongling/Desktop/哥伦布项目/视频多模态/评测数据/video_qa_prompt_1409_v241219_lst.json_Doubao-1.5-thinking-pro-m-250415_64frame.json'
    f_eval_score = f_model_answer + "_gpt4_eval_score.json"
    multi_main(f_model_answer, f_eval_score)
    eval_score_calculate(f_eval_score)

    
    

