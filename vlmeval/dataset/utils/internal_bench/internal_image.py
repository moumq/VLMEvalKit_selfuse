
""" 图像问答机评脚本，使用图像（可选）、问题、参考答案和模型回答进行评估 """
import os
import re
import time
import uuid
import json
import argparse
import pandas as pd
from loguru import logger
from tqdm import tqdm
from loguru import logger
from typing import Optional, Callable
from ....smp import *
import numpy as np 
from scipy.stats import wilcoxon

EVALUATION_RESULT_COLUMNS = [
    "正确性_原因", "正确性_分数",
    "相关性_原因", "相关性_分数",
    "全面性_原因", "全面性_分数",
    "流畅度_原因", "流畅度_分数",
    "创造性_原因", "创造性_分数",
    "整体评估_原因", "整体评估_分数"
]

PROMPT_DIR = osp.dirname(__file__)
with open(osp.join(PROMPT_DIR, "mixed_scoring_for_normal_tasks_v1.md"), "r", encoding="utf-8") as f:
    MIXED_SCORING_FOR_NORMAL_TASKS_PROMPT = f.read()
with open(osp.join(PROMPT_DIR, "mixed_scoring_for_creative_tasks_v1.md"), "r", encoding="utf-8") as f:
    MIXED_SCORING_FOR_CREATIVE_TASKS_PROMPT = f.read()

# 五维打分 和 综合评分 的 列名
SCORING_COLUMNS = ["正确性_分数", "相关性_分数", "全面性_分数", "流畅度_分数", "创造性_分数"] + ["整体评估_分数"]

# 将对 以下列中所有可能类别 进行 按类别地 分数统计 和 GSB统计
CATEGORICAL_COLUMNS = ["img_category_1", "img_category_2", "prompt_task", "prompt_skill"]

def evaluate_imagevqa(model, line, system_prompt: str = "You are a helpful assistant."):
    if line['prompt_task'] in ["创作类"]:
        prompt = MIXED_SCORING_FOR_CREATIVE_TASKS_PROMPT.format(line['prompt_task'], line['question'], line['answer'], line['prediction'])
    else:
        prompt = MIXED_SCORING_FOR_NORMAL_TASKS_PROMPT.format(line['prompt_task'], line['question'], line['answer'], line['prediction'])

    data_root = LMUDataRoot()
    message_content = []
    if 'image_path' in line:
        image_path = line['image_path']
    else:
        image_path = osp.join(data_root, 'images/InternalImageVQA', line['index'] + '.jpg')
    assert osp.exists(image_path)
    message_content.append({"type": "image_url", "image_url": {"url" :f"data:image;base64,{encode_image_file_to_base64(image_path)}"}})
    message_content.append({"type": "text", "text": prompt})
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message_content})

    for attempt in range(0, 5):
        try:
            gen = model.generate(messages)
            return gen 
        except Exception as e:
            logger.error(f"评估失败（第 {attempt} 次）：{e}")
            if attempt == 4:
                return None
            wait_time = 1 * (2 ** attempt)
            time.sleep(wait_time)

SCORE_MAPPING = {f"{i}分": i for i in range(1, 6)} | {f"{i}": i for i in range(1, 6)}

def parse_score(score: Optional[str | int]) -> Optional[int | None]:
    """ 解析分数字符串或整数，返回整数或 None

    - 如果提供的是分数字符串，则尝试基于 SCORE_MAPPING 转换成整数。转换失败则返回 None。
    - 如果提供的是整数，则检查是否在 1 到 5 的范围内，如果是则返回该整数，否则返回 None。

    Args:
        score (Optional[str | int]): 分数字符串或整数
    
    Returns:
        Optional[int | None]: 解析后的分数，如果无法解析则返回 None
    """
    if score is None:
        logger.warning("无法解析的分数：None，返回 None。")
        return None
    
    if isinstance(score, str):
        score = score.strip()
        if score in SCORE_MAPPING:
            return SCORE_MAPPING[score]
        else:
            logger.warning(f"无法解析的分数字符串：'{score}'，返回 None。")
            return None
    elif isinstance(score, int):
        if score in range(1, 6):
            return score
        else:
            logger.warning(f"超出范围的分数整数：{score}，必须在 1 到 5 之间，返回 None。")
            return None
    else:
        logger.warning(f"无法解析的奇怪分数类型：{type(score)}，返回 None。")
        return None


def parse_evaluation_result(evaluation_result: str) -> dict:
    """ 解析评估结果字符串，提取 JSON 格式的评估结果。

    返回值要么是一个包含各个维度的原因和分数的字典，要么是 None（如果解析失败或结果为空）。
    
    如果评估结果字符串中包含未定义的键（即不在 COLUMNS_FOR_PARSED_EVALUATION 中），
    则会记录警告日志，并忽略这些键。

    Args:
        evaluation_result (str): 模型返回的评估结果字符串

    Returns:
        dict: 包含评估结果的字典，包含各个维度的原因和分数
    """
    if not evaluation_result:
        logger.error("评估结果为空字符串，无法解析。")
        return None

    try:
        match = re.search(r'\{.*\}', evaluation_result, re.DOTALL)
        if match:
            json_string = match.group(0)
            loaded_json = json.loads(json_string)
            
            # 检查是否有 COLUMNS_FOR_STORING_PARSED_EVALUATION 中未定义的键，分 全部未定义 和 部分未定义 两种情况处理
            invalid_keys = set(loaded_json.keys()) - set(EVALUATION_RESULT_COLUMNS)
            if len(invalid_keys) == len(loaded_json):
                logger.error(f"评估结果中所有键均未定义：{invalid_keys}，无法解析评估结果。")
                return None
            
            if invalid_keys:
                logger.warning(f"评估结果中包含未定义的键：{invalid_keys}，将被忽略。")

            # 只保留 COLUMNS_FOR_STORING_PARSED_EVALUATION 中定义的键
            parsed_result = {}
            for col in EVALUATION_RESULT_COLUMNS:
                if col in loaded_json:
                    parsed_result[col] = parse_score(loaded_json[col]) if col.endswith("_分数") else loaded_json[col]
                else:
                    parsed_result[col] = None
            return parsed_result
        else:
            raise ValueError("评估结果非空，但未找到有效的 JSON 格式字符串。")
    except Exception as e:
        logger.error(f"解析评估结果时发生未知错误：{e}")
        return None

def calc_score_summary(ours_eval_df, col: pd.Series) -> pd.Series:
    is_valid = ours_eval_df[col].notna()
    valid_cnt = is_valid.sum()
    logger.info(f"打分列 {col} 有分数的行数: {valid_cnt}")

    valid_scores = ours_eval_df.loc[is_valid, col]
    return pd.Series({
        "1分占比": round((valid_scores == 1).sum() / valid_cnt * 100, 4),
        "2分占比": round((valid_scores == 2).sum() / valid_cnt * 100, 4),
        "3分占比": round((valid_scores == 3).sum() / valid_cnt * 100, 4),
        "4分占比": round((valid_scores == 4).sum() / valid_cnt * 100, 4),
        "5分占比": round((valid_scores == 5).sum() / valid_cnt * 100, 4),
        ">=3分占比": round((valid_scores >= 3).sum() / valid_cnt * 100, 4),
        ">=4分占比": round((valid_scores >= 4).sum() / valid_cnt * 100, 4),
        "均分": round(valid_scores.mean(), 4)
    })

def calc_score_by_category(ours_eval_df, category_col: str) -> pd.DataFrame:
    category_groups = ours_eval_df.groupby(category_col)[SCORING_COLUMNS].mean().round(4)
    category_groups = category_groups.rename(columns={col: f"{col}_均分" for col in SCORING_COLUMNS})
    category_groups = category_groups.sort_values(by=[f"{col}_均分" for col in SCORING_COLUMNS], ascending=False)
    logger.info(f"按 {category_col} 分组的 五维打分 和 综合评分 的 均分:")
    return category_groups

def drop_nan_pairs(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    """移除两个 Series 中，至少一个值为 NaN 的行，仅保留两个 Series 中都不为 NaN 的行。

    用于处理机评结果中，某些行的某些列为 NaN 的情况。例如，部分样本可能缺失某个分数，
    或 “创造性” 评分在大多数非创造性问题上为 NaN。

    Args:
        s1 (pd.Series): 第一个 Series
        s2 (pd.Series): 第二个 Series

    Returns:
        tuple[pd.Series, pd.Series]: 移除 NaN 后的两个 Series
    """
    mask = s1.notna() & s2.notna()
    return s1[mask], s2[mask]


def count_gsb(s1: pd.Series, s2: pd.Series) -> tuple[int, int, int]:
    """ 计数两列评分中 GSB (Good, Satisfactory, Bad) 三类数据对的数量

    具体计算方式即统计 s1 列和 s2 列中，有多少打分：
    - s1 better than s2 —— Good
    - s1 equal to s2 —— Satisfactory
    - s1 worse than s2 —— Bad

    Args:
        s1 (pd.Series): 第一个 Series
        s2 (pd.Series): 第二个 Series
    
    Returns:
        tuple[int, int, int]: GSB 分类的数量 (Good, Satisfactory, Bad)
    """
    good = ((s1 > s2).sum())
    satisfactory = ((s1 == s2).sum())
    bad = ((s1 < s2).sum())
    return good, satisfactory, bad


def calc_gs_sb(s1: pd.Series, s2: pd.Series) -> float:
    """ 计算两列评分中 (G + S) / (B + S) 的比值

    (G+S)/(B+S) 越大，说明两列评分的前者比后者更好/更高。

    Args:
        s1 (pd.Series): 第一个 Series
        s2 (pd.Series): 第二个 Series

    Returns:
        float: (G + S) / (B + S) 的比值
    """
    if len(s1) != len(s2):
        raise ValueError("s1 和 s2 的长度不一致，无法计算 GSB 比值。")
    
    if len(s1) == 0:
        logger.warning("s1 和 s2 的长度为 0，返回 NaN 作为 (G + S) / (B + S) 的比值。")
        return np.nan

    good, satisfactory, bad = count_gsb(s1, s2)
    # 避免除零，除数为零意味着 s1 比 s2 100% 好或相等，返回无穷大表示这种充分好
    if (bad + satisfactory) == 0:
        return np.inf
    return (good + satisfactory) / (bad + satisfactory)


def calc_p_value_wilcoxon(s1: pd.Series, s2: pd.Series) -> float:
    """ 计算 Wilcoxon 符号秩检验的 G/B 或者 (G+S)/(B+S) 的 p 值

    使用 Wilcoxon 符号秩检验来计算两列评分 Good/Bad 比较的 p 值，考虑分数的大小而非仅仅是相对更好或更差。
    - 零假设：两列评分的分布相同，即两列评分没有系统性的性能差异，任何一个 case 的 Good/Bad 都纯粹是随机波动
    - 备择假设：两列评分的分布不同，即两列评分有系统性的性能差异，任何一个 case 的 Good/Bad 都是有意义的

    计算 G/B 的 p 值等价于计算 (G+S)/(B+S) 的 p 值，因为其零假设均为 两列评分的分布相同，而 Satisfactory 的分数在两列评分中是相同的。

    Args:
        s1 (pd.Series): 第一个 Series
        s2 (pd.Series): 第二个 Series
    
    Returns:
        float: p 值
    """
    if len(s1) != len(s2):
        logger.error("s1 和 s2 的长度不一致，无法进行 Wilcoxon 检验")
        raise ValueError("s1 和 s2 的长度不一致，无法进行 Wilcoxon 检验")
    
    if len(s1) < 2:
        logger.warning("s1 和 s2 的长度小于 2，无法进行 Wilcoxon 检验，返回 NaN 作为 p 值。")
        return np.nan
    
    # alternative="two-sided" 表示备择假设，即两列评分的分布不同
    p_value = wilcoxon(s1, s2, alternative="two-sided").pvalue
    return p_value


def calc_95_ci_bootstrapping(s1: pd.Series, s2: pd.Series, n_iterations: int = 1000) -> tuple[float, float]:
    """ 使用 Bootstrap 方法计算 (G+S)/(B+S) 的 95% 置信区间

    Args:
        s1 (pd.Series): 第一个 Series
        s2 (pd.Series): 第二个 Series
        n_iterations (int): Bootstrap 的迭代次数，默认为 1000
    
    Returns:
        tuple[float, float]: 95% 置信区间的下限和上限
    """
    if len(s1) != len(s2):
        logger.error("s1 和 s2 的长度不一致，无法进行 Bootstrap 检验")
        raise ValueError("s1 和 s2 的长度不一致，无法进行 Bootstrap 检验")
    
    if len(s1) == 0:
        logger.warning("s1 和 s2 的长度为 0，无法计算置信区间")
        return np.nan, np.nan
    
    n_samples = len(s1)
    bootstrap_gsb_scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        s1_sample = s1.iloc[indices]
        s2_sample = s2.iloc[indices]
        gsb_score = calc_gs_sb(s1_sample, s2_sample)
        bootstrap_gsb_scores.append(gsb_score)

    lower_bound = np.percentile(bootstrap_gsb_scores, 2.5)
    upper_bound = np.percentile(bootstrap_gsb_scores, 97.5)
    return lower_bound, upper_bound

def calc_gsb_summary(ours_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.Series:
    def func(s1, s2):
        s1, s2 = drop_nan_pairs(s1, s2)
        good, satisfactory, bad = count_gsb(s1, s2)
        gsb_ratio = calc_gs_sb(s1, s2)
        p_value = calc_p_value_wilcoxon(s1, s2)
        ci_lower, ci_upper = calc_95_ci_bootstrapping(s1, s2)

        return pd.Series({
            "G/S/B": f"{round(good, 0)}/{round(satisfactory, 0)}/{round(bad, 0)}",
            "(G+S)/(B+S)": round(gsb_ratio, 4),
            "p-value": round(p_value, 4),
            "95% CI": f"[{round(ci_lower, 4)}, {round(ci_upper, 4)}]"
        })

    gsb_summary = pd.DataFrame({
        f"{col}_比较": func(ours_df[col], other_df[col])
        for col in SCORING_COLUMNS if col in other_df.columns
    })
    return gsb_summary

def calc_gsb_by_category(ours_eval_df, other_eval_df, category_col: str) -> pd.DataFrame:
    our_category_groups = ours_eval_df.groupby(category_col)
    other_category_groups = other_eval_df.groupby(category_col)

    gsb_results = {}
    for name, group in our_category_groups:
        if name not in other_category_groups.groups:
            logger.warning(f"类别 {name} 在对比评估结果中不存在，跳过该类别。")
            continue

        other_group = other_category_groups.get_group(name)
        if len(other_group) == 0:
            logger.warning(f"类别 {name} 在对比评估结果中没有数据，跳过该类别。")
            continue
        
        gsb_results[name] = calc_gsb_summary(group, other_group)

    gsb_df = pd.concat(gsb_results.values(),
                       keys=gsb_results.keys(),
                       names=[category_col, '统计项'])
    return gsb_df


def parse_ocr_response(raw):
    pattern = r'分析理由：\s*(.*?)\s*最终评分：\s*(\d+)\s*$'
    match = re.search(pattern, raw, re.DOTALL | re.MULTILINE)
    
    if match:
        reason = match.group(1).strip()
        score = int(match.group(2))
        if score in [0, 1]:
            return reason, score
        else:
            return f"解析失败：评分必须是0或1，实际为{match.group(2)}", 0
    else:
        return f"解析失败：未找到匹配格式，原始内容：{raw}", 0

# import hashlib
# import functools
# def load_cache():
#     if os.path.exists(CACHE_FILE):
#         with open(CACHE_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}

# def save_cache(cache):
#     with open(CACHE_FILE, "w", encoding="utf-8") as f:
#         json.dump(cache, f, ensure_ascii=False, indent=2)

# def get_question_key(line):
#     ref=line['answer']
#     pred=line['prediction']
#     all_text = f"REF: {ref}\nPRED: {pred}"
#     return hashlib.md5(all_text.encode("utf-8")).hexdigest()

# def cache_response(func):
#     @functools.wraps(func)
#     def wrapper(model, line, *args, **kwargs):
#         cache = load_cache()
#         q_key = get_question_key(line)

#         if q_key in cache:
#             return cache[q_key]
#         result = func(model, line, *args, **kwargs)
#         cache[q_key] = result
#         save_cache(cache)
#         return result
#     return wrapper

def evluate_ocr(model, line):
    # "你是一个图片OCR识别评测评分专家，根据模型输出与参考答案是否完全匹配、是否包括参考答案中的所有内容，说明你的评分理由并给出分数\n"
    # prompt = (
    #     "参考答案：\n{ref}\n\n"
    #     "模型输出：\n{pred}\n\n"
    #     "你是一个图片OCR识别评测评分专家，根据模型输出与参考答案是否匹配、是否包括参考答案中的所有内容，说明你的评分理由并给出分数\n"
    #     "请按以下格式输出：\n"
    #     "分析理由：[你的逐步分析]|||最终评分：[1 或 0]\n"
    #     "仅输出上述一行内容，不要任何多余文字。"
    # )
    prompt = (
        "参考答案：\n{ref}\n\n"
        "模型输出：\n{pred}\n\n"
        "你是一个严格的图片OCR识别评测专家，需按以下规则评分（尤其注意数值等价性）：\n"
        "1分标准（必须同时满足）：\n"
        "模型输出包含参考答案的所有信息（尤其是数值、专有名词、关键术语、文字），没有遗漏\n"
        "数值类信息若存在格式差异但语义等价，视为匹配（例如：25%与0.25、1/4等价；1000与1,000等价；$5与5美元等价）\n"
        "无任何与参考答案冲突的内容（包括核心语义、逻辑关系、关键数值）\n"
        "允许非核心信息的细微差异（如标点、同义表述替换）\n\n"
        "0分标准（满足任一）：\n"
        "缺失参考答案中的信息（包括数值）\n"
        "存在与参考答案语义不一致的数值和单位表达\n"
        "存在与参考答案冲突的内容（如数值错误：25% vs 30%；逻辑矛盾：增加 vs 减少）\n"
        "模型输出为空或模型输出与参考答案完全无关\n\n"
        "请先分析所有参考答案信息（尤其是数值）的匹配情况，再给出评分：\n"
        "请按照以下格式输出：\n"
        "分析理由：[你的逐步分析]\n"
        "最终评分：[1 或 0]\n"
        "仅输出上述格式内容，不得添加额外文字。"
    )

    outputs = model.generate(prompt.format(ref=line['answer'], pred=line['prediction']))
    # print(outputs)
    # import pdb; pdb.set_trace()
    return parse_ocr_response(outputs)
    # parts = outputs.split("|||")
    # if len(parts) != 2:
    #     return "解析失败", 0
    # reason, score_str = parts
    # score = 1 if "1" in score_str else 0
    # return reason.strip(), score