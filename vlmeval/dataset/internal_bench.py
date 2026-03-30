from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .image_base import ImageBaseDataset
import pandas as pd
from PIL import Image 
import io 
from .utils.internal_bench.internal_video import single_main
from .utils.internal_bench.internal_image import *
import re 
import requests
from ..utils import track_progress_rich
from tqdm import tqdm
import ast


FAIL_MSG = 'Failed to obtain answer via API.'

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 如果返回 4xx 或 5xx 会抛出异常
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return 1, save_path
    except requests.exceptions.RequestException as e:
        print(f"Download URL ERROR: {e}")
        return -1, url

import base64

def image_url_to_base64(url):
    response = requests.get(url)
    response.raise_for_status()  # 如果请求失败会抛出异常
    image_data = response.content
    encoded = base64.b64encode(image_data).decode('utf-8')
    return encoded


class InternalOCR(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        "InternalOCR": "InternalOCR_local.tsv"
    }

    def load_data(self, dataset):
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == '':
            url = dataset + '.tsv'
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        self.data_path = data_path
        if osp.exists(data_path):
            pass
        else:
            test_datas = pd.read_excel('/mmu_mllm_hdd_2/dinghaojie/Benchmarks/vlmeval/ocr_internal_test.xlsx')
            tsv_datas = []
            for i in tqdm(range(len(test_datas))):
                line = test_datas.iloc[i]
                if not osp.exists(osp.join(data_root, 'images/InternalOCR', f"{line['raw_index']}.jpg")):
                    decode_base64_to_image_file(image_url_to_base64(line['img_src']), osp.join(data_root, 'images/InternalOCR', f"{line['raw_index']}.jpg"))
                tsv_datas.append(dict(
                    index = line['index'],
                    # image = image_url_to_base64(line['img_src']),
                    image_path = osp.join(data_root, 'images/InternalOCR', f"{line['raw_index']}.jpg"),
                    question = line['question'],
                    answer = line['answer'],
                    category = line['category']
                ))
            tsv_datas = pd.DataFrame(tsv_datas)
            dump(tsv_datas, data_path)

        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        judge_model_name = judge_kwargs.pop('model', 'gpt-4o')
        storage = eval_file.replace(f'.{suffix}', f'_{judge_model_name}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{judge_model_name}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        data = load(eval_file)

        if not osp.exists(storage):
            data = load(eval_file)
            # codeflicker-fix: LOGIC-Issue-001/54dno1ajsgy1cfev5iow
            model = build_judge(model=judge_model_name, max_tokens=2048,
                                system_prompt='You are a helpful assistant.', **judge_kwargs)
            assert model.working(), ('InternalOCR evaluation requires a working judge\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            lines = [line for line in lines if str(line['think']).strip()] # 清理空预测
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    evluate_ocr,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, (reason, score) in zip(indices, new_results):
                    ans[k] = {
                        "reason": reason,
                        "score": score
                    }
                data['eval'] = [ans[idx] for idx in data['index']]
                dump(data, storage)
        data = load(storage)
        lt = len(data)

        model_stats = {
            "total": 0,
            "correct": 0,
            "category_stats": {}  # 每个分类的统计
        }

        for i in range(lt):
            line = data.iloc[i]
            # import pdb; pdb.set_trace()
            try:
                score = int(ast.literal_eval(line['eval'])['score'])
            except:
                score = 0
            model_stats["total"] += 1
            if  score == 1:
                model_stats["correct"] += 1

            category = line['category']
            if category not in model_stats["category_stats"]:
                model_stats["category_stats"][category] = {"total": 0, "correct": 0}
            model_stats["category_stats"][category]["total"] += 1
            if score == 1:
                model_stats["category_stats"][category]["correct"] += 1
        
        for category in model_stats["category_stats"]:
            model_stats["category_stats"][category]["acc"] = model_stats["category_stats"][category]["correct"] / model_stats["category_stats"][category]["total"]
        model_stats["acc"] = model_stats["correct"] / model_stats["total"]

        dump(model_stats, eval_file.replace('.xlsx', '_score.json'))
        return model_stats


class InternalImageVQA(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        "InternalImageVQA": "InternalImageVQA.tsv",
        "InternalImageVQA_Subset": "InternalImageVQA_Subset.tsv",
        "Internal_OCR": "Internal_OCR.tsv"
    }

    def load_data(self, dataset):
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == '':
            url = dataset + '.tsv'
        file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
        return self.prepare_tsv(url, file_md5)

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        self.data_path = data_path
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        elif self.dataset_name == "Internal_OCR":
            datas = pd.read_excel('/mmu_mllm_hdd_2/dinghaojie/Benchmarks/vlmeval/ocr_internal.xlsx')
            tsv_datas = []
            for i in range(len(datas)):
                line = datas.iloc[i]
                tsv_datas.append(dict(
                    index = line['index'],
                    image = image_url_to_base64(line['img_src']),
                    question = line['question'],
                    answer = line['参考答案']
                ))
            tsv_datas = pd.DataFrame(tsv_datas)
            dump(tsv_datas, data_path)
        else:
            datas = pd.read_excel('/mmu_mllm_hdd_2/dinghaojie/Benchmarks/vlmeval/img_qa_prompt_reference_1397.xlsx')
            tsv_datas = []
            for i in range(len(datas)):
                line = datas.iloc[i]
                tsv_datas.append(dict(
                    index = line['id'],
                    image = image_url_to_base64(line['img_url']),
                    question = line['prompt'],
                    answer = line['reference'],
                    img_category_1 = line['img_category_1'],
                    img_category_2 = line['img_category_2'],
                    prompt_task = line['prompt_task'],
                    prompt_skill = line['prompt_skill']
                ))
            tsv_datas = pd.DataFrame(tsv_datas)
            dump(tsv_datas, data_path)

        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        judge_model_name = judge_kwargs.pop('model', 'gpt-4.1')
        storage = eval_file.replace(f'.{suffix}', f'_{judge_model_name}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{judge_model_name}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        data = load(eval_file)

        if not osp.exists(storage):
            data = load(eval_file)
            # codeflicker-fix: LOGIC-Issue-001/54dno1ajsgy1cfev5iow
            model = build_judge(model=judge_model_name, max_tokens=4096, **judge_kwargs)
            assert model.working(), ('InternalImageVQA evaluation requires a working judge\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    evaluate_imagevqa,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = v
                data['eval'] = [ans[idx] for idx in data['index']]
                dump(data, storage)

        data = load(storage)
        lt = len(data)

        for i in range(lt):
            line = data.iloc[i]
            parsed_result = parse_evaluation_result(line['eval'])
            if parsed_result:
                for col, value in parsed_result.items():
                    data.loc[i, col] = value


        scores_summary = pd.DataFrame({col: calc_score_summary(data, col) for col in SCORING_COLUMNS})
        img_category_1_scores = calc_score_by_category(data, "img_category_1")
        img_category_2_scores = calc_score_by_category(data, "img_category_2")
        prompt_task_scores = calc_score_by_category(data, "prompt_task")
        prompt_skill_scores = calc_score_by_category(data, "prompt_skill")

        dump(img_category_1_scores, eval_file.replace('.xlsx', '_img_category_1五维评分.csv'))
        dump(img_category_2_scores, eval_file.replace('.xlsx', '_img_category_2五维评分.csv'))
        dump(prompt_task_scores, eval_file.replace('.xlsx', '_prompt_task五维评分.csv'))
        dump(prompt_skill_scores, eval_file.replace('.xlsx', '_prompt_skill五维评分.csv'))

        doubao_df = load('/mmu_mllm_hdd_2/dinghaojie/Benchmarks/vlmeval/vlmeval/dataset/utils/internal_bench/img_qa_answer_128_doubao_eval_20250626_161425.xlsx')
        qwen_df = load('/mmu_mllm_hdd_2/dinghaojie/Benchmarks/vlmeval/vlmeval/dataset/utils/internal_bench/img_qa_answer_128_qwen_eval_20250626_161614.xlsx')
        doubao_gsb_summary = calc_gsb_summary(data, doubao_df)
        qwen_gsb_summary = calc_gsb_summary(data, qwen_df)

        dump(doubao_gsb_summary, eval_file.replace('.xlsx', '_Doubao_GSB.csv'))
        dump(qwen_gsb_summary, eval_file.replace('.xlsx', '_Qwen_GSB.csv'))

        return doubao_gsb_summary

class InternalVideo(VideoBaseDataset):
    TYPE = 'Video-VQA'

    def __init__(self, dataset='InternalVideo', use_subtitle=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['InternalVideo']

    def prepare_dataset(self, dataset_name='InternalVideo_Subset', repo_id='lmms-lab/Video-MME'):
        dataset_path = LMUDataRoot()
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        if os.path.exists(data_file):
            return dict(data_file=data_file, root=dataset_path)
        tsv_df = []
        os.makedirs(osp.join(dataset_path, 'images', self.dataset_name), exist_ok=True)
        for raw_line in open('/mmu_mllm_hdd_2/dinghaojie/Benchmarks/vlmeval/video_qa_prompt_1409_v241219.json'):
            line = json.loads(raw_line)
            question = line['prompt']
            url = line['视频URL']
            retcode, video_path = download_file(url, osp.join(dataset_path, 'images', self.dataset_name, line['视频名称']))
            if retcode == 1:
                tsv_df.append(dict(
                    index = line['data_id'],
                    video_path = video_path,
                    video = line['视频名称'],
                    question = line['prompt'],
                    answer = line['参考答案'],
                    question_type = line['考察能力'],
                ))

        tsv_df = pd.DataFrame(tsv_df)
        dataset_path = LMUDataRoot()
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        dump(tsv_df, data_file)
        return dict(data_file=data_file, root=dataset_path)

    def frame_paths(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = osp.join(self.data_root, video_path)
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            if len(vid) < self.nframe:
                nframe = len(vid)
            else:
                nframe = self.nframe
            step_size = len(vid) / (nframe + 1)
            indices = [int(i * step_size) for i in range(1, nframe + 1)]
            frame_paths = self.frame_paths(video_path[:-4], len(indices))
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path[:-4], len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video_path'], video_llm)

        message = []
        if video_llm:
            message.append(dict(type='video', value=osp.join(line['video_path'])))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=line['question']))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        # model = 'gpt-4o-mini'
        model = 'gpt-4-turbo'
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        data = load(eval_file)

        if not osp.exists(storage):
            data = load(eval_file)
            # codeflicker-fix: LOGIC-Issue-001/54dno1ajsgy1cfev5iow
            model = build_judge(model=judge_kwargs.pop('model', 'gpt-4-turbo'), max_tokens=4096,
                                system_prompt='You are a helpful assistant.', **judge_kwargs)
            assert model.working(), ('InternalVideo evaluation requires a working judge\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    single_main,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = v
                data['eval'] = [ans[idx] for idx in data['index']]
                dump(data, storage)
            
        data = load(storage)
        lt = len(data)

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

        for i in range(lt):
            line = data.iloc[i]
            eval_score_dict = process(str(line['eval']))
            

            for i in dimensions:
                key = i + "_分数"
                if key in eval_score_dict.keys() and eval_score_dict[key].replace('分', '').isdigit():
                    dimensions_score_lst[i].append(int(eval_score_dict[key].replace('分', '')))

            for j in skills:
                if line['question_type'] == j:
                    key = "整体评估" + "_分数"
                    if key in eval_score_dict.keys() and eval_score_dict[key].replace('分', '').isdigit():
                        skills_score_lst[j].append(int(eval_score_dict[key].replace('分', '')))

        total_score, total_len = 0, 0
        for key in dimensions_score_lst:
            total_score += sum(dimensions_score_lst[key])
            total_len += len(dimensions_score_lst[key])
            dimensions_score_lst[key] = sum(dimensions_score_lst[key]) / len(dimensions_score_lst[key])
        dimensions_score_lst['AVG'] = total_score / total_len

        total_score, total_len = 0, 0
        for key in skills_score_lst:
            total_score += sum(skills_score_lst[key])
            total_len += len(skills_score_lst[key])
            skills_score_lst[key] = sum(skills_score_lst[key]) / len(skills_score_lst[key])
        skills_score_lst['AVG'] = total_score / total_len

        rating = {
            '评测维度细分': dimensions_score_lst,
            '评测能力细分': skills_score_lst
        }

        dump(rating, tgt_file)
        return rating