# Keye / Judge 迁移变更记录

## 本次改动范围

### 1. 新增统一 Keye API 实现
- 新增 `VLMEvalKit/vlmeval/api/keye_fastapi.py`
- 目标：在 `VLMEvalKit` 内提供与内部 `k-vlmevalkit` 对齐的服务化评测入口，统一支持图像与视频输入
- 关键点：
  - `VIDEO_LLM = True`
  - `allowed_types = ['text', 'image', 'video']`
  - 支持 `video_total_pixels` / `nframes` / `max_frames` / `fps` / `is_vllm`
  - 保留 `think` / `no_think` / `is_xiaomi` 等推理后缀逻辑
  - 支持 `api_base_list` 多地址轮询

### 2. 导出 KeyeFastAPI
- 更新 `VLMEvalKit/vlmeval/api/__init__.py`
- 新增：
  - `from .keye_fastapi import KeyeFastAPI`
  - `__all__` 中加入 `KeyeFastAPI`

### 3. 在 config.py 中注册 Keye API 系列模型
- 更新 `VLMEvalKit/vlmeval/config.py`
- 新增环境变量读取：
  - `KEYE_API_KEY`
  - `KEYE_API_BASE`
  - `KEYE_API_BASE_LIST`
- 扩展 `keye_series`，新增：
  - `Keye-VL-API`
  - `Keye-VL-API-think`
  - `Keye-VL-API-nothink`
  - `Keye-VL-API-video`
- 说明：保留原有 `KeyeChat` 本地模型入口，同时补充服务化入口，满足“本地/远端并存”的兼容需求

### 4. 补 run.py / run_api.py 的 judge_backend 透传
- 更新 `VLMEvalKit/run.py`
  - CLI 新增 `--judge-backend`
  - 构建 `judge_kwargs` 时显式透传 `judge_backend`
  - 修复 `MMReason` 默认 judge 赋值尾随逗号问题，避免变成 tuple
- 更新 `VLMEvalKit/run_api.py`
  - CLI 新增 `--judge-backend`
  - `get_judge_kwargs()` 中透传 `judge_backend`

### 5. judge mmu 后端状态
- 已存在并可用：`VLMEvalKit/vlmeval/dataset/utils/judge_util.py`
- 当前 `build_judge(**kwargs)` 已支持：
  - `judge_backend='mmu'`
  - `MMU_MODEL_MAP`
  - `mmu_gpt.MMUGPT`
- 因此本轮重点是把 CLI 到 `build_judge` 的参数链路补完整，而不是重复实现 judge 核心逻辑

## 设计取舍

### 为什么保留 KeyeChat，同时新增 KeyeFastAPI
- `KeyeChat`：面向本地权重加载/本地推理
- `KeyeFastAPI`：面向 OpenAI-compatible 服务评测
- 两者职责不同，不能互相替代

### 为什么不单独新增 KeyeVideoFastAPI
- 用户要求统一成单一服务类
- 因此直接在 `KeyeFastAPI` 中吸收视频能力，减少配置分叉与后续维护成本

## 已知注意事项
- `KeyeFastAPI` 当前请求体里的 `model` 仍取 `self.client.models.list().data[0].id`，与内部参考实现保持一致，依赖服务端暴露默认模型
- 如果后续需要固定路由到某个服务端模型名，可再为 `KeyeFastAPI` 增加显式 `model` 参数并在 `config.py` 中透传
- 当前静态诊断里仍有仓库原有的大量类型问题；本轮仅处理与本次迁移直接相关的功能接线

## 建议后续动作
1. 用真实 Keye 服务地址验证：
   - 图像集：`Keye-VL-API`
   - 视频集：`Keye-VL-API-video`
2. 用 `--judge-backend mmu` 跑一个需要 judge 的数据集，确认 CLI → `build_judge()` 链路正确
3. 再继续替换部分 dataset 中仍存在的硬编码 judge 使用方式
