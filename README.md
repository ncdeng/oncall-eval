# OnCall 值班助手评测工程

针对「实验室 GPU 服务器智能值班助手」项目的自建回归评测，为四个核心设计决策提供可复现的量化依据：

| 实验 | 验证的设计决策 | 数据集 | 指标 |
| --- | --- | --- | --- |
| A 检索命中 | 值班文档按标题层级切分入库（vs 整篇入库） | `eval/gold_qa.json`（100 题金标准） | 文档级 Recall@1 / Recall@3、答案要点覆盖率 |
| B 按需检索路由 | 检索作为工具由 Agent 按意图决定调用（vs 固定前置检索） | `eval/ondemand_cases.json`（100 条，need/skip 各半） | 路由准确率、无谓检索次数 |
| C 有据作答 / 无据拒答 | 知识库无依据时明确拒答不编造 | `eval/refusal_cases.json`（15 无答案 + 15 可答对照） | 无答案题拒答率、可答题回答率与要点正确率 |
| D 动态重规划 | Planner-Executor-Replanner（vs 一次性线性计划） | `eval/replan_tasks.json`（16 个排障任务，L1/L2/L3 分层） | 根因定位成功率、收敛步数 |

## 目录结构

```
kb/                  18 篇值班知识库文档（GPU 显存 / 训练异常 / 环境配置 / 网络 / 存储 / 调度规范等）
eval/                四套评测数据（构成见上表）
run_eval.py          评测脚本（dry / smoke / 分实验 / 全量）
results.json         最近一次评测结果（各实验带 run_at 时间戳与 _meta 环境信息）
results-v1-small.json  第一版小规模评测集（5 篇 / 25 题 / 20 条 / 5 任务）的存档结果
```

## 设计要点

- **金标准与私有知识库绑定**：每题标注应命中的文档与答案要点关键词（均为手册原文子串）；18 篇手册间故意保留主题交叉（OOM、权限、镜像源等多篇出现）作为检索干扰。
- **路由标注埋入边界样本**：含技术词的概念科普、带报错术语的翻译请求、情绪吐槽等，防止路由准确率虚高。
- **拒答题带貌似相关的干扰**：无答案题会召回主题接近的片段（如 SLURM 问题召回 GPU 使用规范），考验模型顶住干扰拒答；判定采用规则优先 + LLM 兜底，可答题要点命中阈值为 min(2, 关键词数)。
- **排障任务的关键证据须从观测中提取参数**（PID / 节点名 / 容器名 / 服务名 / 端口号）才能获取，用于区分一次性线性计划与动态重规划；任务按难度分层（L1 单跳 / L2 一次参数提取 / L3 多跳或换方向）。
- **失败隔离**：传输失败、JSON 解析失败、schema 不符的样本单独计入 invalid，不混入能力分数。

方法学对标：检索金标准与私有库绑定的做法与 CRUD-RAG / RGB 一致（先建库再出题）；need/skip 路由标注协议参考 RetrievalQA；拒答评测对应 RGB 的负面拒绝测试与 CRAG 的 missing-vs-hallucinated 区分；排障任务形态借鉴 AIOpsLab / ITBench（规模对照：AIOpsLab 48 任务、ITBench 约 100 场景）。

## 运行

```powershell
pip install -r requirements.txt

# 环境变量（setx 或写入 .env）：
#   SILICONFLOW_API_KEY   硅基流动（embedding：BAAI/bge-m3）
#   CHAT_BASE_URL / CHAT_API_KEY / CHAT_MODEL   OpenAI 兼容对话端点
#   可选：CHAT_API=responses / CHAT_PROXY / REASONING_EFFORT / EMBED_MODEL

python run_eval.py --dry          # 数据结构校验（不调 API）
python run_eval.py --smoke        # 两端点连通性
python run_eval.py --only retrieval|ondemand|refusal|replan
python run_eval.py                # 全量四实验
```
