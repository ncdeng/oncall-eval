import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
KB_DIR = ROOT / "kb"
EVAL_DIR = ROOT / "eval"

# Embedding 走硅基流动（国内直连），Chat 走中转站（可经本机代理），配置来自环境变量：
#   SILICONFLOW_API_KEY  硅基流动 key
#   EMBED_MODEL          可选，默认 BAAI/bge-m3
#   CHAT_BASE_URL        中转站地址，如 https://xxx/v1
#   CHAT_API_KEY         中转站 key
#   CHAT_MODEL           模型名，如 gpt-5.6-terra
#   CHAT_API             可选：chat（默认）或 responses
#   CHAT_PROXY           可选：chat 请求代理，如 http://127.0.0.1:7890；不设则自动读系统代理
#   REASONING_EFFORT     可选：默认 medium，中转不支持时自动去掉
EMBED_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"

EMB_CLIENT = None
CHAT_CLIENT = None
EMBED_MODEL = DEFAULT_EMBED_MODEL
CHAT_MODEL = ""
CHAT_API = "chat"
REASONING_EFFORT = "medium"
_NO_TEMP = False
_NO_EFFORT = False

TOOLS_DESC = (
    "query_log(查错误/运行日志)、query_process_rank(查进程资源占用排行，metric=cpu/gpu/mem)、"
    "query_process_detail(查某进程详情，需要进程名或PID)、query_dmesg(查系统内核日志)、"
    "query_monitor(查监控指标，需要服务名或节点名)、query_disk(查磁盘使用)、"
    "query_docs(检索运维知识库)、get_current_time(取当前时间)"
)


def get_env(name):
    v = os.environ.get(name, "").strip()
    if v:
        return v
    # setx 设置的用户级变量对已启动的进程不可见，从注册表兜底读取
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = str(winreg.QueryValueEx(k, name)[0]).strip()
                if v:
                    return v
        except OSError:
            pass
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def detect_system_proxy():
    if sys.platform != "win32":
        return ""
    try:
        import winreg

        k = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        )
        if not winreg.QueryValueEx(k, "ProxyEnable")[0]:
            return ""
        server = str(winreg.QueryValueEx(k, "ProxyServer")[0]).strip()
        if not server:
            return ""
        if "=" in server:
            entries = dict(p.split("=", 1) for p in server.split(";") if "=" in p)
            host = entries.get("https") or entries.get("http") or ""
            return f"http://{host}" if host else ""
        return f"http://{server}"
    except OSError:
        return ""


def init_embedding_client():
    """初始化只依赖 embedding 配置的客户端，供纯检索实验使用。"""
    global EMB_CLIENT, EMBED_MODEL
    from openai import OpenAI

    sf_key = get_env("SILICONFLOW_API_KEY")
    if not sf_key:
        print("ERROR: 缺少配置：SILICONFLOW_API_KEY（用 setx 设置或写入 oncall-eval/.env）", file=sys.stderr)
        sys.exit(2)
    EMBED_MODEL = get_env("EMBED_MODEL") or DEFAULT_EMBED_MODEL
    EMB_CLIENT = OpenAI(api_key=sf_key, base_url=EMBED_BASE_URL)
    print(f"Embedding 配置就绪：embedding={EMBED_MODEL}@硅基流动（直连）")


def init_clients():
    global EMB_CLIENT, CHAT_CLIENT, EMBED_MODEL, CHAT_MODEL, CHAT_API, REASONING_EFFORT
    from openai import OpenAI

    sf_key = get_env("SILICONFLOW_API_KEY")
    chat_base = get_env("CHAT_BASE_URL")
    chat_key = get_env("CHAT_API_KEY")
    CHAT_MODEL = get_env("CHAT_MODEL")
    CHAT_API = (get_env("CHAT_API") or "chat").lower()
    REASONING_EFFORT = (get_env("REASONING_EFFORT") or "medium").lower()
    EMBED_MODEL = get_env("EMBED_MODEL") or DEFAULT_EMBED_MODEL

    missing = [
        n
        for n, v in [
            ("SILICONFLOW_API_KEY", sf_key),
            ("CHAT_BASE_URL", chat_base),
            ("CHAT_API_KEY", chat_key),
            ("CHAT_MODEL", CHAT_MODEL),
        ]
        if not v
    ]
    if missing:
        print(f"ERROR: 缺少配置：{', '.join(missing)}（用 setx 设置或写入 oncall-eval/.env）", file=sys.stderr)
        sys.exit(2)

    proxy = get_env("CHAT_PROXY") or detect_system_proxy()
    http_client = None
    if proxy:
        import httpx

        try:
            http_client = httpx.Client(proxy=proxy, timeout=300.0)
        except TypeError:
            http_client = httpx.Client(proxies=proxy, timeout=300.0)

    EMB_CLIENT = OpenAI(api_key=sf_key, base_url=EMBED_BASE_URL)
    CHAT_CLIENT = OpenAI(api_key=chat_key, base_url=chat_base.rstrip("/"), http_client=http_client)
    print(
        f"配置就绪：embedding={EMBED_MODEL}@硅基流动（直连），"
        f"chat={CHAT_MODEL}@{chat_base}（{CHAT_API} 接口，effort={REASONING_EFFORT}，"
        f"代理={proxy or '无'}）"
    )


def load_kb_chunks(strategy):
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        prefix = path.name[:2]
        text = path.read_text(encoding="utf-8")
        doc_title = ""
        m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if m:
            doc_title = m.group(1).strip()
        if strategy == "whole":
            chunks.append({"text": text.strip(), "doc": prefix, "file": path.name, "title": doc_title})
        elif strategy == "heading":
            parts = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
            for part in parts:
                part = part.strip()
                if not part or part.startswith("# "):
                    continue
                hm = re.match(r"^##\s+(.+)", part)
                sub = hm.group(1).strip() if hm else ""
                ctx = f"{doc_title} - {sub}\n{part}" if doc_title else part
                if len(ctx) < 60 and chunks and chunks[-1]["doc"] == prefix:
                    chunks[-1]["text"] += "\n" + part
                else:
                    chunks.append({"text": ctx, "doc": prefix, "file": path.name, "title": sub})
    return chunks


def load_json(name):
    return json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))


def embed(texts):
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        for attempt in range(3):
            try:
                resp = EMB_CLIENT.embeddings.create(model=EMBED_MODEL, input=batch)
                out.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    arr = np.array(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr


def chat(prompt, temperature=0.0):
    global _NO_TEMP, _NO_EFFORT
    for attempt in range(6):
        try:
            if CHAT_API == "responses":
                kwargs = {}
                if not _NO_EFFORT and REASONING_EFFORT:
                    kwargs["reasoning"] = {"effort": REASONING_EFFORT}
                r = CHAT_CLIENT.responses.create(model=CHAT_MODEL, input=prompt, **kwargs)
                content = r.output_text
            else:
                kwargs = {}
                if not _NO_TEMP:
                    kwargs["temperature"] = temperature
                if not _NO_EFFORT and REASONING_EFFORT:
                    kwargs["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
                r = CHAT_CLIENT.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
                content = r.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("空回复")
            return content.strip()
        except Exception as e:
            msg = str(e).lower()
            if "temperature" in msg and not _NO_TEMP:
                _NO_TEMP = True
                continue
            if ("reasoning" in msg or "effort" in msg) and not _NO_EFFORT:
                _NO_EFFORT = True
                continue
            if attempt == 5:
                raise
            time.sleep(2 ** (attempt + 1))


def parse_json(text):
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)


def retrieve(chunks, mat, query, k=3):
    q = embed([query])[0]
    sims = mat @ q
    idx = np.argsort(-sims)[:k]
    return [chunks[i] for i in idx]


def exp_retrieval():
    gold = load_json("gold_qa.json")["items"]
    qmat = embed([it["question"] for it in gold])
    result = {}
    for strategy in ["whole", "heading"]:
        chunks = load_kb_chunks(strategy)
        mat = embed([c["text"] for c in chunks])
        sims = qmat @ mat.T
        r1 = r3 = cov = 0
        for i, item in enumerate(gold):
            idx = np.argsort(-sims[i])[:3]
            docs = [chunks[j]["doc"] for j in idx]
            ok_docs = {item["expected_doc"], *item.get("acceptable_docs", [])}
            if docs[0] in ok_docs:
                r1 += 1
            if ok_docs & set(docs):
                r3 += 1
            joined = " ".join(chunks[j]["text"] for j in idx).lower()
            kws = item["answer_keywords"]
            hit = sum(1 for kw in kws if kw.lower() in joined)
            cov += hit / len(kws)
        n = len(gold)
        result[strategy] = {
            "chunks": len(chunks),
            "recall@1": round(r1 / n, 3),
            "recall@3": round(r3 / n, 3),
            "answer_coverage": round(cov / n, 3),
            "n": n,
        }
        print(f"  策略 {strategy}: recall@1={result[strategy]['recall@1']} recall@3={result[strategy]['recall@3']}", flush=True)
    result["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


def exp_ondemand():
    cases = load_json("ondemand_cases.json")["items"]

    def route_one(c):
        prompt = (
            "你是实验室 GPU 服务器值班助手的检索路由。判断回答下面这句用户输入是否需要查询值班知识库"
            "（故障处理手册、报错对照、环境配置、服务器使用规范等）。"
            "服务器故障/报错/资源/环境/使用流程类问题需要检索；"
            "问候、闲聊、感谢、问时间、算术、翻译、改写、通用概念科普等无需检索。"
            "只输出一个词：need 或 skip。\n"
            f"用户输入：{c['query']}"
        )
        try:
            ans = re.sub(r"no\s+need", "skip", chat(prompt).lower().strip())
        except Exception as e:
            return c, None, str(e)[:80]
        if ans.startswith("need"):
            pred = True
        elif ans.startswith("skip"):
            pred = False
        else:
            hits = re.findall(r"need|skip", ans)
            pred = bool(hits) and hits[-1] == "need"
        return c, pred, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(route_one, cases))

    correct = 0
    fixed_waste = sum(1 for c in cases if not c["should_retrieve"])
    ondemand_waste = 0
    ondemand_skips = 0
    errors = []
    invalid = []
    for c, pred, err in outs:
        if err is not None:
            invalid.append({"id": c["id"], "error": err})
            continue
        if pred == c["should_retrieve"]:
            correct += 1
        else:
            errors.append({"id": c["id"], "query": c["query"], "pred": pred, "gold": c["should_retrieve"]})
        if not pred:
            ondemand_skips += 1
        if pred and not c["should_retrieve"]:
            ondemand_waste += 1
    n = len(cases) - len(invalid)
    return {
        "n": n,
        "invalid_n": len(invalid),
        "route_accuracy": round(correct / n, 3) if n else None,
        "fixed_wasted_retrievals": fixed_waste,
        "ondemand_wasted_retrievals": ondemand_waste,
        "retrievals_saved": ondemand_skips,
        "errors": errors,
        "invalid": invalid,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def exp_refusal():
    cases = load_json("refusal_cases.json")["items"]
    chunks = load_kb_chunks("heading")
    mat = embed([c["text"] for c in chunks])
    qmat = embed([c["question"] for c in cases])

    REFUSE_PAT = re.compile(r"未收录|暂未收录|无法回答|资料不足|没有(找到)?相关|不在(值班)?知识库|联系管理员|超出.{0,6}范围")

    def judge_refused(gen):
        head = gen[:600]
        rule_hit = bool(REFUSE_PAT.search(head))
        # 规则命中且未给出具体方案特征（命令/路径/参数）时直接判拒答，否则 LLM 兜底
        has_solution = bool(re.search(r"`|nvidia-smi|export |sudo |df |CUDA_|--\w+|/data|~/", head))
        if rule_hit and not has_solution:
            return True
        if not rule_hit and has_solution:
            return False
        verdict = chat(
            "判断下面这段值班助手的回复属于【拒答】还是【作答】。"
            "拒答=明确表示资料不足/未收录/无法回答/建议联系管理员，且没有给出具体技术方案；"
            "作答=给出了具体的技术判断或操作建议。只输出一个词：拒答 或 作答。\n"
            f"回复：{gen[:600]}"
        )
        return "拒答" in verdict

    def answer_one(item):
        i, case = item
        try:
            idx = np.argsort(-(mat @ qmat[i]))[:3]
            ctx = "\n---\n".join(chunks[j]["text"][:500] for j in idx)
            gen = chat(
                "你是实验室 GPU 服务器值班助手。仅依据下面的值班手册资料回答用户问题；"
                "若资料不足以回答该问题，明确说明知识库暂未收录、建议联系管理员，绝不编造答案。\n"
                f"资料：\n{ctx}\n\n问题：{case['question']}\n回答："
            )
            refused = judge_refused(gen)
        except Exception as e:
            return case, "", None, str(e)[:80]
        return case, gen, refused, None

    with ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(answer_one, enumerate(cases)))

    unans_total = unans_refused = 0
    ans_total = ans_answered = ans_correct = 0
    details = []
    invalid = []
    for case, gen, refused, err in outs:
        if err is not None:
            invalid.append({"id": case["id"], "error": err})
            continue
        if case.get("unanswerable"):
            unans_total += 1
            if refused:
                unans_refused += 1
            else:
                details.append({"id": case["id"], "type": "should_refuse_but_answered", "gen": gen[:150]})
        else:
            ans_total += 1
            if not refused:
                ans_answered += 1
                kws = case.get("answer_keywords", [])
                need = min(2, len(kws)) if kws else 0
                if sum(1 for kw in kws if kw.lower() in gen.lower()) >= need:
                    ans_correct += 1
                else:
                    details.append({"id": case["id"], "type": "answered_but_low_keyword", "gen": gen[:150]})
            else:
                details.append({"id": case["id"], "type": "should_answer_but_refused", "gen": gen[:150]})
    return {
        "n": len(cases) - len(invalid),
        "invalid_n": len(invalid),
        "unanswerable_n": unans_total,
        "refusal_rate_on_unanswerable": round(unans_refused / unans_total, 3) if unans_total else None,
        "answerable_n": ans_total,
        "answer_rate_on_answerable": round(ans_answered / ans_total, 3) if ans_total else None,
        "correct_rate_on_answerable": round(ans_correct / ans_total, 3) if ans_total else None,
        "mistakes": details,
        "invalid": invalid,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_tool(task, kb_ctx, tool, args):
    args_str = json.dumps(args, ensure_ascii=False).lower() if not isinstance(args, str) else args.lower()
    if tool == "get_current_time":
        return time.strftime("%Y-%m-%d %H:%M:%S")
    if tool == "query_docs":
        chunks, mat = kb_ctx
        top = retrieve(chunks, mat, str(args), k=2)
        return " | ".join(c["text"][:200] for c in top)
    rules = task["tool_behavior"].get(tool)
    if not rules:
        return "该工具在本场景无数据。"
    for rule in rules:
        if all(kw.lower() in args_str for kw in rule["need"]):
            return rule["response"]
    return "无有效信息。"


def kw_hit(text, kw):
    """kw 为字符串或同义词列表（列表任一命中即算命中该组）。"""
    if isinstance(kw, list):
        return any(k.lower() in text for k in kw)
    return kw.lower() in text


def judge(conclusion, task):
    text = conclusion.lower()
    hits = sum(1 for kw in task["root_cause_keywords"] if kw_hit(text, kw))
    return hits >= task.get("min_hit", 1)


def attribution_hit(conclusion, task):
    """责任人/归属定位命中情况：只记录展示，不参与成败判定。"""
    kws = task.get("attribution_keywords", [])
    if not kws:
        return None
    text = conclusion.lower()
    return sum(1 for kw in kws if kw_hit(text, kw))


def solve_linear(task, kb_ctx):
    plan_prompt = (
        "你是运维排障规划器。给定告警，一次性生成完整排查计划。可用工具：" + TOOLS_DESC + "。\n"
        "重要：query_process_detail/query_monitor 等工具需要具体参数（进程名/PID/服务名/节点名），"
        "请在计划中给出尽可能具体的参数值。\n"
        "输出 JSON 数组，每个元素形如 {\"tool\":\"query_log\",\"args\":\"...\"}。只输出 JSON。\n"
        f"告警：{task['alert']}"
    )
    try:
        plan = parse_json(chat(plan_prompt))
        if isinstance(plan, dict):
            plan = next((v for v in plan.values() if isinstance(v, list)), [plan])
        if not isinstance(plan, list):
            raise ValueError("plan 非列表")
    except Exception:
        plan = [{"tool": "query_log", "args": task["alert"]}]
    obs = []
    for step in plan[:6]:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool", "")
        args = step.get("args", "")
        obs.append(f"{tool}({args}) -> {run_tool(task, kb_ctx, tool, args)}")
    concl_prompt = (
        "根据以下排查步骤与观测，给出故障根因结论（简明，一段话）。\n"
        f"告警：{task['alert']}\n观测：\n" + "\n".join(obs) + "\n结论："
    )
    conclusion = chat(concl_prompt)
    return conclusion, len(plan[:6])


def solve_replan(task, kb_ctx, max_rounds=6):
    history = []
    conclusion = ""
    for _ in range(max_rounds):
        hist_text = "\n".join(history) if history else "（暂无）"
        prompt = (
            "你是运维排障智能体，目标是定位告警根因。可用工具：" + TOOLS_DESC + "。\n"
            "重要：query_process_detail/query_monitor 等工具需要具体参数（进程名/PID/服务名/节点名），"
            "必须从前面的观测里提取真实值填入，不要凭空猜。\n"
            f"告警：{task['alert']}\n已执行步骤与观测：\n{hist_text}\n"
            "决定下一步：若已能定位根因，输出 {\"action\":\"finish\",\"conclusion\":\"...\"}；"
            "否则输出 {\"action\":\"call\",\"tool\":\"...\",\"args\":\"...\"}。只输出 JSON。"
        )
        try:
            act = parse_json(chat(prompt))
            if isinstance(act, list):
                act = next((x for x in act if isinstance(x, dict)), None)
            if not isinstance(act, dict):
                raise ValueError("act 非对象")
        except Exception:
            break
        if act.get("action") == "finish":
            conclusion = act.get("conclusion", "")
            break
        tool = act.get("tool", "")
        args = act.get("args", "")
        res = run_tool(task, kb_ctx, tool, args)
        history.append(f"{tool}({args}) -> {res}")
    if not conclusion:
        concl_prompt = (
            "根据以下观测给出故障根因结论（简明）。\n"
            f"告警：{task['alert']}\n观测：\n" + ("\n".join(history) or "无") + "\n结论："
        )
        conclusion = chat(concl_prompt)
    return conclusion, len(history)


def exp_replan():
    data = load_json("replan_tasks.json")
    tasks = data["tasks"]
    chunks = load_kb_chunks("heading")
    mat = embed([c["text"] for c in chunks])
    kb_ctx = (chunks, mat)

    def run_task(task):
        try:
            lc, lsteps = solve_linear(task, kb_ctx)
            rc, rsteps = solve_replan(task, kb_ctx)
        except Exception as e:
            print(f"  {task['id']}: INVALID ({str(e)[:60]})", flush=True)
            return {"id": task["id"], "difficulty": task.get("difficulty", "?"), "invalid": str(e)[:120]}
        lok, rok = judge(lc, task), judge(rc, task)
        print(f"  {task['id']}: linear={'OK' if lok else 'X'} replan={'OK' if rok else 'X'}", flush=True)
        return {
            "id": task["id"], "difficulty": task.get("difficulty", "?"),
            "linear_success": lok, "replan_success": rok,
            "linear_steps": lsteps, "replan_steps": rsteps,
            "linear_attribution": attribution_hit(lc, task), "replan_attribution": attribution_hit(rc, task),
            "linear_conclusion": lc[:180], "replan_conclusion": rc[:180],
        }

    with ThreadPoolExecutor(max_workers=4) as ex:
        details = list(ex.map(run_task, tasks))

    valid = [d for d in details if "invalid" not in d]
    invalid = [d for d in details if "invalid" in d]
    n = len(valid)
    linear_ok = sum(d["linear_success"] for d in valid)
    replan_ok = sum(d["replan_success"] for d in valid)
    by_diff = {}
    for d in valid:
        b = by_diff.setdefault(d["difficulty"], {"n": 0, "linear": 0, "replan": 0})
        b["n"] += 1
        b["linear"] += d["linear_success"]
        b["replan"] += d["replan_success"]
    return {
        "n": n,
        "invalid_n": len(invalid),
        "linear_success": linear_ok,
        "replan_success": replan_ok,
        "linear_rate": round(linear_ok / n, 3) if n else None,
        "replan_rate": round(replan_ok / n, 3) if n else None,
        "avg_replan_steps": round(sum(d["replan_steps"] for d in valid) / n, 2) if n else None,
        "by_difficulty": by_diff,
        "details": details,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def smoke():
    v = embed(["连通性测试"])
    print(f"embedding OK：{EMBED_MODEL}，维度 {v.shape[1]}")
    t0 = time.time()
    ans = chat("只回复两个字：正常")
    print(f"chat OK：{CHAT_MODEL} -> {ans[:50]}（耗时 {time.time()-t0:.1f}s）")


def dry():
    print("== DRY 模式：数据加载与规模校验（不调用 API）==\n")
    for strategy in ["whole", "heading"]:
        chunks = load_kb_chunks(strategy)
        print(f"分片策略 {strategy:8s}: {len(chunks)} 个片段")
    gold = load_json("gold_qa.json")["items"]
    ond = load_json("ondemand_cases.json")["items"]
    rep = load_json("replan_tasks.json")["tasks"]
    print(f"\n金标准问答:   {len(gold)} 条")
    print(f"按需检索标注: {len(ond)} 条（need {sum(c['should_retrieve'] for c in ond)} / skip {sum(not c['should_retrieve'] for c in ond)}）")
    try:
        ref = load_json("refusal_cases.json")["items"]
        una = sum(1 for c in ref if c.get("unanswerable"))
        print(f"拒答评测集:   {len(ref)} 条（无答案 {una} / 可答 {len(ref) - una}）")
    except FileNotFoundError:
        print("拒答评测集:   （未创建）")
    diffs = {}
    for t in rep:
        diffs[t.get("difficulty", "?")] = diffs.get(t.get("difficulty", "?"), 0) + 1
    print(f"排障任务:     {len(rep)} 个（难度分布 {diffs}）")
    print(f"知识库文档:   {len(list(KB_DIR.glob('*.md')))} 篇")
    print("\n数据结构 OK。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="只测两个 API 端点连通性")
    ap.add_argument("--only", choices=["retrieval", "ondemand", "refusal", "replan"])
    args = ap.parse_args()
    if args.dry:
        dry()
        return
    init_clients()
    if args.smoke:
        smoke()
        return
    results = {}
    t0 = time.time()
    if args.only in (None, "retrieval"):
        print("跑实验 A：检索命中率 ...", flush=True)
        results["retrieval"] = exp_retrieval()
    if args.only in (None, "ondemand"):
        print("跑实验 B：按需 vs 固定检索路由 ...", flush=True)
        results["ondemand"] = exp_ondemand()
    if args.only in (None, "refusal"):
        print("跑实验 C：有据作答 / 无据拒答 ...", flush=True)
        results["refusal"] = exp_refusal()
    if args.only in (None, "replan"):
        print("跑实验 D：线性 vs 动态重规划 ...", flush=True)
        results["replan"] = exp_replan()
    out = ROOT / "results.json"
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(results)
    existing["_meta"] = {
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "reasoning_effort": None if _NO_EFFORT else REASONING_EFFORT,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print("\n== 结果 ==")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n已写入 results.json")


if __name__ == "__main__":
    main()
