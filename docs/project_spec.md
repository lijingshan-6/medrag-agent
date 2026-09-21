> Architecture-oriented development document. The historical metrics in section 1.3 are superseded. For v0.2.0 behavior and measured results, use the repository README and `docs/benchmark-v1.1-report.md`.

# VeritasMed — 自校验医学文献智能问答系统

> 基于 LangGraph 的主动式 RAG 系统，集成多级检索、自校验生成与安全 MCP 接口

---

## 1. 项目概述

### 1.1 背景

大型语言模型在医学问答场景中面临两个核心挑战：**检索质量不稳定**（专业术语匹配困难、多跳推理覆盖不足）和**生成幻觉**（模型倾向于用参数化知识填补上下文空缺，产生无文献支撑的声明）。现有 RAG 系统多采用"检索-生成"的线性静态管道，无法自主应对这两类失败模式。

### 1.2 目标

构建一个面向生物医学文献（PubMed / PMC）的问答系统，其核心特征为：

- **自主检索修正**：当检索到的文档无法支撑问题时，系统自动重写查询并重试
- **可验证输出**：每条答案经由独立的忠实度审计节点校验，不通过则重新生成（含智能门控防止误拦截）
- **安全集成**：通过 MCP（Model Context Protocol）接口向 Claude Desktop / Claude Code 暴露工具，具备完整的安全防护层

### 1.3 核心指标

**标准集（50题 Golden Dataset）**

| 指标 | P2 Hybrid | P3 Hybrid+Reranker | P4-Agentic |
|------|-----------|-------------------|------------|
| Recall@5 | **100.0%** | **100.0%** | — |
| MRR@20 | **1.000** | **1.000** | — |
| Faithfulness | — | 0.405 | 0.401 |
| Relevance | — | 0.996 | **1.000** |
| Correctness | — | 0.916 | **0.920** |
| Composite | — | 0.772 | **0.774** |

**Hard Set（39题，P4-Agentic v4 smart-gate 版本）**

| Composite | Faithfulness | Relevance | Correctness |
|-----------|-------------|-----------|-------------|
| **0.818** | 0.951 | 0.819 | 0.683 |

详细评估结果见 [`docs/evaluation_report.md`](evaluation_report.md)。

---

## 2. 系统架构

### 2.1 整体拓扑

```
┌──────────────────────────────────────────────────────────────────┐
│                    Claude Desktop / Claude Code                  │
└─────────────────────────────┬────────────────────────────────────┘
                              │  MCP (stdio)
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  VeritasMed MCP Server (FastMCP 2.x)                            │
│                                                                  │
│  安全中间件 (5层):                                                │
│  auth → rate_limit → injection_guard → pii_redaction → audit    │
│                                                                  │
│  工具层:                                                          │
│  ask_agent · search_literature · evaluate_query · search_visual  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  LangGraph 主动式推理环路                                          │
│                                                                  │
│  START → route → retrieve → rerank → grade ──────► generate     │
│                    ▲           │              │         │        │
│                    │    (相关,分≥阈值)   (不相关,          │        │
│                    │           │        iter<1)         ▼        │
│                    └────── rewrite ◄───────────      check       │
│                                                        │         │
│                              (忠实 OR 智能门控) ──► END            │
│                                    (不忠实,regen<1) → inc_regen  │
│                                                        │         │
│                                                   → generate     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  检索层                                                           │
│  BGE-M3 dense (1024-d) → Qdrant → RRF 融合 (dense-only)        │
│  BGE-Reranker-v2-m3 交叉编码器重排序                              │
│  语料: ~44,768 块 (PubMed abstracts + PMC 全文)                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 LangGraph 节点职责

| 节点名（图中） | Python 函数 | LLM 模式 | 职责 |
|-------------|------------|----------|------|
| `route` | `route_query` | llm_fast (thinking=OFF) | 查询分类：factual / synthesis / multihop |
| `retrieve` | `hybrid_retrieve` | — | BGE-M3 dense 混合检索，候选集 top-20 |
| `rerank` | `rerank_chunks` | — | BGE-Reranker 交叉编码器，压缩至 top-5 |
| `grade` | `grade_relevance` | llm_think (direct output) | 相关性评分 0–1，生成重写提示 |
| `rewrite` | `rewrite_query` | llm_think (direct output) | 查询重写（MeSH 扩展 / 子问题分解） |
| `generate` | `generate_answer_node` | llm_fast (thinking=OFF) | 结构化 JSON 答案生成，内联引用 |
| `check` | `check_faithfulness` | llm_think (direct output) | 逐项忠实度审计，标记幻觉声明 |
| `inc_regen` | `increment_regen` | — | 重生成计数器自增（防无限循环） |
| `append_history` | `append_history` | — | 将已完成 Q&A 追加到 state["history"] |
| `summarize_gate` | lambda passthrough | — | 判断是否需要压缩历史 |
| `summarize` | `summarize_history` | llm_fast (thinking=OFF) | L2 滚动记忆压缩，≤200词摘要 |

---

## 3. 关键设计决策

### 3.1 双 LLM 策略

LLM 后端通过 `LLM_BACKEND` 环境变量选择（默认 `mimo`）：

```
llm_fast  — MiMo-V2.5 API (thinking=OFF, temp=0.2, max_tokens=4096)
           Ollama 回退: qwen3.5:9b (thinking=OFF, num_ctx=4096)
           用于: route, generate, summarize
           目标: 低延迟、确定性输出

llm_think — MiMo-V2.5-Pro API (thinking=OFF, temp=0.6, max_tokens=4096)
           Ollama 回退: qwen3.5:9b (reasoning=OFF, num_ctx=6144)
           用于: grade, rewrite, check
           目标: 深度推理，更高忠实度审计质量
```

**设计理由**：grade / rewrite / check 使用更大的上下文和独立提示，但要求模型直接返回短 JSON。实测 `qwen3.5:9b` 的显式 reasoning 会耗尽请求时限或截断 JSON，因此本地与 MiMo 路径都关闭隐藏推理；generate 节点同样直接输出，保证可解析性和延迟边界。

### 3.2 混合检索与 RRF 融合

P2 Hybrid 管道在 50 题评测集上达到 R@5=100%、MRR@20=1.000，优于纯稠密检索（R@5=98%）。

```
查询 → BGE-M3 encode (M3Embedder, 1024-d dense + sparse lexical weights)
      ├── Qdrant dense 查询  → top-20 by cosine similarity
      └── Qdrant sparse 查询 → top-20 by sparse dot product
            ↓
      RRF 融合 (k=60)：score(d) = Σ 1/(k + rank_i(d))
            ↓
      top-20 候选集 → BGE-Reranker 交叉编码器 → top-5
```

> **实现说明**：`embedder.py` 使用 `FlagEmbedding.inference.embedder.encoder_only.m3.M3Embedder`（直接导入子模块，绕过会导致 Windows 崩溃的 `FlagEmbedding.__init__`）。该类能同时产生 dense 向量和 sparse 词汇权重，hybrid 检索双路均激活。索引文件（`data/index_cache/sparse.jsonl`，49.7 MB）已包含真实 sparse 向量。

### 3.3 两级记忆架构

```
L1 — LangGraph SqliteSaver (data/checkpoints/agent.db)
     粒度: 每个 LangGraph step 的完整状态快照
     用途: 崩溃恢复、多轮会话连续性
     键:   thread_id（每个用户会话唯一）

L2 — 滚动摘要（每 10 轮触发，HISTORY_SUMMARIZE_EVERY=10）
     LLM: llm_fast
     输出: state["summary"]，≤200词，保留医学关键信息
     用途: 防止长会话超出上下文窗口
```

### 3.4 自校验终止条件

```
重写上限: MAX_REWRITES = 1（最多 2 次检索尝试）
重生成上限: MAX_REGEN = 1（最多 2 次生成尝试）

动态相关性阈值 (_GRADE_THRESHOLDS):
  factual:   0.5（简单题不需要积极 rewrite）
  synthesis: 0.6（标准阈值）
  multihop:  0.7（多跳题更激进 rewrite）

智能 regen 门控 (REGEN_CONFIDENCE_SKIP=0.3):
  若首次生成答案有 citations 且 confidence ≥ 0.3，
  即使忠实度检查报告 unfaithful，也跳过 regen。
  原因: faithfulness checker 对细微医学答案存在高假阳性率。
```

硬上限通过独立的 `inc_regen` 节点（而非边函数）递增计数器，确保状态更新被 SqliteSaver 持久化，避免无限循环。

---

## 4. 安全设计

### 4.1 威胁模型（本地部署场景）

| 威胁 | 影响 | 缓解措施 |
|------|------|----------|
| 提示注入（用户查询中嵌入指令） | 高（可能产生虚假"权威"医学建议） | injection_guard: 11条正则 + XML边界标签 |
| 速率滥用（自动化高频调用） | 中（耗尽本地 GPU/CPU 资源） | rate_limit: 令牌桶 30 rpm 全局 / 10 rpm 生成 |
| 未授权访问（同机其他进程） | 中（数据暴露） | auth: HMAC 恒时比较，本地 token |
| PII 泄露到审计日志 | 高（GDPR / HIPAA） | pii: 正则脱敏，日志仅存 SHA-256(query)[:16] |
| 语料投毒（检索文档中的恶意指令） | 中 | XML 边界标签隔离 + 系统提示显式声明 |

详细说明见 [`docs/mcp_security.md`](mcp_security.md)。

---

## 5. 评估方法论

### 5.1 Golden Dataset

- **标准集**：50题，人工构建并由 Phase 3 验证（每题关联到语料库中的精确源块）
- **分布**：Cardiology(7) · Neurology(9) · Radiology(19) · Oncology(7) · General(4) · Infectious Disease(2) · Pharmacology(2)
- **难度**：Easy(13) · Medium(37)
- **锁定**：SHA-256 哈希文件 `data/eval/golden_dataset.sha256` 防止意外修改

- **Hard Set**：39题（A-多跳×4 · B-术语歧义×10 · C-否定反事实×10 · D-跨域合成×15）
- **Hard Set SHA-256**：`33ee0351...f7b6092`

### 5.2 评估指标

**检索评估**（scripts/08_eval_retrieval.py）
- **Recall@K**：源块是否出现在 top-K 结果中
- **MRR@20**：Mean Reciprocal Rank

**答案质量评估**（scripts/09、11_eval_agent.py）：

| 维度 | 定义 |
|------|------|
| Faithfulness | 答案中每项声明是否有检索文档支撑 |
| Relevance | 答案是否直接回应问题 |
| Correctness | 答案与金标准答案的信息重叠程度 |

评判 LLM：`mimo-v2.5-pro`（通过独立 API 端点，`JUDGE_BASE_URL` / `JUDGE_API_KEY` 配置）

### 5.3 主要结论

**检索**：混合检索（P2）在所有难度和类别上均达到 R@5=100%，纯稠密检索（P1）在 Cardiology 类别存在 14.3% 的缺失，验证了精确术语匹配对医学检索的重要性。

**Agentic vs Static（标准集）**：P4-Agentic 在标准集上 grade→rewrite 循环触发率为 0%，Agentic 增益主要来自忠实度过滤（+0.004 correctness）。

**Hard Set v4**：智能 regen 门控将 Composite 从 P4-v1 的 0.685 提升至 0.818，超越 P3 基线 0.748（+0.070）。关键优化：智能门控（confidence + citations 门控 regen）、max_tokens=4096 防截断、`_invoke_with_retry()` 空响应保护。

---

## 6. 技术栈

| 层次 | 组件 | 版本/规格 |
|------|------|----------|
| LLM（默认） | MiMo-V2.5 / MiMo-V2.5-Pro | OpenAI-compatible API |
| LLM（备选） | Qwen3-8B via Ollama | `LLM_BACKEND=ollama` |
| 嵌入模型 | BAAI/bge-m3 (sentence_transformers) | dense 1024-d，CPU/CUDA auto |
| 重排序模型 | BAAI/bge-reranker-v2-m3 (CrossEncoder) | CUDA fp16 / CPU float32 auto |
| 向量数据库 | Qdrant | Docker，localhost:6333 |
| 主动推理框架 | LangGraph 0.2 | StateGraph + SqliteSaver |
| API 后端 | FastAPI | REST + WebSocket |
| MCP 服务器 | FastMCP 2.x | stdio transport |
| 语料来源 | PubMed abstracts + PMC OA full-text | 44,768 chunks (评估时) |
| 评判模型 | MiMo-V2.5-Pro | 独立 API 端点 |
| 运行环境 | Python 3.12, conda, Windows + RTX 4060 | |

---

## 7. 项目结构

```
src/medrag/
├── agent/
│   ├── graph.py          # LangGraph StateGraph + SqliteSaver（图组装）
│   ├── nodes.py          # 11 个节点函数
│   ├── state.py          # AgentState TypedDict
│   ├── prompts.py        # 全部 LLM 提示模板
│   ├── llms.py           # 双 LLM 工厂（fast / think，mimo/ollama 后端）
│   ├── utils.py          # strip_thinking() 等工具函数
│   └── generator.py      # Week 1 基线：单次无 Agent 循环的生成器
├── api/
│   ├── app.py            # FastAPI 入口（CORS、import 顺序）
│   ├── models.py         # Pydantic 模型（REST + WebSocket 共用）
│   ├── _helpers.py       # 共享工具函数
│   └── routes/           # ask, search, document, chunk, history, corpus
├── index/
│   ├── embedder.py       # BGEM3Embedder（dense 1024-d）
│   ├── indexer.py        # Qdrant 批量 upsert
│   └── qdrant_setup.py   # collection 初始化
├── ingest/
│   ├── pubmed.py         # PubMed 抓取与解析
│   ├── pmc.py            # PMC OA 全文抓取
│   └── chunker.py        # 文本分块（滑动窗口，64 token overlap）
├── retrieval/
│   ├── retriever.py      # DenseRetriever + RetrievedChunk
│   ├── hybrid.py         # HybridRetriever（RRF 融合，dense-only）
│   ├── reranker.py       # BGEReranker（CrossEncoder）
│   ├── hyde.py           # HyDERetriever（假设文档嵌入）
│   └── multi_query.py    # MultiQueryRetriever（多查询扩展）
└── mcp_server/
    ├── server.py         # FastMCP 服务器 + 4 个工具
    └── security/
        ├── auth.py           # HMAC token 认证
        ├── rate_limit.py     # 令牌桶限流器
        ├── audit.py          # JSON-Lines 审计日志
        ├── pii.py            # PII 脱敏
        └── injection_guard.py # 注入检测 + XML 隔离

data/
├── golden/               # 50 题 golden dataset + SHA-256 锁定
├── eval/                 # 检索评估 + 答案评估 + 报告
└── checkpoints/          # LangGraph SqliteSaver（运行时生成）

scripts/                  # 编号流水线脚本（01–14）
frontend/                 # React + TypeScript 前端（独立服务）

docs/
├── project_spec.md       # 本文档：项目规格与设计说明
├── architecture.md       # 系统架构详解与节点参考
├── frontend_design.md    # 前端设计与 API 参考
├── mcp_security.md       # 威胁模型与 5 层安全中间件
├── evaluation_report.md  # 完整评估报告（含 v4 结果）
├── hard_set_report.md    # Hard Set 构造说明与 Stage 1 结果
└── security_test_report.md # MCP 安全单元测试报告
```

---

## 8. 局限性与未来方向

### 当前局限

- **忠实度上限约 0.40（标准集）**：MiMo / Qwen 模型在参数化知识丰富的医学域有"补充"倾向，需要更强的引用约束策略
- **重排序延迟**：BGE-Reranker 在 CPU 上处理 20 对需要约 15s，是端到端延迟的主要瓶颈；GPU（CUDA）可降至 ~0.2s
- **单节点 Qdrant**：无水平扩展，适合原型验证，生产需集群部署
- **视觉检索 stub**：`search_visual` 工具尚未实现，PMC 图表索引待建
- **Dense+Sparse RRF 已启用**：`BGEM3Embedder` 改用 `M3Embedder`（FlagEmbedding 子模块），能产生真实 sparse 权重

### 未来方向

- **引用感知生成**：在生成提示中强制要求每句话都标注来源块 ID
- **GPU 重排序**：将 BGE-Reranker 迁移到 GPU（设置 `RERANKER_DEVICE=cuda`）
- **稀疏检索恢复**：切换回支持 sparse 的 FlagEmbedding（解决 Windows 兼容性后），启用真正的 dense+sparse RRF
- **多模态检索**：构建 PMC 图表向量索引，支持放射学图像和数据表格检索
- **流式 MCP 响应**：利用 FastMCP SSE 支持，在 generate 节点完成时立即推送中间结果
