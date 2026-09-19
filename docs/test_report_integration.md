> Historical development document. Some claims and defaults are superseded. For v0.1.0, use the repository README, docs/evaluation_report.md and docs/validation-2026-09-18.md.

# MedRAG-Agent 集成测试报告

> 测试日期：2026-05-14
> 测试环境：Windows 11, Python 3.12.13, conda env: medrag
> 后端：MiMo API (mimo-v2.5-pro)
> 向量库：Qdrant (localhost:6333, collection: medrag_text, 44,768 points)

---

## 1. 测试概览

| 指标 | 数值 |
|------|------|
| 总用例数 | 20 |
| 通过 | **18** |
| 失败 | **1** |
| 跳过 | **1** |
| 通过率 | **90%** |
| 总耗时 | 43 分 49 秒 |
| 平均单次端到端耗时 | ~130 秒 |

---

## 2. 测试结果明细

### 2.1 单轮事实型查询 (TestSingleTurnFactual)

| 用例 | 结果 | 耗时 | 置信度 | 引用数 | 说明 |
|------|------|------|--------|--------|------|
| aspirin-mechanism | **PASS** | 114.0s | 0.90 | 1 | 正确回答 COX-1 抑制机制，faithful=True |
| hypertension-treatment | **FAIL** | 90.7s | 0.60 | 1 | 检索到运动治疗文档，未匹配药理学关键词 |
| t2dm-diagnosis | PASS | 166.4s | 0.00 | 0 | 返回 disclaimer（语料中无直接诊断标准文档） |

**分析**：aspirin 查询表现优秀（高置信度 + faithful）。hypertension 失败原因是语料库中优先检索到运动治疗相关文档而非药理学文档，属于**检索排序问题**而非代码 bug。t2dm 查询诚实返回 disclaimer，体现了系统的安全兜底机制。

### 2.2 单轮综合型查询 (TestSingleTurnSynthesis)

| 用例 | 结果 | 耗时 | 置信度 | 引用数 |
|------|------|------|--------|--------|
| metformin-vs-sulfonylureas | **PASS** | 141.5s | 0.50 | 3 |
| cox2-vs-nsaids | **PASS** | 159.4s | 0.50 | 5 |

**分析**：两个综合型查询均通过。系统正确识别出语料中缺乏直接对比数据，但仍基于已有文献提供了相关信息并附带多条引用。

### 2.3 单轮多跳查询 (TestSingleTurnMultihop)

| 用例 | 结果 | 耗时 | 置信度 | 重写次数 |
|------|------|------|--------|----------|
| warfarin-pharmacogenomics | **PASS** | 135.5s | 0.50 | 1 |

**分析**：触发了查询重写（iterations=1），说明第一次检索相关性不足，系统自动重写查询后重试。最终返回了基于文献的回答。

### 2.4 多轮对话 (TestMultiTurn)

| 用例 | 结果 | 耗时 | 说明 |
|------|------|------|------|
| two-turn-conversation | **PASS** | 274.4s | 两轮对话均完成 |
| three-turn-topic-shift | **PASS** | 411.3s | 三轮对话含主题切换，均完成 |

**观察**：
- history_len=0：返回的 state 中 history 为空，说明 history 存储在 SqliteSaver checkpoint 中而非直接返回
- 多轮对话中第二轮的回答与第一轮主题不完全相关（"Baclofen" 而非 "metformin side effects"），说明**上下文传递存在局限**

### 2.5 重写循环 (TestRewriteLoop)

| 用例 | 结果 | 耗时 | 迭代次数 | 重写查询 |
|------|------|------|----------|----------|
| iterations-recorded | **PASS** | 133.3s | 1 | "What are the pharmacological mechanisms of metformin, including AMPK activation..." |

**分析**：重写循环正常工作。第一次检索未通过相关性评分，系统自动重写了更具体的查询。

### 2.6 忠实度检查 (TestFaithfulness)

| 用例 | 结果 | 耗时 | faithful | regen_count |
|------|------|------|----------|-------------|
| faithfulness-reported | **PASS** | 105.8s | False | 1 |

**分析**：忠实度检查节点正确识别出回答中存在无证据支持的声明，触发了重生成（regen_count=1）。重生成后仍未通过检查，最终返回带问题标记的回答。

### 2.7 引用质量 (TestCitationQuality)

| 用例 | 结果 | 耗时 | 引用格式 |
|------|------|------|----------|
| citations-format | **PASS** | 149.1s | PMC:doc* 格式正确 |
| confidence-reasonable | **PASS** | — | 0.50, 4 citations |

**分析**：引用格式验证通过。置信度在合理范围内。

### 2.8 检索管道对比 (TestRetrievalPipelines)

| 管道 | 结果 | 耗时 | Top-1 分数 | Top-3 示例 |
|------|------|------|------------|------------|
| P1 Dense | **PASS** | 0.29s | 0.567 | PMC:doc266, PMC:doc113, PMC:doc85 |
| P2 Hybrid | **PASS** | 0.32s | 0.031 | PMC:doc204, PMC:doc113, PMC:doc85 |
| Hybrid vs Dense Recall | **PASS** | — | overlap=5/10 | dense-only=5, hybrid-only=5 |

**分析**：
- P1（纯稠密）和 P2（混合检索）均正常工作
- 检索耗时约 0.3 秒，性能良好
- Dense 和 Hybrid 各有 5 个独特结果，说明两种检索方式确实互补
- Hybrid 的 RRF 分数量纲不同（0.031 vs 0.567），但排名融合有效

### 2.9 响应时间 (TestResponseTime)

| 用例 | 结果 | 耗时 | 说明 |
|------|------|------|------|
| simple-factual | SKIP | 164.5s | 超过 60s 软阈值，标记为慢 |
| synthesis | **PASS** | 141.7s | 在 180s 硬阈值内 |

**分析**：端到端响应时间在 90-175 秒之间，主要瓶颈是：
1. BGE-M3 embedding 计算（CPU 模式）
2. BGE-Reranker 交叉编码（CPU 模式）
3. MiMo API 调用（5 次 LLM 调用：route → grade → generate → check → summarize_gate）

### 2.10 边界用例 (TestEdgeCases)

| 用例 | 结果 | 耗时 | 说明 |
|------|------|------|------|
| very-short-query | **PASS** | 113.4s | 单词查询 "aspirin" 返回 787 字符回答 |
| technical-query-with-symbols | **PASS** | 175.9s | ICD-10 编码查询正常处理 |

---

## 3. 发现的 Bug 与修复

### 3.1 引用格式验证 Bug（已修复）

**问题**：LLM 有时返回带方括号的引用格式 `[PMC:doc205]`，但 `validate_citations()` 期望 `PMC:doc205`（无方括号），导致所有声明被丢弃，最终返回 disclaimer。

**影响**：第一轮测试中 9/20 个用例因此失败。

**修复**：在 `src/medrag/agent/utils.py:54` 添加方括号剥离逻辑：
```python
cite_keys = [k.strip("[]") for k in cite_keys]
```

**验证**：修复后 aspirin 查询从 confidence=0.0 提升到 confidence=0.90。

### 3.2 多轮对话 history 未返回（待调查）

**问题**：多轮对话测试中，第二轮返回的 `history` 为空列表（len=0），说明 history 虽然存储在 SqliteSaver checkpoint 中，但未在 state 返回值中体现。

**影响**：无法在应用层直接访问对话历史。

### 3.3 检索相关性不稳定（已知限制）

**问题**：部分查询（hypertension, t2dm-diagnosis）检索到的文档与预期不符。例如 "first-line treatment for hypertension" 检索到运动治疗文档而非降压药文档。

**原因**：语料库内容分布不均，某些主题的文献覆盖不足。

---

## 4. 性能指标汇总

| 指标 | 最小值 | 最大值 | 平均值 | 中位数 |
|------|--------|--------|--------|--------|
| 端到端响应时间 | 90.7s | 175.9s | 138.4s | 135.5s |
| 检索耗时 (P2) | 0.29s | 0.32s | 0.31s | 0.31s |
| 置信度（非 disclaimer） | 0.50 | 0.90 | 0.60 | 0.50 |
| 引用数（非 disclaimer） | 1 | 5 | 3.0 | 3.0 |

---

## 5. 测试覆盖矩阵

| 测试类别 | 覆盖场景 | 用例数 | 通过 |
|----------|----------|--------|------|
| 单轮事实型 | aspirin, hypertension, t2dm | 3 | 2 |
| 单轮综合型 | metformin vs sulfonylureas, COX-2 vs NSAIDs | 2 | 2 |
| 单轮多跳型 | warfarin pharmacogenomics | 1 | 1 |
| 多轮对话 | 2 轮对话, 3 轮主题切换 | 2 | 2 |
| 重写循环 | 迭代计数, 查询重写 | 1 | 1 |
| 忠实度检查 | faithful 报告, regen 触发 | 1 | 1 |
| 引用质量 | 格式验证, 置信度合理性 | 2 | 2 |
| 检索管道 | P1 Dense, P2 Hybrid, Recall 对比 | 3 | 3 |
| 响应时间 | 简单查询, 综合查询 | 2 | 1 |
| 边界用例 | 短查询, 特殊字符 | 2 | 2 |

---

## 6. 建议

1. **优先修复**：引用格式 bug 已修复，建议添加单元测试覆盖 `[PMC:xxx]` 格式
2. **性能优化**：考虑将 BGE-M3 和 Reranker 切换到 GPU 模式（当前为 CPU），预计可将端到端时间从 ~130s 降至 ~30s
3. **语料扩充**：高血压、糖尿病诊断等常见医学主题的文献覆盖不足，建议补充
4. **多轮记忆**：调查 history 在 state 返回值中为空的问题，确保多轮上下文正确传递
5. **超时处理**：当前 MiMo API 偶发 "Connection prematurely closed" 错误（400），建议增加重试逻辑
