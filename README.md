# Vector Storage (第三方魔改版)

基于 [SillyTavern 官方 Vector Storage 扩展](https://github.com/SillyTavern/Extension-VectorStorage) 魔改的第三方版本，新增 **Reranker 精排管线** 与 **RAG 诊断系统**。

> ⚠️ 这是个人用途的第三方修改版，跟随官方上游结构，可能存在未知问题。纯插件端实现，**不修改 SillyTavern 主程序**，可正常 git pull 更新 ST。

---

## ✨ 相比官方版的新增功能

### 1. Reranker 精排（二审）

经典 RAG 的"宽进严出"策略：Embedding 海选负责召回率，Cross-Encoder 重排负责精准度。

```
查询文本
   │
   ▼
① 海选：向量检索 topK = 海选数(默认30)，阈值强制 0（可选）
   │
   ▼
② 精排：POST {rerank_url}/v1/rerank （本地 llama.cpp / 云端均可，CPU 友好）
   │
   ▼
③ 截断：按各路径原有数量设置取前 N 条注入上下文
```

开启后，原有的数量设置含义变为「最终保留数」：

| 检索路径 | 海选数量 | 最终保留数 |
|---|---|---|
| 聊天消息 | 全局海选数 | `Insert#` |
| 文件附件 | 全局海选数 | `Retrieve chunks` |
| Data Bank | 全局海选数 | `Retrieve chunks` |
| World Info | 全局海选数 | `Max Entries` |

**配置方法**（以本地 llama.cpp 为例）：

```bash
# 下载 GGUF 格式 reranker 模型后启动服务（CPU 即可运行）
llama-server -m bge-reranker-v2-m3-Q8_0.gguf --reranking --port 8080
```

推荐模型：
- `bge-reranker-v2-m3`（~600MB，多语言，社区口碑最好）
- `Qwen3-Reranker-0.6B`（与 Qwen3 系列同门）

设置面板中填入 Endpoint URL（如 `http://127.0.0.1:8080`），点 **Test Reranker** 验证连通性。任何兼容 `/v1/rerank` 格式的后端均可使用（llama.cpp / SiliconFlow / Jina / TEI 等）。

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| Recall count | 30 | 大海选数量，越大召回越保险，CPU 耗时略增 |
| Ignore score threshold | ✓ | 海选阶段无视向量阈值，让 reranker 决定去留 |
| Query truncation | 2000 字符 | 送入 reranker 前截断查询文本 |
| API key | 空 | 可选 Bearer token |

**降级保护**：rerank 服务不可用时自动回退到原始向量排序，仅提示一次，不阻塞生成。

### 2. 诊断系统（专治静默失败）

RAG 最坑的就是"坏了但你不知道"。本版内置五层可观测性：

- **状态徽章**：面板实时显示上次拦截结果（✓ 注入 N 条 / ✗ 失败）
- **Run Diagnostics 一键自检**：依次检测 向量查询连通性 → reranker 连通性 → 分块器完整性（含 emoji 断裂专项测试）
- **Last Recall 召回报告**：展示最近一次检索的完整决策表——每条候选的排名、rerank 分数、最终去向（INSERTED / cut），直接回答"为什么召回的是它"
- **阈值差分调试**：勾选后每次检索额外发一次 threshold=0 查询，列出"被 score threshold 杀掉的候选"，调参不再靠猜
- **诊断日志**：200 条环形缓冲，跨会话持久化，支持按级别过滤、导出 JSONL

**入库完整性校验（安全网）**：vectorize 时自动检测孤立代理对（emoji 被从中间切断）、空块、碎块、重复内容并告警——只告警不改动数据，清洗仍建议用上游工具（如 rag_toolkits）。插入完成后自动回读比对数量，抓"静默丢数据"。

### 3. 修复的 Bug

- **聊天路径 score_threshold 失效**：ST 服务端 `/api/vector/query` 返回的 `hashes` 未按阈值过滤（只有 `metadata` 过滤了），而插件恰好只读 hashes——导致聊天检索的分数阈值从未生效过，低于阈值的结果照样占满名额。现已改从 metadata 取 hash，阈值真正生效。
  - 注意：此修复会让阈值行为变严格。若之前依赖"阈值形同虚设"的效果，请把 Score threshold 调低。

---

## 📦 安装

在 SillyTavern 的扩展管理中通过 git URL 安装本仓库即可。

> **目录结构兼容性说明**：本插件的模块导入路径已针对魔改版 SillyTavern 目录布局做适配（`script.js` 位于 public 根目录、核心库直接挂载在 `/scripts/` 下、第三方扩展无路径回退）。若在官方原版 ST 上使用，需将 `index.js` 顶部的 import 深度改回官方标准（少一级）。

## 🔧 开发

单文件结构（`index.js`），与官方扩展保持一致，方便 diff 上游更新：

```
index.js        # 全部逻辑
settings.html   # 设置面板 UI
style.css       # 样式（召回消息标记 + 诊断面板）
webllm.js       # WebLLM 支持（继承自官方）
manifest.json   # 扩展清单
```
