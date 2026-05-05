# VideoGenDoctor修订稿蓝图


---

## 1. 推荐重新定位

当前稿件最大的问题不是系统无效，而是：

1. 主张过强，证据不足
2. repair 评估自我闭环
3. 缺少外部强 baseline
4. benchmark 规模太小，taxonomy 覆盖不足

因此建议把论文从：

> “一个已经可靠解决 controllable video generation failure diagnosis and repair 的系统”

改成：

> “一个 evidence-grounded diagnostic interface + actionable repair protocol + controlled benchmark/fixture，并辅以更强的外部对照与独立修复验证”

更激进、也更适合 ICML 的版本则是：

> “首个将 controllable video generation 的 failure diagnosis、localized evidence、repair planning 与 repair verification 统一到同一可执行协议中的 benchmark+system paper，并通过强 VLM baseline 与独立人类评估证明结构化接口相对端到端 VLM agent 的优势”

如果你真的要按 reviewer 要求“生成一篇更强版本”，建议采用下面这个版本的标题与叙事。

---

## 2. 建议标题

### 版本 A：系统 + benchmark 并重

`VideoGenDoctor: Evidence-Grounded Failure Diagnosis, Localized Repair Planning, and Human-Validated Closed-Loop Evaluation for Controllable Video Generation`

### 版本 B：更偏 benchmark / protocol

`VideoGenDoctor-Bench: A Benchmark and Structured Interface for Failure Diagnosis, Evidence Localization, and Repair Planning in Controllable Video Generation`

### 版本 C：更偏调试接口

`From Scores to Debuggable Reports: Failure-as-Code Diagnosis and Closed-Loop Repair Evaluation for Controllable Video Generation`

---

## 3. 假设性摘要（参考写法）


### Abstract Draft

Controllable video generation systems increasingly expose structured conditioning signals such as character identities, camera trajectories, required props, and multi-shot specifications. Yet current evaluation pipelines remain largely score-based: they return scalar judgments of quality, coherence, or text alignment, but do not identify which failure occurred, where it occurred, or how it should be corrected. This gap limits the usefulness of evaluation in real generation workflows, where creators need localized, auditable, and actionable feedback rather than global scores.

We present **VideoGenDoctor**, a structured framework for controllable-video debugging that transforms generated videos into **failure-as-code diagnosis records**, grounds each record in **localized temporal evidence**, compiles diagnoses into **repair plans**, and evaluates repair success in a **closed loop**. Each diagnosis record contains a failure code, temporal span, affected target, confidence score, and evidence keyframes, enabling both machine-readability and human auditability. To support systematic evaluation, we introduce **VideoGenDoctor-Bench-v1**, a benchmark that combines controlled perturbations with real generator failures across multiple source videos and generators, together with human-verified failure annotations and blind repair judgments.

We evaluate VideoGenDoctor along four axes: failure-code detection, evidence localization, repair-plan quality, and human-validated closed-loop repair success. Compared with strong end-to-end VLM baselines that directly produce structured reports, VideoGenDoctor achieves higher code-level F1, more precise temporal localization, and stronger repair usefulness under blind human evaluation, while remaining more controllable and auditable than free-form agent-style outputs. In particular, VideoGenDoctor improves macro-F1 from `[baseline]` to `[ours]`, tIoU@0.5 from `[baseline]` to `[ours]`, and human Pass@2 from `[baseline]` to `[ours]`. These results suggest that structured diagnosis interfaces can provide a practical alternative to score-only evaluation and a more reliable substrate for controllable video repair.

### 摘要关键点

这版摘要有几个刻意强化的点：

1. 不再只说“修复成功率”，而是加了 `human-validated closed-loop repair success`
2. 不再只和内部 ablation 比，而是明确写 `strong end-to-end VLM baselines`
3. benchmark 不再是 72-video 小夹具，而是 `controlled perturbations + real generator failures + multiple generators`
4. 把贡献重点从“性能强”转成“structured interface 更 controllable / auditable”

---

## 4. 引言重写方向

### 4.1 第一段：痛点

不要只说“video generation 还有 failure”。要明确说：

- 现有评测输出的是 scalar scores
- 真实生产需要的是 debug interface
- 生成视频调试本质上需要回答四个问题：
  1. 出了什么错？
  2. 错发生在哪？
  3. 为什么判断是这个错？
  4. 应该如何修？

可直接写：

> Existing video evaluation pipelines are optimized for ranking or aggregate scoring, but real generation workflows require **debuggable feedback**: users need to know what failed, where the failure is visible, what evidence supports the diagnosis, and what repair action should be attempted next.

### 4.2 第二段：现有方法缺什么

明确批评：

- VBench/EvalCrafter/VideoScore/T2V-CompBench 缺 localized diagnosis
- end-to-end VLM reports 虽然灵活，但：
  - 输出不稳定
  - 难审计
  - patch action 不可执行
  - 不易与后续 pipeline 对接

### 4.3 第三段：你的核心主张

核心主张不要写成“我们做了一个复杂系统”，而要写成：

> The key thesis of this paper is that controllable-video evaluation should be reframed as a **structured diagnosis interface** rather than a score-only judgment problem.

### 4.4 第四段：贡献

建议重写为四条：

1. failure-as-code representation
2. localized evidence protocol
3. repair-as-patch compilation and closed-loop evaluation
4. benchmark with controlled + real failures, plus human repair validation

---

## 5. 建议贡献写法

### Contributions Draft

1. **Structured diagnosis interface.** We introduce a failure-as-code representation for controllable video generation, in which each diagnosis record includes a failure code, localized temporal span, affected target, confidence score, and evidence keyframes. This representation connects evaluation, explanation, and repair planning within a single auditable schema.

2. **Evidence-grounded repair protocol.** We propose a repair-as-patch formulation that maps diagnosis records to structured repair actions, such as identity anchoring, prop injection, camera override, and temporal regeneration. This enables closed-loop evaluation beyond scalar scoring.

3. **Benchmark and evaluation protocol.** We introduce VideoGenDoctor-Bench-v1, which combines controlled perturbations with real generator failures and supports evaluation of failure-code detection, evidence localization, repair-plan quality, and human-validated repair success.

4. **Strong empirical comparison.** We compare VideoGenDoctor against both internal ablations and strong end-to-end VLM baselines for diagnosis and repair planning, and show that structure improves reliability, auditability, and repair usefulness.

---

## 6. 方法部分应如何改写

当前方法并不是强算法创新，所以方法部分应该刻意强调：

- 统一接口
- 可审计性
- 可组合性
- 可复现协议

而不是暗示你发明了某种全新学习算法。

### 6.1 Method Positioning

建议写：

> VideoGenDoctor is not a single learned model. Instead, it is a structured protocol that combines feature-based screening, semantic verification, evidence aggregation, and patch compilation into a reproducible diagnostic interface.

### 6.2 failure-as-code record

正文中可显式给出：

\[
r_i = (c_i, t_0^i, t_1^i, y_i, \sigma_i, K_i, T_i, a_i)
\]

其中：

- \(c_i\): failure code
- \((t_0^i, t_1^i)\): evidence span
- \(y_i\): affected target
- \(\sigma_i\): confidence
- \(K_i\): evidence keyframes
- \(T_i\): optional track / object evidence
- \(a_i\): compiled repair action

### 6.3 两个新强调点

方法中要新增两个 reviewer 会买账的概念：

1. **Auditability**
   - 每个 patch 必须能追溯到具体 code 与证据区间
2. **Interface stability**
   - 可以替换 detector/judge/generator，而不改 schema

---

## 7. 实验部分必须新增什么

这是最关键的部分。如果真按 reviewer 标准重做，实验应该长成下面这样。

### 7.1 数据集升级

当前 72 个视频不够。理想稿件至少应写成：

#### VideoGenDoctor-Bench-v1

- `N = [300~1000]` videos
- multiple generators:
  - CogVideoX
  - Stable Video Diffusion / SVD
  - another multi-shot controllable pipeline
- data composition:
  - controlled perturbation subset
  - real failure subset
- split protocols:
  - source-level split
  - generator-level split
  - leave-one-perturbation-out

### 7.2 标注协议升级

必须有：

- 2 annotators on a subset
- Cohen’s kappa or Krippendorff’s alpha
- adjudication protocol
- blind repair evaluation

建议实验写法：

> We collect dual annotations for `[x]%` of the benchmark and report Cohen’s \(\kappa\) on code verification and span overlap agreement. Repair evaluation is conducted through blind human pass/fail annotation, where raters only see the specification and output videos, without method identity.

### 7.3 baseline 必须扩展

至少加三类：

#### Diagnosis baselines

1. `GPT-4V free-form report`
2. `GPT-4V structured report`
3. `Open-source VLM structured report`

#### Repair baselines

4. `VLM-to-patch` end-to-end
5. `score + LLM prompt-to-patch`
6. `Patch-only`
7. `Patch+Judge`

#### 你自己的

8. `VideoGenDoctor default`

### 7.4 指标必须扩展

除了已有的：

- macro-F1
- micro-F1
- tIoU@0.3/0.5
- Top-K keyframe hit
- Pass@1/2

还应加：

- per-code precision / recall / F1
- per-group tIoU
- confusion matrix
- perturbation-type repair gain
- human Pass@1/2
- patch usefulness score
- bootstrap 95% CI

---

## 8. 结果部分应如何组织

建议结果部分按下面顺序写。

### 8.1 Main diagnosis table

表 1：

- methods × macro-F1 / micro-F1 / tIoU@0.5 / Top-3 hit
- 包括强 VLM baseline

重点叙述：

- 如果你的系统优于 VLM baseline：
  - 说明结构化接口确实带来性能与稳定性收益
- 如果只差不多：
  - 强调 auditability, lower variance, controllability, patch executability

### 8.2 Per-code analysis

表 2：

- 8 个 observed codes 的 P/R/F1
- 再加 per-code tIoU@0.5

应明确写：

- 哪些 code 最难
- motion-related failures 是否更难
- frame-drop / jitter / segment-break 是否易混淆

### 8.3 Confusion matrix

图 3：

- 用 heatmap 展示 confusion matrix

### 8.4 Human repair evaluation

表 4：

- Score-only
- Patch-only
- Patch+Judge
- VLM-to-patch
- VideoGenDoctor

指标：

- system Pass@1/2
- human Pass@1/2
- agreement between system and human

这是整篇论文最重要的一张表。

### 8.5 Real-failure subset

表 5：

- controlled subset vs real-failure subset
- diagnosis F1 / tIoU
- patch usefulness

目标是证明：

> 系统不只是适配 deterministic perturbation

---

## 9. 推荐新增图表目录

如果按强版本重写，建议图表如下：

### Main paper

1. System overview
2. Example diagnostic report
3. Annotation interface
4. Confusion matrix
5. Human repair evaluation comparison
6. Real vs controlled subset comparison

### Appendix

1. Full taxonomy
2. Annotation tool
3. Diagnostic report full page
4. Prompt templates for VLM baselines
5. Blind human evaluation protocol
6. Per-code metrics full table
7. Bootstrap CI table

---

## 10. 假设性结果叙述模板

下面是你未来补到真实结果后，可以直接替换数值的写法。

### 10.1 Main result paragraph

> Table 2 compares VideoGenDoctor with both internal ablations and strong end-to-end VLM baselines. Among all methods, structured VideoGenDoctor achieves the strongest overall diagnosis performance, reaching macro-F1 of `[x]` and tIoU@0.5 of `[y]`, outperforming the strongest free-form VLM baseline by `[delta1]` and `[delta2]`, respectively. These gains suggest that explicit failure schemas and evidence-grounded aggregation provide a more stable substrate for controllable-video diagnosis than direct free-form report generation.

### 10.2 Human repair evaluation paragraph

> Automatic verifier Pass@1/2 correlates with blind human repair judgments, but systematically overestimates repair quality for weaker baselines. VideoGenDoctor shows the highest agreement between system-confidence pass rate and human-verified pass rate, indicating that its repair loop is not only self-consistent but also better aligned with external perceptual judgments.

### 10.3 Real-failure paragraph

> Performance drops from the controlled subset to the real-failure subset for all methods, confirming that real generator failures are substantially harder than deterministic perturbations. Nevertheless, VideoGenDoctor maintains a clear advantage over direct VLM baselines, particularly on identity and prop-related failures, suggesting that the structured interface improves robustness beyond the synthetic perturbation setting.

### 10.4 Per-code paragraph

> Per-code analysis reveals that motion-related failures remain the most challenging category. In particular, MO_JITTER, MO_FRAME_DROP, and MO_SEGMENT_BREAK exhibit higher mutual confusion and lower span localization precision, likely because they share overlapping temporal cues. By contrast, identity and prop-related failures are easier to localize and repair once anchored to explicit targets.

---

## 11. 讨论部分应该怎么写

讨论不要再泛泛而谈“未来可以扩展到音频/3D”。应当明确回应 reviewer 的担忧。

### 建议新增三个 subsection

#### 11.1 Why structure helps beyond free-form VLM agents

讨论：

- structure 不只是更强，而是：
  - 更稳定
  - 更可控
  - 更易审计
  - 更利于 patch execution

#### 11.2 Why independent human repair validation matters

讨论：

- automatic verifier 有价值，但不足
- human blind pass/fail 是闭环修复必须的外部锚点

#### 11.3 Controlled perturbations vs real failures

讨论：

- controlled perturbation 的作用是可复现性与 error isolation
- real failures 的作用是外推性
- 两者都需要

---

## 12. 局限性部分应该怎么升级

当前 limitations 太弱。理想版本要主动承认：

1. 真实生成失败比 deterministic perturbation 更复杂
2. taxonomy 不是完备真理，而是 operational schema
3. patch template 在 generator 接口受限时可能失效
4. 人工 blind evaluation 成本高，不易大规模做
5. 多失败共现时 patch interaction 仍未充分解决

参考写法：

> Our benchmark should not be interpreted as a complete ontology of controllable-video failures. Rather, it is an operational schema that makes diagnosis, evidence, and repair comparable across methods. Extending this schema to broader generators, longer videos, and more open-ended failure modes remains an important direction for future work.

---

## 13. 结论应该怎么写

结论不要再以“我们在 72 个视频上取得了 xx 分数”做收束。更强版本的结论应该落在“接口意义”上。

### Conclusion Draft

We argued that controllable-video evaluation should move beyond scalar scoring toward **debuggable, evidence-grounded interfaces**. VideoGenDoctor instantiates this idea through failure-as-code diagnosis records, localized evidence spans, executable repair plans, and closed-loop verification. Across controlled perturbations, real generator failures, strong VLM baselines, and blind human repair judgments, our results suggest that structure improves not only diagnostic accuracy but also auditability and repair usefulness. More broadly, this work positions controllable-video evaluation as a debugging problem rather than a pure scoring problem, and offers an initial protocol for making that perspective measurable and reproducible.

---

## 14. 你真正需要补的实验清单

这是最实用的部分。按优先级排序：

### P0：不补这些，稿子核心站不住

1. 强 VLM diagnosis baseline
2. 强 VLM repair baseline
3. blind human repair evaluation
4. subset-level annotation agreement
5. per-code / per-group metrics
6. bootstrap confidence interval

### P1：增强外推性

7. real-failure subset
8. multi-generator evaluation
9. source-level split / leave-one-perturbation-out

### P2：增强资源价值

10. taxonomy coverage expansion
11. release benchmark annotation UI + report UI + prompts
12. release human evaluation protocol

---

## 15. 最推荐的实际路线

如果你后续真要朝接收概率更高的方向推进，建议按下面路线执行：

### Route A：最现实的短期修订

- 保留当前系统
- 扩展 baseline
- 加 human blind repair eval
- 加 per-code/confusion matrix/CI
- 加小规模 real-failure subset

适合：较短周期重投

### Route B：中期版本

- benchmark 扩展到几百视频
- 3 个 generator
- 双人标注子集
- taxonomy 覆盖扩大

适合：系统/benchmark 强化版投稿

### Route C：最强版本

- 除了 benchmark/system，还加入一个明确的方法创新
- 例如：
  - span refinement model
  - diagnosis aggregation model
  - patch selection policy model
  - repair verification calibration model

适合：真正往 ICML 主会强方法稿靠

---

## 16. 最后结论

如果完全按 reviewer 的要求倒推，一篇更强的论文不应只是“把现有稿子语气变保守”，而应是：

1. **重新定位**为 structured diagnostic interface paper
2. **补强 baseline** 到 strong VLM agent 级别
3. **补强 repair evaluation** 到 blind human validated
4. **补强 benchmark** 到 controlled + real failures + multi-generator
5. **补强分析** 到 per-code / CI / confusion / perturbation breakdown



