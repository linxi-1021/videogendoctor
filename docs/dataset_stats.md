# Dataset Statistics: `out/dataset_v0`

## 1. 数据集概览

- 源视频数量：6
- 扰动类型数量：6
- seeds：2
- 最终生成视频数量：72
- manifest 文件：`out/dataset_v0/manifest.jsonl`
- 视频目录：`out/dataset_v0/videos/`
- ShotIR 覆盖：72 / 72 条样本均带有 `shotir_path`

生成数量计算方式：

\[
6 \text{ source videos} \times 6 \text{ perturbation types} \times 2 \text{ seeds} = 72
\]

---

## 2. 源视频覆盖情况

| 源视频 | 样本数 |
|---|---:|
| `video1` | 12 |
| `video2` | 12 |
| `video3` | 12 |
| `video4` | 12 |
| `video5` | 12 |
| `video6` | 12 |
| **总计** | **72** |

---

## 3. 扰动类型分布

| 扰动类型 | 每个源视频生成数 | 总数 | 自动标注失败码 |
|---|---:|---:|---|
| `remove_anchor` | 2 | 12 | `ID_FACE_DRIFT`, `ID_BODY_DRIFT` |
| `drop_props_required` | 2 | 12 | `AL_PROP_MISSING` |
| `change_camera_movement` | 2 | 12 | `CA_MOVE_WRONG` |
| `extend_duration_or_merge` | 2 | 12 | `MO_SEGMENT_BREAK`, `MO_EVENT_MISSING` |
| `temporal_jitter_or_frame_drop` | 2 | 12 | `MO_JITTER`, `MO_FRAME_DROP` |
| `compression_artifacts` | 2 | 12 | `ST_COMPRESSION_ARTIFACT`, `ST_COLOR_SHIFT` |
| **总计** |  | **72** |  |

---

## 4. 自动标注失败码覆盖统计

> 说明：这里统计的是 `manifest.jsonl` 中 `failure_codes` 的出现次数，不是人工复核后的最终真值。

| 失败码 | 出现次数 | 来源扰动 |
|---|---:|---|
| `ID_FACE_DRIFT` | 12 | `remove_anchor` |
| `ID_BODY_DRIFT` | 12 | `remove_anchor` |
| `AL_PROP_MISSING` | 12 | `drop_props_required` |
| `CA_MOVE_WRONG` | 12 | `change_camera_movement` |
| `MO_SEGMENT_BREAK` | 12 | `extend_duration_or_merge` |
| `MO_EVENT_MISSING` | 12 | `extend_duration_or_merge` |
| `MO_JITTER` | 12 | `temporal_jitter_or_frame_drop` |
| `MO_FRAME_DROP` | 12 | `temporal_jitter_or_frame_drop` |
| `ST_COMPRESSION_ARTIFACT` | 12 | `compression_artifacts` |
| `ST_COLOR_SHIFT` | 12 | `compression_artifacts` |

- 唯一失败码总数：10
- failure code 总标注次数：120

---

## 5. ShotIR 覆盖情况

所有样本均保留了对应源视频的 `ShotIR` 路径：

| 源视频 | ShotIR |
|---|---|
| `video1` | `data/video1_shotir.json` |
| `video2` | `data/video2_shotir.json` |
| `video3` | `data/video3_shotir.json` |
| `video4` | `data/video4_shotir.json` |
| `video5` | `data/video5_shotir.json` |
| `video6` | `data/video6_shotir.json` |

说明：扰动数据集中的每个派生样本都继承了其源视频的 `shotir_path`，便于后续评分、patch 生成与对齐分析。

---

## 6. 文件命名规则

生成后的视频命名格式为：

```text
<source_id>__<perturbation_type>__s<seed>.mp4
```

示例：

- `video1__remove_anchor__s0.mp4`
- `video3__compression_artifacts__s1.mp4`
- `video6__temporal_jitter_or_frame_drop__s0.mp4`

---

## 7. 当前数据集结论

当前 `dataset_v0` 满足最小可用实验集要求：

- 样本总量达到 72 条
- 6 类扰动分布完全均衡
- 10 个自动失败码覆盖均衡
- 6 个源视频全部带有 ShotIR
- 适合进入下一步人工标注与系统评测

---

## 8. 下一步建议

1. 对 `out/dataset_v0/manifest.jsonl` 对应视频进行抽检，确认视频都可播放。  
2. 开始人工标注，生成 `out/annotations.jsonl`。  
3. 基于 `shotir_path` 和视频路径批量运行评分流程，生成预测结果。  
4. 运行失败码 F1、证据定位等评测脚本。
