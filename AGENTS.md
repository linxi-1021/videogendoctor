# Repository Instructions

使用中文回答。

## 论文修改的 Git 工作流

当用户要求修改论文或论文相关资源时，默认执行以下流程，并在不与用户最新指令冲突的前提下优先遵守：

1. 先检查当前 Git 状态。
2. 如果用户在本次对话开始前已经修改了论文相关文件，且这些修改尚未提交，则先创建一个“修改前”快照提交。
3. 再进行论文修改。
4. 修改后运行必要的编译或校验。
5. 创建一个“修改后”快照提交。
6. 将相关提交推送到 GitHub 远程仓库。

## 论文相关文件范围

默认优先纳入版本管理和提交的文件包括但不限于：

- `docs/paper1_videogendoctor/latex/main.tex`
- `docs/paper1_videogendoctor/latex/references.bib`
- `docs/paper1_videogendoctor/neurIPS/main.tex`
- `docs/paper1_videogendoctor/neurIPS/checklist.tex`
- `docs/paper1_videogendoctor/figures/` 下与论文直接相关的图表、图片、矢量图、导出 PDF、源 PPTX、SVG 等 figure 资源
- 其他被本次论文任务直接修改的附录、表格计划、审稿说明或投稿材料

如果工作区中同时存在与论文无关的大量改动，默认只提交与本次论文任务直接相关的文件，不把整个工作区一起提交。

## Figure 资源要求

当论文正文或附录引用了新的 figure，或现有 figure 被更新时：

1. 相关 figure 文件必须一并纳入版本管理。
2. 提交论文主稿时，应同步提交对应的 figure 资源。
3. 推送到 GitHub 时，论文文本文件和本次修改涉及的 figure 文件应一并推送，避免仓库中出现“TeX 已更新但图未上传”的不一致状态。

## 提交信息约定

若无用户特别要求，提交信息应清楚说明是“修改前快照”还是“修改后结果”，并简要概括本次论文改动主题。

## 远程同步

若仓库已配置 GitHub 远程仓库，则在完成本地提交后默认执行推送。
如果推送失败，应向用户说明失败原因和当前本地提交状态。
