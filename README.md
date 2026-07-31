# Noise Source Studio

## Phase 4：模型验证中心

模型验证页面已经提供完整的 manifest 驱动验证流程，并继续复用当前激活模型的单一
`InferenceSession`：

- “验证配置 / 总体结果 / 标签分析 / 组合分析 / 分组分析 / 样本明细”六个页签；
- UTF-8 BOM 清单、中文路径、相对/绝对路径、动态模型标签列和附加元数据；
- 后台顺序推理、逐文件错误隔离、暂停、继续、停止和模型/设备锁定；
- Exact Match、Micro/Macro/Weighted 指标、Hamming Loss、标签 TP/FP/TN/FN、
  组合混淆矩阵、Structured Top-K 和动态分组指标；
- 10,000 条结果虚拟表格、典型误判筛选、图表/矩阵联动和按需波形；
- 历史验证恢复、已有批量结果转验证，以及不依赖外部网络资源的 `report.html`。

验证清单格式、指标分母和输出文件说明见
[验证数据契约](docs/validation_manifest.md)。

Noise Source Studio（噪声源智能识别平台）是一款面向噪声源多标签识别的 Windows
桌面应用。当前 Phase 3 已在冻结的商业 GUI 框架上完成模型包管理、单文件推理和专业批量预测。

训练仓库中的 CSV 解析、STFT、模型结构与 PyTorch 细节不会复制到本仓库。GUI 只通过独立交付的
`noise_source_runtime` wheel 使用这些能力，并长期复用一个 `InferenceSession`。

## 环境要求

- Windows 10/11
- Python 3.11（建议 64 位）
- 与目标 CPU/CUDA 设备兼容的 PyTorch
- 训练侧交付的 `noise_source_identification-*.whl`
- 通过完整性校验的模型包目录

## 创建开发环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果 PowerShell 禁止激活脚本，可直接使用 `.\.venv\Scripts\python.exe` 执行后续命令。

## 安装推理运行时

先根据设备安装兼容的 PyTorch，再安装训练侧交付的 runtime wheel：

```powershell
python -m pip install -e ".[inference]"
python -m pip install <交付目录>\runtime\noise_source_identification-0.1.0-py3-none-any.whl
```

CUDA 环境应使用 PyTorch 官方安装选择器生成的命令。本项目不锁定 CUDA wheel URL，也不提交模型文件。

## 启动

```powershell
python -m noise_source_studio
```

安装后也可运行：

```powershell
noise-source-studio
```

配置、日志、模型注册表和输出默认保存在当前 Windows 用户的应用数据目录，不写入源码目录。

## 模型包导入

模型管理页接受包含以下文件的模型包目录：

```text
model.pt
manifest.json
metrics.json
preprocess.json
labels.json
README.md
sha256.txt
```

应用先调用 runtime 的 `verify_model_package` 校验结构和 SHA256，再检查 package schema、runtime
版本、动态标签和 prediction mode。导入采用临时目录复制、二次校验、原子重命名和 JSON 注册流程。
同一时间只有一个活动模型，活动模型会在后台加载并长期保留 session。

## 单文件预测

- 只接受满足 runtime 严格 DATA 契约的 `.csv` 文件。
- 信号预览、CSV 解析和推理均在后台执行。
- structured 主显示使用 `label_marginal_probabilities`，最终结论使用 runtime 返回的
  `decoded_label_vector`、`predicted_combination` 和 `predicted_sources`。
- multilabel 主显示使用 `multilabel_probabilities`，最终标签由 runtime 的实际 thresholds 判定。
- 用户显式点击导出后才写 prediction JSON；inference contract 为可选项。

## 批量预测

- 支持多文件、文件夹、递归扫描、拖拽、自然排序、规范化绝对路径去重和队列编辑。
- 文件夹扫描在后台执行；添加阶段仅做存在性、类型、后缀和可读性检查。
- 一个批次锁定模型名称、版本、包路径、runtime 版本和设备，并复用当前已加载的 session。
- 单工作线程按队列顺序调用 `session.predict_file`，不会为每个 CSV 重新加载模型，也不会并发
  调用同一个 GPU session。
- 暂停为协作式暂停：当前文件完成后暂停，不启动下一个文件；继续后只处理 pending 文件。
- 停止为协作式停止：当前文件完成后停止，尚未运行的文件统一标记为 `stopped`。再次开始时可将
  stopped 项恢复为 pending。
- 单个 CSV 的解析或推理错误只标记该文件失败并继续；session 关闭、模型卸载、CUDA 设备失效等
  不可恢复错误会终止批次。
- 失败项可以单独或全部重试，`retry_count` 会保留。
- 页面内置“文件任务 / 预测结果 / 统计分析”结果中心；完成后自动进入预测结果。
- 结果表格使用 `QTableView`、`QAbstractTableModel` 和筛选代理，支持万级结果虚拟化展示、
  关键词/状态/组合/噪声源/可信度筛选、低置信度筛选、排序和 2～5 项对比。
- 统计页提供组合、噪声源、置信度与错误分布，点击柱形可回到结果页并应用对应筛选。
- 选择结果时才在后台重新读取原始 CSV 波形，批量任务对象不保存完整波形。
- 可打开历史批量结果目录，恢复表格、详情和统计，不执行模型推理。
- 批次完成、停止或部分失败后自动写出：

```text
outputs/
└── batch_<日期时间>_<task-id>/
    ├── task.json
    ├── summary.json
    ├── predictions.csv
    └── errors.csv
```

CSV 使用 UTF-8 BOM；列表和字典字段保存为合法 JSON 字符串。导出失败不会清除内存结果，可重新
选择目录导出副本。

## 计算设备

- 系统设置提供自动选择、CPU 和实际探测到的 `CUDA:N`，并持久化
  `device_preference` 与 `allow_cpu_fallback`。
- 自动模式会在后台对每个 CUDA 执行设备名称、tensor 计算和 synchronize 完整探测；探测失败时
  使用 CPU，并显示非阻塞回退提示。
- 明确选择 CPU 时不会运行 CUDA 探测，也不会被已安装的 CUDA 环境覆盖。
- 明确选择 CUDA 时不会静默回退；用户可以选择切换到 CPU、取消或查看日志。
- 顶部状态显示当前 session 的实际设备，而不是配置策略或 PyTorch CUDA 构建版本。
- 设备切换使用“候选 session 成功后原子替换，再关闭旧 session”，失败时保留旧 session。
- 单文件、批量、模型验证及模型加载期间禁止切换设备。

完整契约和当前 runtime 兼容说明见
[`docs/device_selection.md`](docs/device_selection.md)。

## 质量检查

```powershell
python -m ruff check .
python -m pytest
```

真实黄金样本测试：

```powershell
$env:NOISE_STUDIO_MODEL_PACKAGE="<模型包目录>"
$env:NOISE_STUDIO_GOLDEN_CSV="<黄金 CSV>"
$env:NOISE_STUDIO_GOLDEN_RESULT="<golden_result.json>"
python -m pytest tests\integration\test_golden_model.py
```

真实三文件批量一致性测试：

```powershell
$env:NOISE_STUDIO_MODEL_PACKAGE="<模型包目录>"
$env:NOISE_STUDIO_BATCH_CSV_DIR="<至少包含三个严格 DATA CSV 的目录>"
python -m pytest tests\integration\test_batch_consistency.py
```

未配置环境变量时，真实集成测试会明确跳过；普通自动化测试使用 fake runtime，不要求 CUDA。

## 当前已实现

- 商业浅色主窗口、顶部状态、八个主页面、统一导航和高 DPI SVG 图标
- JSON 配置、平台用户数据目录、轮转日志和全局异常处理
- 后台 CPU/CUDA 完整探测、设备策略持久化、CPU 回退和 session 原子切换
- 模型包校验、原子导入、注册、激活、恢复、完整性复查和安全删除
- 单一长期 `InferenceSession` 与 runtime 适配层
- 单文件严格 CSV 解析、信号预览、后台推理、动态结果展示和显式导出
- 批量队列、顺序推理、暂停/继续/停止、错误隔离、失败重试和四文件导出
- 批量结果检索、动态详情、按需波形、统计联动、历史恢复和多结果对比
- fake runtime 自动化测试和环境变量驱动的真实模型集成测试

## 尚未实现

- SQLite 任务历史与未完成任务自动恢复
- 多线程模型 forward、多 GPU 和远程推理
- 用户权限、自动更新和安装包

## 项目目录

```text
noise-source-studio/
├── src/noise_source_studio/
│   ├── application.py             # Qt 应用生命周期与异常处理
│   ├── common/                    # 公共异常和平台路径
│   ├── domain/                    # 稳定领域模型、批量状态与接口
│   ├── infrastructure/
│   │   ├── config/                # JSON 配置
│   │   ├── inference/             # runtime 适配层与批量 worker
│   │   └── logging/               # 轮转日志
│   ├── services/                  # 模型、单文件/批量预测与导出编排
│   └── presentation/              # 主窗口、页面、组件、图标和样式
├── tests/
│   └── integration/               # 可选真实模型测试
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```
