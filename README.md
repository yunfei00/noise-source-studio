# Noise Source Studio

噪声源智能识别平台（Noise Source Studio）是一款面向噪声源多标签识别的 Windows
桌面应用。当前 Phase 2 在冻结的商业 GUI 骨架上接入正式
`noise_source_runtime`，支持模型包管理、长期模型会话和单 CSV 真实推理。

训练仓库实现、CSV 解析、STFT、模型结构和 PyTorch 细节不会复制到本仓库；
GUI 仅通过单独交付的 runtime wheel 使用这些能力。

## 环境要求

- Windows 10/11
- Python 3.11（建议 64 位）
- 与目标 CPU/CUDA 设备兼容的 PyTorch
- 训练侧交付的 `noise_source_identification-*.whl`
- 通过完整性校验的模型包目录

## 创建开发环境

在 PowerShell 中运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果 PowerShell 禁止执行激活脚本，可以直接使用
`.\.venv\Scripts\python.exe` 执行后续命令。

## 安装推理运行时

1. 安装 GUI 项目：

   ```powershell
   python -m pip install -e ".[dev]"
   ```

2. 根据设备安装 PyTorch。CPU 环境可使用项目的通用可选依赖：

   ```powershell
   python -m pip install -e ".[inference]"
   ```

   CUDA 环境请根据本机驱动和 CUDA 版本使用 PyTorch 官方安装选择器生成的命令。
   本项目不锁定 CUDA wheel URL。

3. 安装训练侧交付的 runtime wheel。路径仅用于开发环境安装，不写入源码：

   ```powershell
   python -m pip install <交付目录>\runtime\noise_source_identification-0.1.0-py3-none-any.whl
   ```

   wheel 自身声明完整运行依赖；如果已经安装兼容的 CPU/CUDA PyTorch，pip 会复用它。

4. 启动软件：

   ```powershell
   python -m noise_source_studio
   ```

安装后也可以使用：

```powershell
noise-source-studio
```

首次运行会在当前 Windows 用户的应用数据目录中创建配置、日志、模型和输出目录。
运行数据、模型、注册表和推理结果不会写入源码目录。

## 模型包导入

在“模型管理”页面选择包含以下文件的模型包目录：

```text
model.pt
manifest.json
metrics.json
preprocess.json
labels.json
README.md
sha256.txt
```

应用通过 runtime 的 `verify_model_package` 校验结构和 SHA256，再检查 package schema、
runtime 版本、动态标签和 prediction mode。导入采用：

```text
临时目录复制 → 二次完整性校验 → 原子重命名 → JSON 注册
```

同一名称和版本不会重复导入；复制或校验失败会清理临时目录。同一时间只能有一个活动
模型。活动模型在后台加载并长期保留 session，切换模型和退出应用时会关闭旧 session。

## 单文件推理

- 仅接受 `.csv` 文件。
- 文件必须满足 runtime 的严格 DATA 段契约。
- 信号预览直接使用 runtime CSV 解析结果。
- 大信号只对绘图数据抽样，不改变送入模型的原始数据。
- 模型加载、CSV 解析和推理均通过 Qt 工作线程执行。
- structured 模式显示 `label_marginal_probabilities`，最终组合完全由组合概率 argmax
  和 `decoded_label_vector` 决定。
- multilabel 模式显示 `multilabel_probabilities`，最终标签按各标签实际阈值执行
  `probability >= threshold`。
- 预测完成不会自动写报告；只有用户点击“导出结果”后才写 prediction JSON，可选同时
  导出 inference contract。

## 质量检查

运行 Ruff：

```powershell
python -m ruff check .
```

运行普通测试：

```powershell
python -m pytest
```

真实黄金样本集成测试默认跳过。配置交付路径后运行：

```powershell
$env:NOISE_STUDIO_MODEL_PACKAGE="<模型包目录>"
$env:NOISE_STUDIO_GOLDEN_CSV="<黄金 CSV>"
$env:NOISE_STUDIO_GOLDEN_RESULT="<golden_result.json>"
python -m pytest tests\integration\test_golden_model.py
```

## 当前已实现

- 专业浅色主题主窗口、顶部状态、八个主页面和统一导航
- JSON 配置、平台用户数据目录、轮转日志和全局异常处理
- 模型包校验、原子导入、JSON 注册、激活、完整性复查和未激活模型删除
- 启动时后台恢复活动模型及顶部/工作台状态联动
- runtime 适配层和单一长期 `InferenceSession`
- Qt 后台模型加载、严格 CSV 解析和单文件推理
- DATA 原始信号预览、解析统计和动态标签概率
- structured/multilabel 两种结果语义与详细结果
- 显式 prediction JSON 及可选 inference contract 导出
- fake runtime 单元测试和环境变量驱动的黄金模型集成测试

## 尚未实现

- 批量预测
- 完整模型验证工作流
- SQLite 任务历史
- 多文件并发推理
- 远程推理、用户权限和自动更新
- 安装包

## 项目目录

```text
noise-source-studio/
├── src/noise_source_studio/
│   ├── application.py             # Qt 应用生命周期与异常处理
│   ├── common/                    # 公共异常和平台路径
│   ├── domain/                    # GUI 稳定数据模型与接口
│   ├── infrastructure/
│   │   ├── config/                # JSON 配置
│   │   ├── inference/             # runtime wheel 适配层
│   │   └── logging/               # 轮转日志
│   ├── services/                  # 模型、预测和后台任务编排
│   └── presentation/              # 主窗口、页面、组件、图标和样式
├── tests/
│   └── integration/               # 可选黄金模型测试
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```
