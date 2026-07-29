# Noise Source Studio

噪声源智能识别平台（Noise Source Studio）是一款面向噪声源多标签识别的桌面应用。
当前版本为 `0.1.0`，完成了工程基础设施和可运行的 GUI 骨架；真实模型加载、数据预处理
和推理流程将在确认模型契约后接入。

## 环境要求

- Windows 10/11
- Python 3.11（建议 64 位）
- 不需要 CUDA 或真实模型文件

## 创建开发环境

在 PowerShell 中运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

如果 PowerShell 禁止执行激活脚本，也可以直接使用
`.\.venv\Scripts\python.exe` 执行后续命令。

## 启动软件

```powershell
python -m noise_source_studio
```

安装后也可以使用命令行入口：

```powershell
noise-source-studio
```

首次运行会在当前 Windows 用户的应用数据目录中创建配置、日志、模型和输出目录。
这些运行数据不会写入源码目录。设置页面可查看和修改默认目录。

## 质量检查

运行测试：

```powershell
python -m pytest
```

运行 Ruff 检查与格式检查：

```powershell
python -m ruff check .
python -m ruff format --check .
```

## 当前功能

- 专业浅色主题主窗口、顶部状态栏、左侧导航和底部状态栏
- 工作台、单文件预测、批量预测、模型验证、模型管理、任务历史、系统日志和系统设置
  八个页面的可扩展界面骨架
- JSON 配置读取、保存和恢复默认值
- 控制台与按大小轮转的文件日志
- 全局未捕获异常记录与面向用户的错误提示
- 推理服务协议及明确报错的未配置实现
- 配置、日志、导航、模块导入和应用创建的自动化测试

## 当前限制

- 尚未接入 `best.pt` 或其他真实模型
- 尚未定义 CSV/信号文件预处理契约
- 不执行真实推理、验证或报表导出
- 批处理、暂停和停止按钮目前只展示工作流骨架

项目在接入真实推理前不会生成随机概率或虚构业务数据。
推理平台GUI
