# Changelog

本项目遵循语义化版本。

## [Unreleased]

### Added

- 集成正式 `noise_source_runtime` wheel 适配层。
- 增加模型包校验、原子导入、JSON 注册、激活、完整性检查和安全删除。
- 增加长期 `InferenceSession` 生命周期和启动时后台恢复。
- 实现严格 DATA CSV 的后台解析、信号预览和单文件真实推理。
- 实现 structured 与 multilabel 动态结果展示。
- 增加 prediction JSON 及可选 inference contract 显式导出。
- 增加 fake runtime 单元测试和黄金模型可选集成测试。

### Notes

- Phase 2 暂不包含批量预测、完整模型验证、SQLite 任务历史或并发推理。

## [0.1.0] - 2026-07-29

### Added

- 初始化 PySide6 商业 GUI 架构。
- 完成工作台、单文件预测、批量预测、模型验证、模型管理、任务历史、系统日志和系统设置八个主页面。
- 完成配置、路径、日志和异常处理基础设施。
- 完成统一导航、工作台快捷入口和商业界面收尾。
- 增加推理接口及明确报错的未配置占位实现。

### Notes

- 暂未接入真实模型、模型预处理或推理流程。
