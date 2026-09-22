# 计算设备选择与回退

## 配置与实际设备

用户配置保存在平台应用数据目录的 `settings.json` 中：

```json
{
  "device_preference": "auto",
  "allow_cpu_fallback": true
}
```

`device_preference` 只表示用户策略，允许 `auto`、`cpu` 和探测到的
`cuda:N`。`resolved_device` 来自当前成功加载的 `InferenceSession`，不会写入
配置。顶部状态和系统设置中的“实际设备”始终使用 `resolved_device`。

旧配置中的 `default_device` 会在读取时迁移为 `device_preference`。

## CUDA 完整探测

探测在 GUI 后台任务中执行，页面不导入 PyTorch。每个 CUDA 设备需要依次通过：

1. `torch.cuda.is_available()`；
2. 设备索引和设备数量检查；
3. 设备名称及可安全读取的总显存；
4. 在目标 `cuda:N` 创建小 tensor；
5. 执行轻量算术；
6. `torch.cuda.synchronize(index)`。

任一步失败都会把该 CUDA 标记为不可用，简短错误显示给用户，完整 traceback
写入日志。CPU 始终可选。明确选择 CPU 时不会执行 CUDA 探测。

## 选择和回退

- `auto`：选择第一个完整探测成功的 CUDA；没有可用 CUDA 时非阻塞地回退 CPU。
- `cpu`：直接使用 `device="cpu"` 创建 session，不查询 CUDA。
- `cuda:N`：完整探测指定设备后严格加载；失败时不静默回退。启用 CPU 回退后，
  由用户在“切换到 CPU / 取消 / 查看日志”中明确选择。

切换采用候选 session：

```text
创建并检查新 session
→ 原子替换当前 session
→ 关闭旧 session
```

候选创建或检查失败时会关闭候选并保留旧 session。单文件、批量、模型验证或模型
加载占用设备期间，设备设置不能切换。

## 当前 runtime 兼容边界

当前安装的 `noise_source_runtime.device.resolve_device("auto")` 只依据
`torch.cuda.is_available()`，没有 tensor 与 synchronize 探测。本应用不会把
`auto` 直接传给 runtime，而是在 GUI 适配层完整探测后传入具体的 `cpu` 或
`cuda:N`。

后续重新构建 runtime wheel 时建议补充稳定公共接口：

- `probe_devices() -> DeviceProbeReport`；
- 支持对具体 `cuda:N` 执行 tensor/synchronize 健康检查；
- 让 `resolve_device("auto")` 使用完整健康检查，而不是只判断
  `torch.cuda.is_available()`；
- 返回结构化错误摘要，同时保留异常链供调用方记录日志。

GUI 端不复制 runtime 的模型、预处理或 forward 代码。
