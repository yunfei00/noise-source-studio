# 验证数据契约

## 清单格式

正式验证入口使用 CSV manifest，不从父目录名称推导真实标签。文件使用 UTF-8 或
UTF-8 BOM。

推荐格式：

```csv
file_path,true_combination,frequency_mhz,mix_ratio,group,notes
data/001/a.csv,001,600,1,single,
data/011/b.csv,011,650,1_2,double,
data/110/c.csv,110,700,3_2,double,
```

- `file_path` 必填。相对路径默认相对于 manifest 所在目录，也可以在界面指定数据根目录；
  绝对路径直接使用。
- `true_combination` 的顺序必须与当前激活模型 `labels` 完全一致，只能包含 `0` 和 `1`，
  长度必须等于标签数量，默认不允许全零标签。
- 也可以省略 `true_combination`，改为提供全部模型标签列。标签列在 CSV 中的顺序必须与
  模型标签顺序一致，值只能是 `0` 或 `1`。
- 同时提供组合列和逐标签列时，两者必须一致。
- 其余列作为元数据保存，可用于动态分组分析。
- 同一路径重复出现会提示；同一路径出现不同真实标签属于严重错误。
- 缺失文件可以选择“停止”，或“跳过并记录为 skipped”。

## 指标分母

模型准确率指标默认只使用“推理成功且真实标签有效”的样本。推理失败数和失败率单独展示，
不会混入 Exact Match、Precision、Recall 或 F1 的分母。

- Exact Match：预测标签向量与真实标签向量完全一致。
- 过预测：预测阳性标签数量大于真实阳性标签数量。
- 欠预测：预测阳性标签数量小于真实阳性标签数量。
- Source Count Accuracy：预测阳性标签数量等于真实阳性标签数量的样本比例。
- Hamming Loss：错误标签位数除以成功样本数与标签数的乘积。
- 分母为零的单项指标在界面显示“—”；宏平均与训练仓库一致，按
  `zero_division=0` 将该标签计为 0。

Structured 模型的最终离散标签使用 `decoded_label_vector`。Multilabel 模型使用
`multilabel_probabilities >= thresholds`。两种模式均不在 GUI 中重新实现预处理、
STFT 或模型 forward。

## 输出目录

```text
validation_<datetime>_<task-id>/
├── task.json
├── summary.json
├── sample_results.csv
├── label_metrics.csv
├── combination_metrics.csv
├── confusion_matrix.csv
├── group_metrics.csv
├── errors.csv
└── report.html
```

CSV 使用 UTF-8 BOM；JSON 和 HTML 使用 UTF-8。`report.html` 为自包含报告，不引用外部网络
资源，也不嵌入原始信号。历史结果打开时只读取这些缓存文件，不重新执行模型推理。
