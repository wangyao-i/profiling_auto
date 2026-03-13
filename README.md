# 昇腾NPU性能分析工具

一个专门针对昇腾NPU上vLLM应用的性能分析工具，能够自主分析性能数据并生成详细报告。

## 功能特性

### 数据支持
- ✅ msprof CSV报告分析
- ✅ vLLM JSON追踪文件分析
- ✅ 目录批量处理

### 性能指标分析
- 📊 **AI-Core利用率**：计算核心使用效率
- 📈 **内存使用情况**：峰值和平均内存占用
- 🔄 **数据传输**：H2D/D2H传输耗时占比
- ⚡ **Kernel性能**：Top 10耗时算子分析
- 📋 **vLLM阶段**：prefill/decode/sampling/scheduler时间分布

### 分析结果输出
- 📄 **文本报告**：详细的性能分析摘要
- 📊 **可视化图表**：直观的性能指标图表
- 🌐 **交互式HTML**：可交互的网页报告

### 性能瓶颈识别
- 自动检测低AI-Core利用率（<70%）
- 自动检测高数据传输占比（>5%）
- 提供优化建议

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python ascend_profiler.py <输入路径> [选项]
```

### 参数说明

- `<输入路径>`：可以是单个文件（.csv或.json）或包含多个性能文件的目录
- `-o, --output`：指定输出目录（可选）

### 示例

1. 分析单个msprof报告：
   ```bash
   python ascend_profiler.py msprof_report.csv
   ```

2. 分析单个vLLM追踪文件：
   ```bash
   python ascend_profiler.py vllm_trace.json
   ```

3. 分析包含多个文件的目录：
   ```bash
   python ascend_profiler.py profiling_data/
   ```

4. 指定输出目录：
   ```bash
   python ascend_profiler.py msprof_report.csv -o my_analysis
   ```

## 输出内容

分析完成后，工具会在输出目录生成以下内容：

### 文本报告
- `performance_report.txt`：详细的性能分析摘要

### 可视化图表
- `ai_core_utilization.png`：AI-Core利用率柱状图
- `data_transfer.png`：数据传输与计算占比饼图
- `vllm_phase_distribution.png`：vLLM阶段时间分布柱状图

### 交互式报告
- `interactive_report.html`：可在浏览器中打开的交互式分析报告

## 性能指标说明

### AI-Core利用率
- **良好**：>70%
- **一般**：50%-70%
- **较差**：<50%

### 数据传输占比
- **良好**：<5%
- **一般**：5%-10%
- **较差**：>10%

### vLLM阶段分布
- **理想状态**：decode阶段占比最高
- **优化重点**：如果prefill或sampling阶段占比过高，可针对性优化

## 自定义配置

工具支持通过修改代码进行自定义配置：

1. **瓶颈阈值调整**：在`_comprehensive_analysis`方法中修改
2. **图表样式**：在`generate_visualizations`方法中修改
3. **报告内容**：在`generate_report`方法中修改

## 注意事项

1. 确保msprof报告包含必要的列：
   - `aic_compute_time`：AI-Core计算时间
   - `start_time`/`end_time`：时间戳
   - `memory_usage`：内存使用情况（可选）
   - `h2d_time`/`d2h_time`：数据传输时间（可选）
   - `kernel_name`：Kernel名称（可选）

2. vLLM追踪文件需符合Chrome Trace格式

3. 对于大型报告，分析时间可能较长

## 扩展功能

工具设计支持扩展，可通过以下方式添加新功能：

- 添加新的数据解析器：在`load_data`中扩展
- 添加新的分析指标：在`analyze`中添加新的分析方法
- 添加新的可视化图表：在`generate_visualizations`中添加
- 添加新的报告格式：在`generate_report`中添加

## 版本历史

- v1.0.0：初始版本，支持msprof和vLLM数据分析

## 许可证

MIT License
