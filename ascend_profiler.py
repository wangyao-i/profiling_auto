#!/usr/bin/env python3
import argparse
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class AscendProfiler:
    def __init__(self, input_path, output_dir=None):
        self.input_path = input_path
        self.output_dir = output_dir or f"profiler_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data = {}
        self.analysis_results = {}
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载性能数据"""
        if os.path.isfile(self.input_path):
            if self.input_path.endswith('.csv'):
                self.data['msprof'] = self._load_msprof_csv()
            elif self.input_path.endswith('.json'):
                self.data['vllm'] = self._load_vllm_trace()
            else:
                print(f"不支持的文件格式: {self.input_path}")
        elif os.path.isdir(self.input_path):
            # 加载目录中的所有支持文件
            for file in os.listdir(self.input_path):
                file_path = os.path.join(self.input_path, file)
                if file.endswith('.csv'):
                    self.data['msprof'] = self._load_msprof_csv(file_path)
                elif file.endswith('.json'):
                    self.data['vllm'] = self._load_vllm_trace(file_path)
        else:
            raise FileNotFoundError(f"输入路径不存在: {self.input_path}")
    
    def _load_msprof_csv(self, file_path=None):
        """加载msprof CSV报告"""
        file_path = file_path or self.input_path
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载msprof报告: {file_path}")
            return df
        except Exception as e:
            print(f"加载msprof报告失败: {e}")
            return None
    
    def _load_vllm_trace(self, file_path=None):
        """加载vLLM追踪文件"""
        file_path = file_path or self.input_path
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"成功加载vLLM追踪文件: {file_path}")
            return data
        except Exception as e:
            print(f"加载vLLM追踪文件失败: {e}")
            return None
    
    def analyze(self):
        """执行性能分析"""
        if 'msprof' in self.data:
            self._analyze_msprof()
        if 'vllm' in self.data:
            self._analyze_vllm()
        
        # 综合分析
        self._comprehensive_analysis()
    
    def _analyze_msprof(self):
        """分析msprof数据"""
        df = self.data['msprof']
        results = {}
        
        # AI-Core利用率分析
        if 'aic_compute_time' in df.columns:
            aic_time = df['aic_compute_time'].sum()
            total_time = df['end_time'].max() - df['start_time'].min() if 'start_time' in df.columns and 'end_time' in df.columns else aic_time
            results['ai_core_utilization'] = (aic_time / total_time * 100) if total_time > 0 else 0
        
        # 内存分析
        if 'memory_usage' in df.columns:
            results['peak_memory'] = df['memory_usage'].max()
            results['avg_memory'] = df['memory_usage'].mean()
        
        # 数据传输分析
        if 'h2d_time' in df.columns and 'd2h_time' in df.columns:
            h2d_total = df['h2d_time'].sum()
            d2h_total = df['d2h_time'].sum()
            total_time = df['end_time'].max() - df['start_time'].min() if 'start_time' in df.columns and 'end_time' in df.columns else (h2d_total + d2h_total)
            results['h2d_percentage'] = (h2d_total / total_time * 100) if total_time > 0 else 0
            results['d2h_percentage'] = (d2h_total / total_time * 100) if total_time > 0 else 0
        
        # Kernel分析
        if 'kernel_name' in df.columns:
            kernel_stats = df.groupby('kernel_name')['aic_compute_time'].agg(['count', 'sum', 'mean', 'std']).sort_values('sum', ascending=False)
            results['top_kernels'] = kernel_stats.head(10).to_dict()
        
        self.analysis_results['msprof'] = results
    
    def _analyze_vllm(self):
        """分析vLLM追踪数据"""
        data = self.data['vllm']
        results = {}
        
        # 分析vLLM阶段
        phases = ['prefill', 'decode', 'sampling', 'scheduler']
        phase_times = {}
        
        if isinstance(data, dict) and 'traceEvents' in data:
            events = data['traceEvents']
            print(f"共加载 {len(events)} 个事件")
            
            # 先收集所有事件名称，用于调试
            all_event_names = list(set(e.get('name', '') for e in events if 'name' in e))
            print(f"事件名称列表（前20个）: {all_event_names[:20]}")
            
            # 处理带有dur字段的完整事件和B/E配对事件
            for phase in phases:
                print(f"\n正在分析阶段: {phase}")
                
                # 匹配包含阶段名称的事件（不区分大小写）
                phase_events = [e for e in events if phase.lower() in e.get('name', '').lower()]
                print(f"找到 {len(phase_events)} 个与 '{phase}' 相关的事件")
                
                if phase_events:
                    # 计算所有带有dur字段的事件的总时间
                    dur_events = [e for e in phase_events if 'dur' in e]
                    dur_time = sum(e['dur'] for e in dur_events)
                    print(f"  - 带有dur字段的事件: {len(dur_events)} 个, 总时间: {dur_time} ns")
                    
                    # 处理B/E配对事件（更健壮的匹配）
                    # 先按id分组
                    events_by_id = {}
                    for e in phase_events:
                        event_id = e.get('id', '')
                        if event_id:
                            if event_id not in events_by_id:
                                events_by_id[event_id] = []
                            events_by_id[event_id].append(e)
                    
                    be_duration = 0
                    matched_pairs = 0
                    
                    for event_id, id_events in events_by_id.items():
                        if len(id_events) >= 2:
                            # 查找开始和结束事件
                            begin_event = next((e for e in id_events if e.get('ph') == 'B'), None)
                            end_event = next((e for e in id_events if e.get('ph') == 'E'), None)
                            
                            if begin_event and end_event:
                                start_ts = begin_event.get('ts', 0)
                                end_ts = end_event.get('ts', 0)
                                if end_ts > start_ts:
                                    be_duration += (end_ts - start_ts)
                                    matched_pairs += 1
                    
                    print(f"  - B/E配对事件: {matched_pairs} 对, 总时间: {be_duration} ns")
                    
                    # 尝试基于线程和时间戳匹配B/E事件（如果没有id字段）
                    thread_events = {}
                    for e in phase_events:
                        thread_id = e.get('tid', e.get('pid', 'unknown'))
                        if thread_id not in thread_events:
                            thread_events[thread_id] = []
                        thread_events[thread_id].append(e)
                    
                    thread_be_duration = 0
                    thread_matched = 0
                    
                    for thread_id, thread_event_list in thread_events.items():
                        # 按时间戳排序
                        sorted_events = sorted(thread_event_list, key=lambda x: x.get('ts', 0))
                        
                        # 简单的B/E匹配（假设B后紧跟E）
                        i = 0
                        while i < len(sorted_events) - 1:
                            if sorted_events[i].get('ph') == 'B' and sorted_events[i+1].get('ph') == 'E':
                                start_ts = sorted_events[i].get('ts', 0)
                                end_ts = sorted_events[i+1].get('ts', 0)
                                if end_ts > start_ts:
                                    thread_be_duration += (end_ts - start_ts)
                                    thread_matched += 1
                                i += 2  # 跳过这对B/E事件
                            else:
                                i += 1
                    
                    if thread_matched > 0:
                        print(f"  - 基于线程的B/E配对: {thread_matched} 对, 总时间: {thread_be_duration} ns")
                        be_duration += thread_be_duration
                    
                    # 合并所有类型的事件时间
                    total_phase_time = dur_time + be_duration
                    print(f"  - 阶段总时间: {total_phase_time} ns")
                    
                    if total_phase_time > 0:
                        phase_times[phase] = total_phase_time
            
            total_time = sum(phase_times.values())
            print(f"\n所有阶段总时间: {total_time} ns")
            
            if total_time > 0:
                results['phase_distribution'] = {p: (t / total_time * 100) for p, t in phase_times.items()}
                print("阶段时间分布:")
                for phase, percentage in results['phase_distribution'].items():
                    print(f"  {phase}: {percentage:.2f}%")
            else:
                # 如果没有找到匹配的事件，尝试直接搜索所有事件
                print("\n警告：没有找到与vLLM阶段相关的事件！")
                print("请检查trace文件格式或尝试使用更通用的分析工具。")
        
        self.analysis_results['vllm'] = results
    
    def _comprehensive_analysis(self):
        """综合分析所有数据"""
        results = {}
        
        # 识别性能瓶颈
        bottlenecks = []
        
        if 'msprof' in self.analysis_results:
            msprof = self.analysis_results['msprof']
            
            # 检查AI-Core利用率
            if msprof.get('ai_core_utilization', 100) < 70:
                bottlenecks.append(f"AI-Core利用率低: {msprof['ai_core_utilization']:.2f}%")
            
            # 检查数据传输
            if msprof.get('h2d_percentage', 0) > 5:
                bottlenecks.append(f"H2D传输耗时过高: {msprof['h2d_percentage']:.2f}%")
            if msprof.get('d2h_percentage', 0) > 5:
                bottlenecks.append(f"D2H传输耗时过高: {msprof['d2h_percentage']:.2f}%")
        
        results['bottlenecks'] = bottlenecks
        self.analysis_results['comprehensive'] = results
    
    def generate_report(self):
        """生成文本报告"""
        report_path = os.path.join(self.output_dir, 'performance_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("昇腾NPU性能分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入路径: {self.input_path}\n\n")
            
            # msprof分析结果
            if 'msprof' in self.analysis_results:
                f.write("=== MSPROF分析结果 ===\n")
                msprof = self.analysis_results['msprof']
                
                if 'ai_core_utilization' in msprof:
                    f.write(f"AI-Core利用率: {msprof['ai_core_utilization']:.2f}%\n")
                
                if 'peak_memory' in msprof:
                    f.write(f"峰值内存: {msprof['peak_memory']:.2f} MB\n")
                    f.write(f"平均内存: {msprof['avg_memory']:.2f} MB\n")
                
                if 'h2d_percentage' in msprof:
                    f.write(f"H2D传输占比: {msprof['h2d_percentage']:.2f}%\n")
                if 'd2h_percentage' in msprof:
                    f.write(f"D2H传输占比: {msprof['d2h_percentage']:.2f}%\n")
                
                if 'top_kernels' in msprof:
                    f.write("\nTop 10耗时Kernel:\n")
                    for kernel, stats in msprof['top_kernels']['sum'].items():
                        f.write(f"  {kernel}: {stats:.2f} us\n")
                
                f.write("\n")
            
            # vLLM分析结果
            if 'vllm' in self.analysis_results:
                f.write("=== VLLM阶段分析 ===\n")
                vllm = self.analysis_results['vllm']
                
                if 'phase_distribution' in vllm:
                    for phase, percentage in vllm['phase_distribution'].items():
                        f.write(f"{phase}: {percentage:.2f}%\n")
                
                f.write("\n")
            
            # 综合分析
            if 'comprehensive' in self.analysis_results:
                comprehensive = self.analysis_results['comprehensive']
                
                if comprehensive.get('bottlenecks'):
                    f.write("=== 性能瓶颈分析 ===\n")
                    for bottleneck in comprehensive['bottlenecks']:
                        f.write(f"- {bottleneck}\n")
                
                f.write("\n")
        
        print(f"文本报告已生成: {report_path}")
    
    def generate_visualizations(self):
        """生成可视化图表"""
        # AI-Core利用率图表
        if 'msprof' in self.analysis_results:
            msprof = self.analysis_results['msprof']
            
            if 'ai_core_utilization' in msprof:
                plt.figure(figsize=(10, 6))
                plt.bar(['AI-Core利用率'], [msprof['ai_core_utilization']], color='#4CAF50')
                plt.ylabel('利用率 (%)')
                plt.title('AI-Core利用率分析')
                plt.ylim(0, 100)
                plt.text(0, msprof['ai_core_utilization'] + 2, f"{msprof['ai_core_utilization']:.1f}%", ha='center')
                plt.savefig(os.path.join(self.output_dir, 'ai_core_utilization.png'))
                plt.close()
            
            # 数据传输占比
            if 'h2d_percentage' in msprof and 'd2h_percentage' in msprof:
                labels = ['H2D传输', 'D2H传输', '计算']
                sizes = [msprof['h2d_percentage'], msprof['d2h_percentage'], 100 - msprof['h2d_percentage'] - msprof['d2h_percentage']]
                colors = ['#FF9800', '#F44336', '#2196F3']
                
                plt.figure(figsize=(10, 6))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title('数据传输与计算占比')
                plt.axis('equal')
                plt.savefig(os.path.join(self.output_dir, 'data_transfer.png'))
                plt.close()
            
            # vLLM阶段分布
            if 'vllm' in self.analysis_results and 'phase_distribution' in self.analysis_results['vllm']:
                vllm = self.analysis_results['vllm']
                phases = list(vllm['phase_distribution'].keys())
                percentages = list(vllm['phase_distribution'].values())
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(phases, percentages, color=['#9C27B0', '#3F51B5', '#00BCD4', '#8BC34A'])
                plt.xlabel('vLLM阶段')
                plt.ylabel('时间占比 (%)')
                plt.title('vLLM阶段时间分布')
                plt.ylim(0, 100)
                
                for bar, percentage in zip(bars, percentages):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{percentage:.1f}%", ha='center')
                
                plt.savefig(os.path.join(self.output_dir, 'vllm_phase_distribution.png'))
                plt.close()
        
        print(f"可视化图表已生成在: {self.output_dir}")
    
    def generate_interactive_report(self):
        """生成交互式HTML报告"""
        html_path = os.path.join(self.output_dir, 'interactive_report.html')
        
        html_content = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>昇腾NPU性能分析报告</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                h2 {
                    color: #555;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                }
                .metric {
                    display: inline-block;
                    background-color: #f0f0f0;
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                    text-align: center;
                    width: 200px;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }
                .metric-label {
                    font-size: 14px;
                    color: #666;
                }
                .chart-container {
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #fafafa;
                    border-radius: 5px;
                }
                .bottleneck {
                    background-color: #ffebee;
                    padding: 10px;
                    margin: 10px 0;
                    border-left: 4px solid #f44336;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>昇腾NPU性能分析报告</h1>
                <p>分析时间: {{analysis_time}}</p>
                <p>输入路径: {{input_path}}</p>
        """
        
        # 填充基本信息
        html_content = html_content.replace('{{analysis_time}}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        html_content = html_content.replace('{{input_path}}', self.input_path)
        
        # 填充关键指标
        html_content += "\n                <h2>关键性能指标</h2>\n"
        if 'msprof' in self.analysis_results:
            msprof = self.analysis_results['msprof']
            if 'ai_core_utilization' in msprof:
                html_content += f"\n                <div class='metric'>\n                    <div class='metric-value'>{msprof['ai_core_utilization']:.1f}%</div>\n                    <div class='metric-label'>AI-Core利用率</div>\n                </div>\n"
            if 'peak_memory' in msprof:
                html_content += f"\n                <div class='metric'>\n                    <div class='metric-value'>{msprof['peak_memory']:.1f} MB</div>\n                    <div class='metric-label'>峰值内存</div>\n                </div>\n"
            if 'h2d_percentage' in msprof:
                html_content += f"\n                <div class='metric'>\n                    <div class='metric-value'>{msprof['h2d_percentage']:.1f}%</div>\n                    <div class='metric-label'>H2D传输占比</div>\n                </div>\n"
            if 'd2h_percentage' in msprof:
                html_content += f"\n                <div class='metric'>\n                    <div class='metric-value'>{msprof['d2h_percentage']:.1f}%</div>\n                    <div class='metric-label'>D2H传输占比</div>\n                </div>\n"
        
        # 性能瓶颈
        if 'comprehensive' in self.analysis_results and self.analysis_results['comprehensive'].get('bottlenecks'):
            html_content += "\n                <h2>性能瓶颈</h2>\n"
            for bottleneck in self.analysis_results['comprehensive']['bottlenecks']:
                html_content += f"\n                <div class='bottleneck'>{bottleneck}</div>\n"
        
        # 图表部分
        html_content += "\n                <h2>性能图表</h2>\n"
        
        # AI-Core利用率图表
        if 'msprof' in self.analysis_results and 'ai_core_utilization' in self.analysis_results['msprof']:
            html_content += "\n                <div class='chart-container'>\n                    <h3>AI-Core利用率</h3>\n                    <canvas id='aiCoreChart' width='400' height='200'></canvas>\n                </div>\n"
        
        # 数据传输占比
        if 'msprof' in self.analysis_results and 'h2d_percentage' in self.analysis_results['msprof'] and 'd2h_percentage' in self.analysis_results['msprof']:
            html_content += "\n                <div class='chart-container'>\n                    <h3>数据传输与计算占比</h3>\n                    <canvas id='dataTransferChart' width='400' height='200'></canvas>\n                </div>\n"
        
        # vLLM阶段分布
        if 'vllm' in self.analysis_results and 'phase_distribution' in self.analysis_results['vllm']:
            html_content += "\n                <div class='chart-container'>\n                    <h3>vLLM阶段时间分布</h3>\n                    <canvas id='vllmPhaseChart' width='400' height='200'></canvas>\n                </div>\n"
        
        # JavaScript图表代码
        html_content += "\n                <script>\n"
        
        # AI-Core利用率图表
        if 'msprof' in self.analysis_results and 'ai_core_utilization' in self.analysis_results['msprof']:
            utilization = self.analysis_results['msprof']['ai_core_utilization']
            html_content += f"\n                    const aiCoreCtx = document.getElementById('aiCoreChart').getContext('2d');\n                    new Chart(aiCoreCtx, {{\n                        type: 'bar',\n                        data: {{\n                            labels: ['AI-Core利用率'],\n                            datasets: [{{\n                                label: '利用率 (%)',\n                                data: [{utilization}],\n                                backgroundColor: '#4CAF50',\n                                borderColor: '#45a049',\n                                borderWidth: 1\n                            }}]\n                        }},\n                        options: {{\n                            scales: {{\n                                y: {{\n                                    beginAtZero: true,\n                                    max: 100,\n                                    title: {{\n                                        display: true,\n                                        text: '利用率 (%)'\n                                    }}\n                                }}\n                            }},\n                            plugins: {{\n                                legend: {{ display: false }},\n                                title: {{\n                                    display: true,\n                                    text: 'AI-Core利用率分析'\n                                }} \n                            }}\n                        }}\n                    }});\n"
        
        # 数据传输占比
        if 'msprof' in self.analysis_results and 'h2d_percentage' in self.analysis_results['msprof'] and 'd2h_percentage' in self.analysis_results['msprof']:
            h2d = self.analysis_results['msprof']['h2d_percentage']
            d2h = self.analysis_results['msprof']['d2h_percentage']
            compute = 100 - h2d - d2h
            html_content += f"\n                    const dataTransferCtx = document.getElementById('dataTransferChart').getContext('2d');\n                    new Chart(dataTransferCtx, {{\n                        type: 'pie',\n                        data: {{\n                            labels: ['H2D传输', 'D2H传输', '计算'],\n                            datasets: [{{\n                                data: [{h2d}, {d2h}, {compute}],\n                                backgroundColor: ['#FF9800', '#F44336', '#2196F3'],\n                                borderWidth: 1\n                            }}]\n                        }},\n                        options: {{\n                            plugins: {{\n                                title: {{\n                                    display: true,\n                                    text: '数据传输与计算占比'\n                                }} \n                            }}\n                        }}\n                    }});\n"
        
        # vLLM阶段分布
        if 'vllm' in self.analysis_results and 'phase_distribution' in self.analysis_results['vllm']:
            vllm = self.analysis_results['vllm']
            phases = list(vllm['phase_distribution'].keys())
            percentages = list(vllm['phase_distribution'].values())
            html_content += f"\n                    const vllmPhaseCtx = document.getElementById('vllmPhaseChart').getContext('2d');\n                    new Chart(vllmPhaseCtx, {{\n                        type: 'bar',\n                        data: {{\n                            labels: {json.dumps(phases)},\n                            datasets: [{{\n                                label: '时间占比 (%)',\n                                data: {json.dumps(percentages)},\n                                backgroundColor: ['#9C27B0', '#3F51B5', '#00BCD4', '#8BC34A'],\n                                borderColor: ['#7B1FA2', '#303F9F', '#0097A7', '#689F38'],\n                                borderWidth: 1\n                            }}]\n                        }},\n                        options: {{\n                            scales: {{\n                                y: {{\n                                    beginAtZero: true,\n                                    max: 100,\n                                    title: {{\n                                        display: true,\n                                        text: '时间占比 (%)'\n                                    }}\n                                }},\n                                x: {{\n                                    title: {{\n                                        display: true,\n                                        text: 'vLLM阶段'\n                                    }}\n                                }} \n                            }},\n                            plugins: {{\n                                title: {{\n                                    display: true,\n                                    text: 'vLLM阶段时间分布'\n                                }} \n                            }}\n                        }}\n                    }});\n"
        
        html_content += "\n                </script>\n            </div>\n        </body>\n        </html>"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"交互式HTML报告已生成: {html_path}")
    
    def run(self):
        """运行完整分析流程"""
        print("开始加载性能数据...")
        self.load_data()
        
        print("开始性能分析...")
        self.analyze()
        
        print("生成性能报告...")
        self.generate_report()
        
        print("生成可视化图表...")
        self.generate_visualizations()
        
        print("生成交互式报告...")
        self.generate_interactive_report()
        
        print(f"\n分析完成！结果已保存到: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='昇腾NPU性能分析工具')
    parser.add_argument('input', help='输入文件或目录路径（支持msprof CSV和vLLM JSON）')
    parser.add_argument('-o', '--output', help='输出目录路径')
    
    args = parser.parse_args()
    
    profiler = AscendProfiler(args.input, args.output)
    profiler.run()

if __name__ == '__main__':
    main()
