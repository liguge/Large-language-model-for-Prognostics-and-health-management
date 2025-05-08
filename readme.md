# Large language model for Prognostics and health management

> 用于预测性维护与健康管理的大型语言模型

## 创新点角度

1. **微调大模型的方法。（Adapter Learning和Prompt Learning）**
2. **如何将数据输入到大模型中，结合NLP技术；主要利用大模型本身可以持续学习，强化学习的特点。（提示模板，数据混合等）**
3. **LLMs整合到一个系统，解决其幻觉问题。**
4. **馈入大模型的数据集，筛选扩充有意义的数据，构建语料库。**
5. **多模态语料库。**
6. **大模型作为assistant。（辅助之前算法的诊断，主要用作特征提取）**

## 论文整理

| 序号 | 英文标题                                                     | 中文标题                                                     |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Integrating LLMs for Explainable Fault Diagnosis in Complex Systems | 大语言模型在复杂系统中的可解释故障诊断集成                   |
| 2    | Joint Knowledge Graph and Large Language Model for Fault Diagnosis and Its Application in Aviation Assembly | 联合知识图谱与大语言模型的故障诊断及其在航空装配中的应用     |
| 3    | Joint KEmpirical study on fine-tuning pre-trained large language models for fault diagnosis of complex systems | 大语言模型微调在复杂系统故障诊断中的实证研究                 |
| 4    | Large-Scale Visual Language Model Boosted by Contrast Domain Adaptation for Intelligent Industrial Visual Monitoring | 基于对比域适应的大规模视觉语言模型用于智能工业视觉监控       |
| 5    | Large Model for Rotating Machine Fault Diagnosis Based on a Dense Connection Network With Depthwise Separable Convolution | 基于密集连接网络与深度可分离卷积的旋转机械故障诊断大模型     |
| 6    | Diff-MTS: Temporal-Augmented Conditional Diffusion-Based AIGC for Industrial Time Series Toward the Large Model Era | Diff-MTS：面向大模型时代的工业时间序列的时序增强条件扩散生成模型 |
| 7    | FD-LLM: LARGE LANGUAGE MODEL FOR FAULT DIAGNOSIS OF MACHINES | FD-LLM：用于机器故障诊断的大语言模型                         |
| 8    | ParInfoGPT: An LLM-based two-stage framework for reliability assessment of rotating machine under partial information | ParInfoGPT：基于大语言模型的两阶段部分信息下旋转机械可靠性评估框架 |
| 9    | Brain-like Cognition-Driven Model Factory for IIoT Fault Diagnosis by Combining LLMs With Small Models | 基于类脑认知驱动的模型工厂：结合大语言模型与小模型用于工业物联网故障诊断 |
| 10   | Industrial-generative pre-trained transformer for intelligent manufacturing systems | 工业生成预训练变换器用于智能制造系统                         |
| 11   | Survey on Foundation Models for Prognostics and Health Management in Industrial Cyber-Physical Systems | 工业网络物理系统中预测性健康管理的基础模型综述               |
| 12   | SafeLLM: Domain-Specific Safety Monitoring for Large Language Models: A Case Study of Offshore Wind Maintenance | SafeLLM：特定领域的大语言模型安全监控——海上风电维护案例研究  |
| 13   | Domain-specific large language models for fault diagnosis of heating, ventilation, and air conditioning systems by labeled-data-supervised fine-tuning | 基于标记数据监督微调的特定领域大语言模型用于暖通空调系统故障诊断 |
| 14   | GPT-based equipment remaining useful life prediction         | 基于 GPT 的设备剩余使用寿命预测                              |
| 15   | Remaining Useful Life Prediction: A Study on Multidimensional Industrial Signal Processing and Efficient Transfer Learning Based on Large Language Models | 剩余使用寿命预测：基于大语言模型的多维工业信号处理与高效迁移学习研究 |
| 16   | Advancing multimodal diagnostics: Integrating industrial textual data and domain knowledge with large language models | 推进多模态诊断：将工业文本数据与领域知识和大语言模型相结合   |
| 17   | Blockchain-Enabled Large Language Models for Prognostics and Health Management Framework in Industrial Internet of Things | 基于区块链的大语言模型用于工业物联网中的预测性健康管理框架   |
| 18   | Large Language Model Agents as Prognostics and Health Management Copilots | 大语言模型代理作为预测性健康管理的辅助工具                   |
| 19   | Evaluating the Performance of ChatGPT in the Automation of Maintenance Recommendations for Prognostics and Health Management | 评估 ChatGPT 在预测性健康管理维护建议自动化中的表现          |
| 20   | Consultation on Industrial Machine Faults with Large language Models | 利用大语言模型进行工业机器故障咨询                           |
| 21   | LLM-TSFD: An industrial time series human-in-the-loop fault diagnosis method based on a large language model | LLM-TSFD：基于大语言模型的工业时间序列人机协作故障诊断方法   |
| 22   | Fault Diagnosis and System Maintenance Based on Large Language Models and Knowledge Graphs | 基于大语言模型和知识图谱的故障诊断与系统维护                 |
| 23   | Explainable Fault Diagnosis of Control Systems Using Large Language Models | 利用大语言模型实现控制系统的可解释故障诊断                   |
| 24   | Large Language Models for Fault Detection in Buildings’ HVAC Systems | 大语言模型用于建筑暖通空调系统的故障检测                     |
| 25   | Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis | 大语言模型辅助的机器人故障诊断细粒度知识图谱构建             |
| 26   | AdditiveLLM: Large Language Models Predict Defects in Additive Manufacturing | AdditiveLLM：大语言模型预测增材制造中的缺陷                  |
| 27   | FaultGPT: Industrial Fault Diagnosis Question Answering System by Vision Language Models | FaultGPT：基于视觉语言模型的工业故障诊断问答系统             |
| 28   | DiagLLM: Multimodal Reasoning with Large Language Model for Explainable Bearing Fault Diagnosis | DiagLLM：基于大语言模型的多模态推理用于可解释的轴承故障诊断  |
| 29   | FD-LLM: Large language model for fault diagnosis of complex equipment | FD-LLM：用于复杂设备故障诊断的大语言模型                     |
| 30   | Leveraging Pre-Trained GPT Models for Equipment Remaining Useful Life Prognostics | 利用预训练的 GPT 模型进行设备剩余使用寿命预测                |
| 31   | Language Pre-training Guided Masking Representation Learning for Time Series Classification | 基于语言预训练引导的掩码表示学习用于时间序列分类             |
| 32   | Multimodal Large Language Model-Based Fault Detection and Diagnosis in Context of Industry 4.0 | 基于多模态大语言模型的工业 4.0 故障检测与诊断                |
| 33   | LLM-based framework for bearing fault diagnosis              | 基于大语言模型的轴承故障诊断框架                             |
| 34   | Pre-Trained Large Language Model Based Remaining Useful Life Transfer Prediction of Bearing | 基于预训练大语言模型的轴承剩余使用寿命迁移预测               |
| 35   | LLM-R: A Framework for Domain-Adaptive Maintenance Scheme Generation Combining Hierarchical Agents and RAG | LLM-R：结合层次化代理和 RAG 的领域自适应维护方案生成框架     |
| 36   | Leveraging large self-supervised time-series models for transferable diagnosis in cross-aircraft type Bleed Air System | 利用大规模自监督时间序列模型实现跨机型引气系统的可迁移诊断   |
| 37   | Large-Scale Visual Language Model Boosted by Contrast Domain Adaptation for Intelligent Industrial Visual Monitoring | 基于对比域适应的大规模视觉语言模型用于智能工业视觉监控       |
| 38   | Leveraging Large Language Models to Empower Bayesian Networks for Reliable Human-Robot Collaborative Disassembly Sequence Planning in Remanufacturing | 利用大语言模型增强贝叶斯网络以实现可靠的再制造中人机协作拆卸序列规划 |
| 39   | Personalizing Vision-Language Models With Hybrid Prompts for Zero-Shot Anomaly Detection | 利用混合提示个性化视觉语言模型用于零样本异常检测             |
| 40   | BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation | BearLLM：基于先验知识增强的轴承健康管理框架与统一振动信号表示 |
| 41   | FaultExplainer Leveraging large language models for interpretable fault detection and diagnosis | 故障解释器：利用大语言模型实现可解释的故障检测与诊断         |
| 42   | The Interpretable Reasoning and Intelligent Decision-Making Based on Event Knowledge Graph With LLMs in Fault Diagnosis Scenarios | 基于事件知识图谱与大语言模型的可解释推理与智能决策在故障诊断场景中的应用 |
| 43   | 面向机械设备通用健康管理的智能运维大模型}}$ | 面向机械设备通用健康管理的智能运维大模型                     |
| 44   | An outline of Prognostics and health management Large Model: Concepts, Paradigms, and challenges | 预测与健康管理大模型概述：概念、范式与挑战                   |
| 45   | Channel attention residual transfer learning with LLM fine-tuning for few-shot fault diagnosis in autonomous underwater vehicle propellers | 面向自主水下航行器螺旋桨小样本故障诊断的通道注意力残差迁移学习与大型语言模型微调方法 |
| 46   | An End-to-End General Language Model (GLM)-4-Based Milling Cutter Fault Diagnosis Framework for Intelligent Manufacturing | 基于端到端通用语言模型（GLM） - 4 的智能制造铣刀故障诊断框架 |
| 47   | ChatGPT-like large-scale foundation models for prognostics and health management A survey and roadmaps | 面向预测与健康管理的类ChatGPT大规模基础模型：综述与路线图    |
