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

序号 |英文标题                                                     |中文标题                                                     |发表年份 |期刊                                                         
---- |------------------------------------------------------------ |------------------------------------------------------------ |-------- |------------------------------------------------------------ 
1   |How Can Large Language Models Help Humans in Design And Manufacturing |大型语言模型如何助力人类的设计与制造                         |2023     |ArXiv                                                        
2   |Foundational Models for Fault Diagnosis of Electrical Motors |~电机故障诊断的基础模型                                      |2023     |ArXiv                                                        
3   |Fault Diagnosis and System Maintenance Based on Large Language Models and Knowledge Graphs |基于大语言模型和知识图谱的故障诊断与系统维护                 |2023     |2023 5th International Conference on Robotics, Intelligent Control and Artificial Intelligence (RICAI) 
4   |Evaluating the Performance of ChatGPT in the Automation of Maintenance Recommendations for Prognostics and Health Management |评估 ChatGPT 在预测性健康管理维护建议自动化中的表现          |2023     |Annual Conference of the PHM Society                         
5   |Industrial-generative pre-trained transformer for intelligent manufacturing systems |工业生成预训练变换器用于智能制造系统                         |2023     |IET Collaborative Intelligent Manufacturing                  
6   |Large Language Models in Fault Localisation                  |（软件代码故障定位）大型语言模型在故障定位中的应用           |2023     |ArXiv                                                        
7    |FD-LLM: Large Language Model for Fault Diagnosis of Machines |FD-LLM：用于机器故障诊断的大语言模型                         |2024     |ArXiv                                                        
8   |Multimodal Large Language Model-Based Fault Detection and Diagnosis in Context of Industry 4.0 |基于多模态大语言模型的工业 4.0 故障检测与诊断                |2024     |Electronics                                                  
9   |Large Language Models for Test-Free Fault Localization       |（软件代码故障定位）无需测试的大语言模型用于故障定位         |2024     |ICSE '24: Proceedings of the 46th IEEE/ACM International Conference on Software Engineering 
10   |LLM-R: A Framework for Domain-Adaptive Maintenance Scheme Generation Combining Hierarchical Agents and RAG |LLM-R：结合层次化代理和 RAG 的领域自适应维护方案生成框架     |2024     |ArXiv                                                        
11    |Diff-MTS: Temporal-Augmented Conditional Diffusion-Based AIGC for Industrial Time Series Toward the Large Model Era |Diff-MTS：面向大模型时代的工业时间序列的时序增强条件扩散生成模型 |2024     |IEEE Transactions on Cybernetics                             
12   |FaultExplainer Leveraging large language models for interpretable fault detection and diagnosis |故障解释器：利用大语言模型实现可解释的故障检测与诊断         |2024     |ArXiv                                                        
13   |✅ChatGPT-like large-scale foundation models for prognostics and health management A survey and roadmaps |面向预测与健康管理的类ChatGPT大规模基础模型：综述与路线图    |2024     |Reliability Engineering & System Safety                      
14   |DiagLLM: Multimodal Reasoning with Large Language Model for Explainable Bearing Fault Diagnosis |DiagLLM：基于大语言模型的多模态推理用于可解释的轴承故障诊断  |2024     |Electronics                                                  
15   |BearingFM Towards a foundation model for bearing fault diagnosis by domain knowledge and contrastive learning |~面向轴承故障诊断的基础模型：基于领域知识与对比学习方法      |2024     |International Journal of Production Economics                
16   |Empirical study on fine-tuning pre-trained large language models for fault diagnosis of complex systems |针对复杂系统故障诊断的预训练大语言模型微调实证研究           |2024     |Reliability Engineering & System Safety                      
17   |Empowering digital twins with large language models for global temporal feature learning |利用大语言模型为数字孪生赋能，以实现全球时序特征学习         |2024     |Journal of Manufacturing Systems                             
18    |Joint Knowledge Graph and Large Language Model for Fault Diagnosis and Its Application in Aviation Assembly |联合知识图谱与大语言模型的故障诊断及其在航空装配中的应用     |2024     |IEEE Transactions on Industrial Informatics                  
19   |✅Generative artificial intelligence and data augmentation for prognostic and health management Taxonomy, progress, and prospects |生成式人工智能与数据增强在预测性健康管理中的应用：分类体系、研究进展与前景展望 |2024     |Expert Systems with Applications                             
20    |Integrating LLMs for Explainable Fault Diagnosis in Complex Systems |~大语言模型在复杂系统中的可解释故障诊断集成                  |2024     |ArXiv                                                        
21   |Evaluation and Improvement of Fault Detection for Large Language Models |（软件代码）大型语言模型故障检测的评估与改进                 |2024     |ArXiv                                                        
22   |CausalKGPT: Industrial structure causal knowledge-enhanced large language model for cause analysis of quality problems in aerospace product manufacturing |面向航空航天产品制造质量问题因果分析的产业结构因果知识增强大语言模型 |2024     |Advanced Engineering Informatics                             
23   |Large Language Models for Fault Detection in Buildings’ HVAC Systems |大语言模型用于建筑暖通空调系统的故障检测                     |2024     |Energy Informatics: 4th Energy Informatics Academy Conference 
24   |Large-Scale Visual Language Model Boosted by Contrast Domain Adaptation for Intelligent Industrial Visual Monitoring |基于对比域适应的大规模视觉语言模型用于智能工业视觉监控       |2024     |IEEE Transactions on Industrial Informatics                  
25    |Brain-like Cognition-Driven Model Factory for IIoT Fault Diagnosis by Combining LLMs With Small Models |基于类脑认知驱动的模型工厂：结合大语言模型与小模型用于工业物联网故障诊断 |2024     |IEEE Internet of Things Journal                              
26    |ParInfoGPT: An LLM-based two-stage framework for reliability assessment of rotating machine under partial information |ParInfoGPT：基于大语言模型的两阶段部分信息下旋转机械可靠性评估框架 |2024     |Reliability Engineering & System Safety                      
27   |Explainable Fault Diagnosis of Control Systems Using Large Language Models |利用大语言模型实现控制系统的可解释故障诊断                   |2024     |2024 IEEE Conference on Control Technology and Applications (CCTA) 
28    |Joint KEmpirical study on fine-tuning pre-trained large language models for fault diagnosis of complex systems |大语言模型微调在复杂系统故障诊断中的实证研究                 |2024     |Reliability Engineering & System Safety                      
29   |Remaining Useful Life Prediction: A Study on Multidimensional Industrial Signal Processing and Efficient Transfer Learning Based on Large Language Models |剩余使用寿命预测：基于大语言模型的多维工业信号处理与高效迁移学习研究 |2024     |ArXiv                                                        
30   |GPT-based equipment remaining useful life prediction         |基于 GPT 的设备剩余使用寿命预测                              |2024     |Proceedings of the ACM Turing Award Celebration Conference   
31   |Consultation on Industrial Machine Faults with Large language Models |利用大语言模型进行工业机器故障咨询                           |2024     |ArXiv                                                        
32   |Blockchain-Enabled Large Language Models for Prognostics and Health Management Framework in Industrial Internet of Things |基于区块链的大语言模型用于工业物联网中的预测性健康管理框架   |2024     |International Conference on Blockchain and Trustworthy Systems 
33    |Large-Scale Visual Language Model Boosted by Contrast Domain Adaptation for Intelligent Industrial Visual Monitoring |基于对比域适应的大规模视觉语言模型用于智能工业视觉监控       |2024     |IEEE Transactions on Industrial Informatics                  
34   |Advancing multimodal diagnostics: Integrating industrial textual data and domain knowledge with large language models |推进多模态诊断：将工业文本数据与领域知识和大语言模型相结合   |2024     |Expert Systems with Applications                             
35   |Large Language Model Agents as Prognostics and Health Management Copilots |大语言模型代理作为预测性健康管理的辅助工具                   |2024     |Annual Conference of the PHM Society                         
36    |Large Model for Rotating Machine Fault Diagnosis Based on a Dense Connection Network With Depthwise Separable Convolution |基于密集连接网络与深度可分离卷积的旋转机械故障诊断大模型     |2024     |IEEE Transactions on Instrumentation and Measurement         
37   |✅Survey on Foundation Models for Prognostics and Health Management in Industrial Cyber-Physical Systems |工业网络物理系统中预测性健康管理的基础模型综述               |2024     |IEEE Transactions on Industrial Cyber-Physical Systems       
38   |SafeLLM: Domain-Specific Safety Monitoring for Large Language Models: A Case Study of Offshore Wind Maintenance |SafeLLM：特定领域的大语言模型安全监控——海上风电维护案例研究  |2024     |ArXiv                                                        
39   |A Survey on Intelligent Network Operations and Performance Optimization Based on Large Language Models |（智能网络优化）基于大语言模型的智能网络运维与性能优化研究综述 |2025     |IEEE Communications Surveys & Tutorials                      
40   |Domain-specific large language models for fault diagnosis of heating, ventilation, and air conditioning systems by labeled-data-supervised fine-tuning |基于标记数据监督微调的特定领域大语言模型用于暖通空调系统故障诊断 |2025     |Applied Energy                                               
41   |Large language model assisted fine-grained knowledge graph construction for robotic fault diagnosis |大语言模型辅助的机器人故障诊断细粒度知识图谱构建             |2025     |Advanced Engineering Informatics                             
42   |✅Large scale foundation models for intelligent manufacturing applications a survey |面向智能制造应用的大规模基础模型综述                         |2025     |Journal of Intelligent Manufacturing                         
43   |Multi large language model collaboration framework for few-shot link prediction in evolutionary fault diagnosis event graphs |用于演化故障诊断事件图小样本链接预测的多大型语言模型协作框架 |2025     |Journal of Process Control                                   
44   |Running Gear Global Composite Fault Diagnosis Based on Large Model |基于大模型的跑步装备全局复合故障诊断                         |2025     |IEEE Transactions on Industrial Informatics                  
45   |UniFault: A Fault Diagnosis Foundation Model from Bearing Data |UniFault: 基于轴承数据的故障诊断基础模型                     |2025     |ArXiv                                                        
46   |[Bridging the Gap: LLM-Powered Transfer Learning for Log Anomaly Detection in New Software Systems](https://nkcs.iops.ai/wp-content/uploads/2025/03/LogSynergy.pdf) |（日志异常检测）基于大语言模型的跨系统日志异常检测迁移学习方法 |2025     |International Conference on Data Engineering                 
47   |Intelligent Fault Diagnosis for CNC Through the Integration of Large Language Models and Domain Knowledge Graphs |通过大语言模型与领域知识图谱的融合实现数控机床的智能故障诊断 |2025     |Engineering                                                  
48   |✅A survey on potentials, pathways and challenges of large language models in new-generation intelligent manufacturing |新一代智能制造中大型语言模型的潜力、路径与挑战研究调查       |2025     |Robotics and Computer-Integrated Manufacturing               
49   |An End-to-End General Language Model (GLM)-4-Based Milling Cutter Fault Diagnosis Framework for Intelligent Manufacturing |基于端到端通用语言模型（GLM） - 4 的智能制造铣刀故障诊断框架 |2025     |Sensors                                                      
50   |面向旋转机械装备的智能故障诊断通用基础模型研究               |面向旋转机械装备的智能故障诊断通用基础模型研究               |2025     |西安交通大学学报                                             
51   |AdditiveLLM: Large Language Models Predict Defects in Additive Manufacturing |AdditiveLLM：大语言模型预测增材制造中的缺陷                  |2025     |ArXiv                                                        
52   |FaultGPT: Industrial Fault Diagnosis Question Answering System by Vision Language Models |FaultGPT：基于视觉语言模型的工业故障诊断问答系统             |2025     |ArXiv                                                        
53   |FD-LLM: Large language model for fault diagnosis of complex equipment |FD-LLM：用于复杂设备故障诊断的大语言模型                     |2025     |Advanced Engineering Informatics                             
54   |Leveraging Pre-Trained GPT Models for Equipment Remaining Useful Life Prognostics |利用预训练的 GPT 模型进行设备剩余使用寿命预测                |2025     |Electronics                                                  
55   |Language Pre-training Guided Masking Representation Learning for Time Series Classification |基于语言预训练引导的掩码表示学习用于时间序列分类             |2025     |Proceedings of the AAAI Conference on Artificial Intelligence 
56   |LLM-TSFD: An industrial time series human-in-the-loop fault diagnosis method based on a large language model |LLM-TSFD：基于大语言模型的工业时间序列人机协作故障诊断方法   |2025     |Expert Systems with Applications                             
57   |Pre-Trained Large Language Model Based Remaining Useful Life Transfer Prediction of Bearing |基于预训练大语言模型的轴承剩余使用寿命迁移预测               |2025     |ArXiv                                                        
58   |Leveraging large self-supervised time-series models for transferable diagnosis in cross-aircraft type Bleed Air System |利用大规模自监督时间序列模型实现跨机型引气系统的可迁移诊断   |2025     |Advanced Engineering Informatics                             
59   |Leveraging Large Language Models to Empower Bayesian Networks for Reliable Human-Robot Collaborative Disassembly Sequence Planning in Remanufacturing |利用大语言模型增强贝叶斯网络以实现可靠的再制造中人机协作拆卸序列规划 |2025     |IEEE Transactions on Industrial Informatics                  
60   |Personalizing Vision-Language Models With Hybrid Prompts for Zero-Shot Anomaly Detection |利用混合提示个性化视觉语言模型用于零样本异常检测             |2025     |IEEE Transactions on Cybernetics                             
61   |BearLLM: A Prior Knowledge-Enhanced Bearing Health Management Framework with Unified Vibration Signal Representation |BearLLM：基于先验知识增强的轴承健康管理框架与统一振动信号表示 |2025     |Proceedings of the AAAI Conference on Artificial Intelligence 
62   |The Interpretable Reasoning and Intelligent Decision-Making Based on Event Knowledge Graph With LLMs in Fault Diagnosis Scenarios |基于事件知识图谱与大语言模型的可解释推理与智能决策在故障诊断场景中的应用 |2025     |IEEE Transactions on Instrumentation and Measurement         
63   |✅面向机械设备通用健康管理的智能运维大模型                    |面向机械设备通用健康管理的智能运维大模型                     |2025     |机械工程学报                                                 
64   |✅An outline of Prognostics and health management Large Model: Concepts, Paradigms, and challenges |预测与健康管理大模型概述：概念、范式与挑战                   |2025     |Mechanical Systems and Signal Processing                     
65   |Channel attention residual transfer learning with LLM fine-tuning for few-shot fault diagnosis in autonomous underwater vehicle propellers |面向自主水下航行器螺旋桨小样本故障诊断的通道注意力残差迁移学习与大型语言模型微调方法 |2025     |Ocean Engineering                                            
66   |A knowledge-graph enhanced large language model-based fault diagnostic reasoning and maintenance decision support pipeline towards industry 5.0 |面向工业5.0的知识图谱增强型大语言模型故障诊断推理与维护决策支持流程 |2025     |International Journal of Production Research                 
67   |LLM-based framework for bearing fault diagnosis              |基于大语言模型的轴承故障诊断框架                             |2025     |Mechanical Systems and Signal Processing                     
68   ||||
69   ||||
70   ||||
71   ||||
72   ||||
73   ||||
74   ||||
75   ||||
76   ||||
77   ||||
78   ||||
79   ||||
80   ||||
81   ||||
82   ||||
83   ||||
84   ||||
85   ||||
86   ||||
87   ||||
88   ||||
89   ||||
90   ||||
91   ||||
92   ||||
93   ||||
94   ||||
95   ||||
96   ||||
97   ||||
98   ||||
99   ||||
100  ||||