# Large language model for Prognostics and health management

![Awesome](https://img.shields.io/badge/Awesome-Yes-brightgreen)  ![Last update](https://img.shields.io/badge/Last%20update-20250513-pink)  ![Paper number](https://img.shields.io/badge/Paper%20Number-67-red)

> 用于预测性维护与健康管理的大型语言模型

## AI解读

**Kimi大模型：**

> ### 研究现状
> 1. **研究热度持续上升**：从2023年到2025年，相关论文数量不断增加，表明大型语言模型在预测性维护与健康管理领域的应用研究热度在持续上升，越来越多的研究者开始关注这一方向。
>2. **多学科交叉融合**：论文发表在多个不同领域的期刊和会议上，如ArXiv、IEEE Transactions系列、Reliability Engineering & System Safety、Expert Systems with Applications等，这些期刊涵盖了计算机科学、机械工程、电气工程、控制科学等多个学科领域，体现了该领域多学科交叉融合的特点。
> 3. **研究重点集中在特定方法和技术**：
>     - **模型微调和优化**：许多论文关注如何通过Adapter Learning和Prompt Learning等方法对大型语言模型进行微调，以更好地适应PHM领域的特定任务和数据特点。
>     - **多模态融合**：研究者们尝试将不同类型的数据（如文本、图像、时间序列等）结合起来，利用多模态语料库来提高故障诊断和健康管理的准确性。
>     - **知识图谱与大模型结合**：将知识图谱与大型语言模型相结合，以利用领域知识增强模型的推理和诊断能力，成为研究热点之一。
>     - **幻觉问题解决**：针对大型语言模型的幻觉问题，研究者们探索如何将LLMs整合到一个系统中，通过与其他算法或机制的结合来减少幻觉现象，提高模型输出的可靠性。
> 4. **应用领域广泛**：应用对象涵盖了电机、轴承、旋转机械、暖通空调系统、增材制造设备、航空装配系统、工业物联网等广泛的工业设备和系统，表明大型语言模型在PHM领域的应用具有广泛的适用性和潜力。
> 5. **存在一些关键挑战**：尽管取得了一定的进展，但仍然存在一些关键挑战，如数据质量和数据扩充问题、模型的可解释性、幻觉问题、以及在不同领域和场景中的适配性等，这些问题有待进一步研究解决。
> 
>### 未来研究进展
> 1. **方法创新和优化**：
>     - **持续改进微调方法**：不断探索更有效的Adapter Learning和Prompt Learning等微调方法，以进一步提高大型语言模型在PHM任务中的性能。
>     - **多模态融合的深化**：加强对多模态数据融合的研究，开发更先进的技术来处理和整合不同类型的数据，充分挖掘多模态信息的潜力。
>    - **知识图谱与大模型的深度融合**：进一步深化知识图谱与大型语言模型的结合，探索如何更有效地利用领域知识来增强模型的推理和决策能力。
>     - **幻觉问题的解决**：持续探索解决幻觉问题的新方法和新机制，提高模型输出的准确性和可靠性，使其更适用于实际的PHM应用场景。
> 2. **拓展应用领域和场景**：
>     - **探索新的工业领域**：将大型语言模型应用于更多不同类型的工业设备和系统，如新型智能制造业中的先进设备、能源领域的复杂系统等，不断拓展其应用范围。
>     - **针对特定场景的定制化**：根据不同的工业场景和需求，开发定制化的大型语言模型解决方案，以满足特定领域的PHM要求。
> 3. **加强跨学科合作**：由于该领域的跨学科特点，未来需要进一步加强计算机科学、机械工程、电气工程、控制科学等不同学科领域的研究者之间的合作，共同攻克大型语言模型在PHM领域的关键问题，推动相关技术的发展。
>4. **提升模型的可解释性**：随着大型语言模型在PHM领域的应用越来越广泛，模型的可解释性将变得越来越重要。未来研究将致力于开发新的方法和技术，提高模型的可解释性，使工业领域的工程师和技术人员能够更好地理解和信任模型的输出。
> 5. **构建更高质量的数据集**：数据是大型语言模型的基础，未来需要构建更高质量、更大规模的PHM数据集和语料库，以支持模型的训练和优化。同时，研究如何对数据进行有效的筛选、扩充和预处理，以提高数据的质量和可用性。
>6. **开发标准化的评估指标和方法**：目前对于大型语言模型在PHM领域的性能评估还缺乏统一的标准和指标。未来需要开发标准化的评估指标和方法，以便更客观地比较和评价不同模型和方法的性能，为实际应用提供更可靠的参考依据。
> 7. **推动实际应用和产业化**：随着研究的不断深入和技术的逐渐成熟，未来将更加注重大型语言模型在PHM领域的实际应用和产业化发展。这包括与工业企业的合作，将研究成果转化为实际的PHM产品和解决方案，推动相关技术在工业界的广泛应用。

------

**Deepseek大模型：**

> #### 1. **期刊偏好与发表平台**  
> - **核心期刊/会议**：  
>   - **IEEE Transactions系列**（如 *IEEE Transactions on Industrial Informatics*, *IEEE Transactions on Cybernetics* 等）是发表大模型在PHM领域应用的核心平台，尤其关注工业场景的技术融合与系统集成。  
>   - **ArXiv** 作为预印本平台，收录了大量早期探索性研究（如微调方法、框架设计），反映该领域技术迭代迅速。  
>   - **Reliability Engineering & System Safety** 和 **Expert Systems with Applications** 侧重于PHM的实际应用与可靠性评估，偏好结合领域知识和大模型方法的实证研究。  
>   - **机械工程领域期刊**（如 *机械工程学报*）聚焦于设备级诊断与通用基础模型设计。  
>
> - **新兴趋势**：  
>   - 2025年的论文更多发表在 **应用能源类期刊**（如 *Applied Energy*）和 **智能制造类期刊**（如 *Journal of Intelligent Manufacturing*），表明研究逐渐向实际工业场景落地。  
>
> ---
>
> #### 2. **研究角度与切入点**  
> - **技术方法**：  
>   - **微调策略**（Adapter/Prompt Learning）是主流，旨在提升预训练模型对PHM领域数据的适应性。  
>   - **多模态融合**（文本、时间序列、图像）是重要方向，需解决异构数据对齐与联合表征问题。  
>   - **知识增强**（结合知识图谱、领域规则）被广泛用于提升模型的可解释性和可靠性。  
>   - **RAG技术**（检索增强生成）用于缓解幻觉问题，尤其在故障诊断的复杂推理场景中。  
>
> - **应用场景**：  
>   - **复杂系统诊断**（如航空装配、增材制造）要求模型具备跨设备迁移能力。  
>   - **剩余寿命预测**（RUL）是核心任务，需结合时序信号处理与大模型的时序建模能力。  
>   - **小样本与零样本学习**（如自主水下航行器螺旋桨）依赖迁移学习与提示工程。  
>
> ---
>
> #### 3. **应用对象与行业覆盖**  
> - **设备类型**：  
>   - **旋转机械**（轴承、电机、涡轮机）是研究重点（占比约30%），因其高故障率与工业重要性。  
>   - **复杂系统**（暖通空调、数控机床、航空装配系统）强调多组件协同诊断。  
>   - **新兴领域**（增材制造、工业物联网、区块链集成）体现技术前沿探索。  
>
> - **行业分布**：  
>   - **制造业**（40%）：涵盖传统制造（轴承/电机）与智能制造（数字孪生、工业4.0）。  
>   - **能源与建筑**（20%）：如暖通空调系统故障检测、海上风电维护。  
>   - **航空航天**（15%）：对可靠性要求极高，需结合因果推理与安全监控。  
>
> ---
>
> #### 4. **研究趋势与挑战**  
> - **趋势**：  
>   - **从方法探索到系统集成**：早期研究（2023-2024）聚焦微调与多模态方法，2025年更多关注端到端框架（如LLM-TSFD、BearLLM）。  
>   - **跨学科深化**：与知识图谱、贝叶斯网络、区块链等技术结合，形成复合解决方案。  
>   - **轻量化与可迁移**：研究转向小模型与大模型的协同（如模型工厂、层次化代理）。  
>
> - **挑战**：  
>   - **数据质量与标注成本**：高质量PHM数据集稀缺，需依赖半监督与自监督学习。  
>   - **模型可靠性**：幻觉问题在安全关键场景（如航空）尤为突出，需结合规则引擎与人类反馈。  
>   - **工业适配性**：实时性、算力限制与边缘部署需求推动模型压缩与硬件协同设计。  
>
> ---
>
> #### 5. **潜在价值与未来方向**  
> - **工业价值**：  
>   - 减少停机损失（预估提升设备可用性10-20%），降低维护成本（通过预测性策略替代定期检修）。  
>   - 支持个性化维护（如基于用户日志的故障咨询）与全球化运维（跨区域/机型迁移）。  
>
> - **学术方向**：  
>   - **通用基础模型**：构建覆盖多设备类型的PHM大模型（如UniFault、GLM-4）。  
>   - **因果推理**：探索故障根因分析与决策链生成（如CausalKGPT）。  
>   - **人机协作**：开发基于LLM的交互式诊断助手（如FaultGPT、PHM Copilots）。  
>
> ### 研究现状与未来研究进展总结
>
> ---
>
> #### **一、研究现状分析**  
> 通过分析67篇论文的标题、期刊和发表年份（2023-2025），可得出以下结论：  
>
> ##### 1. **技术方法聚焦方向**  
> - **主流技术**：  
>   - **微调与适配**：超半数论文（如FD-LLM、BearLLM）关注预训练大模型在PHM任务中的微调策略，包括Adapter Learning（适配器学习）和Prompt Learning（提示学习），以提升领域适应性。  
>   - **多模态融合**：约30%的论文（如DiagLLM、Multimodal LLM）探索文本、时序信号（振动/温度）、图像数据的联合建模，解决工业多源异构数据对齐问题。  
>   - **知识增强**：20%的论文（如CausalKGPT、Joint Knowledge Graph）结合知识图谱或领域规则，增强模型的可解释性和因果推理能力。  
>   - **检索增强生成（RAG）**：LLM-R、FaultGPT等论文利用外部知识库缓解模型幻觉问题，提升诊断建议的可靠性。  
>
> - **新兴技术**：  
>   - **轻量化部署**：2025年论文（如LLM-TSFD）提出大模型与小模型协同的“模型工厂”架构，适应边缘计算场景。  
>   - **时序建模**：Diff-MTS等研究将扩散模型与LLMs结合，提升工业时序信号（如振动数据）的生成与预测能力。  
>
> ##### 2. **应用场景与对象**  
> - **核心任务**：  
>   - **故障诊断**（60%论文）：覆盖旋转机械（轴承、电机）、暖通空调系统、航空装配等场景。  
>   - **剩余寿命预测（RUL）**（20%论文）：如GPT-based RUL Prediction，结合时序信号与大语言模型的长期依赖建模。  
>   - **小样本/零样本诊断**（15%论文）：针对自主水下航行器、增材制造等数据稀缺场景，依赖迁移学习与提示工程。  
>
> - **行业分布**：  
>   - **制造业**（45%）：传统设备（轴承/电机）与智能制造（数字孪生、工业4.0）并重。  
>   - **航空航天**（18%）：高可靠性需求驱动因果推理与安全监控研究（如SafeLLM）。  
>   - **能源与建筑**（15%）：聚焦暖通空调系统、海上风电维护等场景。  
>
> ##### 3. **期刊发表偏好**  
> - **高影响力平台**：  
>   - **IEEE Transactions系列**（占比25%）：如*IEEE Transactions on Industrial Informatics*、*IEEE Transactions on Cybernetics*，偏好工业场景的技术融合（如数字孪生、区块链）。  
>   - **ArXiv**（占比30%）：收录早期探索性研究（微调框架、多模态方法），反映技术快速迭代特性。  
>   - **应用类期刊**（占比20%）：如*Reliability Engineering & System Safety*、*Expert Systems with Applications*，侧重实证研究与可靠性验证。  
>   - **机械工程期刊**（如《机械工程学报》）：聚焦设备级通用基础模型（如UniFault）。  
>
> ---
>
> #### **二、未来研究进展与趋势**  
> ##### 1. **技术深化方向**  
> - **模型轻量化与边缘部署**：需解决工业场景的实时性与算力限制，推动模型压缩（如蒸馏技术）与硬件协同设计。  
> - **因果推理与根因分析**：结合因果发现算法（如CausalKGPT），从相关性诊断转向因果链生成。  
> - **动态持续学习**：适应设备老化、工况变化等动态环境，开发增量学习与在线微调框架。  
>
> ##### 2. **场景扩展方向**  
> - **跨领域迁移能力**：构建通用PHM基础模型（如GLM-4），支持不同设备类型（旋转机械→数控机床）和行业（制造→医疗）的迁移。  
> - **人机协作交互**：开发基于LLM的交互式诊断助手（如FaultGPT），结合人类专家反馈优化决策链。  
> - **安全关键场景**：如核电站、航天器，需融合形式化验证与安全约束，确保模型输出绝对可靠。  
>
> ##### 3. **数据与评估挑战**  
> - **低成本数据构建**：依赖自监督学习与合成数据（如Diff-MTS生成的时序信号），减少标注依赖。  
> - **标准化评估体系**：当前研究缺乏统一指标（如诊断准确性、实时性、可解释性评分），需建立跨场景Benchmark。  
>
> ##### 4. **跨学科融合趋势**  
> - **与新兴技术结合**：  
>   - **数字孪生**：利用LLMs生成虚拟设备的动态行为模拟（如Empowering Digital Twins）。  
>   - **区块链**：确保诊断记录的可追溯性与防篡改（如Blockchain-Enabled LLMs）。  
>   - **工业5.0**：探索人本协作（如人机协同拆卸规划）与可持续发展目标（如能耗优化）。  
>
> ---
>
> ### **总结**  
> - **现状**：当前研究以“微调+多模态+知识增强”为核心，聚焦高价值工业设备（轴承、航空系统）的故障诊断与寿命预测，IEEE Transactions系列和ArXiv为主要发表平台。  
> - **未来**：技术将向轻量化、因果推理、动态学习演进，场景扩展至跨领域迁移与安全关键系统，并深度融入数字孪生、区块链等新兴技术。  
> - **挑战**：数据质量、模型可靠性、评估标准化仍是瓶颈，需学术界与工业界联合攻关。
>
> ### 总结  
> 大型语言模型在PHM领域的研究呈现“技术-场景-行业”三重驱动：  
> 1. **技术驱动**：微调、多模态、知识增强为核心方法，RAG与轻量化是突破点。  
> 2. **场景驱动**：从单一设备诊断转向复杂系统与跨领域迁移。  
> 3. **行业驱动**：覆盖高价值工业场景（如航空、能源），追求可靠性与经济性平衡。  
> 未来需重点关注模型可靠性提升、低成本数据构建与工业级落地验证。

------

**文心一言大模型：**

> ### 研究现状总结
>
> 1. 时间线发展：
>    - **2023年**：研究起步阶段，主要集中于探索大型语言模型在故障诊断、维护推荐、智能制造系统等基础应用。例如，《How Can Large Language Models Help Humans in Design And Manufacturing》和《Foundational Models for Fault Diagnosis of Electrical Motors》等论文，为后续研究奠定了理论基础。
>    - **2024年**：研究进入快速发展期，研究范围显著扩展，涵盖多模态故障检测、领域自适应维护方案生成、数字孪生赋能、知识图谱与大型语言模型结合等前沿领域。此阶段发表的论文数量大幅增加，表明研究热度持续升温。
>    - **2025年（预测）**：研究将进一步深化，聚焦于语言模型在故障诊断中的潜力挖掘、新一代智能制造中的应用探索，以及跨学科技术的融合创新。尽管目前仅提取到部分2025年的论文信息，但已可预见该领域的研究将持续活跃。
> 2. 主要期刊与会议：
>    - 论文发表在多个权威期刊和会议上，包括但不限于ArXiv、Electronics、IEEE Transactions系列期刊、Reliability Engineering & System Safety等。这些平台的选择反映了研究领域的多学科交叉特性，以及研究成果的广泛传播和应用价值。
> 3. 研究热点聚焦：
>    - **大型语言模型的应用探索**：研究如何利用大型语言模型进行故障诊断、维护推荐、智能制造系统优化等。
>    - **多模态故障检测与诊断**：结合视觉、文本等多模态数据，提升故障检测的准确性和效率。
>    - **领域自适应与迁移学习**：研究如何使大型语言模型更好地适应特定领域的知识和任务需求。
>    - **数字孪生与知识图谱的融合**：探索数字孪生技术与大型语言模型的结合，以及知识图谱在故障诊断中的应用。
>
> ### 未来研究方向展望
>
> 1. 深化模型应用与优化：
>    - 进一步挖掘大型语言模型在故障检测和维修中的潜力，探索其在更复杂、更精细任务中的应用。
>    - 优化模型的微调方法和训练策略，提高模型在特定领域和任务中的性能表现。
> 2. 跨学科技术融合：
>    - 加强大型语言模型与其他技术（如数字孪生、知识图谱、强化学习等）的融合创新，推动智能制造和工业4.0的发展。
>    - 探索多模态数据融合的新方法，提升故障检测和诊断的全面性和准确性。
> 3. 领域自适应与泛化能力：
>    - 研究如何提高大型语言模型在跨领域、跨任务中的自适应能力和泛化性能。
>    - 探索迁移学习、领域自适应等技术在大型语言模型中的应用，降低模型在不同场景下的部署成本。
> 4. 可解释性与安全性研究：
>    - 加强大型语言模型在故障检测和诊断中的可解释性研究，提高模型的透明度和可信度。
>    - 研究模型的安全性和鲁棒性，确保模型在复杂工业环境中的稳定运行和可靠应用。
>
> 综上所述，大型语言模型在故障检测和维修领域的研究正呈现出蓬勃发展的态势。未来，随着技术的不断进步和应用场景的不断拓展，该领域的研究将更加深入和广泛，为智能制造和工业4.0的发展提供有力支持。

------

## 创新点角度

1. **微调大模型的方法。（Adapter Learning和Prompt Learning）**
2. **如何将数据输入到大模型中，结合NLP技术；主要利用大模型本身可以持续学习，强化学习的特点。（提示模板，数据混合等）**
3. **LLMs整合到一个系统，解决其幻觉问题。**
4. **馈入大模型的数据集，筛选扩充有意义的数据，构建语料库。**
5. **多模态语料库。**
6. **大模型作为assistant/agent。（辅助之前算法的诊断）**
7. **Retrieval-Augmented Generation**
8. **将向量数据转变为标量数据**

------

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
50   |Research on General Foundation Model of Intelligent Fault Diagnosis for Rotating Machines               |面向旋转机械装备的智能故障诊断通用基础模型研究               |2025     |西安交通大学学报                                             
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
63   |Research on Foundation Model for General Prognostics and Health Management of Machinery                    |✅面向机械设备通用健康管理的智能运维大模型                     |2025     |机械工程学报                                                 
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





------

其他Github项目推荐：

- LLM-based-PHM: https://github.com/CHAOZHAO-1/LLM-based-PHM     [Zhaochao](https://github.com/CHAOZHAO-1) 
- Awesome-TimeSeries-SpatioTemporal-LM-LLM: https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM
