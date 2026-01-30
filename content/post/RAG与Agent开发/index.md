+++
date = '2026-1-24T13:40:43+08:00'
title = 'RAG与Agent开发'

+++

## Ollama使用

### Ollama部署

正常下载运行即可，简单流畅

### 代码调用Ollama的本地模型

把`base_url`改为`http://localhost:11434/v1`

把`model`改为对应的本地模型名称，如：qwen3:4b

## OpenAI库

### OpenAI库基本使用

获取客户端对象：`api-key`和`base-url`

调用模型：

`model`：选用的模型

`messages`：提供给模型的消息

`system角色`：人设，幕后导演，影响所有回答

`assistant角色`：代表AI助手的回答，让AI记得之前说过什么

`user角色`：代表用户，发送问题、指令或需求

处理结果

```python
from openai import OpenAI

#1.获取client对象，OpenAI类对象
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

#2.调用模型
response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role":"system","content":"你是一个python编程专家，并且不说废话，简单回答"},
        {"role":"assistant","content":"好的，我是编程专家，并且话不多，你要问什么？"},
        {"role":"user","content":"输出1-10的数字，使用python代码"}
    ]
)
#3.处理结果
print(response.choices[0].message.content)
```

### OpenAI库的流式输出

  ```python
  from openai import OpenAI
  
  #1.获取client对象，OpenAI类对象
  client = OpenAI(
      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
  )
  
  #2.调用模型
  response = client.chat.completions.create(
      model="qwen3-max",
      messages=[
          {"role":"system","content":"你是一个python编程专家，并且话很多"},
          {"role":"assistant","content":"好的，我是编程专家，并且话很多，你要问什么？"},
          {"role":"user","content":"输出1-10的数字，使用python代码"}
      ],
      stream=True
  )
  #3.处理结果
  #print(response.choices[0].message.content)
  for chunk in response:
      print(
          chunk.choices[0].delta.content,
          end=" ",    #每段之间以空格分割
          flush=True
      )
  ```

### OpenAI库附带历史消息调用模型

```python
from openai import OpenAI

#1.获取client对象，OpenAI类对象
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

#2.调用模型
response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role":"system","content":"你是AI助理，回答很简洁"},
        {"role":"user","content":"小明有2条宠物狗"},
        {"role":"assistant","content":"好的"},
        {"role": "user", "content": "小红有3条宠物猫"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "总共有几只宠物"},
    ],
    stream=True
)
#3.处理结果
#print(response.choices[0].message.content)
for chunk in response:
    print(
        chunk.choices[0].delta.content,
        end=" ",    #每段之间以空格分割
        flush=True
    )
```

## 大模型Prompt工程

### 提示词优化基本思想

Zero-shot：是指在训练阶段不存在与测试阶段完全相同的类别，但是模型可以使用训练过的知识来推广到测试集中的新类别上。
在模型训练中，它可以通过已知类别特征来推理识别新类别。例如，模型已知马（四脚兽）、虎（有条纹）、熊猫（黑白色）的特征，在被告知 “斑马是四脚兽、有黑白色的条纹” 后，就能推理识别出斑马。
在提示词优化中，它是基于模型已有的预训练能力，不提供任何示例，仅通过语言描述任务的要求、目标和约束，让模型直接生成结果，核心是 “用语言定义任务，信任并调用模型的预训练知识”。比如，只通过指令 “请判断‘’包围的用户评论中的情感倾向，输出正面或负面。‘这款代餐鸡胸肉饱腹感很强，吃起来也不柴，很推荐！’”，模型就能完成情感判断。

Few-shot：是指模型在学习了一定类别大量数据的基础上，面对新类别时，仅需少量样本就能快速完成学习；当样本数量仅为 1 时，称为 one-shot learning（单样本学习），也属于 Few-shot 的特例。
在模型训练中，常采用相似度判断的方式。例如，基于少量企鹅样本，通过相似度计算来判断未知图片中是否包含企鹅。
在提示词优化中，它主要是提供少量示例，让模型参考示例的格式、逻辑来完成任务，以此提升回答的对齐精度。比如，给模型两个提取产品信息的示例后，它就能按相同的 JSON 格式，从新的产品描述中提取出产品名称和核心卖点。

### 提示词优化案例-金融文本分类任务

 ```python
 from conda.core.link import messages
 from openai import OpenAI
 
 # 1. 获取 client 对象，OpenAI 类对象（DashScope OpenAI 兼容模式）
 client = OpenAI(
     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
 )
 
 examples_data = {  # 示例数据
     "新闻报道": "今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。投资者密切关注美联储可能的政策调整，以适应市场的不确定性。",
     "财务报告": "本公司年度财务报告显示，在去年实现了稳步增长的盈利，同时资产负债表呈现强劲的状况。公司还增加了现金储备和管理层的有效战略执行，为未来的持续增长奠定基础。",
     "公司公告": "本公司宣布完成最新一轮并购交易，收购了一家在人工智能领域领先的公司。这一战略举措将有助于扩大我们的业务领域，提高市场竞争力。",
     "分析师报告": "最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。云计算、人工智能和数字化转型被认为是引领行业发展的关键因素，投资者应关注相关领域的投资机会。",
 }
 
 # 分类列表
 examples_types = ["新闻报道", "财务报告", "公司公告", "分析师报告"]
 
 # 提问数据
 questions = [
     "今日，央行发布公告宣布降低利率，以刺激经济增长。这一降息举措将影响贷款利率，并在未来几个季度内对金融市场产生影响。",
     "ABC公司今日发布公告称，已成功完成对XYZ公司股权的收购交易。本次交易是ABC公司在扩大业务范围、加强市场竞争力方面的重要举措。据悉，此次收购将进一步巩固ABC公司的市场地位。",
     "公司资产负债表显示，公司偿债能力强劲，现金流充足，为未来投资和扩张提供了坚实的财务基础。",
     "最新的分析报告指出，可再生能源行业预计将在未来几年经历持续增长，投资者应关注这一领域的投资机会。",
     "小明喜欢小新哟",
 ]
 
 """
 [
     {"role":"system","content":"你是金融专家，将文本分类为['新闻报道','财务报告','公司公告','分析师报告'], 不清楚的分类为'不清楚'"},
     {"role":"user","content":"今日，央行发布公告宣布降低......"},
     {"role":"assistant","content":"新闻报道"},
     ...
 ]
 """
 
 messages = [
     {"role":"system","content":"你是金融专家，将文本分类为['新闻报道','财务报告','公司公告','分析师报告'], 不清楚的分类为'不清楚'"},
 ]
 
 for key, value in examples_data.items():
     messages.append({"role":"user","content":value})
     messages.append({"role":"assistant","content":key})
 
 # 向模型提问
 for q in questions:
     response = client.chat.completions.create(
         model = "qwen3-max",
         messages = messages + [{"role":"user","content":f"按照示例，回答这段文本的分类类别:{q}"}]
     )
 
     print(response.choices[0].message.content)
 ```

### Json数据格式

Text文本，非结构化，抽取信息不方便
CSV（固定分隔符）文本，结构化，抽取信息方便，但是有一定风险
JSON 是一种带格式的字符串，它的核心作用是方便不同程序之间进行数据交互。

在 Python 中，JSON 和常见的数据结构可以直接对应：JSON 对象对应 Python 字典，JSON 数组对应包含多个字典的 Python 列表。

使用 Python 的 `json` 模块可以轻松实现两者的转换：
- `json.dumps(字典或列表, ensure_ascii=False)`：将 Python 数据结构转为 JSON 字符串，`ensure_ascii=False` 可以保证中文正常显示。
- `json.loads(json字符串)`：将 JSON 字符串转回为 Python 数据结构。

### 提示词优化案例-金融词信息抽取

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    #base_url="http://localhost:11434/v1"
)

examples_data = {
    "是": [
        ("公司ABC发布了季度财报，显示盈利增长。", "财报披露，公司ABC利润上升。"),
        ("公司ITCAST发布了年度财报，显示盈利大幅度增长。", "财报披露，公司ITCAST更赚钱了。")
    ],
    "不是": [
        ("黄金价格下跌，投资者抛售。", "外汇市场交易额创下新高。"),
        ("央行降息，刺激经济增长。", "新能源技术的创新。")
    ]
}

questions = [
    ("利率上升，影响房地产市场。", "高利率对房地产有一定的冲击。"),
    ("油价大幅度下跌，能源公司面临挑战。", "未来智能城市的建设趋势越加明显。"),
    ("股票市场今日大涨，投资者乐观。", "持续上涨的市场让投资者感到满意。")
]

"""
    {"role": "system",      "content": f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："},

    {"role": "user",        "content": "句子1：[公司ABC发布了季度财报，显示盈利增长。]句子2：[财报披露，公司ABC利润上升。]"},
    {"role": "assistant",   "content": "是"},
    {"role": "user",        "content": "句子1：[公司ITCAST发布了年度财报，显示盈利大幅度增长。]句子2：[财报披露，公司ITCAST更赚钱了。]"},
    {"role": "assistant",   "content": "是"},
    {"role": "user",        "content": "句子1：[黄金价格下跌，投资者抛售。]句子2：[外汇市场交易额创下新高。]"},
    {"role": "assistant",   "content": "不是"},
    {"role": "user",        "content": "句子1：[央行降息，刺激经济增长。]句子2：[新能源技术的创新。]"},
    {"role": "assistant",   "content": "不是"}, 

    {"role": "user",        "content": f"按照上述示例，回答这2个句子的情况。句子1: [...]，句子2: [...]"}
"""

messages = [
    {"role": "system",
     "content": f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："},
]

for key, value in examples_data.items():
    for t in value:
        messages.append(
            {"role": "user", "content": f"句子1：[{t[0]}]，句子2：[{t[1]}]"}
        )
        messages.append(
            {"role": "assistant", "content": key}
        )

for q in questions:
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages + [{"role": "user", "content": f"句子1：[{q[0]}]，句子2：[{q[1]}]"}]
    )

    print(response.choices[0].message.content)
```

## RAG开发

### LangChain的简介

它是 Python 的第三方库，是开发大语言模型（LLM）相关业务功能的 “集成工具集”，可以简化 LLM 应用的开发流程，后续也是学习 RAG（检索增强生成）开发的核心框架。
核心功能 API，LangChain 提供的能力包括：
提示词优化：辅助提升提示词效果的工具接口；
模型调用：支持对接各类 LLM 模型的统一接口；
会话记忆：实现对话上下文的存储与管理；
文档管理分析：对各类文档进行处理、检索、分析的功能；
Agent 智能体构建：支持开发具备自主决策能力的智能体；
功能链式执行：将多个功能按逻辑串联执行的能力。

### LangChain环境部署

### RAG工作流程

这是关于RAG（检索增强生成）的核心说明，可整理为以下结构化内容：

### RAG的核心定义
RAG是一种优化大模型输出的技术：在向模型提问前，先从已有知识库/文档中检索相关信息，将这些信息整合到提问中，让模型基于更精准、信息量更足的输入给出回答。

RAG的工作流程：

1. **离线流程（知识预处理，对应“R”）**
   需预先处理知识库：将文档私有知识加载后，分割为合适长度的片段，再通过向量化（转为向量表示）存储到向量数据库中，完成检索前的准备。
2. **在线流程（增强+生成，对应“A+G”）**
   用户提问后，先从向量数据库中检索相关知识片段，将这些片段与原问题融合为新的Prompt，再交给大模型生成回答。

RAG的核心价值：

1. **解决知识实效性问题**：弥补大模型训练数据的“截止时间”限制，可接入最新文档（如财报、政策），让模型输出贴合最新信息。
2. **降低模型幻觉**：回答基于检索到的事实性资料，而非模型自身记忆，大幅减少编造信息的概率。
3. **成本与效率优势**：无需重新训练模型，仅需更新知识库即可实现知识迭代，相比微调（Fine-tuning）成本更低、效率更高。

### 向量的基本概念

`text-embedding-v1`文本嵌入模型

文本向量是文本的“数学身份证”，它通过将文字的语义信息转换为固定长度的数字列表，让计算机能够理解文字的含义，并基于这些数字进行语义相似度计算。

文本向量的关键要点

1. **向量的生成**
   通过“文本嵌入模型”（如text-embedding-v1）完成“文本→向量”的转换（即文本嵌入过程）。
2. **向量的匹配方式**
   借助算法（如余弦相似度）计算不同向量之间的距离，以此判断文本语义的相似程度。
3. **向量的维度**
   - 维度代表模型用多少个“抽象语义特征”描述文本，每个维度对应一个特征的强度；
   - 维度越多，语义匹配的精准度越高，但同时会增加计算和存储的性能压力，需根据场景平衡精度与成本。

### 余弦相似度

### LangChain调用大语言模型

### LangChain聊天模型应用

### LangChain嵌入模型应用

### LangChain通用提示词模板

### Fewshot提示词模板

### ChatPromptTemplate





























