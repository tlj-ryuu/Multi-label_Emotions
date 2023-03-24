# 自然语言NLP基础知识

**BERT**：Bidirectional Encoder Representations from Transformers，基于transformer的双向编码表示技术。是用于自然语言处理的预训练技术，由Google提出。2018年，雅各布·德夫林和同事创建并发布了BERT。

BERT的核心部分是一个[Transformer模型](https://zh.m.wikipedia.org/wiki/Transformer模型)，其中编码层数和自注意力头数量可变。结构与Vaswani等人的实现几乎“完全一致”。

BERT在两个任务上进行预训练： 语言模型（15%的token被掩盖，BERT需要从上下文中进行推断）和下一句预测（BERT需要预测给定的第二个句子是否是第一句的下一句）。训练完成后，BERT学习到单词的上下文嵌入。代价昂贵的预训练完成后，BERT可以使用较少的资源和较小的数据集在下游任务上进行微调，以改进在这些任务上的性能。

https://zh.m.wikipedia.org/zh-hans/BERT



**Transformer模型**（直译为“变换器”）是一种采用[自注意力机制](https://zh.m.wikipedia.org/wiki/注意力机制)的[深度学习](https://zh.m.wikipedia.org/wiki/深度学习)模型，这一机制可以按输入数据各部分重要性的不同而分配不同的权重。该模型主要用于[自然语言处理](https://zh.m.wikipedia.org/wiki/自然语言处理)（NLP）与[计算机视觉](https://zh.m.wikipedia.org/wiki/计算机视觉)（CV）领域。[[1\]](https://zh.m.wikipedia.org/zh-hans/Transformer模型#cite_note-1)

与[循环神经网络](https://zh.m.wikipedia.org/wiki/循环神经网络)（RNN）一样，Transformer模型旨在处理自然语言等顺序输入数据，可应用于[翻译](https://zh.m.wikipedia.org/wiki/统计机器翻译)、文本摘要等任务。而与RNN不同的是，Transformer模型能够一次性处理所有输入数据。注意力机制可以为输入序列中的任意位置提供上下文。如果输入数据是自然语言，则Transformer不必像RNN一样一次只处理一个单词，这种架构允许更多的[并行计算](https://zh.m.wikipedia.org/wiki/并行计算)，并以此减少训练时间。[[2\]](https://zh.m.wikipedia.org/zh-hans/Transformer模型#cite_note-:0-2)

Transformer模型于2017年由[谷歌大脑](https://zh.m.wikipedia.org/wiki/谷歌大脑)的一个团队推出[[2\]](https://zh.m.wikipedia.org/zh-hans/Transformer模型#cite_note-:0-2)，现已逐步取代[长短期记忆](https://zh.m.wikipedia.org/wiki/长短期记忆)（LSTM）等RNN模型成为了NLP问题的首选模型。[[3\]](https://zh.m.wikipedia.org/zh-hans/Transformer模型#cite_note-wolf2020-3)并行化优势允许其在更大的数据集上进行训练。这也促成了[BERT](https://zh.m.wikipedia.org/wiki/BERT)、[GPT](https://zh.m.wikipedia.org/wiki/OpenAI)等预训练模型的发展。这些系统使用了[维基百科](https://zh.m.wikipedia.org/wiki/维基百科)、[Common Crawl](https://zh.m.wikipedia.org/w/index.php?title=Common_Crawl&action=edit&redlink=1)（英语：[Common Crawl](https://en.wikipedia.org/wiki/Common_Crawl)）等大型语料库进行训练，并可以针对特定任务进行微调。[[4\]](https://zh.m.wikipedia.org/zh-hans/Transformer模型#cite_note-:6-4)[[5\]](https://zh.m.wikipedia.org/zh-hans/Transformer模型#cite_note-:7-5)

https://zh.m.wikipedia.org/zh-hans/Transformer%E6%A8%A1%E5%9E%8B



**注意力机制**（英语：attention）是[人工神经网络](https://zh.m.wikipedia.org/wiki/人工神经网络)中一种模仿[认知注意力](https://zh.m.wikipedia.org/wiki/注意)的技术。这种机制可以增强神经网络输入数据中某些部分的权重，同时减弱其他部分的权重，以此将网络的关注点聚焦于数据中最重要的一小部分。数据中哪些部分比其他部分更重要取决于上下文。可以通过[梯度下降法](https://zh.m.wikipedia.org/wiki/梯度下降法)对注意力机制进行训练。



https://zh.m.wikipedia.org/zh-hans/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6





**迁移学习**(Transfer learning) ：顾名思义就是就是把已学训练好的模型参数迁移到新的模型来帮助新模型训练。考虑到大部分数据或任务是存在相关性的，所以通过迁移学习我们可以将已经学到的模型参数（也可理解为模型学到的知识）通过某种方式来分享给新模型从而加快并优化模型的学习效率不用像大多数网络那样从零学习（starting from scratch，tabula rasa）。



**baseline**：主要关注自己提出的方法，比如最原始最简单的方法出来的结果（参照物）。然后再这个基础上改进，增加各种组件，可以看出提升了多少，通过baseline我们可以知道这个方法能不能work，有多少提升。

**benchmark**：主要对比别人的方法，这个方法不一定是最好的，但一定是最有代表性且被广泛认可的（一种标准和规则）。其所用的数据就是 benchmark data ，其方法就是benchmark method，你提出的方法的在benchmark data上得出的结果与benchmark method的结果对比才能知道你的方法是否足够好。

当然有些时候文章中的baseline指代其他论文提出的方法，所以阅读论文的时候要结合语境，灵活理解。



**tokenization**，也叫word segmentation,是一种操作，它按照特定需求，把文本切分成一个字符串序列(其元素一般称为**token**，或者叫词语)。一般来说，我们要求序列的元素有一定的意义，比如“text mining is time-consuming”需要处理成"text mining/ is/ time-consuming"，其中"text mining"表示"文本挖掘"。



**词袋向量**：

忽略词的顺序和语法，混合在一个“袋子”中，每个句子或每篇短文档对应一个“袋子”，可以用来概括文档的本质内容。



**n-gram**：可以忽略如“at the”这种经常出现但是没意义的，可以设定25%阈值过滤。

停用词：



NLTK提供了很多有用的nlp手段，如分词，比较合理的停用词，以及分成n-gram之类的以及词干还原工具，词性归并。



词干还原以及大小写转换能够使得词汇表减少，是词汇表归一化的手段。



**TF-IDF**：可以用于量化词项的重要程度



词向量

计算词向量的方法有两种，一般都是大型公司已经预训练好的，无需我们再去训练，对于特定领域使用相应预训练好的就行。

输入词项的独热向量表示与权重的点积代表词向量嵌入





RNN



双向RNN



LSTM



seq2seq



attention机制

上面几个自然语言处理实战上都有





梯度消失问题不仅影响[多层](https://zh.wikipedia.org/wiki/深度学习)[前馈网络](https://zh.wikipedia.org/wiki/前馈神经网络)，[[3\]](https://zh.wikipedia.org/zh-hans/梯度消失问题#cite_note-3)还影响[循环网路](https://zh.wikipedia.org/w/index.php?title=循环神经网路&action=edit&redlink=1)。[[4\]](https://zh.wikipedia.org/zh-hans/梯度消失问题#cite_note-4)循环网路是通过将前馈网路深度展开来训练，在网路处理的输入序列的每个时间步骤中，都会产生一个新的层。

当所使用的激励函数之导数可以取较大值时，则可能会遇到相关的**梯度爆炸问题（exploding gradient problem）**。
