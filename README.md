# Multi-label_Emotions
### Multi-label emotion classification for complex classification systems on GoEmotions

💻 **The final project including complete code is in the 'Multi-label_Emotions' folder** :books:
___

*Abstract:* 
> With the development of Internet technology and the popularity of social media platforms, more and more active netizens flood into these platforms to express various opinions. These comments contain a large number of values, emotions and other valuable information to be mined, and the analysis of these information can be applied to depression
emotion recognition, public opinion detection and other useful tasks. A comment often
contains multiple emotions, and often contains special language such as emoticons, abbreviations and slang, which makes multi-label emotion classification with a large number of emotions on social media still a challenging task.
> 
> The purpose of this paper is to use the correlation of emotion categories from the
hierarchical relationship of different emotions to establish a multi-level model to obtain
better classification results than a single level under a complex system with many emotion
categories. The strategy of multi-label classification in this paper adopts BR method, which
models a multi-label classification task as a number of single-label classification tasks, and
uses a deep learning-based model to construct a binary classifier for each level of emotion. In terms of classifier selection, the classification effect of five commonly used deep
learning models including CNN, LSTM, BiLSTM, self-Attention, and Attention+BiLSTM
under the same emotion is analyzed and compared, and the model with the best effect is
selected as the base model to construct a multi-level model. In addition, aiming at the
empty label problem predicted by the multi-level model, in order to maximize the
advantages of multi-level, the multi-level model was improved and optimized by using
Integer Linear Programming (ILP). Finally, on the GoEmotions dataset, we divided three
levels of emotion systems, from coarse to fine granularity, which are ternary emotion
system, ekman emotion system, and 27 fine-grained emotion systems, totaling 36 emotions. For each emotion, we trained the above five base models, totaling 180 models.
> 
> Through comparative experiments, the classification effect of the five base models on
36 emotions is similar, and the effect of LSTM is slightly better. LSTM is used as the base
model to construct a single-level model, a multi-level model, and an improved multi-level model using ILP only in the third layer, on the test set of 5427 samples. The multi-level
model improves the accuracy and F1 value by about 0.05 compared with the single-level
model. The effect of the improved multi-level model is also improved compared with the
multi-level model. The model using ILP only in the third layer improves the accuracy by
about 0.02 and F1 by about 0.03 compared with the original multi-level model, and the
model using ILP from the second layer improves the accuracy and F1 by about 0.05
compared with the original multi-level model. Compared with the single layer, the
improvement is 0.1. Finally, through the comparison of multi-label prediction and actual
viewing analysis, the effectiveness of the multi-level model compared with the single-level
model is further verified.
> 
> **Keywords:** Multi-label emotion classification; Deep learning; Multi-level; Integer linear
programming
