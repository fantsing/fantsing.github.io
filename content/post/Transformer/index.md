+++
date = '2025-10-15T13:52:23+08:00'
title = 'Transformer'

+++

Transformer is a typical Seq2Seq model. 



Attention mechanism: Selectively focus on the important parts and ignore the unimportant ones. 



Embedding: The process of mapping textual information into numerical information. 



Queries (query vectors), Keys (key vectors), Values (value vectors) 



Step 1 is to create three new vectors q, k, and v for each input vector of the encoder. 



Step 2 is to calculate a correlation score (Score)
$$
Score_{1.1}=\mathbf{q}_1 \cdot \mathbf{k}_1=|q_1||k_1|\cos\theta\\
Score_{1.2}=\mathbf{q}_1 \cdot \mathbf{k}_2=|q_1||k_2|\cos\theta\\
...
$$
Step 3 is to divide the correlation score (Score) by 8, which will make the gradient during the model training more stable. 



Step 4 involves obtaining the weight factors through Softmax. 



Step 5 is to multiply each value vector by the corresponding Softmax score.
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^{\mathrm{T}}}{\sqrt{d_k}} \right) V
$$
Multi-heads self-attention mechanism 



Encoder-decoder, initially developed to solve the problem of text translation 



Cross-attention mechanism
