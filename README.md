# Adversarial Anomaly Detection
Unsupervised Adversarial Anomaly Detection

# Task: Anomaly detection in multivariate time series variable.
- About 150 features in 1 input sample per 1 sec.
- t1: [f1, f2, ... , f150]
- t2: [f1, f2, ... , f150]
- ...
- tn: [f1, f2, ... , f150]

# SEADNet
- Sequence Embedding, Anomaly Detection Network

Note) I assumed you have 2 GPUs. If not, change gpu setting code.

Stacked LSTM + Deep Auto Encoder + Adversarial Discriminator + Latent Reencoder

Input
- Muti-variate time series (Feature dimension: 150, 1 Event per sec)

Output
- Anomaly score

Stacked LSTM
- Many to One
- Create sequence latent for sequence anomaly
- Sequence length: 20

Deep Auto Encoder
- Create local anomaly latent

Decoder
- Concat sequence latent and local latent
- Reconstruct using concatenated latent variable

Test
- Normal: mean 0.0, stdev 1.0, 100,000 samples, sigmoid standardazation for harder discrimination
- Anomaly: mean 1.0, stddev 1.0, 100 samples, sigmoid standardazation for harder discrimination
- Training only normal samples
- Discriminate anomaly samples at test time

# STADNet
- Spatio-Temporal, Anomaly Detection Network


Note) I assumed you have 2 GPUs. If not, change gpu setting code.


1D/2D Convolution + Adversarial Discriminator

Input
- Muti-variate time series (Feature dimension: 150, Window size: 20)
- Reshape to [B, 150, 20, 1]

Output
- Anomaly score

Deep Auto Encoder
- Create local anomaly latent

Decoder
- Concat sequence latent and local latent
- Reconstruct using concatenated latent variable

Test
- Normal: mean 0.0, stdev 1.0, 100,000 samples, sigmoid standardazation for harder discrimination
- Anomaly: mean 1.0, stddev 1.0, 100 samples, sigmoid standardazation for harder discrimination
- Training only normal samples
- Discriminate anomaly samples at test time

# Always welcome a nice idea: Code or Text what ever!
