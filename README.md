# AdversarialAnomalyDetection
Unsupervised Adversarial Anomaly Detection

SEADNet
- Sequence Embedding, Anomaly Detection Network

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
- Normal: mean 0.0, stdev 1.0, 100,000 samples
- Anomaly: mean 1.0, stddev 1.0, 100 samples
- Training only normal samples
- Discriminate anomaly samples at test time

Always welcom a nice idea: Code or Text what ever!
