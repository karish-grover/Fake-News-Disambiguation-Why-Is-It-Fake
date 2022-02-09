# Fake News Disambiguation: Why Is It Fake?
This project considers the fake news detection problem under a more realistic scenario on social media. Given just a source short-text tweet, we aim to predict whether the given tweet is fake or not. We further aim to gen- erate an explanation for this prediction. This paper presents **Attention-enhanced Multi-channel Recurrent Convolutional Network (AMRCN)**, for explainable fake news detection. We explain our final predictions by **highlighting the essential words in the short tweet text**. Experimental results and extensive ablation studies show that our model outperforms the baseline systems on two benchmarking datasets.

**Here is the link to the blog post**: https://medium.com/@karish19471/fake-news-disambiguation-why-is-it-fake-e65c049181e5

## Problem Statement
Given the short text tweet content, we aim to classify this tweet as fake or real, i.e., binary classification. Further, we aim to explain why this tweet is **fake/actual by highlighting tokens/phrases from the tweet content** that contribute relatively more to the final prediction.

## Dataset Collection and Preprocessing
We utilize two well-known datasets compiled by (Ma et al., 2017), `Twitter15` and `Twitter16`, for our work. These datasets contain source tweets and their corresponding sequences of retweet users. These datasets are balanced towards the classes and consist of short length tweets. We remove the stop words from the tweets, replace all URLs with the token `URL`, and stem the tweets in the dataset for proceeding with the experiments. You can download the datasets from the [link](https://drive.google.com/drive/folders/1gFJsimozHJpyfE5ClCIgSftGrhtd1QWr?usp=sharing).

## Baselines
We evaluate the dataset collected on several baselines. The baseline models have been implemented in the notebook [ML_Project_Baselines.ipynb](ML_Project_Baselines.ipynb).


- **NB** (Multinomial Naive Bayes)
- **LR** (Logistic Regression)
- **DT** (Decision Trees)
- **SVM** (Support Vector Machines)
- **RF** (Random Forest - Bagging)
- **XGB** (XGBoost - Boosting)
- **CNN**(Convolutional Neural Networks)
- **LSTM** (Long Short Term Memory cells)
- **GRU** (Gated Recurrent Units)
- **Bi-RNN** (Bi-directional Recurrent Neural Networks)
- **RCNN** (Recurrent Convolutional Neural Networks). 

The neural network baselines have an embedding layer
containing the `100D` Glove embeddings of the tokens. We use F1 score,
accuracy, precision and recall metrics to get a better idea of the
models' performance on both the datasets. All the models were trained on
a `80:10:10 train-test-val` split of the dataset.
<p align ="center">
<img width="594" alt="Screenshot 2022-02-09 at 6 22 26 PM" src="https://user-images.githubusercontent.com/64140048/153204867-0741557b-6378-4767-a88c-cdaa0ba0cbb5.png"></p>


## Methodology
The model architecture is shown below. It can be divided into two broad parts: (1) CNN-based Representation, (2) Attention enhanced Word-level Encoder, (3) Multiple Channels, and (4) Explainability of the Prediction. To know more about the model architecture, refer to the [paper](ML__PG__Project.pdf). The model has been implemented in the notebook [ML_Project_Final.ipynb](ML_Project_Final.ipynb).

<p align="center">
<img width="636" alt="Screenshot 2022-02-09 at 6 37 39 PM" src="https://user-images.githubusercontent.com/64140048/153207365-3519e24d-0621-4bb8-8dfc-bb44956f2e07.png"></p>


## Explainability in Fake News Detection
We use the attention weights at the end of the model pipeline to highlight the most relevant words for detecting whether a piece of news is fake or not. A visualization of these attention weights is shown below for 4 example input tweets from the **Twitter15** dataset. The visualization of attention weights has been implemented in the notebook [ML_Project_Attn.ipynb](ML_Project_Attn.ipynb).

<p align="center">
<img width="681" alt="Screenshot 2022-02-09 at 6 30 43 PM" src="https://user-images.githubusercontent.com/64140048/153206196-3ffadea4-931a-4a3e-9cd5-ec575f0a8f8d.png"></p>

## Results
The following table lays down the ablation study for our model `AMRCN`. The final model `AMRCN w/ Channels` outperforms all the baseline models in terms of the prediction F1 score. In addition to generating explainability, `AMRCN` performs better than the baseline models in the binary classification task. This highlights that attention (i.e. explainability) has a side-product of boosted performance, and is intuitive. If a model can explain its prediction, it will directly or indirectly make better predictions. Our attention-enhanced framework proves to be quite effective in giving explanations and, at the same time distinguishing between fake and real tweets.

<p align="center">
<img width="705" alt="Screenshot 2022-02-09 at 6 44 44 PM" src="https://user-images.githubusercontent.com/64140048/153208492-a1caa5fe-32e3-414b-8622-709ceb5031e5.png"></p>


## Credits
This project was done as a part of the **Machine Learning 2021 course at IIITD**. Kindly drop issues if you have trouble running codes. 

