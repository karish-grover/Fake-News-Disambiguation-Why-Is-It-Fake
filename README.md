# Fake News Disambiguation: Why Is It Fake?
This project considers the fake news detection problem under a more realistic scenario on social media. Given just a source short-text tweet, we aim to predict whether the given tweet is fake or not. We further aim to gen- erate an explanation for this prediction. This paper presents **Attention-enhanced Multi-channel Recurrent Convolutional Network (AMRCN)**, for explainable fake news detection. We explain our final predictions by **highlighting the essential words in the short tweet text**. Experimental results and extensive ablation studies show that our model outperforms the baseline systems on two benchmarking datasets.

**Here is the link to the blog post**: https://medium.com/@karish19471/fake-news-disambiguation-why-is-it-fake-e65c049181e5

## Problem Statement
Given the short text tweet content, we aim to classify this tweet as fake or real, i.e., binary classification. Further, we aim to explain why this tweet is **fake/actual by highlighting tokens/phrases from the tweet content** that contribute relatively more to the final prediction.

## Dataset Collection and Preprocessing
We utilize two well-known datasets compiled by (Ma et al., 2017), `Twitter15` and `Twitter16`, for our work. These datasets contain source tweets and their corresponding sequences of retweet users. These datasets are balanced towards the classes and consist of short length tweets. We remove the stop words from the tweets, replace all URLs with the token `URL`, and stem the tweets in the dataset for proceeding with the experiments. You can download the datasets from the [link](https://drive.google.com/drive/folders/1gFJsimozHJpyfE5ClCIgSftGrhtd1QWr?usp=sharing).

## Baselines
We evaluate the dataset collected on several baselines:

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


<!--  |                              | **Twitter15**                             ||||**Twitter16**                       ||||
 | ---------------------------- |--------- |---------- |--------- |-------- |--------- |---------- |--------- |------- |
 | **Method**                   | **Acc**  | **Prec**  | **Rec**  | **F1**  | **Acc**  | **Prec**  | **Rec**  | **F1** |
 | **NB** *w/* `Countvectors`     | 0.9301   |  0.9404   | 0.9080   | 0.9239  | 0.8932   |  0.8444   | 0.9047   | 0.8735 |
 | **NB** *w/* `Word TFIDF`       | 0.9247   |  0.9523   | 0.8888   | 0.9195  | 0.9029   |  0.8888   | 0.8888   | 0.8888 |
 | **NB** *w/* `NGram Vectors`    | 0.9086   |  0.9523   | 0.8602   | 0.9039  | 0.8640   |  0.9333   | 0.7924   | 0.8571 |
 | **NB** *w/* `CharVectors`      | 0.8924   |  0.9523   | 0.8333   | 0.8888  | 0.9126   |  0.9555   | 0.8601   | 0.9052 |
 | **LR** *w/* `Countvectors`     | 0.9086   |  0.9047   | 0.8941   | 0.8994  | 0.8932   |  0.8222   | 0.9251   | 0.8705 |
 | **LR** *w/* `Word TFIDF`       | 0.9301   |  0.9047   | 0.9382   | 0.9212  | 0.9029   |  0.8444   | 0.9268   | 0.8837 |
 | **LR** *w/* `NGram Vectors`    | 0.9408   |  0.9047   | 0.9620   | 0.9325  | 0.8252   |  0.8888   | 0.7547   | 0.8163 |
 | **LR** *w/* `CharVectors`      | 0.9086   |  0.8928   | 0.9036   | 0.8982  | 0.9126   |  0.8222   | 0.9736   | 0.8915 |
 | **SVM** *w/* `Countvectors`    | 0.9354   |  0.9047   | 0.9501   | 0.9268  | 0.9126   |  0.8444   | 0.9501   | 0.8941 |
 | **SVM** *w/* `Word TFIDF`      | 0.9408   |  0.9047   | 0.9620   | 0.9325  | 0.8155   |  0.9111   | 0.7321   | 0.8118 |
 | **SVM** *w/* `NGram Vectors`   | 0.9193   |  0.8452   | 0.9726   | 0.9044  | 0.8252   |  0.9555   | 0.7288   | 0.8269 |
 | **SVM** *w/* `CharVectors`     | 0.9408   |  0.9166   | 0.9506   | 0.9333  | 0.9223   |  0.8444   | 0.9743   | 0.9047 |
 | **RF** *w/* `Countvectors`     | 0.9354   |  0.8809   | 0.9736   | 0.9250  | 0.9029   |  0.8001   | 0.9729   | 0.8780 |
 | **RF** *w/* `Word TFIDF`       | 0.9193   |  0.8452   | 0.9726   | 0.9044  | 0.8834   |  0.7777   | 0.9459   | 0.8536 |
 | **RF** *w/* `CharVectors`      | 0.8978   |  0.8809   | 0.8915   | 0.8862  | 0.9223   |  0.8444   | 0.9743   | 0.9047 |
 | **XGB** *w/* `Countvectors`    | 0.8655   |  0.7738   | 0.9154   | 0.8387  | 0.8155   |  0.7555   | 0.8095   | 0.7816 |
 | **XGB** *w/* `Word TFIDF`      | 0.8602   |  0.7619   | 0.9142   | 0.8311  | 0.8446   |  0.7555   | 0.8717   | 0.8095 |
 | **XGB** *w/* `CharVectors`     | 0.8548   |  0.8690   | 0.8202   | 0.8439  | 0.9029   |  0.8666   | 0.9069   | 0.8863 |
 | **CNN**                      | 0.9247   |  0.9074   | 0.9607   | 0.9301  | 0.8155   |  0.8001   | 0.7826   | 0.7912 |
 | **CNN** *w/o* `Dropout`        | 0.9139   |  0.8796   | 0.9693   | 0.9223  | 0.8446   |  0.8222   | 0.8222   | 0.8222 |
 | **CNN** *w/* `TrainEmb`        | 0.9193   |  0.8796   | 0.9793   | 0.9268  | 0.8252   |  0.8222   | 0.7872   | 0.8043 |
 | **CNN** *w/* `AvgPool`         | 0.8978   |  0.8703   | 0.9494   | 0.9082  | 0.7864   |  0.7111   | 0.7804   | 0.7441 |
 | **LSTM**                     | 0.8602   |  0.8611   | 0.8942   | 0.8773  | 0.7669   |  0.8001   | 0.7058   | 0.7501 |
 | **LSTM** *w/* `TrainEmb`       | 0.8978   |  0.9166   | 0.9082   | 0.9124  | 0.8834   |  0.8222   | 0.9024   | 0.8604 |
 | **GRU**                      | 0.8333   |  0.8333   | 0.8737   | 0.8530  | 0.8737   |  0.7777   | 0.9210   | 0.8433 |
 | **GRU** *w/* `TrainEmb`        | 0.9139   |  0.9074   | 0.9423   | 0.9245  | 0.8737   |  0.8666   | 0.8478   | 0.8571 |
 | **BiGRU**                    | 0.8602   |  0.8055   | 0.9456   | 0.8701  | 0.8349   |  0.7111   | 0.8888   | 0.7901 |
 | **BiGRU** *w/* `TrainEmb`      | 0.9032   |  0.8981   | 0.9326   | 0.9150  | 0.8932   |  0.8444   | 0.9047   | 0.8735 |
 | **RCNN** *w/* `TrainEmb`       | 0.8590   |  0.8571   | 0.8450   | 0.8510  | 0.8657   |  0.8101   | 0.9275   | 0.8648 |
 | **RCNN** *w/* `Multiple CNN`   | 0.9127   |  0.9189   | 0.9066   | 0.9127  | 0.9261   |  0.8676   | 0.9672   | 0.9147 |
   -->
  
