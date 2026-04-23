# Assignment II — Transfer Learning for NLP (Text Classification)

The task is to classify a medical transcript into the correct **medical specialty**.

### Objectives

1. Explore the dataset and understand its main characteristics.
2. Build the required baseline model: frozen **BERT + LSTM(50)**.
3. Propose a stronger model and explain why it should work better.
4. Compare both models using suitable metrics for multiclass classification.
5. Critically discuss the results from both theoretical and practical perspectives.

## 2. Load and Clean Data

using only 2columns

- `medical_specialty` as the target label
- `transcription` as the input text

Before training the model, I cleaned the dataset by removing empty rows and clearing spaces to facilitate the subsequent training process.

## 3. Exploratory Data Analysis

I’ll start by analyzing the data to understand what’s in my dataset, its size, the categories, and the variations in text length. This is to ensure that any class imbalance doesn’t affect the evaluation results, and because text length can impact the overall performance of the model.

### 3.1 Class Distribution

I’m using bar charts—a simple and easy-to-understand visual format—to illustrate the sample sizes, especially when there are many categories.

### 3.2 Transcript Length Distribution

This histogram illustrates the detailed distribution of transcript lengths. Using a histogram makes it easier to set a maximum word count when deploying the model later, thereby preventing records from being truncated.

### 3.3 Length Variation Across Major Classes

Box plots are used here to compare the length of transcribed text across various high-frequency specialties and to display the median, range, and outliers for each group. Compared to a standard histogram, this comparison provides more detailed information. It reveals whether records for certain specialties are typically longer and more detailed than those for other specialties.

## 4. Workflow

The full experiment follows a clear end-to-end pipeline, shown in `images/workflow_diagram.svg`.

**Raw CSV → Column Selection → Data Cleaning → Label Encoding → Train/Validation/Test Split → Tokenization → Baseline Model → Proposed Model → Evaluation → Critique**

This workflow is easy and useful with each step has a specific purpose. The data is first cleaned, then converted into a form suitable for modeling, then used to train two different approaches, and finally evaluated with multiple metrics.

## 5. Label Encoding and Split

Since the target labels are text values, they must be converted into numerical form before training. This is achieved through label encoding, which maps each medical specialty to a unique integer ID. The preprocessed dataset is then split into a **training set**, a **validation set**, and a **test set**. A hierarchical splitting method is used to ensure that the class distributions across the three subsets remain as consistent as possible.

## 6. Dataset Class

Use a custom PyTorch `Dataset` class to prepare text for Transformer-based models. This class handles tokenization, truncation, padding, and label formatting, encapsulating this logic within the dataset for ease of use.

## 7. Baseline Model — Frozen BERT + LSTM(50)

This approach uses a baseline model that generates context embedding vectors using a pre-trained **BERT** model; however, the BERT parameters are frozen. The vectors are then fed into an **LSTM with 50 hidden units**, and the final prediction is made via a classification task. Note: Freezing BERT helps reduce computational costs for the subsequent model while also minimizing overfitting in the data.

## 8. Proposed Model — Biomedical BERT

Recommended model:'Bio_ClinicalBERT'

Because this model supports **end-to-end fine-tuning**, enabling pre-trained language representations to be directly adapted to classification tasks. Second, it is based on biomedical or clinical pre-training, which helps it understand medical terminology, abbreviations, and writing styles more effectively. Third, it employs class-weighted loss to reduce bias toward majority classes.

## 9. Results and Brief Discussion

To compare the two models fairly, I used several evaluation metrics instead of relying only on accuracy. In this task, **macro F1** is especially important because the dataset is imbalanced, so performance on small classes should not be hidden by the large classes. I also report **macro precision**, **macro recall**, and **weighted F1** to provide a more complete view of model performance.

From the results table, the baseline model achieved a slightly higher **accuracy** (**0.3463**) than the proposed model (**0.3409**). However, this does not mean that the baseline is better overall. When looking at more balanced metrics, the proposed model performed much better. Its **macro F1** increased from **0.0421** to **0.2824**, and its **weighted F1** also improved from **0.2110** to **0.2647**.

This result suggests that the baseline model was strongly biased toward a small number of majority classes. By comparision, the proposed model gave more balanced predictions across different specialties. Therefore, although the proposed model did not achieve the highest accuracy, it was still the better model for current task because it deal with the class imbalance more effectively.

Overall, the result shows that **accuracy alone is not enough** for this assignment. For an imbalanced multiclass medical classification problem, **macro F1 gives a more meaningful picture of model quality**.

## 10. Model Critique

### Theoretical Perspective

From a theoretical point of view, the baseline model is a reasonable starting point, but it also has clear limitations. In this study, BERT is used as a frozen encoder, so it works mainly as a fixed contextual feature extractor. This design is useful for building a simple baseline, and it is consistent with the original BERT framework, where pretrained representations can be used either as fixed features or through task-specific fine-tuning [1]. However, once BERT is frozen, its internal representations cannot adapt to the current medical specialty classification task. This means that most task-specific learning must be handled by the LSTM and the final classifier.

Another limitation is that the LSTM only has 50 hidden units, which gives the baseline model limited learning capacity. This may not be enough for a difficult multiclass task and clinical expressions.This is also consistent with the result of the baseline model in this experiment, where the macro F1 score is very low. In other words, the baseline may still capture common patterns, but it struggles to make balanced predictions across all classes.

The proposed model is theoretically stronger because it allows end-to-end fine-tuning and uses a domain-related pretrained model. This is important because biomedical and clinical language is different from general English. BioBERT was introduced exactly to address this domain gap, and the authors reported that pretraining on biomedical corpora helped the model perform better on biomedical text mining tasks than general BERT [2]. In a similar way, ClinicalBERT was developed for clinical notes and was shown to produce useful clinical text representations and outperform baselines on a clinical prediction task [3]. Based on these studies, it is reasonable that the proposed model in this assignment achieved a much higher macro F1 score than the baseline.

### Practical Perspective

From a practical perspective, the proposed model is more useful for this task, even though its accuracy is slightly lower than the baseline. In an imbalanced multiclass problem, accuracy alone can be misleading because a model may appear acceptable simply by performing well on the largest classes. In contrast, the much better macro F1 score of the proposed model shows that it makes more balanced predictions across specialties. For a medical text classification task, this is more meaningful than only improving overall accuracy, because the system should not ignore less frequent specialties.

At the same time, the stronger model also comes with higher cost. Fine-tuning a transformer requires more computation, more memory, and more careful hyperparameter selection than using frozen embeddings [1]. Therefore, the baseline still has practical value as a lightweight benchmark. It is easier to train and easier to reproduce. However, when the goal is better performance on specialized medical text, domain-specific pretrained models such as BioBERT or ClinicalBERT are usually a more suitable choice [2][3].

### Existing Limitations

Although the proposed model performed better overall, several limitations still remain. First, class imbalance is still a challenge, especially for classes with very small numbers of samples. Second, some medical record are long, so truncation may remove useful information before the text reaches the model.  Finally, the dataset is still limited in size for a multiclass medical NLP task, so the model may not generalize well to the real clinical settings.

Overall, the comparison in this assignment supports the idea that stronger task adaptation and domain-specific pretraining are useful for medical NLP [1][2][3]. At the same time, the remaining errors suggest that there is still room for improvement in handling rare classes, long documents, and more realistic clinical decision support.

## References

[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” _NAACL-HLT_, 2019.

[2] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang, “BioBERT: a pre-trained biomedical language representation model for biomedical text mining,” _Bioinformatics_, vol. 36, no. 4, pp. 1234–1240, 2020.

[3] K. Huang, J. Altosaar, and R. Ranganath, “ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission,” arXiv:1904.05342, 2019.
