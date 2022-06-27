# Removing Zero-Variance Units of Deep Models for COVID-19 Detection

This is the code of the paper "Removing Zero-Variance Units of Deep Models for COVID-19 Detection" we use python programming language, for the training of the model we use keras and tensorflow as backend.

---

**Abstract**: Deep Learning has been used for several applications including the analysis of medical images. Some transfer learning works show that an improvement of the performance is obtained if a pre-trained model on imagenet is transferred to a new task. Taking into account this, we propose a method that use a pre-trained model on imagenet to fine-tune it for Covid-19 detection. After the fine-tune process, the units that produce variance equals to zero are removed from the model. Finally, we test the features of the penultimate layer in different classifiers removing those that are less important according to f-test. The results produce models with less units than the transferred model. Also, we study the attention of the neural network for the classification. Noise and metadata printed in medical images can bias the performance of the neural network and it obtains poor performance when the model is tested on new data. We study the bias of medical images when raw and masked images are used for training of deep models using a transfer learning strategy. Additionally, we test the performance on novel data in both models: raw and masked data.

---

The classes of the dataset need to be saved one per folder:


```
project
| model_compression.py
| xai.py
|
└--KAGGLE_V3
|   └-- COVID-19
|   └-- Lung_Opacity
|   └-- Normal
|   └-- Viral_Pneumonia
└--KAGGLE_V1
   └--COVID-19
   └--Normal
   └--Viral_Pneumonia
   ...
 ```
 
 ---
 For both tasks we receive two arguments:
 
 `--dataset`: name of the folder with the used dataset
 
 `--experiment_id`: experiment id to save the figures and logs
 
 ---
 Citation:
 <pre>
 @article{garcia2017removing,
  title={Removing Zero-Variance Units of Deep Models for COVID-19 Detection},
  author={García-Ramírez, Jesús and Escalante Ramírez, Boris and Olveres Montiel, Jimena},
  journal={Pre-print},
  year={2022}
}
</pre>
