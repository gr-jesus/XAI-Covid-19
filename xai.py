from os import walk
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Add, Input, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotNormal
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc 
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import matplotlib.pyplot as plt
import argparse

# Define the input arguments with a argparse
parser = argparse.ArgumentParser(description='Compress a pre-trained model')
parser.add_argument('--dataset', type=str, help='name of the folder with the used dataset')
parser.add_argument('--experiment_id', type=str, help='Experiment id to save the figures and logs')
args=parser.parse_args()

#-----------------------------------------------------------------------------------------------------------
# Some additional functions 
#-----------------------------------------------------------------------------------------------------------
def readFolder(mypath):
  '''
  Read the files from a folder

  params:
    mypath: path from we want to read the file name

  return: a list with the files name in the folder
  '''
  return next(walk(mypath), (None, None, []))[2]

def read_image(name):
  '''
  Read and preprocess a image

  params:
    name: path of the image to pre-process

  return:
    pre-processed image for give to the model
  '''
  img = image.load_img(name, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

def print_clasification_report(y_pred, y_test):
  '''
  Print the classification report, accuracy, recall, and the confusion matrix

  params:
    y_pred: prediction of the trained model
    y_test: true labels to evaluate the model
  '''
  print('Accuracy:', accuracy_score(y_test, y_pred))
  print('------------------Classification Report------------------')
  print(classification_report(y_test, y_pred, target_names=folders))
  print('------------------Confusion Matrix------------------')
  print(confusion_matrix(y_test, y_pred))
  print('------------------AUC------------------')
  fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=2)
  print('Area Under ROC curve',auc(fpr, tpr))

def evaluate_features(train_data, y_train, test_data, y_test, n_features):
  '''
  Evaluate features with a SVM

  params:
    train_data: data to train the SVM
    y_train: labels to train the svm
    test_data: data to evaluate the SVM
    y_test: labels to evaluate the SVM
  '''
  fit=SelectKBest(f_classif, k=n_features).fit(train_data, y_train)
  X_train_new=fit.transform(train_data)
  X_test_new=fit.transform(test_data)
  clf_compress=SVC(gamma='auto', kernel='poly', degree=3)
  clf_compress.fit(X_train_new, y_train)
  y_pred=clf_compress.predict(X_test_new)
  return accuracy_score(y_test, y_pred)

def compress_fc_no_previous_flatten(weigths, index):
  '''
  Compress a fully connected layer with a previous fully connected (not flatten)
  params:
  	weigths: the weigths of the fully connected layer
	index: index of the useful units
  return: the new weigth of the layer
  '''
  return [weigths[0][:,index], weigths[1][index]]


def evaluate_layer(layer_name, a_model, data):
  '''
  Evaluate and get the variances of the output in a hidden layer

  params:
    layer_name: the name of the layer to extract the output
    a_model: the model that will be evaluated
    data: the data that will be used to evaluate the model
  return: a numpy array with the variances of each convolutional kernels
  '''
  K.clear_session()
  aux_model=Model(inputs=a_model.input, outputs=a_model.get_layer(layer_name).output)
  #predict_out=aux_model.predict(data, batch_size=32)
  #variances=np.var(predict_out[:,:,:,i])
  variances=[]
  for batch in np.array_split(data, 200):
    predict_out=aux_model.predict(batch)
    local_variances=[]
    for i in range(predict_out.shape[3]):
      local_variances.append(np.var(predict_out[:,:,:,i]))
    variances.append(local_variances)
  return np.mean(np.array(variances), axis=0)



experiment_id=args.experiment_id

# Create the folders that will be used to load the dataset, each folder corresponds to a class in the dataset
base_folder=args.dataset
folders=[]
if base_folder=='Kaggle_v3/':
  #folders=['COVID/images/', 'Lung_Opacity/images/', 'Normal/images/', 'Viral_Pneumonia/images']
  folders=['COVID/images/', 'Normal/images/']
  #folders=['Viral_Pneumonia', 'Normal', 'Lung_Opacity', 'COVID']
elif base_folder=='unmasked/':
  #folders=['Viral_Pneumonia', 'Normal', 'Lung_Opacity', 'COVID']
  #folders=['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']
  folders=['COVID/', 'Normal/']
else:
  folders=['COVID-19',  'PNEUMONIA', 'NORMAL']

print(folders)

#Load the dataset
raw=[]
Y=[]
y=0
for folder in folders:
  for image_name in readFolder(base_folder+folder):
    raw.append(read_image(base_folder+folder+'/'+image_name)[0])
    Y.append(y)
  y+=1

#Split the three arrays
# 1) 80% for train and 20% to test
raw_train, raw_test, y_train, y_test=train_test_split(raw, Y, test_size=0.20, stratify=Y)
# 2) from 80% to train -> 80% train, 20% val
raw_train, raw_val, y_train, y_val =train_test_split(raw_train, y_train, test_size=0.20, stratify=y_train)
# convert all the data to numpy arrays
raw_val=np.array(raw_val)
raw_train=np.array(raw_train)
raw_test=np.array(raw_test)
y_val=np.array(y_val)
y_train=np.array(y_train)
y_test=np.array(y_test)


#-----------------------------------------------------------------------------------------------------------------------------
# Fine tune the new model with the last layer randomly initialized
# Callback to save the best model
save_best_cb=ModelCheckpoint(filepath='best_full_VGG16_'+experiment_id+'.h5', monitor='val_accuracy', save_best_only=True)
#checkpoint_cb=ModelCheckpoint(filepath='experiments/checkpoint_{epoch:02d}.h5', save_freq='epoch')
# Parameters for the optimizer
opt=Adam(learning_rate=0.0001)
# Compile and train the model
model = VGG16(weights='imagenet')
dense_out = Dense(units=len(folders), activation='softmax', kernel_regularizer=regularizers.L2(0.05))(model.layers[-2].output)
model = Model(inputs=model.input, outputs=dense_out)
print(model.summary())


model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
#history=model.fit(raw_train, to_categorical(y_train), validation_data=(raw_val, to_categorical(y_val)), batch_size=32, epochs=20, callbacks=[save_best_cb, checkpoint_cb])
history=model.fit(raw_train, to_categorical(y_train), validation_data=(raw_val, to_categorical(y_val)), batch_size=32, epochs=15, callbacks=[save_best_cb])
#load the weights of the best model
model.load_weights('best_full_VGG16_'+experiment_id+'.h5')
# evaluate the model with the testing data
y_pred=np.argmax(model.predict(raw_test), axis=1)
print('------------------Experiment with the entire representation---------------------------')
print_clasification_report(y_pred,y_test)

# Get the output of the fc2 layer
output_fc2=Model(inputs=model.input, outputs=model.get_layer('fc2').output)
x_train=output_fc2.predict(raw_train)
#-------------------------------------------------------------------------------------------------------------------------------

# Validation in the new data

base_folder='Validacion/'
folders=['COVID', 'NORMAL']
#Load the dataset
raw=[]
Y=[]
y=0
for folder in folders:
  for image_name in readFolder(base_folder+folder):
    raw.append(read_image(base_folder+folder+'/'+image_name)[0])
    Y.append(y)
  y+=1


raw=np.array(raw)

#y_pred=np.argmax(model.predict(raw), axis=1)
#print(model.predict(raw))
#print(y_pred)
#y_pred=np.argmax(model.predict(raw)[:,[1,3]], axis=1)
y_pred=np.argmax(model.predict(raw), axis=1)
print(y_pred)

print('------------------Results with new data---------------------------')
#print_clasification_report(y_pred,y)
print_clasification_report(y_pred,Y)
#print(classification_report(Y, y_pred, target_names=folders))

