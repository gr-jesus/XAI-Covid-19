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

def get_reduced_model(weights, layer_names, model_name, n_classes):
  '''
  Return a reduced model with the unused units

  params:
    weights: the weigths of the new reduced model
    layer_names: names of the layers that will be used
    model_name: nem of the model that is used to compress: VGG16
    n_classes: number of classes in the dataset
  '''
  reduced_model=Sequential()
  if 'VGG16':
    # Input layer
    # self.reduced_model.add(InputLayer(input_shape=(224,224,3)))
    # Block_1
    reduced_model.add(Conv2D(input_shape=(224,224,3), filters=weights[0][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-1]))
    reduced_model.add(Conv2D(filters=weights[1][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-2]))
    reduced_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool'))
    # Block_2
    reduced_model.add(Conv2D(filters=weights[2][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-3]))
    reduced_model.add(Conv2D(filters=weights[3][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-4]))
    reduced_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_pool'))
    # Block_3
    reduced_model.add(Conv2D(filters=weights[4][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-5]))
    reduced_model.add(Conv2D(filters=weights[5][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-6]))
    reduced_model.add(Conv2D(filters=weights[6][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-7]))
    reduced_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool'))
    # Block_4 
    reduced_model.add(Conv2D(filters=weights[7][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-8]))
    reduced_model.add(Conv2D(filters=weights[8][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-9]))
    reduced_model.add(Conv2D(filters=weights[9][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-10]))
    reduced_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_pool'))
    # Block_5
    reduced_model.add(Conv2D(filters=weights[10][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-11]))
    reduced_model.add(Conv2D(filters=weights[11][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-12]))
    reduced_model.add(Conv2D(filters=weights[12][0].shape[-1], kernel_size=(3,3), padding='same', activation='relu', name=layer_names[-13]))
    reduced_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block5_pool'))
    # Fully connected layers
    reduced_model.add(Flatten())
    reduced_model.add(Dense(units=weights[13][0].shape[-1], activation='relu', name=layer_names[-14]))
    reduced_model.add(Dense(units=weights[14][0].shape[-1], activation='relu', name=layer_names[-15]))
    #reduced_model.add(Dense(units=n_classes, activation='softmax', name='output'))

    cont=0
    for name in layer_names[::-1]:
    #  print(name, weights[cont][0].shape)
      reduced_model.get_layer(name).set_weights(weights[cont])
      cont+=1
  return reduced_model

#-----------------------------------------------------------------------------------------------------------
# Some additional functions
#-----------------------------------------------------------------------------------------------------------

experiment_id=args.experiment_id

# Create the folders that will be used to load the dataset, each folder corresponds to a class in the dataset
base_folder=args.dataset
folders=[]
if base_folder=='Kaggle_v3/':
  folders=['COVID/images/', 'Lung_Opacity/images/', 'Normal/images/', 'Viral_Pneumonia/images']
  #folders=['Viral_Pneumonia', 'Normal', 'Lung_Opacity', 'COVID']
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
save_best_cb=ModelCheckpoint(filepath='best_full_VGG16.h5', monitor='val_accuracy', save_best_only=True)
checkpoint_cb=ModelCheckpoint(filepath='experiments/checkpoint_{epoch:02d}.h5', save_freq='epoch')
# Parameters for the optimizer
opt=Adam(learning_rate=0.0001)
# Compile and train the model
model = VGG16(weights='imagenet')
dense_out = Dense(units=len(folders), activation='softmax', kernel_regularizer=regularizers.L2(0.05))(model.layers[-2].output)
model = Model(inputs=model.input, outputs=dense_out)
print(model.summary())


model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#history=model.fit(raw_train, to_categorical(y_train), validation_data=(raw_val, to_categorical(y_val)), batch_size=32, epochs=20, callbacks=[save_best_cb, checkpoint_cb])
history=model.fit(raw_train, to_categorical(y_train), validation_data=(raw_val, to_categorical(y_val)), batch_size=32, epochs=30, callbacks=[save_best_cb])
#load the weights of the best model
model.load_weights('best_full_VGG16.h5')
# evaluate the model with the testing data
y_pred=np.argmax(model.predict(raw_test), axis=1)
print('------------------Experiment with the entire representation---------------------------')
print_clasification_report(y_pred,y_test)

# Get the output of the fc2 layer
output_fc2=Model(inputs=model.input, outputs=model.get_layer('fc2').output)
x_train=output_fc2.predict(raw_train)
#-------------------------------------------------------------------------------------------------------------------------------

# Extract the features from the last layer fully connected layer of the VGG16 pre-trained model
output_fc2=Model(inputs=model.input, outputs=model.get_layer('fc2').output)
train_data=output_fc2.predict(raw_train)
val_data=output_fc2.predict(raw_val)
test_data=output_fc2.predict(raw_test)


# output of the model
train_output_entire_model=model.predict(raw_train)
test_output_entire_model=model.predict(raw_test)
val_output_entire_model=model.predict(raw_val)

# Here we apply two filters to the ouput data
scores=[]
features=[]
# First we erase those features with variance=0 (constant features)
fit=VarianceThreshold().fit(train_data)
train_data=fit.transform(train_data)
test_data=fit.transform(test_data)
val_data=fit.transform(val_data)
positions=[i for i, x in enumerate(fit.get_support()) if x]

nw=compress_fc_no_previous_flatten(model.get_layer('fc2').get_weights(), positions)

# As the second filter we apply a greedy search, we evaluate in the SVM every 100
# features, at the end we select the number of features with the highest accuracy
for i in range(100, train_data.shape[1], 100):
  scores.append(evaluate_features(train_data, y_train, val_data, y_val, i))
  features.append(i)

plt.figure(figsize=(8, 6))

# we plot the scores and features
plt.title('1st. stage greedy Search every 100 features')
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.plot(features, scores)
plt.savefig('Features_Filter_1_'+experiment_id+'.png')
plt.clf()

max_features=features[scores.index(max(scores))]
scores=[]
features=[]

# Now, we search between the last threshold to see if there is a lower number
# of features with the same accuracy
for i in range(max_features-100, max_features+1):
  if i>0:
    scores.append(evaluate_features(train_data, y_train, test_data, y_test, i))
    features.append(i)

plt.title('2nd. stage greedy search')
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.plot(features, scores)

plt.savefig('Features_Filter_'+experiment_id+'.png')

n_features=features[scores.index(max(scores))]
fit=SelectKBest(f_classif, k=n_features).fit(train_data, y_train)
positions=[i for i, x in enumerate(fit.get_support()) if x]

#print(nw[0].shape, nw[1].shape)
#print(compress_fc_no_previous_flatten(nw, positions)[0].shape, compress_fc_no_previous_flatten(nw, positions)[1].shape)
nw=compress_fc_no_previous_flatten(nw, positions)

#----------------------------------------------------------------------------
# Now we fit our data with the selected number of features
fit=SelectKBest(f_classif, k=n_features).fit(train_data, y_train)
train_data=fit.transform(train_data)
test_data=fit.transform(test_data)
val_data=fit.transform(val_data)
clf_compress=SVC(gamma='auto', kernel='poly', degree=3)
clf_compress.fit(train_data, y_train)
y_pred=clf_compress.predict(test_data)
print('Support Vector Machine')
print_clasification_report(y_pred, y_test)
#----------------------------------------------------------------------------
output_fc2=Model(inputs=model.input, outputs=model.get_layer('fc1').output)
train_data=output_fc2.predict(raw_train)
fit=VarianceThreshold().fit(train_data)
train_data=fit.transform(train_data)
positions=[i for i, x in enumerate(fit.get_support()) if x]
nw_fc1=compress_fc_no_previous_flatten(model.get_layer('fc1').get_weights(), positions)
nw=[nw[0][positions], nw[1]]

print('fc_out', nw[0].shape, nw[1].shape)
#print(nw_fc1[0].shape, nw_fc1[1].shape)

#conv_layers_names=['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
#'block4_conv1','block4_conv2', 'block4_conv3','block5_conv1', 'block5_conv2', 'block5_conv3']
conv_layers_names=['block5_conv3', 'block5_conv2', 'block5_conv1', 'block4_conv3', 'block4_conv2',
                   'block4_conv1', 'block3_conv3', 'block3_conv2', 'block3_conv1', 'block2_conv2',
                   'block2_conv1', 'block1_conv2', 'block1_conv1']

last_conv='block5_conv3'
maxpool_layer='block5_pool'
new_weights=[]
for l_name in conv_layers_names:
  # Evaluate the layer out
  out=evaluate_layer(l_name, model, raw_train)
  # variances of the feature maps, if !=0 append True, else False, works similar to Zero-Variance support
  var_conv=[]
  for i in out:
    if i==0:
      var_conv.append(False)
    else:
      var_conv.append(True)
  # get the positions where variance =0
  positions=[i for i, x in enumerate(var_conv) if x]
  # Obtain the new weigths and bias according to the variance of the feature maps
  w=model.get_layer(l_name).get_weights()[0][:,:,:,positions]
  b=model.get_layer(l_name).get_weights()[1][positions]
  print(l_name)
  print(positions)


  if last_conv==l_name:
    new_weights.append([w, b])
    units_positions=[]
    number_of_kernels=model.get_layer(maxpool_layer).output_shape[-1]
    kernel_size=model.get_layer(maxpool_layer).output_shape[-2]*model.get_layer(maxpool_layer).output_shape[-3]
    for k in range(kernel_size):
      for j in positions:
        units_positions.append(k*number_of_kernels+j)
    #Update the weights of the fully connected layer
    nw_fc1=[nw_fc1[0][units_positions], nw_fc1[1]]
  else:
    # Insert in the new position
    new_weights.insert(0,[w, b])
    # Modify the weights of the next layer
    new_weights[1][0]=new_weights[1][0][:,:,positions]



#dense_out = Conv2D(filters=new_weights.shape[-1], kernel_size=(3,3), padding='same', activation='relu')(model.get_layer('block5_conv2').output)
#dense_out = MaxPool2D(pool_size=(2,2), strides=(2,2))(dense_out)
#dense_out = Flatten()(dense_out)
#dense_out = Dense(units=nw_fc1[0].shape[1], activation='relu')(dense_out)
#dense_out = Dense(units=nw[0].shape[1], activation='relu')(dense_out)
#compressed_model = Model(inputs=model.input, outputs=dense_out)

new_weights.append(nw_fc1)
new_weights.append(nw)
#for i in new_weights:
#  print(i[0].shape)

conv_layers_names.insert(0, 'fc1')
conv_layers_names.insert(0, 'fc2')
#for i in conv_layers_names[::-1]:
#  print(i)
compressed_model=get_reduced_model(new_weights, conv_layers_names, 'VGG16', len(folders))
print(compressed_model.summary())

#print(compressed_model.summary())
#compressed_model.get_layer('conv2d').set_weights([new_weights, new_bias])
#compressed_model.get_layer('dense').set_weights(nw_fc1)
#compressed_model.get_layer('dense_1').set_weights(nw)

train_data=compressed_model.predict(raw_train)
val_data=compressed_model.predict(raw_val)
test_data=compressed_model.predict(raw_test)
clf_compress=SVC(gamma='auto', kernel='poly', degree=3)
clf_compress.fit(train_data, y_train)
y_pred=clf_compress.predict(test_data)
print('Support Vector Machine Compressed model')
print_clasification_report(y_pred,y_test)

#print(nw[0].shape, nw[1].shape)
#print(nw_fc1[0].shape, nw_fc1[1].shape)
