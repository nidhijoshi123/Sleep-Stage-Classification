from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.layers import LSTM , Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
import Paths as paths
from keras.callbacks import TensorBoard
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



#Build model
def elevenLayerConv1DModel(input_shape,n_classes):
    model = Sequential()

    model.add(Conv1D (kernel_size = (4), filters = 32, strides=1, input_shape=input_shape, 
                     activation='relu')) 

    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D (kernel_size = (4), filters = 64, strides=1, activation='relu')) 
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D (kernel_size = (5), filters = 128, strides=5)) 

    #model.add(BatchNormalization())

    model.add(Bidirectional(LSTM(32, return_sequences=True , input_shape=input_shape)))
    #model.add(BatchNormalization())
    #model.add(Bidirectional(LSTM(32, return_sequences=False , input_shape=input_shape)))
    model.add(Bidirectional(LSTM(32)))
    #model.add(BatchNormalization())

    model.add(Dense(n_classes, activation = 'softmax',name='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model


def removeResiduals(Y,X,remIndY):
    nc = [x for x in range(len(Y)) if x not in remIndY]
    X = [X[k*samples:(k*samples+samples)] for k in nc]
    X = [item for sublist in X for item in sublist] 
    Y = list(filter(lambda a: a != -1, Y))
    return [X,Y]

def removeResiduals2(Y,X,remIndY):
    count = 0
    for val in remIndY:
        val = val - count
        del X[val*samples:(val*samples+samples)]
        count = count + 1
    Y = list(filter(lambda a: a != -1, Y))
    return [X,Y]

def removeResiduals3(Y,X,remIndY):
    count = 0
    for val in remIndY:
        val = val - count
        X = np.delete(X,list(range(val*samples,(val*samples+samples))))
        count = count + 1
    Y = list(filter(lambda a: a != -1, Y))
    return [X,Y]
    


#Fetch X and Y from stored numpy arrays
# =============================================================================
# XTrain = np.load('traindata1d.npy').tolist()
# XTest = np.load('testdata1d.npy').tolist()
# =============================================================================
XTrain = np.load('traindata1dagain.npy')
XTest = np.load('testdata1dagain.npy')
YTrain = np.load('trainlabels1dagain.npy').tolist()
YTest = np.load('testlabels1dagain.npy').tolist()
print('Original lengths: ')
print(len(XTrain))
print(len(XTest))
print(len(YTrain))
print(len(YTest))

samples = paths.samplesPerLabel

#Get rid of -1 in Y
remIndYtrain = [i for i, e in enumerate(YTrain) if e == -1]
remIndYtest = [i for i, e in enumerate(YTest) if e == -1]
# =============================================================================
# YTrain = list(filter(lambda a: a != -1, YTrain))
# YTest = list(filter(lambda a: a != -1, YTest))
# =============================================================================
#Get rid of corresponding -1 data in X
XTrain,YTrain = removeResiduals3(YTrain,XTrain,remIndYtrain)
XTest,YTest = removeResiduals3(YTest,XTest,remIndYtest)
# =============================================================================
# for k in remIndYtrain:
#     del XTrain[k*samples:(k*samples+samples)]
# for l in remIndYtest:    
#     del XTest[l*samples:(l*samples+samples)]
# =============================================================================

print('Modified lengths: ')
print(len(XTrain))
print(len(XTest))
print(len(YTrain))
print(len(YTest))

#Convert all lists to numpy arrays
   
#XTrain = np.array(XTrain)
YTrain = np.array(YTrain)
#XTest = np.array(XTest)
YTest = np.array(YTest)

#Reshape X and Y
XTrain1 =XTrain.reshape(int(len(XTrain) / samples),samples,1)
print(XTrain1.shape)
XTest1 = XTest.reshape(int(len(XTest) / samples),samples,1)
print(XTest1.shape)
YTrain1=YTrain.reshape(int(len(XTrain) / samples ) , 1)
print(YTrain1.shape)
YTest1 = YTest.reshape(int(len(XTest) / samples) , 1)
print(YTest1.shape)

#Convert Y to categorical
YTrain1 = to_categorical(YTrain1)
YTest1 = to_categorical(YTest1)


input_shape=[samples,1]

#Fit the model
model = elevenLayerConv1DModel(input_shape,paths.numberOfCLasses)
#mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#callbacks = [tensorboard,es]

history=model.fit(XTrain1, YTrain1, epochs = 50, batch_size = 32,validation_data=(XTest1,YTest1), callbacks=[es])

#plots and model evaluations

np.save("groundtruthlabelsagain.npy",YTest1)


#convert predicted probablities into labels
Y_pred = model.predict(XTest1)
y_p = np.argmax(Y_pred, axis=1)
np.save("predictedlabelsagain.npy",y_p)

plt.figure()
plt.xlabel('Epochs')
plt.plot(history.history['acc'], 'orange', label='Training accuracy')
plt.plot(history.history['val_acc'], 'blue', label='Validation accuracy')
plt.plot(history.history['loss'], 'red', label='Training loss')
plt.plot(history.history['val_loss'], 'green', label='Validation loss')
plt.legend()
plt.savefig('accuraciesagain.png')

print('Confusion Matrix')
#print(confusion_matrix(YTest1, y_p))
print(confusion_matrix(np.argmax(YTest1, axis=1), y_p))
target_names = ['0', '1', '2', '3', '4']
print(classification_report(np.argmax(YTest1, axis=1), y_p, target_names=target_names))



































