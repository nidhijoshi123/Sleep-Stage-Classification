import matplotlib.pyplot as plt
import numpy as np

YTest1=np.load('groundtruthlabelsagain.npy')
y_p=np.load('predictedlabelsagain.npy')
print(len(y_p))

#plt.figure()
#plt.xlabel('Epochs')
#plt.plot(groundtruth, 'orange', label='Ground Truth labels')
#plt.plot(predicted, 'blue', label='Predicted labels')
#plt.legend()
#plt.savefig('labels.png')

y_indices = []
for i, _ in enumerate(np.reshape(y_p, (-1, 1))):
    y_indices.append(i)

print(len(y_indices))

plt.figure()
plt.xlabel('Label index')
si=0
ei=15004
plt.plot(y_indices[si:ei],np.argmax(YTest1, axis=1)[si:ei] ,'ro',label='Actual labels')
plt.plot(y_indices[si:ei],y_p[si:ei] ,'bx',label='Predicted labels')
plt.legend()
plt.savefig('CMGraphexact.png', dpi=1000)
plt.savefig("CMGraphexact.pdf")


