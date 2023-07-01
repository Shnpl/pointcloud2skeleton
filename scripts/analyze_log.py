import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')

import numpy
def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')
with open('logs/Jun_30-78657b10/log.txt','r') as f:
    mpjpes = []
    losses = []
    epochs = []
    for line in f:
        if 'Epoch' in line:
            epoch = line.split('Epoch:')[-1]
            epoch = epoch.replace(',\n','')
            epochs.append(int(epoch))
        elif 'loss:' in line:
            loss = line.split("loss:")[-1]
            loss = float(loss.replace(',\n',''))
            losses.append(loss)
        # mpjpes.append(line[-1].replace('\n','')))
        # line = line[0].split("Epoch:")
            
            
losses = numpy.array(losses)
epochs = numpy.array(epochs)
length = len(losses)
avg_loss = {}
for i in range(length):
    if str(epochs[i]) not in avg_loss:
        avg_loss[str(epochs[i])] = [losses[i]]
    else:
        avg_loss[str(epochs[i])].append(losses[i])
avg_loss_list = []
for key in avg_loss:
    avg_loss_list.append(numpy.mean(numpy.array(avg_loss[key])))
x = numpy.linspace(0,length,length)
xi =numpy.linspace(0,length,len(avg_loss_list)) 
plt.plot(x,losses)
plt.plot(xi,avg_loss_list)
plt.ylabel('Loss')
plt.show()
