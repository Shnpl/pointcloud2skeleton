import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy
def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')
with open('nohup.out','r') as f:
    mpjpes = []
    epochs = []
    for line in f:
        if 'MPJPE3D:' in line:
            line = line.split("MPJPE3D:")
            mpjpes.append(float(line[-1].replace('\n','')))
            line = line[0].split("Epoch:")
            epoch = line[1].split(',')[0]
            epochs.append(int(epoch))
mpjpes = numpy.array(mpjpes)
epochs = numpy.array(epochs)
length = len(mpjpes)
avg_mpjpe = {}
for i in range(length):
    if str(epochs[i]) not in avg_mpjpe:
        avg_mpjpe[str(epochs[i])] = [mpjpes[i]]
    else:
        avg_mpjpe[str(epochs[i])].append(mpjpes[i])
avg_mpjpe_list = []
for key in avg_mpjpe:
    avg_mpjpe_list.append(numpy.mean(numpy.array(avg_mpjpe[key])))
x = numpy.linspace(0,length,length)
xi =numpy.linspace(0,length,10) 
plt.plot(avg_mpjpe_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
