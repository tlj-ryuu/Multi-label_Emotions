import matplotlib.pyplot as plt


loss = [0.3146,0.2304,0.2021,0.1913,0.1791,0.1779,0.1669,0.1510,0.1362,0.1217]
acc = [0.9052,0.9175,0.9264,0.9311,0.9367,0.9358,0.9393,0.9451,0.9508,0.9579]
val_loss = [0.2316,0.2013,0.1822,0.1803,0.1781,0.1811,0.1983,0.1970,0.1950,0.2090]
val_acc = [0.9248,0.9351,0.9377,0.9399,0.9399,0.9397,0.9349,0.9375,0.9351,0.9329]


epochs = [1,2,3,4,5,6,7,8,9,10]

#plot1 acc
plt.subplot(1,2,1)

plt.plot(epochs,acc,label='acc',marker='.',color='b')
plt.plot(epochs,val_acc,label='val_acc',linestyle=':',marker='.',color='r')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
#plot2 loss
plt.subplot(1,2,2)

plt.plot(epochs,loss,label='loss',marker='.',color='g')
plt.plot(epochs,val_loss,label='val_loss',linestyle=':',marker='.',color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
