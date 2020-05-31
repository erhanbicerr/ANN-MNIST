import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

df = tf.keras.datasets.mnist

(X_train, y_train),(X_test, y_test) = df.load_data()

X_train = X_train/255;
X_test = X_test/255;
# values are between 0-1
# white colors are near to 1, black 0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
    
    ])

model.compile(optimizer="adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"]
              )

r = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=12)


plt.figure()
plt.plot(r.history["accuracy"], label = "Accuracy on Train Data")
plt.plot(r.history["val_accuracy"], label = "Accuracy on Test Data")
plt.legend()
plt.show()


plt.figure()
plt.plot(r.history["loss"], label = "Loss on Train Data")
plt.plot(r.history["val_loss"], label = "Loss on Test Data")
plt.legend()
plt.show()

pred = model.predict(X_test)
p_test = []

for i in range(0,len(pred)):
    p_test = np.append(p_test,np.argmax(pred[i]))

missclassified = np.where(p_test!=y_test)

for i in missclassified[0]:
    plt.figure()
    plt.imshow(X_test[i], cmap="gray")
    plt.title("True Label: %s Prediction is: %s " % (y_test[i],p_test[i]))
    

    
