import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

class_names = [0,1,2,3,4,5,6,7,8,9]


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu", name="HL-1"),
    Dropout(0.2),
    Dense(64, activation="relu", name="HL-2"),
    Dropout(0.2),
    Dense(32, activation="relu", name="HL-3"),
    Dropout(0.2),
    Dense(16, activation="relu", name="HL-4"),
    Dropout(0.2),
    Dense(10, activation="softmax", name="output_layer")
])


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))


test_loss,test_acc = model.evaluate(x_test,y_test)
print("Test accuracy:",test_acc)

print(model.summary())

