# Importing the main libraries needed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Controlling the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Starting the DL model
model = Sequential()
# adding two convolutional layers to extract the features in the photos
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',  input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# To reduce the input of the layers
model.add(MaxPooling2D(pool_size=(2, 2)))

# Turning off some nodes to avoid over fitting
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Controlling the training information
modelInfo = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=7178 // 64)
# Saving the model as json and weights
modelJson = model.to_json()
with open('Models/model.json', 'w') as json_file:
    json_file.write(modelJson)

model.save_weights('model.weights.h5')
