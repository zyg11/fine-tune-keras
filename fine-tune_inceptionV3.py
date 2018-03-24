from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras import backend as k

# create the base pre-trained model
base_model=InceptionV3(weights='imagenet',include_top=False)

# add a global spatial average pooling layer
x=base_model.output
x=GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x=Dense(1024,activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions=Dense(200,activation='softmax')(x)

# this is the model we will train
model=Model(inputs=base_model.input,outputs=predictions)
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable=False
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
# train the model on the new data for a few epochs
model.fit_generator()
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze: