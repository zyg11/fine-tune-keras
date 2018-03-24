# fine-tuning方式二：要调整权重
# fine-tune分三个步骤：
# - 搭建vgg-16并载入权重，将之前定义的全连接网络加在模型的顶部，并载入权重
# - 冻结vgg16网络的一部分参数
# - 模型训练
from keras import  applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout,Activation,Flatten,Dense,GlobalAveragePooling2D
#载入model权重和网络
from  keras.applications.vgg16 import VGG16
model=VGG16(weights='imagenet',include_top=False)

#新加层
x=model.output
# 最有问题的层:flatten层
# x=Flatten(name='flatten')(x)
# 尝试一：x = Flatten()(x)
# 尝试二：x = GlobalAveragePooling2D()(x)
# 尝试三：from keras.layers import Reshape
#x = Reshape((4,4, 512))(x)
#  TypeError: long() argument must be a string or a number, not 'NoneType'
#最后这样操作
x=GlobalAveragePooling2D()(x)
x=Dense(256,activation='relu',name='fcl')(x)
x=Dropout(0.5)(x)
predictions=Dense(5,activation='softmax')(x)

#然后将最后一个卷积块前的卷积层参数冻结：
from keras.models import Model
vgg_model=Model(inputs=model.input,outputs=predictions)

# 冻结vgg16网络的一部分参数
for layer in vgg_model.layers[:25]:
    layer.trainable=False
# compile the model with a SGD/momentum optimizer
vgg_model.compile(loss='binary_crossentropy',#binary_crossentropy
                  optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
                  metrics=['accuracy'])
#准备数据
train_data_dir='E:/keras_data/data1/train'
test_data_dir='E:/keras_data/data1/test'
img_width,img_height=150,150
nb_train_samples=500
nb_test_samples=100
epochs=2
batch_size=16#

#图片预处理生成器
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
test_datagen=ImageDataGenerator(
    rescale=1./255
)
#图片生成器
train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height,img_width),
    batch_size=32,
    class_mode='categorical'
)
test_generotor=test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height,img_width),
    batch_size=32,
    class_mode='categorical'
)
#训练
vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=test_generotor,
    validation_steps=nb_test_samples//batch_size
)