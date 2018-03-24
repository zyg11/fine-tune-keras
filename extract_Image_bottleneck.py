# 载入图片；
# 灌入pre-model的权重；
# 得到bottleneck feature
#如何提取bottleneck feature
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten,Activation

from keras.applications.vgg16 import VGG16
model=VGG16(include_top=False,weights='imagenet')

#载入图片
#图像生成器初始化
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
datad_gen=ImageDataGenerator(rescale=1./255)
#训练集生成器
generator=datad_gen.flow_from_directory(
    'E:/keras_data/data1/train',
    target_size=(150,150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)
#验证集图像生成器
generator=datad_gen.flow_from_directory(
    'E:/keras_data/data1/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False,
)
#载入pre_model的权重
model.load_weights('E:/keras_data/data1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
#获得bottleneck feature
bottleneck_feature_train=model.predict_generator(generator,500)
# 核心，steps是生成器要返回数据的轮数，
# 每个epoch含有500张图片，与model.fit(samples_per_epoch)相对
np.save(open('E:/keras_data/data1/bottleneck_feature_train.npy','w'),bottleneck_feature_train)

bottleneck_feature_test=model.predict_generator(generator,100)
# 与model.fit(nb_val_samples)相对，一个epoch有800张图片，验证集
np.save(open('E:/keras_data/data1/bottleneck_feature_test.npy','w'),bottleneck_feature_test)