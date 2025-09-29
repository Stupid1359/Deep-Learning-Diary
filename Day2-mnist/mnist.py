import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0],"GPU")

from tensorflow.keras import datasets, layers,models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# 导入数据
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 归一化
train_images, test_images = train_images / 255.0, test_images / 255.0
print(train_images.shape,test_images.shape,train_labels.shape,test_labels.shape)
"""
((60000, 28, 28), (10000, 28, 28), (60000,), (10000,))
"""

# 显示
def show():
    plt.figure(figsize=(20,10))

    for i in range(20):
        plt.subplot(5,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])

    plt.show()
# show()

# 调整格式
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# train_images, test_images = train_images / 255.0, test_images / 255.0


# CNN
model = models.Sequential([
    # 输入层
    layers.Input(shape=(28,28,1)),
    # 卷积层1, 卷积核3*3
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    # 池化层1, 2*2采样
    layers.MaxPooling2D((2,2)),
    # 卷积层2, 卷积核3*3
    layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    # 池化层2, 2*2采样
    layers.MaxPooling2D((2,2)),
    # Flatten层, 连接卷积层与全连接层
    layers.Flatten(),
    # 全连接层, 特征进一步提取
    layers.Dense(64,activation='relu'),
    # 输出层, 输出预期结果
    layers.Dense(10, activation='softmax')
])

# 打印网络结构
model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# 保存模型到一个文件
model.save('mnist_model.keras')

# plt.imshow(test_images[1])
# plt.show()
# pre = model.predict(test_images)
# print(pre[1])

# 预处理上传的图片
def prepare_image(image_path):
    # 加载图片，并调整大小为28x28像素
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    # 转换为NumPy数组
    img_array = image.img_to_array(img)
    # 归一化处理，和训练时的处理一致
    img_array = img_array / 255.0
    # 添加一个批次维度，模型的输入是一个批次（即多个图像），所以需要添加这个维度
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 对上传的图片进行预测
def predict_image(prepared_model, image_path):
    # 预处理图片
    img_array = prepare_image(image_path)
    # 可选：显示图片
    plt.imshow(img_array[0], cmap='gray')
    plt.show()
    # 预测
    predictions = prepared_model.predict(img_array)
    predicted = np.argmax(predictions[0])  # 获取预测的类别索引
    return predicted


# 上传图片路径
picture_path = 'picture/img3.png'  # 替换为你的图片路径

# 对图片进行预测
predicted_class = predict_image(model, picture_path)

print(f"Predicted class: {predicted_class}")