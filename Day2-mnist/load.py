import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 获取第一张图像
first_image = train_images[0]  # 第一张图像

# 将图像从 (28, 28) 变为 (28, 28, 1) 以适应保存格式
first_image = np.expand_dims(first_image, axis=-1)

# 可选：显示图像
plt.imshow(first_image.squeeze(), cmap='gray')  # 使用 .squeeze() 来去掉单一维度
plt.show()

# 保存图像为 PNG 文件
image.save_img('picture/image.png', first_image)

print("第一张图像已保存为 'picture/image.png'")
