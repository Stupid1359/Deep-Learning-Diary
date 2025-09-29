import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2

model_path = 'mnist_model.keras'
image_path = 'picture/img_2.png'

# 加载训练好的模型
def load_model(prepared_model_path):
    return tf.keras.models.load_model(prepared_model_path)


# 预处理上传的图片
def prepare_image(image_path, target_size=(28, 28)):
    """
    预处理上传的图片，调整尺寸，二值化，并反转颜色（背景黑，数字白）。
    参数：
    - image_path: 图片路径
    - target_size: 目标尺寸，默认为 (28, 28)
    返回：
    - 预处理后的图片（NumPy 数组）
    """
    try:
        # 加载图片，并调整大小
        img = image.load_img(image_path, target_size=target_size)
        # 转换为 NumPy 数组
        img_array = image.img_to_array(img)

        # 将图像转换为灰度图
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # 计算白色像素的占比
        white_pixels = np.sum(img_array > 128)  # 白色像素
        total_pixels = img_array.size  # 总像素数
        white_ratio = white_pixels / total_pixels

        # 如果白色像素占比超过 flip_threshold，则反转图像
        if white_ratio > 0.5:
            img_array = cv2.bitwise_not(img_array)  # 反转颜色

        # 指数变换
        c = 1  # 常数
        a = 50  # 根据图像的动态范围调整
        img_array_exp = c * (np.exp(img_array / a) - 1)

        # 归一化处理
        img_array_exp = img_array_exp / np.max(img_array_exp)

        # 将处理后的图像转换为 [0, 255] 范围，确保保存为有效的图像格式
        img_array_exp = np.uint8(img_array_exp * 255)
        # 显示图像
        plt.imshow(img_array_exp, cmap='gray')
        plt.axis('off')  # 不显示坐标轴
        plt.show()

        # 添加一个批次维度
        inverted_img = np.expand_dims(img_array_exp, axis=0)
        return inverted_img

    except Exception as e:
        print(f"图像加载或预处理出错：{e}")
        return None


# 对上传的图片进行预测
def predict_image(model, pimage_path):
    # 预处理图片
    img_array = prepare_image(pimage_path, target_size=(28, 28))

    # 如果返回值为 None，说明预处理出错
    if img_array is None:
        print("图像预处理失败，无法进行预测！")
        return None, None

    # 显示预处理后的图像
    plt.imshow(img_array[0], cmap='gray')
    plt.axis('off')
    plt.show()

    # 预测
    predictions = model.predict(img_array)
    # 获取预测的类别索引和对应的概率
    predicted_class = np.argmax(predictions[0])  # 最有可能的类别索引
    predicted_probability = np.max(predictions[0])  # 对应的最大概率

    return predicted_class, predicted_probability


def main():
    # 加载模型
    model = load_model(model_path)

    # 加载图片并显示
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # 对图片进行预测
    predicted_class, predicted_probability = predict_image(model, image_path)

    # 输出最有可能的类别和它的概率
    if predicted_class is not None and predicted_probability is not None:
        print(f"最有可能的数字: {predicted_class}")
        print(f"概率: {predicted_probability:.4f}")


# 运行示例
if __name__ == "__main__":
    main()
