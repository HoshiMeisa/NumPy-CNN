from PIL import Image
import numpy as np

# filepath = '/home/kana/LinuxData/CNN/dataset/Car/train/flip+GANO/9'
# savepath = '/home/kana/LinuxData/CNN/dataset/Car/train/rotate+flip+GANO/9'

x = 'rotate+flip+GABL'
a = 1120
b = 3080+280+280+280+280+280+280+280+280+280+280+280+280
c = b + 140
for i in range(10):
    for j in range(140):
        if i == 9 and j == 56:
            continue
        img = Image.open(f'/home/kana/LinuxData/CNN/dataset/Car/train/{x}/{str(i)}' + "/" + str(j+a) + ".jpg", mode='r')
        img = np.asarray(img)
        img1 = img.copy()
        img2 = img.copy()

        img1[:, :, 0] = np.asarray((img1[:, :, 1] / 1.45), dtype=int)
        img1 = Image.fromarray(img1)
        img1.save(f'/home/kana/LinuxData/CNN/dataset/Car/train/green_{x}/{str(i)}' + '/' + str(j+b) + ".jpg")

        img2[:, :, 0] = np.asarray((img2[:, :, 2] / 1.45), dtype=int)
        img2 = Image.fromarray(img2)
        img2.save(f'/home/kana/LinuxData/CNN/dataset/Car/train/blue_{x}/{str(i)}' + '/' + str(j+c) + ".jpg")




# for i in range(140):
#     img = Image.open(filepath + "/" + str(i + 280 + 140 + 140) + ".jpg", mode='r')
#     img.transpose(Image.FLIP_LEFT_RIGHT).save(savepath + '/' + str(i + 280 + 140 + 140 + 140) + ".jpg")

# for j in range(10):
#     for i in range(140):
#         img = Image.open(f'/home/kana/LinuxData/CNN/dataset/Car/train/GABL/{str(j)}' + "/" + str(i+280) + ".jpg", mode='r')
#         img.rotate(10, translate=(0, -25), fillcolor="black").save(f'/home/kana/LinuxData/CNN/dataset/Car/train/rotate+GABL/{str(j)}' + '/' + str(i+1120+140+140+140) + ".jpg")

# for i in range(140):
#     img = Image.open(filepath + "/" + str(i+140) + ".jpg", mode='r')
#     img = img.filter(ImageFilter.GaussianBlur(3))
#     img.save(savepath + '/' + str(i+280+140) + ".jpg")


# def gaussian_noise(img, mean, sigma):
#     """
#     此函数用将产生的高斯噪声加到图片上
#     传入:
#         img   :  原图
#         mean  :  均值
#         sigma :  标准差
#     返回:
#         gaussian_out : 噪声处理后的图片
#         noise        : 对应的噪声
#     """
#     # 将图片灰度标准化
#     img = img / 255
#     # 产生高斯 noise
#     Noise = np.random.normal(mean, sigma, img.shape)
#     # 将噪声和图片叠加
#     gaussian_out = img + Noise
#     # 将超过 1 的置 1，低于 0 的置 0
#     gaussian_out = np.clip(gaussian_out, 0, 1)
#     # 将图片灰度范围的恢复为 0-255
#     gaussian_out = np.uint8(gaussian_out * 255)
#     # 将噪声范围搞为 0-255
#     # noise = np.uint8(noise*255)
#     return gaussian_out, Noise  # 这里也会返回噪声，注意返回值
# #
# #
# for i in range(140):
#     img = Image.open(filepath + "/" + str(i) + ".jpg", mode='r')
#     img = np.asarray(img)
#     img, x = gaussian_noise(img, mean=0, sigma=0.18)
#     img = Image.fromarray(img)
#     img.save(savepath + '/' + str(i + 280 + 140 + 140) + ".jpg")

# img = np.asarray(img)
# out, noise = gaussian_noise(img, mean=0, sigma=0.15)
# # cv2.imshow('noise', noise)
# cv2.imshow('out', out)
# cv2.waitKey(0)
# k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
# if k == 27:  # 键盘上Esc键的键值
#     cv2.destroyAllWindows()

# img = Image.open('/home/kana/LinuxData/CNN/dataset/Car/train/origin/0bus/1.jpg')
#
# img.show()
