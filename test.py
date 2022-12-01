from PIL import Image
import numpy as np

x = 'rotate+GANO'
a = 1400
b = 3080+280+280+280+280+280+280+280+280+280
c = b + 140
for i in range(10):
    for j in range(140):
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
