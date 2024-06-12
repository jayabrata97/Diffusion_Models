import cv2
import glob


def number(filename):
    return int(filename[3:-4])


img_array = []
for filename in sorted(glob.glob("x0*.png"), key=number):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# print(type(img_array[0]))
# print(img_array[len(img_array) - 1])
img_array = img_array[::-1]
# print(img_array[0])

out = cv2.VideoWriter(
    "MNIST_generation.avi",
    cv2.VideoWriter_fourcc(*"DIVX"),
    30,
    size,
)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
