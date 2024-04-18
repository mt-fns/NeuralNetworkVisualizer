import pygame
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



pygame.init()

screen = display = pygame.display.set_mode((500, 500))
x = np.arange(0, 300)
y = np.arange(0, 300)
X, Y = np.meshgrid(x, y)
Z = X + Y
Z = 255*Z/Z.max()
print(Z.shape)

path = 'digit-recognizer/train.csv'
data = pd.read_csv(path)
data = np.array(data)
m, n = data.shape  # m = number of images
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

def greyscale(surface: pygame.Surface):
	arr = pygame.surfarray.array3d(surface)
	# calulates the avg of the "rgb" values, this reduces the dim by 1
	mean_arr = np.mean(arr, axis=2)
	# restores the dimension from 2 to 3
	mean_arr3d = mean_arr[..., np.newaxis]
	# repeat the avg value obtained before over the axis 2
	new_arr = np.repeat(mean_arr3d[:, :, :], 3, axis=2)
	# return the new surface
	return pygame.surfarray.make_surface(new_arr)

data_train = data[100:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

img_test = X_train[:, [0]].reshape((28, 28))

plt.gray()
plt.imshow(img_test, interpolation='nearest')
plt.show()


surf = pygame.surfarray.make_surface(img_test)
surf = greyscale(surf)
surf = pygame.transform.scale(surf, (400, 400))
surf = pygame.transform.rotate(surf, 270)
surf = pygame.transform.flip(surf, True, False)
crashed = False

while not crashed:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			crashed = True
	display.blit(surf, (0, 0))
	pygame.display.update()


pygame.quit()