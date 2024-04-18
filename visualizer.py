import time
import random
import pygame
import pandas as pd
import numpy as np
from main import NeuralNetwork

class Neuron:
    def __init__(self, id):
        self.id = id
        self.color = (0, 0, 0)

class Weight:
    def __init__(self, id, position, color):
        self.id = id
        self.position = position
        self.color = color
        self.end = None
        self.vector = None
        self.unit_vec = None # for animation purposes
        self.ratio = None
        self.animate = False
        self.width = 1
class Network:
    def __init__(self, shape):
        self.shape = shape  # tuple (shape of nn)
        self.neuron_list = []
        self.weight_list = None

    def draw_neurons(self):
        x_pos = 200
        layer_id = 1
        position_list = []

        for layer in self.shape:
            y_pos = 0
            y_negative = 0
            y_middle = int(screen.get_height() / 2)
            y_increment = 40
            for neuron in range(layer):
                new_neuron = Neuron((layer_id, neuron + 1))

                if new_neuron.id[1] == layer / 2:
                    new_neuron.position = (x_pos, y_middle)

                elif new_neuron.id[1] < layer / 2:
                    y_negative += y_increment
                    y_multiplier = (layer / 2) - new_neuron.id[1]
                    new_neuron.position = (x_pos, y_middle - y_increment * y_multiplier)

                elif new_neuron.id[1] > layer / 2:
                    y_pos += y_increment
                    new_neuron.position = (x_pos, y_middle + y_pos)
                self.neuron_list.append(new_neuron)
                position_list.append(new_neuron.position)
            x_pos += 250
            layer_id += 1
        return position_list

    def animate_weights(self, step, total_frame, current_layer):
        weight_positions = []
        done_animating = False
        for weight in self.weight_list:  # weight is a param that hasnt been updated (no vector added)
            if weight.animate:
                if not weight.unit_vec:
                    weight_vec = [weight.position[1][0] - weight.position[0][0], weight.position[1][1] - weight.position[0][1]]
                    unit_vec = np.array(weight_vec) / np.linalg.norm(weight_vec)
                    weight.unit_vec = [unit_vec[0], unit_vec[1]]
                    weight.vector = [weight.position[1][0] - weight.position[0][0], weight.position[1][1] - weight.position[0][1]]
                    weight.ratio = weight.vector[0] / weight.unit_vec[0]
                    x = weight.position[0][0] + weight.unit_vec[0]
                    y = weight.position[0][1] + weight.unit_vec[1]
                    end_pos = (x, y)
                    weight.end = end_pos
                else:
                    if step == total_frame:
                        done_animating = True
                    if current_layer == weight.id:
                        increment = (weight.ratio / total_frame) * step
                        weight_anim = [weight.position[0][0] + weight.unit_vec[0] * increment, weight.position[0][1] + weight.unit_vec[1] * increment]
                        weight.end = weight_anim

            weight_positions.append(weight)

        self.weight_list = weight_positions
        return weight_positions, done_animating

    def draw_weights(self):
        weights = []
        for neuron in self.neuron_list:
            for other in self.neuron_list:
                if neuron.id[0] + 1 == other.id[0]:
                    weight_id = neuron.id[0]
                    new_weight = Weight(weight_id, (neuron.position, other.position), "WHITE")
                    random_bool = self.bool_based_on_probability(0.25)
                    if random_bool:
                        animated_weight = Weight(weight_id, (neuron.position, other.position), "YELLOW")
                        animated_weight.animate = True
                        animated_weight.width = 2
                        weights.append(animated_weight)
                    new_weight.end = new_weight.position[1]
                    weights.append(new_weight)
        return weights

    def load_data(self, path):
        data = pd.read_csv(path)
        data = np.array(data)
        m, n = data.shape  # m = number of images
        np.random.shuffle(data)  # shuffle before splitting into dev and training sets

        data_train = data[100:m].T
        Y_train = data_train[0]
        X_train = data_train[1:n]
        _, m_train = X_train.shape
        return X_train, Y_train

    def bool_based_on_probability(self, probability):
        return random.random() < probability

    def update_neurons(self, input, first, output, current):
        first_pixels = input[300:308].T
        last_pixels = input[308:316].T
        input_pixels = np.hstack((first_pixels, last_pixels))
        i = 0
        z = 0
        x = 0
        for neuron in self.neuron_list:
            if current == 1:
                if neuron.id[0] == current:
                    color = int(input_pixels[0][x])
                    neuron.color = (color, color, color)
                    x += 1
            elif current == 2:
                if neuron.id[0] == current:
                    ratio = first[z] / first.max()
                    color = 255 * ratio
                    neuron.color = (color, color, color)
                    z += 1
            elif current == 3:
                if neuron.id[0] == 3:
                    color = 255 * output[i]
                    neuron.color = (color, color, color)
                    i += 1

    def update_weights(self):
        pass


    def gray_scale(self, surface):
        arr = pygame.surfarray.array3d(surface)
        # calulates the avg of the "rgb" values, this reduces the dim by 1
        mean_arr = np.mean(arr, axis=2)
        # restores the dimension from 2 to 3
        mean_arr3d = mean_arr[..., np.newaxis]
        # repeat the avg value obtained before over the axis 2
        new_arr = np.repeat(mean_arr3d[:, :, :], 3, axis=2)
        # return the new surface
        return pygame.surfarray.make_surface(new_arr)

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
circle_size = 15

nn = NeuralNetwork()
saved_params = np.load('saved_nn.npz')
ih_weights = saved_params['array1']
ih_bias = saved_params['array2']
ho_weights = saved_params['array3']
ho_bias = saved_params['array4']
img_index = 0
done_animating = True

neural_network = Network([16, 10, 10])
neuron_list = neural_network.draw_neurons()
x_test, y_test = neural_network.load_data('digit-recognizer/train.csv')
x_test = x_test / 255.



font = pygame.font.Font('freesansbold.ttf', 25)

# create a text surface object,
# on which text is drawn on it.
draw_frames = 20
frame_step = 0
layer_number = 2
current_layer = 2
neuron_layer = 1
surf = None
text_prediction = None
text_label = None
text_rec = None
label_rec = None
start = True
input_pixels = None
first_layer = None
output_layer = None
prediction = None

while running:
    screen.fill("black")
    if done_animating and current_layer == layer_number:  # if done animating the final layer
        if not start:  # if weight is done animating but neuron hasnt
            neuron_layer += 1
            neural_network.update_neurons(input_pixels, first_layer, output_layer, neuron_layer)
            start = True
        else:
            for item in neural_network.neuron_list:
                item.color = (0, 0, 0)
            neuron_layer = 1
            start = False
            neural_network.weight_list = None
            frame_step = 0
            current_layer = 1
            img_test = x_test[:, [img_index]]
            first_layer, output_layer, prediction = nn.predict(ih_weights, ih_bias, ho_weights, ho_bias, img_test)
            input_pixels = img_test * 255.

            display_img = input_pixels.reshape((28, 28))
            surf = pygame.surfarray.make_surface(display_img)
            surf = neural_network.gray_scale(surf)
            surf = pygame.transform.scale(surf, (200, 200))
            surf = pygame.transform.rotate(surf, 270)
            surf = pygame.transform.flip(surf, True, False)

            text_prediction = font.render("Prediction: " + str(prediction[0]), True, "GREEN", "BLACK")
            text_rec = text_prediction.get_rect()
            text_rec.center = (900, 400)

            text_label = font.render("Label: " + str(y_test[img_index]), True, "GREEN", "BLACK")
            label_rec = text_label.get_rect()
            label_rec.center = (900, 430)

            if not neural_network.weight_list:
                weight_list = neural_network.draw_weights()
                neural_network.weight_list = weight_list

            neural_network.update_neurons(input_pixels, first_layer, output_layer, neuron_layer)

            img_index += 1
            time.sleep(2)

    elif done_animating and current_layer < layer_number:
        current_layer += 1
        neuron_layer += 1
        frame_step = 0
        neural_network.update_neurons(input_pixels, first_layer, output_layer, neuron_layer)

    if frame_step < draw_frames:
        frame_step += 1

    weight_lines, animation_status = neural_network.animate_weights(frame_step, draw_frames, current_layer)
    done_animating = animation_status

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for weight in weight_lines:
        if not weight.animate:
            pygame.draw.line(screen, weight.color, weight.position[0], weight.end, width=weight.width)

    for weight in weight_lines:
        if weight.animate:
            pygame.draw.line(screen, weight.color, weight.position[0], weight.end, width=weight.width)

    number_list = []
    for neuron in neural_network.neuron_list:
        if neuron.id[0] == 3:
            color = "WHITE"
            neuron_number = neuron.id[1] - 1
            if prediction[0] == neuron_number:
                color = "YELLOW"
            number_text = font.render(str(neuron_number), True, color, "BLACK")
            number_rec = number_text.get_rect()
            number_rec.center = (neuron.position[0] + 30, neuron.position[1])
            number_list.append((number_text, number_rec))
        pygame.draw.circle(screen, "WHITE", neuron.position, circle_size)
        pygame.draw.circle(screen, neuron.color, neuron.position, circle_size - 1)

    y_img = screen.get_height() / 2
    screen.blit(surf, (800, 150))
    screen.blit(text_prediction, text_rec)
    screen.blit(text_label, label_rec)
    for num in number_list:
        screen.blit(num[0], num[1])
    pygame.display.flip()
    clock.tick(60)  # limits FPS to 60


pygame.quit()