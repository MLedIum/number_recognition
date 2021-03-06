from PIL import Image
import random
import os
import numpy
import shutil
import scipy.special

# Задаем переменные
width = height = 8

# Получаем текущую директорию
path = os.path.dirname(os.path.abspath(__file__)) + "\\"
# И директорию с тренировочными фото
trainphotos_dir = path + "trainphotos\\"


class NN:
    def __init__(
        self, inputnodes, hiddennodes, hiddennodes_2, outputnodes, learningrate, epochs
    ):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodes2 = hiddennodes_2
        self.onodes = outputnodes
        self.lr = learningrate
        self.epochs = epochs

        # Генерируем случайные значения весов
        self.weights_inhi = numpy.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)
        )
        self.weights_hihi = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes2, self.hnodes)
        )
        self.weights_hiout = numpy.random.normal(
            0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2)
        )

        # Функция активации
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self):
        epoch_n = 0
        for epoch in range(self.epochs):

            # Каждые 50 эпох выводим текущую эпоху
            if epoch_n == 50:
                print(epoch)
                epoch_n = 0

            for photo in os.listdir(trainphotos_dir):

                # Опустошаем массивы
                colors = list()
                true_answer = list()

                # Загружаем фотографию
                pix = Image.open(trainphotos_dir + photo).load()

                # Наполняем массив данными о цветах пикселей
                for x in range(width):
                    for y in range(height):
                        color = (pix[x, y][0] + pix[x, y][1] + pix[x, y][2]) / 256
                        colors.append(color)

                # Создаем матрицу из входных значений
                inputs_array = numpy.array([colors], float).T

                # Получаем название фото
                photo_name = photo.split(".png")[0]

                # Получаем правильный ответ
                for answers in range(10):
                    if answers == int(photo_name.split("_")[0]):
                        true_answer.append(1.0)
                    else:
                        true_answer.append(0.0)

                # Создаем матрицу желаемых ответов
                target = numpy.array([true_answer], float).T

                # Ужножаем матрицу входных значений на веса
                hidden_inputs = numpy.dot(self.weights_inhi, inputs_array)
                # Функция активации нейронов первого скрытого слоя
                hidden_outputs = self.activation_function(hidden_inputs)

                # Ужножаем матрицу выходных значений нейронов первого скрытого слоя на веса
                hidden2_inputs = numpy.dot(self.weights_hihi, hidden_outputs)
                # Функция активации нейронов второго скрытого слоя
                hidden2_outputs = self.activation_function(hidden2_inputs)

                # Ужножаем матрицу выходных значений нейронов второго скрытого слоя на веса
                final_inputs = numpy.dot(self.weights_hiout, hidden2_outputs)
                # Функция активации нейронов выходного слоя
                final_outputs = self.activation_function(final_inputs)

                # Высчитываем ошибки
                output_errors = target - final_outputs
                hidden2_errors = numpy.dot(self.weights_hiout.T, output_errors)
                hidden_errors = numpy.dot(self.weights_hihi.T, hidden2_errors)

                # Изменяем веса
                self.weights_hiout += self.lr * numpy.dot(
                    (output_errors * final_outputs * (1.0 - final_outputs)),
                    numpy.transpose(hidden2_outputs),
                )

                # print(hidden_outputs)
                # print(self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden2_outputs)))

                self.weights_hihi += self.lr * numpy.dot(
                    (hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),
                    numpy.transpose(hidden_outputs),
                )
                self.weights_inhi += self.lr * numpy.dot(
                    (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                    numpy.transpose(inputs_array),
                )

            epoch_n += 1

    def query(self):

        # Открываем фотографию
        pix = Image.open(path + "photo.png").load()

        colors_q = list()

        # Наполняем массив цветами по порядку (0 - черный, 1 - белый)
        for x in range(width):
            for y in range(height):
                color = (pix[x, y][0] + pix[x, y][1] + pix[x, y][2]) / 765
                colors_q.append(color)

        # Создаем матрицу из входных значений
        inputs = numpy.array([colors_q], float).T

        hidden_inputs = numpy.dot(self.weights_inhi, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden2_inputs = numpy.dot(self.weights_hihi, hidden_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = numpy.dot(self.weights_hiout, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        mansw = 0
        nansw = 0
        ntansm = 0
        for answ in final_outputs:
            if answ > mansw:
                mansw = answ
                ntansm = nansw
            nansw += 1

        print()
        print("Ответ: " + str(ntansm))


# Настройки сети
inputnodes = 64
hiddennodes = 256
hiddennodes_2 = 128
outputnodes = 10
learningrate = 0.1
epochs = 1500

network = NN(inputnodes, hiddennodes, hiddennodes_2, outputnodes, learningrate, epochs)
network.train()

# Функция назначения правильного ответа в случае ошибки нейронной сети
def setTrueAnswer():
    like_answer = input("Введите правильный ответ: ").replace(" ", "")

    if -1 < int(like_answer) < 10:
        pass
    else:
        like_answer = setTrueAnswer()

    return like_answer


while True:
    # Запускаем сканирование изображения
    input("Нажмите Enter для сканирования photo.png  ")
    network.query()

    # Узнаем правильный ли ответ
    question = input("Ответ правильный? 1 - Нет, Enter - Да: ").lower().replace(" ", "")
    if question == "1" or question == "нет":

        # Узнаем количество тренировочных файлов
        numberOfTrainPhotos = len(os.listdir(trainphotos_dir))

        # Копируем фото в папку с тренировочными файлами
        shutil.copy((path + "photo.png"), trainphotos_dir)
        # Переименовываем это фото для дальнейшего обучения сети
        os.rename(
            (trainphotos_dir + "photo.png"),
            (
                trainphotos_dir
                + setTrueAnswer()
                + "_"
                + str(numberOfTrainPhotos + 1)
                + ".png"
            ),
        )
    elif question == "t" or question == "train":
        network.train()
