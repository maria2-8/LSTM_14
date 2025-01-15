import sys,random,math
from collections import Counter
import numpy as np
import sys
from classes_LSTM import Tensor
from classes_LSTM import Embedding, LSTMCell
from classes_LSTM import CrossEntropyLoss, SGD
#надо импортировать классы из другого файла

np.random.seed(0)

# читаем файл
f = open("C:/Users/79215/Dropbox/ПК/Downloads/shakespear.txt",'r')
raw = f.read() # файл читается как одна строка (не по отдельным строкам, как readlines())
f.close()

vocab = list(set(raw)) # список символов (не слов)
word2index = {} # каждому символу сопоставляется его индекс
for i, word in enumerate(vocab):
    word2index[word] = i
indices = np.array(list(map(lambda x: word2index[x], raw))) #массив с индексами символов

# слой, где создаются векторные представления для последовательностей символов
embed = Embedding(vocab_size=len(vocab),dim=512)
# ячейка LSTM
model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
model.w_ho.weight.data *= 0 #это поможет в обучении

# перекрестная энтропия (функция потерь)
criterion = CrossEntropyLoss()
# оптимизатор стохастического градиентного спуска
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

# эта функция сохраняет прогнозы в строке и возвращает их строковую версию
def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
#         output.data *= 25
#         temp_dist = output.softmax()
#         temp_dist /= temp_dist.sum()

#         m = (temp_dist > np.random.rand()).argmax()
        m = output.data.argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s

batch_size = 16
bptt = 25 # граница усечения
n_batches = int((indices.shape[0] / (batch_size)))

# усекаем набор данных до размера, кратного произведению n_batches и batch_size
# это делается для приведения набора данных к прямоугольной форме перед группировкой в тензоры
trimmed_indices = indices[:n_batches*batch_size]
#меняем форму набора данных так, чтобы каждый столбец представлял сегмент начального массива индексов
#число столбцов = batch_size
batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:] #это входные индексы, смещенные на одну строку

# создаем несколько наборов данных, размер каждого из которых равен bptt
n_bptt = int(((n_batches-1) / bptt))
input_batches = input_batched_indices[:n_bptt*bptt].reshape(n_bptt,bptt,batch_size)
target_batches = target_batched_indices[:n_bptt*bptt].reshape(n_bptt, bptt, batch_size)
min_loss = 1000

def train(iterations=400):
    for iter in range(iterations):
        total_loss = 0
        n_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)
        batches_to_train = len(input_batches) #32
        for batch_i in range(batches_to_train):
            hidden = (Tensor(hidden[0].data, autograd=True), Tensor(hidden[1].data, autograd=True))

            losses = list()
            #после каждых bptt шагов выполняется обратное распространение и корректировка весов
            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)

                target = Tensor(target_batches[batch_i][t], autograd=True)
                # вычисляется ошибка с помощью перекрестной энтропии
                batch_loss = criterion.forward(output, target) # генерируется на каждом шаге

                if (t == 0):
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss + losses[-1])

            loss = losses[-1] #записывается только последняя ошибка

            loss.backward() #обратное рапространение
            optim.step() #корректировка весов
            #потом чтение данных продолжается с использованием того же скрытого состояния, что и прежде
            #скрытое состояние поменяется при смене эпохи

            total_loss += loss.data / bptt #общая ошибка

            #ошибка на всей эпохе
            epoch_loss = np.exp(total_loss / (batch_i + 1))
            if (epoch_loss < min_loss):
                min_loss = epoch_loss
                print()

            #информация о ходе обучения сети
            log = "\r Iter:" + str(iter)
            log += " - Alpha:" + str(optim.alpha)[0:5]
            log += " - Batch " + str(batch_i + 1) + "/" + str(len(input_batches))
            log += " - Min Loss:" + str(min_loss)[0:5]
            log += " - Loss:" + str(epoch_loss)

            if (batch_i == 0):
                #генерируем последовательность из 70 символов, начальный символ - 'T'
                log += " - " + generate_sample(n=70, init_char='T').replace("\n", " ")
            if (batch_i % 1 == 0):
                sys.stdout.write(log) #записываем информацию об обучении сети
            optim.alpha *= 0.99

train(100)