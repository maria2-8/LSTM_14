import glob
import re
from collections import Counter
import numpy as np

texts_folder = 'C:/Users/79215/Dropbox/ПК/Downloads/Тексты писателей/Тексты писателей'

#функция для чтения файлов
def read_text(filename):
    with open(filename, encoding='utf-8') as f:
        text = f.read()
        text = text.replace('\n', ' ')
        return text

#список классов
class_names = ["О. Генри", "Стругацкие", "Булгаков", "Саймак", "Фрай", "Брэдберри"]
#кол-во классов
num_classes = len(class_names)

train_texts = []
test_texts = []

for class_name in class_names:
    for filename in glob.glob(f'{texts_folder}/*txt'):
        if class_name in filename:
            if 'Обучающая' in filename:
                train_texts.append(read_text(filename))
            if 'Тестовая' in filename:
                test_texts.append(read_text(filename))

stopwords_file = 'C:/Users/79215/Dropbox/ПК/Downloads/TopicModelingTool_1/TopicModelingTool/swl.txt'
#список стоп-слов
stopwords_list = []
with open(stopwords_file, encoding='utf-8') as f:
    stopwords = [word.strip() for word in f.readlines()]
    stopwords_list.extend(stopwords)

#Токенизация
def tokenize(texts_list: list) -> list:
    all_tokens = []
    for el in texts_list:
        print(len(el))
        tokens = re.findall(r"\w+", el) #убираем знаки препинания, оставляем только буквенные символы
        lower_tokens = []
        for token in tokens:
            token_lower = token.lower()
            lower_tokens.append(token_lower)
        no_sw_tokens = []
        for lt in lower_tokens:
            if lt not in stopwords_list: #убираем стоп-слова
                no_sw_tokens.append(lt)
        all_tokens.append(no_sw_tokens)
    return all_tokens

all_train_tokens = tokenize(train_texts)
all_test_tokens = tokenize(test_texts)

#список со словарями частотности
train_freq_dicts = []
for el in all_train_tokens:
    words_freqs = dict(Counter(el))
    sort_words_freqs = {k:v for k, v in sorted(words_freqs.items(), key=lambda item: item[1], reverse=True)}
    train_freq_dicts.append(sort_words_freqs)

test_freq_dicts = []
for el in all_test_tokens:
    words_freqs = dict(Counter(el))
    sort_words_freqs = {k:v for k, v in sorted(words_freqs.items(), key=lambda item: item[1], reverse=True)}
    test_freq_dicts.append(sort_words_freqs)

train_words_indexes = []
for text, dct in zip(all_train_tokens, train_freq_dicts):
    indices = list(map(lambda x: dct[x], text))
    train_words_indexes.append(indices)

test_words_indexes = []
for text, dct in zip(all_test_tokens, test_freq_dicts):
    indices = list(map(lambda x: dct[x], text))
    test_words_indexes.append(indices)

one_list = []
for text in all_train_tokens:
    one_list += text

train_words_set = set(one_list) #множество из уникальных слов, входящих во все тексты обучающей выборки
vocab_list = list(train_words_set)

print('Размер словаря:')
print(len(vocab_list))

# Формирование обучающей выборки по списку индексов слов
# (разделение на короткие векторы)
def getSetFromIndexes(wordIndexes, xLen, step): # функция принимает последовательность индексов, размер окна, шаг окна
  xSample = [] # Объявляем переменную для векторов
  wordsLen = len(wordIndexes) # Считаем количество слов
  index = 0 # Задаем начальный индекс
  while (index + xLen <= wordsLen):# Идём по всей длине вектора индексов
    xSample.append(wordIndexes[index:index+xLen]) # "Откусываем" векторы длины xLen
    index += step # Смещаемся вперёд на step
  return xSample


#делим на обучающую и тестовую выборку
def createSetsMultiClasses(wordIndexes, xLen, step): # Функция принимает последовательность индексов, размер окна, шаг окна
  # Для каждого из 6 классов
  # Создаём обучающую/проверочную выборку из индексов
  nClasses = len(wordIndexes) # Задаем количество классов выборки
  classesXSamples = []        # Здесь будет список размером "кол-во классов*кол-во окон в тексте*длину окна (например, 6 по 1341*1000)"
  for wI in wordIndexes:      # Для каждого текста выборки из последовательности индексов
    classesXSamples.append(getSetFromIndexes(wI, xLen, step)) # Добавляем в список очередной текст индексов, разбитый на "кол-во окон*длину окна"

  # Формируем один общий xSamples
  xSamples = [] # Здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна (например, 15779*1000)"
  ySamples = [] # Здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной 6"

  for t in range(nClasses): # В диапазоне кол-ва классов(6)
    #xT - видимо один текст с кучей окон
    xT = classesXSamples[t] # Берем очередной текст вида "кол-во окон в тексте*длину окна"(например, 1341*1000)
    for i in range(len(xT)): # И каждое его окно
      xSamples.append(xT[i]) # Добавляем в общий список выборки
      #добавляем каждое окно в xSamples

      #ySamples.append(utils.to_categorical(t, nClasses)) # Добавляем соответствующий вектор класса
      res_vector = np.zeros(nClasses)
      res_vector[t] += 1
      ySamples.append(res_vector)

      #nClasses - число классов (6)
      #[0,0,1,0,0,0]

  xSamples = np.array(xSamples) # Переводим в массив numpy для подачи в нейронку
  ySamples = np.array(ySamples) # Переводим в массив numpy для подачи в нейронку

  return (xSamples, ySamples) #Функция возвращает выборку и соответствующие векторы классов

xLen = 500
step = 50
xTrain_tokens, yTrain_tokens = createSetsMultiClasses(all_train_tokens, xLen, step)
xTest_tokens, yTest_tokens = createSetsMultiClasses(all_test_tokens, xLen, step)
print('Размерность обуч массива (train) с токенами')
print(xTrain_tokens.shape)
print('Размерность обуч массива (test) с токенами')
print(xTest_tokens.shape)

xTrain_ind, yTrain_ind = createSetsMultiClasses(train_words_indexes, xLen, step)
print('Размерность обуч массива (train) с индексами')
print(xTrain_ind.shape)
xTest_ind, yTest_ind = createSetsMultiClasses(test_words_indexes, xLen, step)
print('Размерность обуч массива (test) с индексами')
print(xTest_ind.shape)

def bag_of_words(input_array):
    bag_vectors = []
    for window in input_array:
        bag_vector = np.zeros(len(vocab_list))
        for token in window:
            for i, word in enumerate(vocab_list):
                if token == word:
                    bag_vector[i] += 1
                    #print(bag_vector)
        #print(bag_vector)
        bag_vectors.append(bag_vector)
    bag_vectors = np.array(bag_vectors)
    return bag_vectors

bag_vectors_xtrain = bag_of_words(xTrain_tokens)
bag_vectors_xtest = bag_of_words(xTest_tokens)

print('Длина массива с векторами (train):', ' ', len(bag_vectors_xtrain))
print('Его размерность', ' ', bag_vectors_xtrain.shape)
print('Длина одного окна', ' ', len(bag_vectors_xtrain[0]))

print('Длина массива с векторами (test):', ' ', len(bag_vectors_xtest))
print('Его размерность', ' ', bag_vectors_xtest.shape)
print('Длина одного окна', ' ', len(bag_vectors_xtest[0]))
