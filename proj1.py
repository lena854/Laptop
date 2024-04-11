import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import clearml
from clearml import Task
import pandas as pd


# Установить ключ API вашего проекта
task1 = Task.init(project_name='Подбор ноутбука', task_name='Laptops')

# Загрузить данные из базы данных
data = pd.read_csv("C:/Users/antip/PycharmProjects/18Olga/best_buy_laptops_2024.csv")

# Предварительная обработка текста
def preprocess_text(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stopwords.words('russian')]
    return ' '.join(filtered_words)

# Привязка ключевых слов к категориям ноутбуков
categories = {
    'Большой экран': ['большой экран', 'широкий экран'],
    'Яркий дисплей': ['яркий дисплей', 'яркий экран'],
    'Домашняя версия': ['домашняя версия', 'home edition'],
    'Шустрый': ['шустрый', 'быстрый', 'производительный']
}

# Поиск соответствующих категорий для запроса пользователя
def find_matching_categories(query):
    preprocessed_query = preprocess_text(query)
    matching_categories = []
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in preprocessed_query:
                matching_categories.append(category)
                break
    return matching_categories

# Функция для поиска ноутбуков по признакам
def find_matching_notebooks(user_query, data):
    matching_notebooks = []
    for index, notebook in data.iterrows():
        if 'легкий' in user_query.lower():
            if 1 <= notebook['depth'] <= 2:
                matching_notebooks.append(notebook)
        if 'небольшой' in user_query.lower():
            if notebook['width'] <= 15.4 and notebook['depth'] <= 3:
                matching_notebooks.append(notebook)
        if 'недорогой' in user_query.lower():
            if notebook['price'] <= 1000:
                matching_notebooks.append(notebook)
        if 'классика' in user_query.lower():
            if notebook['type'] == 'классика':
                matching_notebooks.append(notebook)
        if 'ультрабук' in user_query.lower():
            if notebook['portability'] == 'ультрабук':
                matching_notebooks.append(notebook)
        if 'трансформер' in user_query.lower():
            if notebook['type'] == 'трансформер':
                matching_notebooks.append(notebook)
        if 'сенсорный' in user_query.lower():
            # Предположим, что это относится к наличию сенсорного экрана
            # В нашем примере базы данных это не учтено, но вы можете добавить соответствующий признак
            pass
        # Добавьте дополнительные ключевые слова и соответствующие признаки здесь
    # Отправить запрос и рекомендации в ClearML
    task1.comment(f"Запрос пользователя: {user_query}")
    task1.comment(f"Рекомендуемые ноутбуки: {[notebook['model'] for notebook in matching_notebooks]}")
    return matching_notebooks

# Пример запроса пользователя
user_query = "Хочу легкий ноутбук, который можно брать с собой в дорогу"

# Рекомендация ноутбуков для пользователя
recommended_notebooks = find_matching_notebooks(user_query, data)
if recommended_notebooks:
    print("Рекомендуемые ноутбуки:")
    for notebook in recommended_notebooks:
        print(notebook['model'])
else:
    print("Извините, не удалось найти подходящие ноутбуки.")
