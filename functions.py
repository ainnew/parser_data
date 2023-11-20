import re
import json
import string
from itertools import chain

import pandas as pd
import missingno as msno
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

from scipy import stats
from sklearn.preprocessing import PowerTransformer

import pymorphy2
import nltk
from stop_words import get_stop_words
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from fuzzywuzzy import fuzz
import spacy

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.vectorstores import FAISS

# !python -m spacy download ru_core_news_md
nlp = spacy.load("ru_core_news_md")

# nltk.download('punkt', quiet=True)
MA = pymorphy2.MorphAnalyzer()
STOPWORDS_RU = get_stop_words('russian')
WORD = re.compile(r'\w+')

def set_styles(color1 = 'tab:blue', color2 = 'tab:orange', color3 = 'tab:green', cmap = LinearSegmentedColormap.from_list("", ['tab:blue', 'white', 'tab:orange']), styles = []):
    """
    Функция оформления вывода.
    Назначает стили и цвета для отображения датафреймов и графиков.
    Параметры:
        color1 : default 'tab:blue', первый цвет графиков
        color2 : default 'tab:orange', второй цвет графиков
        styles : default '[]', список словарей со стилями для отображения датафрейма
    """
    global df_style
    global c1
    global c2
    global c3
    global cm
    global wc
    df_style = styles
    c1 = color1
    c2 = color2
    c3 = color3
    cm = cmap
    wc = wc

# Назначение цветов и стилей 
color1, color2, color3 = 'Blue', 'DeepPink', 'Lime'
cmap = LinearSegmentedColormap.from_list("", [color1, 'white', color2])
styles = []
x, y = np.ogrid[:1000, :1000]
mask = 255 * ((x - 500) ** 2 + (y - 500) ** 2 > 400 ** 2).astype(int)
wc = WordCloud(
                random_state=42, 
                background_color='white', 
                colormap='Set2', 
                collocations=False,
                max_words=32,
                # stopwords = STOPWORDS_RU, 
                mask=mask)
set_styles(color1, color2, color3, cmap, wc)


# Функции для предварительного просмотра общей статистики по данным
def describe_data(df):
    """
    Функция выводит общую статистическую информацию по нумерическим и ненумерическим данным
    Нумерические данные: count, mean, std, min, max, перцентили (25%, 50%, 75%)
    Ненумерические данные: count, unique, top, freq (частота самого популярного значения)
    Параметры:
        df : датафрейм
    Выводит:
        df : датафрейм
    """
    n_cols = df.select_dtypes(include=['number']).columns
    if not n_cols.empty:
        display(df[n_cols].describe().apply(lambda x: x.map('{:.2f}'.format)))
    o_cols = df.select_dtypes(include=['O']).columns
    if not o_cols.empty:
        display(df[o_cols].describe())


def _show_isna_small(df):
    """
    Вспомогательная функция вывода графиков для функции describe_isna
    """
    fig = plt.figure(figsize=(18,4))
    ax1 = fig.add_subplot(1,2,1)
    msno.bar(df, color=c1, fontsize=8, ax=ax1)
    ax2 = fig.add_subplot(1,2,2)
    msno.heatmap(df, cmap=cm, fontsize=8, ax=ax2)
    plt.tight_layout()
    
    fig = plt.figure(figsize=(18,4))
    ax3 = fig.add_subplot(1,2,1)
    sns.heatmap(df.isnull(), cmap=sns.color_palette([c1, 'white']), ax=ax3)
    ax4 = fig.add_subplot(1,2,2)
    msno.dendrogram(df, fontsize=8, ax=ax4)
    plt.tight_layout()

def _show_isna_big(df):
    """
    Вспомогательная функция вывода графиков для функции describe_isna
    """
    _, ax = plt.subplots(figsize=(18, len(df) / 50))  # Устанавливаем размер графика
    msno.bar(df, sort='ascending', color=c1, fontsize=8, ax=ax)  # Используем параметры figsize и ax для настройки графика
    plt.show()

    _, ax = plt.subplots(figsize=(18, 18))
    msno.heatmap(df, labels=False, cbar=False, cmap=cm, fontsize=8, ax=ax)
    plt.show()  


def describe_isna(df, show = True, format = 'small'):
    """
    Функция выводит форму, размер датафрейма и общую информацию о пропущенных значениях, а также данные о количестве и долях пропущенных значений в каждом столбце.
    Вывод сопровождается четырьмя графиками (barplot, heatmap , matrix plot, dendrogram из библиотеки missingno), два последних из которых выводятся опционально.
    Параметры:
        df : датафрейм
        show_matrix : bool (default True), опциональный вывод графиков matrix plot, dendrogram
    Выводит:
        текстовая информация
        df : датафрейм
        графики barplot, heatmap , matrix plot, dendrogram
    """
    size = df.size
    isna = df.isna().sum().sum()
    print('shape', df.shape)
    print('size', size)
    print('isna', isna)
    print('isna share {:.2%}'.format(isna / size))
    df_isna = pd.concat([df.dtypes, df.count(), df.isna().sum(), (df.isna().sum() / df.shape[0]).map('{:.2%}'.format)], axis=1).rename(columns = {0 : 'dtype', 1 : 'size', 2 : 'isna', 3 : 'isna_share'})
    display(df_isna)
    if show:
        if format == 'small':
            _show_isna_small(df)
        elif  format == 'big':
            _show_isna_big(df)
        else:
            print('Выберите значение параметра format: small или big')
            



def _apply_sigma(df_test, name_test, sigma, how):
    """
    Вспомогательная функция для применения установленного количества сигм к столбцу
    """
    if how == 'both':
        upper_level = df_test[name_test].mean() + sigma * df_test[name_test].std()
        lower_level = df_test[name_test].mean() - sigma * df_test[name_test].std()
        return df_test[(df_test[name_test] < upper_level) & (df_test[name_test] > lower_level)]
    elif how == 'right':
        upper_level = df_test[name_test].mean() + sigma * df_test[name_test].std()
        return df_test[df_test[name_test] < upper_level]
    elif how == 'left':
        lower_level = df_test[name_test].mean() - sigma * df_test[name_test].std()
        return df_test[df_test[name_test] > lower_level]
    else:
        print('There is no how parameter here')



def apply_quantile_sigma_sequence(df, quantile = .999, sigma = 3, how = 'both', interval = 0.95, seq_columns = [], apply_quantile = True):
    """
    Выводит гистограммы распределения значений выбранных столбцов при каждом последовательном изменении сигмами.
    Возвращает нормализованные данные и данные, удаленные при нормализации.
    Каждый график сопровождается дополнительной информацией:
        len - длина датафрейма и ее изменение (в т.ч. остаточная доля)
        skew - коэффициент асимметрии, скос графика влево или вправо
        kurtosis - эксцесс, выпуклость или вогнутость графика
        interval - минимальные и максимальные значения, которые попадают в доверительный интервал
        min-max - минимальные и максимальные значения
        top - 5 популярных значений
    Последовательность столбцов для применения квантилей и сигм задается в параметре seq_columns.
    Опционально могут отрезаться 0.999 квантиль (до преобразования сигмами) по выбранным столбцам.
    В анализе участвуют только нумерические столбцы.
    Параметры:
        df : датафрейм
        quantile : float (default '.999'), значение квантиля
        sigma : int (default '3'), количество применяемых сигм
        how : str (default 'both'), варианты применения сигм: 'both' - применение с двух сторон, 'right' - применение только сверху, 'left' - применение только снизу
        interval : float (0,1) (default '0.95'), вероятность доверительного интервала
        seq_columns : list, список нумерических столбцов в выбранной последовательности
        apply_quantile : bool (default True), применение квантиля до применения сигм
    Выводит:
        текстовая информация
        графики гистограммы
    Возвращает:
        df : датафрейм с нормализованными данными
        df : датафрейм с данными, удаленными при нормализации
    """
    df_outliers = pd.DataFrame()
    
    if len(seq_columns) > 0:
        print('sequence:', seq_columns)
    else:
        for i in df.columns:
            if (df[i].dtype == 'int') | (df[i].dtype == 'float'):
                seq_columns.append(i)
        print('sequence:', seq_columns)
    print('-'*70)
    print('before sequence:', len(df)) 
    df_norm = df.copy()
    
    plt.rcParams['font.size'] = 8
    plt.figure(figsize = (16,4))
    for start, name_start in enumerate(seq_columns):
        inter = stats.norm.interval(interval, loc = df[name_start].mean(), scale = df[name_start].std())
        plt.subplot(1, len(seq_columns), start + 1)
        # sns.histplot(df[name_start], kde=True, kde_kws=dict(cut=3), bins=50, element="step") #, stat="density"
        plt.hist(df[name_start], bins = 100, color = c2, alpha = .8)
        plt.title(f'{name_start.capitalize()} - len {len(df[name_start])}\nkurtosis {round(stats.kurtosis(df[name_start]), 3)}, skew {round(stats.skew(df[name_start]), 3)}\ninterval (probability {interval}) {list(map(lambda n: round(n, 2), inter))}\nmin-max {[df[name_start].min(), df[name_start].max()]}\ntop {df[name_start].value_counts().nlargest(5).index.to_list()}', loc='left', pad=20, fontsize = 8)
        print(f'\t{name_start.capitalize()}: min-max {[df[name_start].min(), df[name_start].max()]}')
    plt.tight_layout()
    print('-'*70)

    if apply_quantile:
        for i in seq_columns:
            df_temp = df_norm.copy()
            Q = df_norm[i].quantile(quantile)
            df_norm = df_norm[df_norm[i] <= Q]
            
            print(f'after applying quantile({quantile}) to {i}: {len(df_norm)}')
            df_outliers = pd.concat([df_outliers, df_temp[~df_temp.index.isin(df_norm.index)].assign(cut = f'{i} {quantile}quantile')], ignore_index=True)
        
        plt.figure(figsize = (16,4))
        for q, name_q in enumerate(seq_columns):
            inter = stats.norm.interval(interval, loc = df_norm[name_q].mean(), scale = df_norm[name_q].std())
            plt.subplot(1, len(seq_columns), q + 1)
            plt.hist(df_norm[name_q], bins = 100, color = c3, alpha = .8)
            plt.title(f'{name_q.capitalize()}  after {quantile} quantile - len {len(df_norm[name_q])} (share {round(len(df_norm) / len(df), 3)})\nkurtosis {round(stats.kurtosis(df_norm[name_q]), 3)}, skew {round(stats.skew(df[name_q]), 3)}\ninterval (probability {interval}) {list(map(lambda n: round(n, 2), inter))}\nmin-max {[df_norm[name_q].min(), df_norm[name_q].max()]}\ntop {df_norm[name_q].value_counts().nlargest(5).index.to_list()}', loc='left', pad=20, fontsize = 8)
            print(f'\t{name_q.capitalize()}: min-max {[df_norm[name_q].min(), df_norm[name_q].max()]}')
        plt.tight_layout()

    for name_i in seq_columns:
        df_temp = df_norm.copy()
        df_norm = _apply_sigma(df_norm, name_i, sigma, how)
        print('-'*70)
        print(f'after applying sigma({sigma}) to {name_i}: {len(df_norm)}')
        plt.figure(figsize = (16,4))
        for j, name_j in enumerate(seq_columns):
            inter_after_sigma = stats.norm.interval(interval, loc = df_norm[name_j].mean(), scale = df_norm[name_j].std())
            plt.subplot(1, len(seq_columns), j + 1)
            plt.hist(df_norm[name_j], bins = 100, color = c1, alpha = .8)
            plt.title(f'{name_j.capitalize()} after {sigma}sigma - len {len(df_norm)} (share {round(len(df_norm) / len(df), 3)})\nkurtosis {round(stats.kurtosis(df_norm[name_j]), 3)}, skew {round(stats.skew(df_norm[name_j]), 3)}\ninterval (probability {interval}) {list(map(lambda n: round(n, 2), inter_after_sigma))}\nmin-max {[df_norm[name_j].min(), df_norm[name_j].max()]}\ntop {df_norm[name_j].value_counts().nlargest(5).index.to_list()}', loc='left', pad=20, fontsize = 8)
            print(f'\t{name_j.capitalize()}: min-max {[df_norm[name_j].min(), df_norm[name_j].max()]}')
        plt.tight_layout()
        df_outliers = pd.concat([df_outliers, df_temp[~df_temp.index.isin(df_norm.index)].assign(cut = f'{name_i} {sigma}sigma')], ignore_index=True)
    print('-'*70)
    print('after sequence:', len(df_norm))
    print('outliers count:', len(df) - len(df_norm), end = '\n\n')
 
    return df_norm, df_outliers





# Функции нормализации числовых столбцов
def norm_num(df_in, col, methods = ['box-cox', 'yeo-johnson', 'log'], inplace = False, quite = False):
    if inplace:
        df = df_in
    else:
        df = df_in.copy()

    if 'box-cox' in methods or not methods:
        if (df[col] > 0).all():
            bc_transform = PowerTransformer(method='box-cox', standardize=False) # box-cox преобразование работает только с позитивными значениями
            df[f'{col}_boxcox'] = bc_transform.fit_transform(df[col].values.reshape(df.shape[0],-1))
        else:
            print('box-cox преобразование работает только с позитивными значениями')
    if 'yeo-johnson' in methods or not methods:
        yj_transform = PowerTransformer(method='yeo-johnson', standardize=False)
        df[f'{col}_yeojohnson'] = yj_transform.fit_transform(df[col].values.reshape(df.shape[0],-1)) # yeo-johnson преобразование
    if 'log' in methods or not methods:
        df[f'{col}_log'] = np.log(df[col].values.reshape(df[col].shape[0],-1)) # логарифмирование
    

    if not quite:
        df_temp = df.loc[:, df.columns.str.startswith(f'{col}_')]
        plt.rcParams['font.size'] = 8
        plt.figure(figsize = (18 / len(df_temp.columns), 3))
        analyze(df[col])
        plt.show()
        plt.figure(figsize = (18, 3))
        for i, c in enumerate(df_temp.columns):
            plt.subplot(1, len(df_temp.columns), i + 1)
            analyze(df_temp[c])
        plt.show()
    return df







# Функция для токенизации и фильтрации текстов
def _reg_tokenize(text):
    """
    Функция токенизации текста на основе регулярных выражений
    """
    words = WORD.findall(text)
    return words

def get_tokens(text):
    """
    Функция токенизации текста, а также фильтрации и нормализации токенов
    Параметры:
        text : str, текст
    Возвращает:
        tokens : list, список отфильтрованных токенов
    """
    text = text.replace(r'([^\w\s]+)', ' \\1 ').strip().lower() # вклинивание пробелов между словами и знаками препинания, приведение к нижнему регистру
    # print(text)
    tokens = _reg_tokenize(text) # токенизация
    # print(tokens)
    # tokens = [element for element in list(chain(*[re.split(r'\W+', element) for element in tokens])) if element != ''] # разделение составных элементов слов
    # print(tokens)
    tokens = list(chain(*[re.findall(r'\d+|\D+', element) if element.isalnum() else element for element in tokens])) # разбиение токенов, состоящих из букв и цифр
    # print(tokens)
    tokens = [MA.parse(token)[0].normal_form for token in tokens] # нормализация токенов
    # print(tokens)
    # tokens = [i for i in tokens if i not in stopwords + ['свой', 'весь', 'ваш']] # фильтрация по стоп-словам
    tokens = [i for i in tokens if i not in STOPWORDS_RU] # фильтрация по стоп-словам
    # print(tokens)
    tokens = [i for i in tokens if MA.parse(i)[0].tag.POS not in ['INTJ', 'PRCL', 'CONJ', 'PREP', 'PRED', 'NPRO']] # фильтрация по частям речи
    # print(tokens, end = '\n\n\n')
    return tokens


def get_marketplace_df(df_name):
    """
    Функция загрузки и обработки данных маркетплейсов
    Параметры:
        df : датафрейм
    Возвращает:
        df - преобразованный датафрейм
    """
    # Загружаем датафрейм
    skipcols = [] #'card_price', 'price', 'reg_price', 'rate', 'rate_count'
    df = pd.read_csv(df_name, usecols=lambda x: x not in skipcols)
    print('Columns', [i for i in df.columns])
    # Чистим и обрабатываем датафрейм
    df = df.drop(df[df['text'].isnull()].index) # удаление строк с пустым описанием
    df['tokenized_text'] = df['text'].apply(get_tokens) # токенизация и фильтрация текстов
    df = df[df['tokenized_text'].str.len() > 1] # фильтруем строки по количеству токенов
    df['specs'] = df['specs'].apply(lambda x: json.loads(str(x)))
    df = df[df['specs'].str.len() > 0]

    # Превращаем словари атрибутов в поля датафрейма
    df_specs = df['specs'].apply(pd.Series)
    df = pd.concat([df.drop(['specs'], axis=1), df_specs], axis=1)
    return df


# Функции фильтрации датафрейма по условию
def get_filtered_df(df, col, expr):
    """
    Функция фильтрует датафрейм по регулярным выражениям к указанному столбцу.
    Параметры:
        df : датафрейм
        col : колонка датафрейма
        expr : регулярное выражение
    Возвращает:
        df_true - датафрейм c положительным результатом
        df_false - датафрейм c отрицательным результатом
    """
    # Отбираем тексты товаров, относящихся к целевым для проверки результата отбора
    mask = df[col].str.contains(expr, flags=re.IGNORECASE, regex=True) #, na=False
    df_true = df[mask].reset_index(drop = True)
    df_false = df[~mask].reset_index(drop = True)
    return df_true, df_false



# Функции для демонстрации облака слов по определенным полям (опционально по строкам) датафрейма
def _show_wordcloud_all(df, columns):
    """
    Функция построения графика
    """
    plt.figure(figsize = (6 * len(columns), 10))
    for i, col in enumerate(columns):
        plt.subplot(1, len(columns), i + 1)
        if df[col].head(10).apply(lambda v: isinstance(v, list)).any():
            wc.generate(' '.join(list(chain.from_iterable(df[col]))))
        else:
            wc.generate(' '.join([k for k in df[col]]))

        plt.imshow(wc, interpolation="bilinear")
        plt.title(f'column {col}', fontsize = 8) 
        plt.axis("off")

def _show_wordcloud_by_row(df, columns):
    """
    Функция построения графика для каждой строки датафрейма
    """
    _, axs  = plt.subplots(len(df), len(columns), figsize = (6 * len(columns), 6 * len(df)))
    for i in range(len(df)):
        for j, col in enumerate(columns):
            if isinstance(df[col][i:i+1].values[0], list):
                wc.generate(' '.join(df[col][i:i+1].values[0]))
            else:
                wc.generate(df[col][i:i+1].values[0])
            if (len(df) == 1):
                axs = axs.reshape(1, -1)
            if (len(columns) == 1):
                axs = axs.reshape(-1, 1)
            axs[i, j].imshow(wc, interpolation="bilinear")
            axs[i, j].set_title(f'Column: {col}, Title: {df.title[i:i+1].values[0]}', fontsize = 8)
            axs[i, j].axis("off")

def show_wordcloud(df, columns, by_row = False):
    """
    Функция демонстрирует облака слов в определенных полях таблицы. Опционально можно задать просмотр по строкам
    Параметры:
        df : датафрейм
        columns : list, список выбранных текстовых полей
        by_row : bool, опционально можно задать вывод отдельно по строкам
    Показывает:
        графики с облаками слов 
    """
    if by_row:
        _show_wordcloud_by_row(df, columns)
    else:
        _show_wordcloud_all(df, columns)




# Функции построения LdaModel для тематического распознавания текстов
def get_corpus(df):
    """
    Функция для получения словаря и корпуса данных
    """
    common_dictionary = Dictionary(df) # подаем список списков слов, формируем слова
    common_corpus = [common_dictionary.doc2bow(text) for text in df]  # превращаем предложение в вектор bow
    return common_dictionary, common_corpus


def get_lda_model(common_dictionary, common_corpus, num_topics):
    """
    Функция для обучения LdaModel
    """
    lda_model = LdaModel(common_corpus, num_topics=num_topics, id2word=common_dictionary) # обучение модели
    return lda_model


def get_topic_weight(df, num_topics):
    """
    Функция получения весов слов для каждой темы и каждого текста с помощью LDA модели.
    Используется для последующей проверки текстов на соответствие нужным темам 
    Параметры:
        s : серия - тексты
        num_topics : int, количество тем для проверки
    Возвращает:
        df_topic_weights - веса слов для каждой темы
        df_doc_topic_weights - веса слов для каждой темы каждого текста
    """
    common_dictionary, common_corpus = get_corpus(df) # получение векторов
    lda_model = get_lda_model(common_dictionary, common_corpus, num_topics) # получение обученной модели
    # получение весов слов для каждой темы
    topic_weights = {}
    for topic_id in range(num_topics):
        words_weights = lda_model.show_topic(topic_id)
        topic_weights[topic_id] = words_weights
    # получение весов слов для каждой темы каждого текста
    doc_topic_weights = []
    for text in df:
        doc_bow = common_dictionary.doc2bow(text)
        doc_topics = lda_model.get_document_topics(doc_bow)
        doc_topic_weights.append([(num, weight) for num, weight in doc_topics])
    df_topic_weights = pd.DataFrame(topic_weights)
    df_topic_weights.columns = ['topic_{}'.format(i) for i in range(num_topics)]
    df_doc_topic_weights = pd.DataFrame(doc_topic_weights, columns=df_topic_weights.T.index)
    return df_topic_weights, df_doc_topic_weights


# Функция анализа распределения данных
def analyze(col):
    """
    Функция десмонстрации распределения данных
    Параметры:
        col : столбец датафрейма
    Показывает:
        график барплот значений
        показатели средней, дисперсии, асимметрии, эксцесса,  теста на нормальность данных Шапиро-Уилка
    """
    plt.rcParams['font.size'] = 8
    plt.style.use('ggplot')
    plt.hist(col, bins=60, color = c1, alpha = .8)
    plt.title(f"{col.name.title()}\nmedian : {round(np.median(col), 2)}\nmean : {round(np.mean(col), 2)}\nvar : {round(np.var(col), 2)}\nskew : {round(stats.skew(col), 2)}\nkurt : {round(stats.kurtosis(col), 2)}\nshapiro : {round(stats.shapiro(col)[0], 2)}\n", fontdict = {'fontsize' : 8}) #stats.normaltest : {normaltest(data)}




# Функция удаляет непопулярные столбцы по трешхолду
def del_na_attr(df, threshold, name):
    """
    Функция получения весов слов для каждой темы и каждого текста с помощью LDA модели.
    Используется для последующей проверки текстов на соответствие нужным темам 
    Параметры:
        df : датафрейм
        threshold : float, трешхолд
    Показывает:
        график гистограммы распределения значений
        график барплот значений
    Возвращает:
        df - преобразованный датафрейм
    """
    df1 = df.isnull().sum() * 100 / len(df)
    plt.rcParams['font.size'] = 8
    ax1 = df1.hist(figsize = (10,3), bins = 100, color = c2, alpha = 0.5)
    ax1.set_title(f"{name}. До применения порога. Гистограмма распределения na-значений по столбцам атрибутов")
    plt.show()
    median_value = df1.median()
    ax2 = (df1.sort_values(ascending = False)).plot.barh(figsize=(8,10), color = c2, alpha = 0.5)
    ax2.text(median_value, len(df1)-1, f'Median: {median_value}', ha='center', va='bottom', color='black')
    ax2.set_title(f"{name}. До применения порога. Барплот na-значений по столбцам атрибутов по возрастанию")
    plt.axvline(median_value, color='y', linestyle='--', linewidth=2)
    plt.show()
    
    # Удаляем непопулярные столбцы атрибутов с пустыми значениями больше порога
    threshold = len(df) * (1 - threshold)
    df = df.dropna(thresh=threshold, axis=1)

    df2 = df.isnull().sum() * 100 / len(df)
    ax3 = (df2 * 100 / len(df)).hist(figsize = (10,3), bins = 100, color = c1, alpha = 0.5)
    ax3.set_title(f"{name}. После применения порога. Гистограмма распределения na-значений по столбцам атрибутов")
    plt.show()
    median_value = df2.median()
    ax4 = (df2.sort_values(ascending = False)).plot.barh(figsize=(8,10), color = c1, alpha = 0.5)
    ax4.text(median_value, len(df2)-1, f'Median: {median_value}', ha='center', va='bottom', color='black')
    ax4.set_title(f"{name}. После применения порога. Барплот na-значений по столбцам атрибутов по возрастанию")
    plt.axvline(median_value, color='y', linestyle='--', linewidth=2)
    plt.show()

    wc.generate_from_frequencies(df.notnull().sum())

    # Отображение wordcloud графика
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'{name}. Ключевые атрибуты') 
    plt.axis('off')
    plt.show()

    return df



# Функции нахождения наиболее популярных значений и слов
def get_col_top_values(df, top = 10, score = True, norm = True, tokens = True):
    """
    Функция нахождения наиболее популярных значений столбцов датафрейма
    Параметры:
        df : датафрейм
        top : int, количество первых по популярности значений для вывода
        score : bool, опционально можно задать вывод популярности значения (на основе метода value_counts())
        norm : bool, опционально можно нормализовать score в долях (на основе метода параметра normalize метода value_counts())
        tokens : bool, опционально можно задать вывод в виде популярности токенов (слов)
    Возвращает:
        df - преобразованный датафрейм, где значениями являются топ-популярных значений столбца
    """
    df_top_values = pd.DataFrame()
    if tokens:
        df = df.map(lambda x: str(x)).map(get_tokens)
    for col in df.columns:
        if tokens:
            s_top_values = df[col].explode().value_counts(normalize = True if norm else False).drop(index=('nan')).nlargest(top)
        else:
            s_top_values = df[col].value_counts(normalize = True if norm else False).nlargest(top)
        if score:
            tuple_top_values = list(zip(s_top_values.index, s_top_values.values if isinstance(s_top_values.values.dtype, int) else s_top_values.values.round(2)))
            s_top_values = pd.Series(tuple_top_values, name=col)
        else:
            s_top_values = pd.Series(s_top_values.index, name=col)
        df_top_values = pd.concat([df_top_values, s_top_values], axis=1)
    return df_top_values



def find_sim_cols(df):
    """
    Функция попарного сравнивнения столбцов на предмет совместно встречаемых данных по условию: данные не должны одновременно встречаться в строках
    Параметры:
        df : датафрейм
    Возвращает:
        columns : list, списки колонок, удовлетворяющих условию
    """
    columns = []
    for col in df.columns:
        a = df.columns.get_loc(col)
        for b in range(a + 1, len(df.columns)):
            mask = df.iloc[:, [a, b]].isnull()
            if (mask.apply(lambda x: x.sum(), axis=1) > 0).all():
                columns.append(list(df.iloc[:, [a, b]].columns))
    return columns



def get_similarities(docs_1, docs_2, model = 'fuzz', threshold = 50):
    """
    Функция нахождения наиболее похожих текстов на основе методов библиотек fuzzywuzzy и spacy
    Параметры:
        docs_1 : текст для сравнения
        docs_2 : текст, с которым сравниваем
        model : 'fuzz' на основе методов fuzzywuzzy, 'spacy' на основе методов spacy, можно расширить функцию для других методов и моделей
        threshold : трешхолд для вывода в таблице наиболее похожих текстов
    Возвращает:
        df - датафрейм с сравниваемыми текстами и оценкой соответствия
    """
    tuples = []
    for doc_1 in docs_1:  
        for doc_2 in docs_2:
            if model == 'fuzz':
                ratio = fuzz.ratio(doc_1, doc_2)
                partial_ratio = fuzz.partial_ratio(doc_1, doc_2)
                token_sort_ratio = fuzz.token_sort_ratio(doc_1, doc_2)
                token_set_ratio = fuzz.token_set_ratio(doc_1, doc_2)
                WRatio = fuzz.WRatio(doc_1, doc_2)
                metrics = [ratio, partial_ratio, token_sort_ratio, token_set_ratio, WRatio]
                sim = sum(metrics) / len(metrics)
            if model == 'spacy':
                doc1 = nlp(doc_1)
                doc2 = nlp(doc_2)
                sim = doc1.similarity(doc2)  * 100
            sim = round(sim, 2)
            thr = threshold
            if sim > thr:
                tuples.append((doc_1, sim, doc_2))
    df_fuzz = pd.DataFrame(tuples, columns=['docs_1', 'sim', 'docs_2'])
    df = df_fuzz.sort_values(['sim', 'docs_1'], ascending=[False, True])
    return df



def get_faiss_similarities(docs_1, docs_2, num, score = True):
    """
    Функция нахождения наиболее похожих текстов на основе методов библиотеки langchain
    Параметры:
        docs_1 : текст для сравнения
        docs_2 : текст, с которым сравниваем
        num : количество наиболее похожих текстов
        score : опционально вывод с оценкой соответствия
    Возвращает:
        df - датафрейм с сравниваемыми текстами и оценкой соответствия, где docs_1 - индексы, а значения с оценками в столбцах по убыванию степени соответсвия
    """
    docs_1 = [Document(page_content=col) for col in docs_1] # list(docs_1)
    db = FAISS.from_documents(docs_1, OpenAIEmbeddings())

    headers = []
    values = []

    for i in docs_2:
        headers.append(i)
        values.append(db.similarity_search_with_score(i, num))
    values = [list(x) for x in zip(*values)]

    df = pd.DataFrame(values, columns = headers)
    if score:
        df = df.applymap(lambda x: (x[0].page_content, round(x[1], 3)))
    else:
        df = df.applymap(lambda x: x[0].page_content)
    df = df.T
    df = df.iloc[df[0].str[1].argsort()]
    df = df.reset_index(names='text_to_compare')
    return df



if __name__ == "__main__":
    print("Hello, World!")