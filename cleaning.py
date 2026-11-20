import pandas as pd

INPUT_PATH = "autokz2019.csv"
OUTPUT_PATH = "autokz2019_clean.csv"

# --- Загрузка и первичный осмотр ---

# Чтение файла
df = pd.read_csv(INPUT_PATH, sep=";", decimal=",")
print("Исходный размер:", df.shape)
print(df.dtypes, "\n")
print("Пропуски:\n", df.isna().sum(), "\n")

# Удаление лишних (почти пустых) столбцов
cols_to_drop = [
    "Наименование дилерского центра", "Форма расчета", "Сегмент",
    "Сегментация Eng", "Локализация производства", "Тип клиента", "Модификация",
]
df = df.drop(columns=cols_to_drop)

# Переименование столбцов на английский
df = df.rename(columns={
    "Год": "year", "Месяц": "month", "Компания": "company", "Бренд": "brand",
    "Модель": "model", "Год выпуска": "model_year",
    "Страна-производитель": "country", "Вид топлива": "fuel_type",
    "Объём двиг, л,": "engine_volume", "Коробка передач": "transmission",
    "Тип привода": "drive_type", "Регион": "region", "Область": "oblast",
    "Количество": "quantity", "Цена, USD": "price_usd",
    "Продажа, USD": "total_usd", "Сегментация 2013": "segment_2013",
    "Класс 2013": "class_2013",
})

# Удаление пустых строк и дубликатов
n0 = len(df)
df = df.dropna(how="all")
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Удалено пустых строк: {n0 - len(df) - n_dup}, явных дубликатов: {n_dup}")

# Удаление лишних пробелов
text_cols = df.select_dtypes(include=["object", "string"]).columns
for c in text_cols:
    df[c] = df[c].astype("string").str.strip()