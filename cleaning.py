import re

import pandas as pd
import numpy as np

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


# --- Глубокая очистка ---

# Исправление сдвинутых строк
shifted = df["fuel_type"].isin(["1,6", "2"])
df.loc[shifted, "transmission"] = df.loc[shifted, "engine_volume"]
df.loc[shifted, "engine_volume"] = df.loc[shifted, "fuel_type"]
df.loc[shifted, "fuel_type"] = pd.NA
print("Исправлено сдвинутых строк:", shifted.sum())

# Замена названий стран на ALPHA-3
country_map = {
    "Республика Казахстан": "KAZ", "Российская Федерация": "RUS",
    "Япония": "JPN", "Турция": "TUR", "Таиланд": "THA", "США": "USA",
    "Германия": "DEU", "UK": "GBR", "Китай": "CHN", "Узбекистан": "UZB",
    "Австрия": "AUT", "Швеция": "SWE", "Корея": "KOR", "Нидерланды": "NLD",
    "Испания": "ESP", "Польша": "POL", "Бельгия": "BEL", "Венгрия": "HUN",
    "Белоруссия": "BLR",
}
df["country"] = df["country"].map(country_map)

# Приведение вида топлива к F / D / E / HYB
fuel_map = {"бензин": "F", "дизель": "D", "электро": "E",
            "электричество": "E", "гибрид": "HYB"}
df["fuel_type"] = df["fuel_type"].str.lower().map(fuel_map)

# Приведение типа привода к 4WD / 2WD / FWD / RWD
def clean_drive(x):
    if pd.isna(x):
        return np.nan
    s = x.lower().replace(" ", "")
    s = s.replace("х", "x")  # кириллическая "х" в латинскую
    if s in {"4wd", "awd", "полный", "quattro", "4motion", "4x4"}:
        return "4WD"
    if s in {"ff", "fwd", "передний", "передний(ff)"}:
        return "FWD"
    if s in {"rwd", "задний"}:
        return "RWD"
    if s in {"2wd"} or s.startswith("4x2"):
        return "2WD"   # привод на одну ось без уточнения
    return np.nan

df["drive_type"] = df["drive_type"].apply(clean_drive)

# Приведение коробки передач к "Автомат" / "Механика"
cyr2lat = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")

def clean_transmission(x):
    if pd.isna(x):
        return np.nan
    s = x.upper().translate(cyr2lat).replace(" ", "")
    if s in {"0", "#Н/Д", "4WD", "ПЕРЕДНИЙ", "TDI", ""}:
        return np.nan                       # значения не про коробку
    if "AMT" in s:
        return "Автомат"                    # роботизированная - тоже без педали сцепления
    if re.search(r"(MT|M/T|MKP|MEX)", s) or re.fullmatch(r"\dM", s):
        return "Механика"
    if re.search(r"(AT|A/T|AKP|CVT|DSG|DCT|PDK|TRONIC|POWERSHIFT|PEДУКТОР|РЕДУКТОР)", s) \
            or re.fullmatch(r"\d{1,2}A?(-СТУП.*)?", s) or s.endswith("AKPP"):
        return "Автомат"
    return np.nan

df["transmission"] = df["transmission"].apply(clean_transmission)
