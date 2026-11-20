import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
os.makedirs("figs", exist_ok=True)

df = pd.read_csv("autokz2019_clean.csv", parse_dates=["sale_date"])
ME = "Mercur Auto"
df["is_mercur"] = df["company"] == ME
df["month"] = df["sale_date"].dt.month


def save(name):
    plt.tight_layout()
    plt.savefig(f"figs/{name}.png")
    plt.close()


def pct(x):
    return f"{x:.1f}%"


# Общая картина рынка
print("Рынок в целом:")
print("Строк:", len(df), "| автомобилей (нетто):", int(df.quantity.sum()),
      "| выручка, млн USD:", round(df.total_usd.sum() / 1e6, 1))
print(df[["price_usd", "engine_volume", "model_year", "quantity"]].describe().round(2))
print("Skew цены:", round(df.price_usd.skew(), 2))

# Распределения цены, объёма двигателя и года выпуска
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df.price_usd, bins=50, ax=ax[0])
ax[0].set(title="Цена за единицу, USD", xlabel="USD", ylabel="Число строк")
sns.histplot(df.engine_volume.dropna(), bins=30, ax=ax[1])
ax[1].set(title="Объём двигателя, л", xlabel="литры", ylabel="")
df.model_year.value_counts().sort_index().plot.bar(ax=ax[2])
ax[2].set(title="Год выпуска", xlabel="", ylabel="")
save("01_distributions")

# Логарифмическая цена (чтобы увидеть дорогие машины)
plt.figure(figsize=(7, 4))
sns.histplot(df.price_usd, bins=50, log_scale=True)
plt.title("Цена за единицу (лог. шкала)")
save("02_price_log")

# Динамика продаж по месяцам
monthly = df.groupby("month").agg(units=("quantity", "sum"), revenue=("total_usd", "sum"))
print("\nПродажи по месяцам:\n", monthly)
fig, ax = plt.subplots(figsize=(8, 4))
monthly.units.plot(marker="o", ax=ax)
ax.set(title="Продажи по месяцам, шт.", xlabel="Месяц 2019", ylabel="шт.")
save("03_monthly_units")

# Бренды и модели
top_brand = df.groupby("brand").quantity.sum().sort_values(ascending=False)
print("\nТоп-10 брендов, шт.:\n", top_brand.head(10))
print("Доля топ-5 брендов:", pct(top_brand.head(5).sum() / top_brand.sum() * 100))
top_model = (df.groupby(["brand", "model"]).quantity.sum()
               .sort_values(ascending=False).head(10))
print("\nТоп-10 моделей:\n", top_model)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
top_brand.head(12).sort_values().plot.barh(ax=ax[0])
ax[0].set(title="Топ-12 брендов по числу проданных авто", xlabel="шт.", ylabel="")
tm = top_model.sort_values()
tm.index = [f"{b} {m}" for b, m in tm.index]  # подписи вида "Lada Granta"
tm.plot.barh(ax=ax[1], color="tab:orange")
ax[1].set(title="Топ-10 моделей", xlabel="шт.", ylabel="")
save("04_brands_models")

# Регионы
reg = df.groupby("oblast").quantity.sum().sort_values(ascending=False)
print("\nРегионы (доля, %):\n", (reg / reg.sum() * 100).round(1).head(8))
plt.figure(figsize=(8, 6))
reg.sort_values().plot.barh()
plt.title("Продажи по областям, шт.")
plt.xlabel("шт.")
plt.ylabel("")
save("05_regions")

# Топливо, коробка, привод, страна
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for a, col, title in zip(ax.flat, ["fuel_type", "transmission", "drive_type", "country"],
                         ["Тип топлива", "Коробка передач", "Привод", "Страна производства"]):
    s = df.groupby(col, observed=True).quantity.sum().sort_values(ascending=False).head(8)
    a.bar(s.index.astype(str), s.values)
    a.set_title(title)
    print(f"\n{title}, доля %:\n", (s / s.sum() * 100).round(1).to_dict())
save("06_categoricals")

# Сегменты
seg = df.groupby("segment_2013").quantity.sum().sort_values()
plt.figure(figsize=(8, 4))
seg.plot.barh()
plt.title("Структура рынка по сегментам 2013, шт.")
plt.ylabel("")
save("07_segments")

# Аномалии (возвраты, опт, дорогие позиции)
print("\nВозвраты:\n", df[df.is_return][["company", "brand", "model", "quantity", "total_usd"]])
bulk = df[df.is_bulk]
print("\nОптовые партии (>10 шт.): строк", len(bulk),
      "| доля авто на рынке:", pct(bulk.quantity.sum() / df.quantity.sum() * 100))
print(bulk.groupby("brand").quantity.sum().sort_values(ascending=False).head(5))
print("\nКрупнейшие партии:\n", bulk.nlargest(5, "quantity")[["company", "brand", "model", "quantity"]])

plt.figure(figsize=(9, 4))
sns.boxplot(data=df[df.brand.isin(top_brand.head(8).index)],
            x="brand", y="price_usd", showfliers=False)
plt.title("Цена по топ-8 брендам (без выбросов)")
plt.xlabel("")
plt.ylabel("USD")
save("08_price_by_brand")

num = df[["price_usd", "engine_volume", "model_year", "quantity"]].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(num, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Корреляции числовых признаков")
save("09_corr")
print("\nКорреляции:\n", num.round(2))