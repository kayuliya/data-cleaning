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
print("=== Рынок в целом ===")
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

# Позиция Mercur Auto
print("\n\n=== Mercur Auto ===")
m = df[df.is_mercur]
mk = df
print("Мерку: авто", int(m.quantity.sum()), "| выручка, млн USD", round(m.total_usd.sum() / 1e6, 2))
print("Доля рынка по шт.:", pct(m.quantity.sum() / mk.quantity.sum() * 100),
      "| по выручке:", pct(m.total_usd.sum() / mk.total_usd.sum() * 100))
print("Ранг по выручке:",
      df.groupby("company").total_usd.sum().rank(ascending=False)[ME])

# Доля в разрезе брендов
brands_me = m.brand.unique()
b = pd.DataFrame({
    "market_units": mk[mk.brand.isin(brands_me)].groupby("brand").quantity.sum(),
    "mercur_units": m.groupby("brand").quantity.sum(),
    "market_rev": mk[mk.brand.isin(brands_me)].groupby("brand").total_usd.sum(),
    "mercur_rev": m.groupby("brand").total_usd.sum(),
})
b["share_units_%"] = (b.mercur_units / b.market_units * 100).round(1)
b["share_rev_%"] = (b.mercur_rev / b.market_rev * 100).round(1)
print("\nДоля по брендам:\n", b)

# Доля в разрезе регионов
r = pd.DataFrame({"market": mk.groupby("oblast").quantity.sum(),
                  "mercur": m.groupby("oblast").quantity.sum()}).fillna(0)
r["share_%"] = (r.mercur / r.market * 100).round(2)
print("\nДоля по областям (где Mercur есть):\n", r[r.mercur > 0])

# Доля в разрезе сегментов
s = pd.DataFrame({"market": mk.groupby("segment_2013").quantity.sum(),
                  "mercur": m.groupby("segment_2013").quantity.sum()}).fillna(0)
s["share_%"] = (s.mercur / s.market * 100).round(2)
s["mercur_mix_%"] = (s.mercur / s.mercur.sum() * 100).round(1)
s["market_mix_%"] = (s.market / s.market.sum() * 100).round(1)
print("\nСегменты:\n", s)

# Рыночный премиум-сегмент (бренды-аналоги Mercur)
premium = ["Audi", "BMW", "Mercedes-Benz", "Porsche", "Volvo", "Land Rover", "Jaguar",
           "Lexus", "Infiniti", "Mini", "Cadillac"]
p = mk[mk.brand.isin(premium)]
print("\nПремиум-рынок (11 брендов): шт.", int(p.quantity.sum()),
      "| выручка, млн", round(p.total_usd.sum() / 1e6, 1))
print("Доля Mercur Auto в премиуме (Audi+Porsche) по шт.:",
      pct(m[m.brand.isin(premium)].quantity.sum() / p.quantity.sum() * 100))
prem_b = p.groupby("brand").quantity.sum().sort_values(ascending=False)
print("Премиум по брендам:\n", prem_b)

# KPI: средний чек (за 1 автомобиль) и структура продаж
def kpi(frame):
    return pd.Series({
        "avg_price_per_unit": frame.total_usd.sum() / frame.quantity.sum(),
        "median_price": frame.price_usd.median(),
        "avg_units_per_deal": frame.quantity.mean(),
        "bulk_share_%": frame.is_bulk.mean() * 100,
        "avg_engine_l": frame.engine_volume.mean(),
        "share_4wd_%": (frame.drive_type == "4WD").sum() / frame.drive_type.notna().sum() * 100,
        "share_auto_%": (frame.transmission == "Автомат").sum() / frame.transmission.notna().sum() * 100,
        "share_petrol_%": (frame.fuel_type == "F").sum() / frame.fuel_type.notna().sum() * 100,
        "share_imported_%": (frame.country != "KAZ").mean() * 100,
        "share_2019_model_%": (frame.model_year == 2019).sum() / frame.model_year.notna().sum() * 100,
    })

k = pd.DataFrame({"Mercur Auto": kpi(m), "Весь рынок": kpi(mk), "Премиум-рынок": kpi(p)}).round(1)
print("\nKPI:\n", k)

# Сравнение с VW на рынке в целом
vw_other = mk[(mk.brand == "Volkswagen") & (~mk.is_mercur)]
print("\nVolkswagen вне Mercur: шт.", int(vw_other.quantity.sum()),
      "| ср. цена", round(vw_other.total_usd.sum() / vw_other.quantity.sum()))
vw_me = m[m.brand == "Volkswagen"]
print("Volkswagen Mercur: ср. цена", round(vw_me.total_usd.sum() / vw_me.quantity.sum()))
print("\nVW по компаниям:\n", mk[mk.brand == "Volkswagen"].groupby("company").quantity.sum())

# Динамика Mercur по месяцам
mm = m.groupby("month").agg(units=("quantity", "sum"), revenue=("total_usd", "sum"))
mm["market_share_%"] = (mm.units / monthly.units * 100).round(2)
print("\nMercur по месяцам:\n", mm)

# Модельный состав Mercur
print("\nТоп-модели Mercur:\n",
      m.groupby(["brand", "model"]).agg(units=("quantity", "sum"),
                                        revenue=("total_usd", "sum"))
        .sort_values("units", ascending=False).head(8))

# Графики Mercur
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
b[["share_units_%", "share_rev_%"]].plot.bar(ax=ax[0], rot=0)
ax[0].set(title="Доля Mercur Auto в рынке брендов, %", xlabel="")
r[r.mercur > 0]["share_%"].sort_values().plot.barh(ax=ax[1], color="tab:green")
ax[1].set(title="Доля Mercur Auto по областям (в шт.), %", xlabel="%", ylabel="")
save("10_mercur_shares")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
s[["mercur_mix_%", "market_mix_%"]].plot.bar(ax=ax[0], rot=30)
ax[0].set(title="Структура продаж по сегментам, %", xlabel="")
k.loc[["avg_price_per_unit"]].T.plot.bar(ax=ax[1], legend=False, rot=0, color="tab:purple")
ax[1].set(title="Средняя цена автомобиля, USD", xlabel="")
save("11_mercur_structure")

fig, ax = plt.subplots(figsize=(8, 4))
mm["units"].plot(marker="o", ax=ax)
ax.set(title="Mercur Auto: продажи по месяцам, шт.", xlabel="Месяц 2019", ylabel="шт.")
save("12_mercur_monthly")

# Проверка устойчивости: доля Mercur в исходных данных (до удаления дубликатов)
raw = pd.read_csv("autokz2019.csv", sep=";", decimal=",")
raw["c"] = raw["Компания"].str.strip().replace({"Mercur Autos": "Mercur Auto"})
rq = raw[raw["Количество"].notna()]
print("\nДоля Mercur в сырых данных (шт.):",
      pct(rq[rq.c == ME]["Количество"].sum() / rq["Количество"].sum() * 100),
      "| (выручка):", pct(rq[rq.c == ME]["Продажа, USD"].sum() / rq["Продажа, USD"].sum() * 100))