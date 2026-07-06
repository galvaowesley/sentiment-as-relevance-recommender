"""Top 5 categorias (site_category_lv1) por volume de opiniões, com percentual
positivo (nota 4-5) x negativo (nota 1-2), usando os 3 splits sem neutros
(train/val/test) de data/processed.
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = "data/processed/B2W-Reviews01_no_neutral_{}.csv"
SPLITS = ["train", "val", "test"]

df = pd.concat(
    [pd.read_csv(BASE.format(s)) for s in SPLITS],
    ignore_index=True,
)

top5 = df["site_category_lv1"].value_counts().head(5).index.tolist()
df_top5 = df[df["site_category_lv1"].isin(top5)].copy()

df_top5["polaridade"] = df_top5["overall_rating"].map(
    lambda r: "Positiva (4-5)" if r >= 4 else "Negativa (1-2)"
)

tabela = (
    df_top5.groupby("site_category_lv1")["polaridade"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .unstack()
    .reindex(top5)
)
tabela["Volume Total"] = df_top5["site_category_lv1"].value_counts().reindex(top5)
tabela = tabela[["Volume Total", "Positiva (4-5)", "Negativa (1-2)"]]
tabela.index.name = "Categoria (site_category_lv1)"

csv_path = "01_eda/notebooks/top5_categorias_sentimento.csv"
tabela.to_csv(csv_path, encoding="utf-8-sig")
print(tabela)

fig, ax = plt.subplots(figsize=(10, 6))
plot_df = tabela[["Positiva (4-5)", "Negativa (1-2)"]].sort_values(
    "Volume Total" if False else "Positiva (4-5)"
)
plot_df = tabela.loc[top5, ["Positiva (4-5)", "Negativa (1-2)"]]
plot_df.plot(
    kind="barh",
    stacked=True,
    color=["#2ca02c", "#d62728"],
    ax=ax,
)

for i, cat in enumerate(top5):
    volume = int(tabela.loc[cat, "Volume Total"])
    ax.text(101, i, f"n={volume:,}".replace(",", "."), va="center", fontsize=9)

ax.set_xlim(0, 115)
ax.set_xlabel("% das opiniões")
ax.set_ylabel("")
ax.set_title(
    "Figura 6 - Top 5 categorias por volume de opiniões:\n% positivas (4-5) x negativas (1-2)"
)
ax.legend(title="Polaridade", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
plt.tight_layout()

fig_path = "01_eda/figures/Figura_6.png"
plt.savefig(fig_path, dpi=150)
print(f"\nFigura salva em: {fig_path}")
print(f"Tabela salva em: {csv_path}")
