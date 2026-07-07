"""Distribuição de opiniões por polo (Positiva 4-5 x Negativa 1-2) usando os 3
splits sem neutros (train/val/test) de data/processed — conjunto total no_neutral.
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = "data/processed/B2W-Reviews01_no_neutral_{}.csv"
SPLITS = ["train", "val", "test"]

df = pd.concat(
    [pd.read_csv(BASE.format(s)) for s in SPLITS],
    ignore_index=True,
)

df["polaridade"] = df["overall_rating"].map(
    lambda r: "Positiva (4-5)" if r >= 4 else "Negativa (1-2)"
)

ordem = ["Negativa (1-2)", "Positiva (4-5)"]
qtd = df["polaridade"].value_counts().reindex(ordem)
pct = df["polaridade"].value_counts(normalize=True).mul(100).round(1).reindex(ordem)

tabela = pd.DataFrame({"Opiniões": qtd, "% do total": pct})
tabela.index.name = "Polo"

total = len(df)
total_fmt = f"{total:,d}".replace(",", ".")
csv_path = "01_eda/notebooks/opinioes_por_polo.csv"
tabela.to_csv(csv_path, encoding="utf-8-sig")
print(f"Total de opiniões (no_neutral, train+val+test): {total_fmt}")
print(tabela)

fig, ax = plt.subplots(figsize=(9, 3))
cores = {"Negativa (1-2)": "#d62728", "Positiva (4-5)": "#2ca02c"}
esquerda = 0
for polo in ordem:
    largura = pct[polo]
    ax.barh(0, largura, left=esquerda, color=cores[polo], label=polo)
    ax.text(
        esquerda + largura / 2, 0,
        f"{polo}\n{qtd[polo]:,} ({pct[polo]}%)".replace(",", "."),
        ha="center", va="center", color="white", fontsize=11, fontweight="bold",
    )
    esquerda += largura

ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel("% das opiniões")
ax.set_title(
    f"Opiniões por Polo (conjunto total no_neutral · {total_fmt} opiniões)",
    fontsize=12, fontweight="bold", pad=12,
)
plt.tight_layout()

fig_path = "01_eda/figures/Opiniões por Polo.png"
plt.savefig(fig_path, dpi=150)
print(f"\nFigura salva em: {fig_path}")
print(f"Tabela salva em: {csv_path}")
