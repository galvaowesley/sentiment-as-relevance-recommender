"""Histograma da distribuição de produtos (product_id) por quantidade de
avaliações nos 3 splits sem neutros (train/val/test), com bins densos.
"""
import pandas as pd
import matplotlib.pyplot as plt

BASE = "data/processed/B2W-Reviews01_no_neutral_{}.csv"
SPLITS = ["train", "val", "test"]

df = pd.concat(
    [pd.read_csv(BASE.format(s), dtype={"product_id": str}) for s in SPLITS],
    ignore_index=True,
)

contagem = df.groupby("product_id").size().reset_index(name="quantidade")

limites = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 300, contagem["quantidade"].max()]
rotulos = ["1", "2", "3", "4", "5", "6 a 10", "11 a 20", "21 a 50", "51 a 100", "101 a 300", "301+"]

contagem["faixa"] = pd.cut(
    contagem["quantidade"], bins=limites, labels=rotulos, include_lowest=True
)

dados_grafico = (
    contagem["faixa"].value_counts().reindex(rotulos).reset_index()
)
dados_grafico.columns = ["Faixa de Avaliações", "Quantidade de Produtos"]
print(f"Total de avaliações (no_neutral, train+val+test): {len(df)}")
print(f"Total de produtos únicos: {contagem['product_id'].nunique()}")
print(dados_grafico)

plt.figure(figsize=(12, 6))
ax = plt.gca()
bars = ax.bar(
    dados_grafico["Faixa de Avaliações"],
    dados_grafico["Quantidade de Produtos"],
    color=plt.cm.viridis(
        [i / (len(dados_grafico) - 1) for i in range(len(dados_grafico))]
    ),
)

for b in bars:
    h = b.get_height()
    ax.annotate(
        f"{int(h)}",
        (b.get_x() + b.get_width() / 2, h),
        ha="center", va="bottom", fontsize=10,
    )

ax.set_title(
    "Distribuição de Produtos por Quantidade de Avaliações (splits sem neutros)",
    fontsize=14, fontweight="bold", pad=15,
)
ax.set_xlabel("Intervalo de Quantidade de Avaliações", fontsize=12, labelpad=10)
ax.set_ylabel("Número de Produtos Únicos (product_id)", fontsize=12, labelpad=10)
ax.grid(axis="y", linestyle="-", alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()

fig_path = "01_eda/figures/Distribuição de Produtos por Quantidade de Avaliações (no_neutral).png"
plt.savefig(fig_path, dpi=150)
print(f"\nFigura salva em: {fig_path}")
