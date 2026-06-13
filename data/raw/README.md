# raw — Corpus Original

Corpus B2W-Reviews01: mais de 130 mil avaliações de produtos da Americanas.com em português brasileiro, coletadas entre janeiro e maio de 2018.

## Download

```bash
# Via Hugging Face Datasets
python -c "from datasets import load_dataset; load_dataset('ruanchaves/b2w-reviews01')"
```

Ou acesse diretamente: https://github.com/americanas-tech/b2w-reviews01

## Arquivo esperado

```
data/raw/B2W-Reviews01.csv   (~50MB, não versionado)
```

## Colunas principais

`submission_date`, `reviewer_id`, `product_id`, `product_name`, `overall_rating`, `recommend_to_a_friend`, `review_title`, `review_text`, `site_category_lv1`, `site_category_lv2`
