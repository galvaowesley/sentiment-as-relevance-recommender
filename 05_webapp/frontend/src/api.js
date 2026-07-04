const API = import.meta.env.VITE_API_URL ?? ''

async function request(path) {
  const res = await fetch(`${API}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const getCategories = () => request('/categories')

export const getSubcategories = (cat1) =>
  request(`/categories/lv2${cat1 ? `?cat1=${encodeURIComponent(cat1)}` : ''}`)

export const countProducts = (params) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null && v !== ''))
  )
  return request(`/products/count?${qs}`)
}

export const getProducts = (params) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null && v !== ''))
  )
  return request(`/products?${qs}`)
}

export const getProduct = (id) => request(`/products/${encodeURIComponent(id)}`)

export const getProductReviews = (id) =>
  request(`/products/${encodeURIComponent(id)}/reviews`)

export const getRecommendations = (productId, topN = 10, alpha) => {
  const qs = new URLSearchParams({ product_id: productId, top_n: topN })
  if (alpha != null) qs.set('alpha', alpha)
  return request(`/recommend?${qs}`)
}
