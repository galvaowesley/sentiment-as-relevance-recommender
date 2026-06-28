import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getProducts, countProducts } from '../../api.js'
import CategorySidebar from '../CategorySidebar.jsx'
import ProductCard from '../ProductCard.jsx'
import Pagination from '../Pagination.jsx'

const PAGE_SIZE = 24

export default function Browse() {
  const [searchParams, setSearchParams] = useSearchParams()

  const category = searchParams.get('category') ?? ''
  const cat2     = searchParams.get('cat2') ?? ''
  const q        = searchParams.get('q') ?? ''
  const page     = Math.max(1, parseInt(searchParams.get('page') ?? '1', 10))

  const [products, setProducts] = useState([])
  const [total, setTotal]       = useState(0)
  const [loading, setLoading]   = useState(true)

  // Fetch total count whenever filters change
  useEffect(() => {
    countProducts({ ...(category ? { category } : {}), ...(cat2 ? { cat2 } : {}), ...(q ? { q } : {}) })
      .then((res) => setTotal(res.total ?? 0))
      .catch(console.error)
  }, [category, cat2, q])

  // Fetch products page
  useEffect(() => {
    setLoading(true)
    getProducts({
      limit: PAGE_SIZE,
      offset: (page - 1) * PAGE_SIZE,
      ...(category ? { category } : {}),
      ...(cat2     ? { cat2 }     : {}),
      ...(q        ? { q }        : {}),
    })
      .then(setProducts)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [category, cat2, q, page])

  function handlePage(p) {
    const next = new URLSearchParams(searchParams)
    next.set('page', String(p))
    setSearchParams(next)
    window.scrollTo(0, 0)
  }

  const heading = cat2 ? cat2 : (category || (q ? `"${q}"` : 'Todos os produtos'))

  return (
    <div className="page browse-page">
      <CategorySidebar />

      <div className="browse-main">
        <div className="browse-toolbar">
          <h2>{heading}</h2>
          {!loading && (
            <span className="browse-count">
              {total} produto{total !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {loading && <div className="loading">Carregando produtos…</div>}

        {!loading && products.length === 0 && (
          <div className="empty-state">
            <h3>Nenhum produto encontrado</h3>
            <p>Tente outra busca ou categoria.</p>
          </div>
        )}

        {!loading && products.length > 0 && (
          <>
            <div className="product-grid">
              {products.map((p) => <ProductCard key={p.product_id} product={p} />)}
            </div>
            <Pagination
              page={page}
              pageSize={PAGE_SIZE}
              total={total}
              onPage={handlePage}
            />
          </>
        )}
      </div>
    </div>
  )
}
