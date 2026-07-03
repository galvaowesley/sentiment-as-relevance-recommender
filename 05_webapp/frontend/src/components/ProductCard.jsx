import { useNavigate } from 'react-router-dom'
import CategoryIcon from './CategoryIcon.jsx'

function Stars({ rating }) {
  const n = Math.round(rating ?? 0)
  return (
    <>
      <span className="stars-filled">{'★'.repeat(Math.max(0, n))}</span>
      <span className="stars-empty">{'☆'.repeat(Math.max(0, 5 - n))}</span>
    </>
  )
}

export default function ProductCard({ product }) {
  const navigate = useNavigate()
  const {
    product_id, product_name, product_brand,
    site_category_lv1, site_category_lv2,
    avg_rating, review_count, num_reviews,
  } = product

  const rating = avg_rating ?? null
  const count  = review_count ?? num_reviews ?? 0

  return (
    <div
      className="product-card"
      onClick={() => navigate(`/product/${encodeURIComponent(product_id)}`)}
    >
      <div className="product-card-icon">
        <CategoryIcon category={site_category_lv1} size={48} />
      </div>

      <div className="product-card-lv2">{site_category_lv2 || site_category_lv1}</div>

      <div className="product-card-name">{product_name}</div>

      {product_brand && (
        <div className="product-card-brand">{product_brand}</div>
      )}

      <div className="product-card-stars">
        {rating !== null ? (
          <>
            <Stars rating={rating} />
            <span className="stars-value">{rating.toFixed(1)}</span>
            <span className="stars-count">({count})</span>
          </>
        ) : (
          <span className="stars-count">{count} aval.</span>
        )}
      </div>
    </div>
  )
}
