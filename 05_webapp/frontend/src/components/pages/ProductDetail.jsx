import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { getProduct, getProductReviews, getRecommendations } from '../../api.js'
import CategoryIcon from '../CategoryIcon.jsx'
import RecommendationsPanel from '../RecommendationsPanel.jsx'

// Default re-ranking weight — mirrors RecommenderConfig.alpha in the backend.
const DEFAULT_ALPHA = 0.7

// ── Helpers ────────────────────────────────────────────────
function Stars({ rating, size = 'sm' }) {
  const n = Math.round(rating ?? 0)
  return (
    <>
      <span className="stars-filled" style={size === 'lg' ? { fontSize: 17, letterSpacing: 1 } : {}}>
        {'★'.repeat(Math.max(0, n))}
      </span>
      <span className="stars-empty" style={size === 'lg' ? { fontSize: 17, letterSpacing: 1 } : {}}>
        {'☆'.repeat(Math.max(0, 5 - n))}
      </span>
    </>
  )
}

function formatDate(raw) {
  if (!raw) return ''
  return raw.slice(0, 10)
}

// Per-review sentiment as predicted by the BERTimbau classifier (positivo/negativo),
// with the model's confidence in that label.
function SentimentPill({ label, confidence }) {
  if (label !== 'positive' && label !== 'negative') {
    return (
      <span className="review-sentiment-badge pending" title="Sem predição de sentimento para esta avaliação">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <circle cx="5" cy="5" r="4.5" stroke="#9B9B9B" strokeWidth="1"/>
          <line x1="5" y1="2.5" x2="5" y2="5.5" stroke="#9B9B9B" strokeWidth="1.2" strokeLinecap="round"/>
          <circle cx="5" cy="7" r="0.6" fill="#9B9B9B"/>
        </svg>
        sem predição
      </span>
    )
  }
  const isPos = label === 'positive'
  const pct = confidence != null ? Math.round(confidence * 100) : null
  return (
    <span
      className={`review-sentiment-badge ${isPos ? 'positive' : 'negative'}`}
      title="Classificação de sentimento do modelo BERTimbau para esta avaliação"
    >
      {isPos ? '😊 Positivo' : '☹️ Negativo'}
      {pct != null && <span className="review-sentiment-conf"> · {pct}%</span>}
    </span>
  )
}

// ── Review filters bar ─────────────────────────────────────
function ReviewFilters({ filter, setFilter, starFilter, setStarFilter }) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 18, alignItems: 'center' }}>
      <span style={{ fontSize: 12, fontWeight: 600, color: '#6B6B6B', marginRight: 2 }}>Filtrar:</span>

      {['all', 'yes', 'no'].map((v) => (
        <button
          key={v}
          onClick={() => setFilter(v)}
          style={{
            padding: '3px 10px',
            borderRadius: 20,
            fontSize: 12,
            fontWeight: filter === v ? 600 : 400,
            border: filter === v ? '1.5px solid #8B5CF6' : '1.5px solid #E5E7EB',
            background: filter === v ? '#F5F0FF' : '#fff',
            color: filter === v ? '#7C3AED' : '#6B6B6B',
            cursor: 'pointer',
            transition: 'all 0.1s',
          }}
        >
          {v === 'all' ? 'Todos' : v === 'yes' ? '✓ Recomenda' : '✗ Não recomenda'}
        </button>
      ))}

      <div style={{ width: 1, height: 20, background: '#E5E7EB', margin: '0 4px' }} />

      {[0, 1, 2, 3, 4, 5].map((s) => (
        <button
          key={s}
          onClick={() => setStarFilter(starFilter === s ? 0 : s)}
          style={{
            padding: '3px 10px',
            borderRadius: 20,
            fontSize: 12,
            fontWeight: starFilter === s ? 600 : 400,
            border: starFilter === s ? '1.5px solid #F59E0B' : '1.5px solid #E5E7EB',
            background: starFilter === s ? '#FFFBEB' : '#fff',
            color: starFilter === s ? '#D97706' : '#6B6B6B',
            cursor: 'pointer',
            transition: 'all 0.1s',
          }}
        >
          {s === 0 ? 'Todas ★' : `${s}★`}
        </button>
      ))}
    </div>
  )
}

// ── Single review item ─────────────────────────────────────
function ReviewItem({ review }) {
  const { review_title, overall_rating, reviewer_id, reviewer_state, submission_date,
          review_text, recommend_to_a_friend, sentiment_label, sentiment_confidence } = review

  const recYes = recommend_to_a_friend?.toLowerCase() === 'yes'
  const recBadgeClass = recYes ? 'yes' : recommend_to_a_friend?.toLowerCase() === 'no' ? 'no' : ''

  return (
    <div className="review-item">
      <div className="review-item-header">
        <div className="review-item-info">
          <div className="review-item-title">{review_title || '(sem título)'}</div>
          <div className="review-item-author">
            {reviewer_id}
            {reviewer_state ? ` · ${reviewer_state}` : ''}
            {submission_date ? ` · ${formatDate(submission_date)}` : ''}
          </div>
        </div>
        <div className="review-item-stars">
          <Stars rating={overall_rating} />
          <span className="stars-value">{overall_rating}</span>
        </div>
      </div>

      {review_text && (
        <p className="review-item-text">{review_text}</p>
      )}

      <div className="review-item-footer">
        {recBadgeClass && (
          <span className={`review-recommend-badge ${recBadgeClass}`}>
            {recYes ? '✓ Recomenda' : '✗ Não recomenda'}
          </span>
        )}
        <SentimentPill label={sentiment_label} confidence={sentiment_confidence} />
      </div>
    </div>
  )
}

// ── Page ───────────────────────────────────────────────────
export default function ProductDetail() {
  const { productId } = useParams()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const [product,       setProduct]       = useState(null)
  const [reviews,       setReviews]       = useState([])
  const [recs,          setRecs]          = useState([])
  const [loadingProd,   setLoadingProd]   = useState(true)
  const [loadingReviews,setLoadingReviews]= useState(true)
  const [loadingRecs,   setLoadingRecs]   = useState(true)
  const [error,         setError]         = useState(null)

  // Review filters
  const [recFilter,  setRecFilter]  = useState('all')
  const [starFilter, setStarFilter] = useState(0)

  // Re-ranking weight — user-pilotable: score = α·sim + (1−α)·S(p). Default matches
  // the backend RecommenderConfig (0.7).
  const [alpha, setAlpha] = useState(DEFAULT_ALPHA)

  // Product + reviews: fetched once per product.
  useEffect(() => {
    setLoadingProd(true); setLoadingReviews(true)
    setError(null); setProduct(null); setReviews([])
    setRecFilter('all'); setStarFilter(0)
    setAlpha(DEFAULT_ALPHA)

    getProduct(productId)
      .then(setProduct)
      .catch((e) => setError(e.message))
      .finally(() => setLoadingProd(false))

    getProductReviews(productId)
      .then(setReviews)
      .catch(() => setReviews([]))
      .finally(() => setLoadingReviews(false))
  }, [productId])

  // Recommendations: re-fetched whenever the product or α changes. Debounced so
  // dragging the α slider doesn't fire a request per intermediate value.
  useEffect(() => {
    setLoadingRecs(true)
    const timer = setTimeout(() => {
      getRecommendations(productId, 10, alpha)
        .then(setRecs)
        .catch(console.error)
        .finally(() => setLoadingRecs(false))
    }, 180)
    return () => clearTimeout(timer)
  }, [productId, alpha])

  const filteredReviews = useMemo(() => {
    return reviews.filter((r) => {
      if (recFilter === 'yes' && r.recommend_to_a_friend?.toLowerCase() !== 'yes') return false
      if (recFilter === 'no'  && r.recommend_to_a_friend?.toLowerCase() !== 'no')  return false
      if (starFilter > 0 && r.overall_rating !== starFilter) return false
      return true
    })
  }, [reviews, recFilter, starFilter])

  // Navigation helpers
  const cat1 = searchParams.get('category') ?? ''
  const cat2 = searchParams.get('cat2') ?? ''
  const pageClass = 'page'

  function goToCat1(c) {
    const next = new URLSearchParams()
    next.set('category', c)
    setSearchParams(next)
    navigate(`/?${next.toString()}`)
  }

  function goToCat2(c1, c2) {
    const next = new URLSearchParams()
    next.set('category', c1)
    next.set('cat2', c2)
    setSearchParams(next)
    navigate(`/?${next.toString()}`)
  }

  if (loadingProd) {
    return (
      <div className={pageClass}>
        <div className="product-detail-outer">
          <div className="loading">Carregando produto…</div>
        </div>
      </div>
    )
  }

  if (error || !product) {
    return (
      <div className={pageClass}>
        <div className="product-detail-outer">
          <div className="empty-state">
            <h3>Produto não encontrado</h3>
            <p>{error ?? 'Este produto não está disponível.'}</p>
          </div>
        </div>
      </div>
    )
  }

  const {
    product_name, product_brand, site_category_lv1, site_category_lv2,
    sentiment_score, avg_rating, review_count, num_reviews, description,
  } = product

  const totalReviews = review_count ?? num_reviews ?? 0
  const pct = Math.round((sentiment_score ?? 0) * 100)

  return (
    <div className={pageClass}>
      <div className="product-detail-outer">

        {/* Breadcrumb */}
        <div className="breadcrumb">
          <button className="breadcrumb-back" onClick={() => navigate(-1)}>← Voltar</button>
          {site_category_lv1 && (
            <>
              <span className="breadcrumb-sep">/</span>
              <button className="breadcrumb-link" onClick={() => goToCat1(site_category_lv1)}>
                {site_category_lv1}
              </button>
            </>
          )}
          {site_category_lv2 && site_category_lv2 !== site_category_lv1 && (
            <>
              <span className="breadcrumb-sep">/</span>
              <button className="breadcrumb-link" onClick={() => goToCat2(site_category_lv1, site_category_lv2)}>
                {site_category_lv2}
              </button>
            </>
          )}
          <span className="breadcrumb-sep">/</span>
          <span className="breadcrumb-current">{product_name}</span>
        </div>

        <div className="product-detail-layout">

            {/* Product card */}
            <div className="product-info-card">
              <div className="product-info-top">
                <div className="product-info-icon">
                  <CategoryIcon category={site_category_lv1} size={72} />
                </div>

                <div className="product-info-meta">
                  {site_category_lv2 && (
                    <div className="product-info-lv2">{site_category_lv2}</div>
                  )}
                  <h1 className="product-info-name">{product_name}</h1>
                  {product_brand && (
                    <div className="product-info-brand">{product_brand}</div>
                  )}

                  <div className="product-info-stars">
                    {avg_rating !== null && avg_rating !== undefined ? (
                      <>
                        <Stars rating={avg_rating} size="lg" />
                        <span className="stars-value" style={{ fontSize: 17, fontWeight: 800, letterSpacing: '-0.02em' }}>
                          {avg_rating.toFixed(1)}
                        </span>
                        <span className="stars-count" style={{ fontSize: 14 }}>
                          · {totalReviews} avaliações
                        </span>
                      </>
                    ) : (
                      <span className="stars-count">{totalReviews} avaliações</span>
                    )}
                  </div>

                  <div className="product-sentiment-badge">
                    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <circle cx="6.5" cy="6.5" r="6" fill="#059669" fillOpacity="0.18"/>
                      <path d="M3.5 6.5l2 2 4-4" stroke="#059669" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    <span>
                      Sentimento positivo S(p):{' '}
                      <strong>{pct}%</strong>
                    </span>
                  </div>
                </div>
              </div>

              {description && (
                <div className="product-description">
                  <div className="section-label">Sobre o produto</div>
                  <p>{description}</p>
                </div>
              )}
            </div>

            {/* Recommendations — right after the product, before reviews */}
            <div className="product-detail-right">
              <RecommendationsPanel
                recommendations={recs}
                loading={loadingRecs}
                alpha={alpha}
                onAlphaChange={setAlpha}
                defaultAlpha={DEFAULT_ALPHA}
              />
            </div>

            {/* Reviews */}
            <div className="reviews-card">
              <h2>
                Avaliações dos clientes
                {!loadingReviews && reviews.length > 0 && (
                  <span style={{ fontWeight: 400, fontSize: 14, color: '#9B9B9B', marginLeft: 8 }}>
                    ({reviews.length} total)
                  </span>
                )}
              </h2>

              {loadingReviews && (
                <div style={{ color: '#9B9B9B', fontSize: 14, marginBottom: 16 }}>Carregando avaliações…</div>
              )}

              {!loadingReviews && reviews.length > 0 && (
                <ReviewFilters
                  filter={recFilter}
                  setFilter={setRecFilter}
                  starFilter={starFilter}
                  setStarFilter={setStarFilter}
                />
              )}

              {!loadingReviews && reviews.length === 0 && (
                <p style={{ color: '#9B9B9B', fontSize: 14 }}>
                  Nenhuma avaliação disponível para este produto nos splits de validação e teste.
                </p>
              )}

              {!loadingReviews && reviews.length > 0 && filteredReviews.length === 0 && (
                <p style={{ color: '#9B9B9B', fontSize: 14 }}>
                  Nenhuma avaliação corresponde ao filtro selecionado.
                </p>
              )}

              {filteredReviews.length > 0 && (
                <div className="reviews-scroll">
                  {filteredReviews.map((r, i) => (
                    <ReviewItem key={i} review={r} />
                  ))}
                </div>
              )}
            </div>

        </div>
      </div>
    </div>
  )
}
