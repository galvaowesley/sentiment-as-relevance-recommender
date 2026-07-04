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

function ScoreBar({ label, value, type }) {
  const pct = Math.round((value ?? 0) * 100)
  return (
    <div className="score-bar-row">
      <div className="score-bar-labels">
        <span className={`score-bar-label ${type}`}>{label}</span>
        <span className={`score-bar-value ${type}`}>
          {type === 'red' ? `${pct}%` : value?.toFixed(2)}
        </span>
      </div>
      <div className="score-bar-track">
        <div
          className={`score-bar-fill ${type}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function RecItem({ rec, rank, onClick }) {
  const {
    product_name, site_category_lv1,
    avg_rating, review_count, num_reviews,
    score, similarity, sentiment_score,
  } = rec

  const count  = review_count ?? num_reviews ?? 0
  const rating = avg_rating ?? null

  return (
    <div className="rec-item" onClick={onClick}>
      {/* Rank + final score */}
      <div className="rec-item-top">
        <span className={`rec-rank-badge ${rank === 1 ? 'rank-1' : 'rank-other'}`}>
          #{rank}
        </span>
        <div className="rec-final-score">
          score final <strong>{score?.toFixed(2)}</strong>
        </div>
      </div>

      {/* Icon + name + stars */}
      <div className="rec-item-product">
        <div className="rec-item-icon">
          <CategoryIcon category={site_category_lv1} size={32} />
        </div>
        <div className="rec-item-product-info">
          <div className="rec-item-name" title={product_name}>{product_name}</div>
          <div className="rec-item-stars">
            {rating !== null ? (
              <>
                <Stars rating={rating} />
                <span className="stars-value">{rating.toFixed(1)}</span>
                <span className="stars-count">· {count} aval.</span>
              </>
            ) : (
              <span className="stars-count">{count} aval.</span>
            )}
          </div>
        </div>
      </div>

      {/* Score breakdown */}
      <div className="score-bars">
        <ScoreBar label="similaridade textual" value={similarity}       type="teal" />
        <ScoreBar label="sentimento S(p)"       value={sentiment_score} type="red"  />
      </div>
    </div>
  )
}

export default function RecommendationsPanel({
  recommendations,
  loading,
  alpha = 0.7,
  onAlphaChange,
  defaultAlpha = 0.7,
}) {
  const navigate = useNavigate()
  const interactive = typeof onAlphaChange === 'function'
  const simPct = Math.round(alpha * 100)
  const sentPct = 100 - simPct

  return (
    <>
      <div className="recs-panel">
        {/* Header */}
        <div className="recs-panel-header">
          <div className="recs-panel-title">Recomendados para você</div>
          <div className="recs-pipeline">
            <span className="pipeline-step step1">1</span>
            <span className="pipeline-label teal">recuperação por similaridade</span>
            <span className="pipeline-sep">→</span>
            <span className="pipeline-step step2">2</span>
            <span className="pipeline-label red">re-rank por sentimento</span>
          </div>
        </div>

        {/* List */}
        <div className="recs-list">
          {loading && (
            <div className="loading" style={{ minHeight: 120 }}>Carregando…</div>
          )}

          {!loading && recommendations.length === 0 && (
            <div style={{ padding: '24px 18px', color: '#9B9B9B', fontSize: 14 }}>
              Nenhuma recomendação disponível.
            </div>
          )}

          {!loading && recommendations.map((rec, i) => (
            <RecItem
              key={rec.product_id}
              rec={rec}
              rank={i + 1}
              onClick={() => navigate(`/product/${encodeURIComponent(rec.product_id)}`)}
            />
          ))}
        </div>
      </div>

      {/* Explainability + interactive α control */}
      <div className="formula-card">
        <div className="formula-card-label">Fórmula do score final</div>
        <div className="formula-expr">score = α · sim + (1−α) · S(p)</div>

        {interactive && (
          <div className="alpha-control">
            <div className="alpha-control-head">
              <span className="alpha-control-title">Ajuste o peso α</span>
              <span className="alpha-value">α = {alpha.toFixed(2)}</span>
            </div>
            <input
              type="range"
              className="alpha-slider"
              min="0"
              max="1"
              step="0.05"
              value={alpha}
              onChange={(e) => onAlphaChange(parseFloat(e.target.value))}
              aria-label="Peso α do re-ranking"
            />
            <div className="alpha-scale">
              <span className="alpha-scale-end red">← só sentimento</span>
              <span className="alpha-scale-end teal">só similaridade →</span>
            </div>
            {Math.abs(alpha - defaultAlpha) > 1e-9 && (
              <button
                type="button"
                className="alpha-reset"
                onClick={() => onAlphaChange(defaultAlpha)}
              >
                restaurar padrão (α = {defaultAlpha.toFixed(2)})
              </button>
            )}
          </div>
        )}

        <div className="formula-note">
          {alpha >= 0.999
            ? 'α = 1 — ranqueado apenas por similaridade textual; o sentimento é ignorado.'
            : alpha <= 0.001
            ? 'α = 0 — ranqueado apenas por S(p); a similaridade é ignorada.'
            : `similaridade textual pesa ${simPct}% · sentimento S(p) pesa ${sentPct}%`}
        </div>
        <div className="formula-note" style={{ marginTop: 4 }}>
          S(p) = proporção de avaliações classificadas como positivas pelo BERTimbau.
        </div>
      </div>
    </>
  )
}
