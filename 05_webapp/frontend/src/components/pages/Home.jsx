import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getCategories } from '../../api.js'
import CategoryIcon from '../CategoryIcon.jsx'

export default function Home() {
  const [categories, setCategories] = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    getCategories()
      .then(setCategories)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="page">
      <div className="home-hero">
        <h1>Bem-vindo ao <span>SentiShop</span></h1>
        <p>Recomendações inteligentes baseadas em análise de sentimento de avaliações reais de clientes.</p>
      </div>

      <div className="home-categories">
        <h2>Navegar por categoria</h2>
        {loading && <div className="loading">Carregando categorias…</div>}
        {!loading && (
          <div className="category-grid">
            {categories.map((cat) => (
              <div
                key={cat}
                className="category-tile"
                onClick={() => navigate(`/browse?category=${encodeURIComponent(cat)}`)}
              >
                <div className="category-tile-icon" style={{ color: 'var(--accent)' }}>
                  <CategoryIcon category={cat} size={36} />
                </div>
                <span className="category-tile-name">{cat}</span>
              </div>
            ))}
            <div
              className="category-tile"
              onClick={() => navigate('/browse')}
              style={{ borderStyle: 'dashed' }}
            >
              <div className="category-tile-icon" style={{ color: 'var(--text-muted)' }}>
                <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
                  <circle cx="28" cy="28" r="18" stroke="currentColor" strokeWidth="2.5" fill="none"/>
                  <line x1="28" y1="18" x2="28" y2="38" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                  <line x1="18" y1="28" x2="38" y2="28" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
              </div>
              <span className="category-tile-name" style={{ color: 'var(--text-muted)' }}>
                Todos os produtos
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
