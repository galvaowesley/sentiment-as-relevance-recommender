import { useRef } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'

export default function Header() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const inputRef = useRef(null)

  function handleSearch(e) {
    if (e.key === 'Enter') {
      const raw = inputRef.current.value.trim()
      if (!raw) return

      // If the query looks like a product_id (numeric, no spaces), go directly to that product
      if (/^\d+$/.test(raw)) {
        navigate(`/product/${encodeURIComponent(raw)}`)
        return
      }

      const next = new URLSearchParams()
      next.set('q', raw)
      setSearchParams(next)
      navigate(`/?${next.toString()}`)
    }
  }

  function handleLogoClick() {
    setSearchParams(new URLSearchParams())
    navigate('/')
  }

  return (
    <header className="header">
      <div className="header-logo" onClick={handleLogoClick} style={{ cursor: 'pointer' }}>
        <div className="header-logo-icon">S</div>
        <span className="header-brand">SentiShop</span>
      </div>

      <div className="header-search">
        <svg className="header-search-icon" width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="7" cy="7" r="5" stroke="#9B9B9B" strokeWidth="1.5"/>
          <line x1="11" y1="11" x2="14" y2="14" stroke="#9B9B9B" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <input
          ref={inputRef}
          type="search"
          placeholder="Buscar por nome ou ID do produto… (Enter)"
          defaultValue={searchParams.get('q') ?? ''}
          onKeyDown={handleSearch}
        />
      </div>

      <span className="header-subtitle">Recomendação por sentimento</span>
    </header>
  )
}
