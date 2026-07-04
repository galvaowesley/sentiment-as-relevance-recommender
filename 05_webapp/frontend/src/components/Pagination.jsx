export default function Pagination({ page, pageSize, total, onPage }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  if (totalPages <= 1) return null

  // Build page number window: always show first, last, current ±2, with ellipsis
  function getPages() {
    const pages = []
    const delta = 2
    const left  = page - delta
    const right = page + delta

    let last = 0
    for (let p = 1; p <= totalPages; p++) {
      if (p === 1 || p === totalPages || (p >= left && p <= right)) {
        if (last && p - last > 1) pages.push('...')
        pages.push(p)
        last = p
      }
    }
    return pages
  }

  const pages = getPages()

  const btnBase = {
    height: 34,
    minWidth: 34,
    padding: '0 10px',
    borderRadius: 8,
    border: '1px solid #E5E7EB',
    background: '#fff',
    fontSize: 13,
    fontWeight: 500,
    cursor: 'pointer',
    fontFamily: 'inherit',
    transition: 'background 0.1s, border-color 0.1s, color 0.1s',
  }

  return (
    <div className="pagination" style={{ paddingBottom: 16 }}>
      {/* Anterior */}
      <button
        style={{ ...btnBase, ...(page <= 1 ? { opacity: 0.4, cursor: 'default' } : {}) }}
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
      >
        ← Anterior
      </button>

      {/* Page numbers */}
      {pages.map((p, i) =>
        p === '...'
          ? <span key={`dot-${i}`} style={{ padding: '0 4px', color: '#9B9B9B', fontSize: 13 }}>…</span>
          : (
            <button
              key={p}
              style={{
                ...btnBase,
                background: p === page ? '#8B5CF6' : '#fff',
                color:      p === page ? '#fff'    : '#4A4A4A',
                borderColor: p === page ? '#8B5CF6' : '#E5E7EB',
                fontWeight: p === page ? 700 : 500,
              }}
              onClick={() => p !== page && onPage(p)}
            >
              {p}
            </button>
          )
      )}

      {/* Próxima */}
      <button
        style={{ ...btnBase, ...(page >= totalPages ? { opacity: 0.4, cursor: 'default' } : {}) }}
        disabled={page >= totalPages}
        onClick={() => onPage(page + 1)}
      >
        Próxima →
      </button>

      <span className="pagination-summary">
        {total} produto{total !== 1 ? 's' : ''} · {totalPages} página{totalPages !== 1 ? 's' : ''}
      </span>
    </div>
  )
}
