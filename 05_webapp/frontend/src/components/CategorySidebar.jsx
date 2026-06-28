import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getCategories, getSubcategories } from '../api.js'

const SIDEBAR_W  = 220
const TOGGLE_W   = 40

export default function CategorySidebar() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [collapsed, setCollapsed] = useState(false)
  const [lv1List, setLv1List]     = useState([])
  const [lv2Map, setLv2Map]       = useState({})
  const [openLv1, setOpenLv1]     = useState(null)

  const cat1 = searchParams.get('category') ?? ''
  const cat2 = searchParams.get('cat2') ?? ''

  useEffect(() => {
    getCategories().then(setLv1List).catch(console.error)
  }, [])

  // Auto-open accordion for the active lv1
  useEffect(() => {
    if (cat1) setOpenLv1(cat1)
  }, [cat1])

  // Lazy-load lv2 when accordion opens
  useEffect(() => {
    if (!openLv1 || lv2Map[openLv1]) return
    getSubcategories(openLv1)
      .then((subs) => setLv2Map((prev) => ({ ...prev, [openLv1]: subs })))
      .catch(console.error)
  }, [openLv1])

  function selectLv1(c) {
    const next = new URLSearchParams()
    if (c) next.set('category', c)
    setSearchParams(next)
    setOpenLv1(c || null)
  }

  function selectLv2(c) {
    const next = new URLSearchParams(searchParams)
    if (c) { next.set('cat2', c) } else { next.delete('cat2') }
    next.delete('page')
    setSearchParams(next)
  }

  function toggleAccordion(c) {
    if (cat1 !== c) {
      selectLv1(c)
    } else {
      setOpenLv1(openLv1 === c ? null : c)
    }
  }

  const w = collapsed ? TOGGLE_W : SIDEBAR_W

  const btnBase = {
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    border: 'none',
    background: 'none',
    cursor: 'pointer',
    fontFamily: 'inherit',
    transition: 'background 0.12s, color 0.12s',
  }

  return (
    <aside
      style={{
        width: w,
        minWidth: w,
        flexShrink: 0,
        background: '#fff',
        borderRight: '1px solid rgba(0,0,0,0.08)',
        transition: 'width 0.2s ease, min-width 0.2s ease',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        alignSelf: 'stretch',
      }}
    >
      {/* Toggle button */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        title={collapsed ? 'Expandir categorias' : 'Colapsar categorias'}
        style={{
          ...btnBase,
          justifyContent: collapsed ? 'center' : 'flex-start',
          gap: 8,
          padding: '11px 12px',
          fontSize: 11,
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          color: '#9B9B9B',
          borderBottom: '1px solid rgba(0,0,0,0.06)',
          flexShrink: 0,
          whiteSpace: 'nowrap',
        }}
      >
        <svg
          width="15" height="15" viewBox="0 0 15 15" fill="none"
          style={{ flexShrink: 0, transform: collapsed ? 'none' : 'none' }}
        >
          <line x1="1.5" y1="3.5"  x2="13.5" y2="3.5"  stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          <line x1="1.5" y1="7.5"  x2="13.5" y2="7.5"  stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          <line x1="1.5" y1="11.5" x2="13.5" y2="11.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        {!collapsed && 'Categorias'}
      </button>

      {/* "Todos" row */}
      {!collapsed && (
        <button
          onClick={() => selectLv1('')}
          style={{
            ...btnBase,
            padding: '9px 14px',
            fontSize: 13,
            color: !cat1 ? '#7C3AED' : '#4A4A4A',
            fontWeight: !cat1 ? 700 : 400,
            background: !cat1 ? '#F5F0FF' : 'none',
            borderLeft: !cat1 ? '3px solid #8B5CF6' : '3px solid transparent',
          }}
        >
          Todos os produtos
        </button>
      )}

      {/* LV1 list */}
      <div style={{ overflowY: 'auto', flex: 1 }}>
        {lv1List.map((c) => {
          const isActive = cat1 === c
          const isOpen   = openLv1 === c
          const subs     = lv2Map[c] ?? []

          return (
            <div key={c}>
              <button
                onClick={() => toggleAccordion(c)}
                title={collapsed ? c : undefined}
                style={{
                  ...btnBase,
                  justifyContent: 'space-between',
                  padding: collapsed ? '9px 0' : '9px 14px',
                  fontSize: 13,
                  color: isActive ? '#7C3AED' : '#4A4A4A',
                  fontWeight: isActive ? 700 : 400,
                  background: isActive ? '#F5F0FF' : 'none',
                  borderLeft: isActive ? '3px solid #8B5CF6' : '3px solid transparent',
                  gap: 4,
                }}
              >
                <span style={{
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  flex: 1,
                  textAlign: collapsed ? 'center' : 'left',
                  fontSize: collapsed ? 10 : 13,
                }}>
                  {collapsed ? c.slice(0, 2) : c}
                </span>
                {!collapsed && (
                  <svg
                    width="10" height="10" viewBox="0 0 10 10" fill="none"
                    style={{
                      flexShrink: 0,
                      transform: isOpen ? 'rotate(180deg)' : 'none',
                      transition: 'transform 0.15s',
                      opacity: 0.45,
                    }}
                  >
                    <path d="M2 3.5l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                )}
              </button>

              {/* LV2 accordion */}
              {!collapsed && isOpen && subs.length > 0 && (
                <div style={{ background: '#FAFAFA', borderLeft: '3px solid #E9D5FF' }}>
                  <button
                    onClick={() => selectLv2('')}
                    style={{
                      ...btnBase,
                      padding: '6px 14px 6px 20px',
                      fontSize: 12,
                      color: isActive && !cat2 ? '#7C3AED' : '#6B6B6B',
                      fontWeight: isActive && !cat2 ? 600 : 400,
                      background: isActive && !cat2 ? '#EDE9FE' : 'none',
                    }}
                  >
                    Todos em {c}
                  </button>
                  {subs.map((s) => (
                    <button
                      key={s}
                      onClick={() => selectLv2(s)}
                      style={{
                        ...btnBase,
                        padding: '6px 14px 6px 20px',
                        fontSize: 12,
                        color: cat2 === s ? '#7C3AED' : '#6B6B6B',
                        fontWeight: cat2 === s ? 600 : 400,
                        background: cat2 === s ? '#EDE9FE' : 'none',
                        textAlign: 'left',
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </aside>
  )
}
