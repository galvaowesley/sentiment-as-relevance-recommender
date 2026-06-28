import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getCategories, getSubcategories } from '../api.js'

export default function CategoryNavBar() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [lv1List, setLv1List] = useState([])
  const [lv2List, setLv2List] = useState([])

  const cat1 = searchParams.get('category') ?? ''
  const cat2 = searchParams.get('cat2') ?? ''

  useEffect(() => {
    getCategories().then(setLv1List).catch(console.error)
  }, [])

  useEffect(() => {
    if (!cat1) { setLv2List([]); return }
    getSubcategories(cat1).then(setLv2List).catch(console.error)
  }, [cat1])

  function selectLv1(cat) {
    const next = new URLSearchParams(searchParams)
    if (cat) { next.set('category', cat) } else { next.delete('category') }
    next.delete('cat2')
    next.delete('page')
    setSearchParams(next)
  }

  function selectLv2(cat) {
    const next = new URLSearchParams(searchParams)
    if (cat) { next.set('cat2', cat) } else { next.delete('cat2') }
    next.delete('page')
    setSearchParams(next)
  }

  return (
    <>
      <nav className="catnav">
        <button
          className={`catnav-tab${!cat1 ? ' active' : ''}`}
          onClick={() => selectLv1('')}
        >
          Todos
        </button>
        {lv1List.map((c) => (
          <button
            key={c}
            className={`catnav-tab${cat1 === c ? ' active' : ''}`}
            onClick={() => selectLv1(c)}
          >
            {c}
          </button>
        ))}
      </nav>

      {cat1 && lv2List.length > 0 && (
        <nav className="catnav2">
          <button
            className={`catnav2-tab${!cat2 ? ' active' : ''}`}
            onClick={() => selectLv2('')}
          >
            Todos em {cat1}
          </button>
          {lv2List.map((c) => (
            <button
              key={c}
              className={`catnav2-tab${cat2 === c ? ' active' : ''}`}
              onClick={() => selectLv2(c)}
            >
              {c}
            </button>
          ))}
        </nav>
      )}
    </>
  )
}
