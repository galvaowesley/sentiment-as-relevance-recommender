/* Maps site_category_lv1 → a simple SVG icon. Falls back to a generic store icon. */

const icons = {
  'Celulares e Smartphones': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="14" y="6" width="28" height="44" rx="6" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <circle cx="28" cy="44" r="2.5" fill="currentColor" opacity=".5"/>
      <rect x="22" y="10" width="12" height="3" rx="1.5" fill="currentColor" opacity=".4"/>
    </svg>
  ),
  'Informática': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="7" y="14" width="42" height="24" rx="3" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <rect x="4" y="38" width="48" height="6" rx="3" fill="currentColor" opacity=".4"/>
      <rect x="21" y="40" width="14" height="2" rx="1" fill="currentColor" opacity=".6"/>
    </svg>
  ),
  'Eletroportáteis': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <path d="M14 34 Q28 10 42 34" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" fill="none"/>
      <rect x="21" y="34" width="14" height="10" rx="2" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <line x1="28" y1="44" x2="28" y2="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  ),
  'Eletrodomésticos': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="10" y="8" width="36" height="40" rx="4" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <circle cx="28" cy="28" r="10" stroke="currentColor" strokeWidth="2" fill="none" opacity=".5"/>
      <circle cx="28" cy="28" r="4" fill="currentColor" opacity=".4"/>
    </svg>
  ),
  'TV e Home Theater': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="6" y="10" width="44" height="28" rx="4" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <line x1="18" y1="44" x2="38" y2="44" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
      <line x1="28" y1="38" x2="28" y2="44" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  ),
  'Áudio': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <path d="M10 30v-4C10 17.16 18.06 10 28 10s18 7.16 18 16v4" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" fill="none"/>
      <rect x="8" y="27" width="8" height="14" rx="4" fill="currentColor" opacity=".5"/>
      <rect x="40" y="27" width="8" height="14" rx="4" fill="currentColor" opacity=".5"/>
    </svg>
  ),
  'Games': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="4" y="17" width="48" height="24" rx="4" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <rect x="10" y="24" width="5" height="4" rx="1" fill="currentColor" opacity=".5"/>
      <rect x="19" y="24" width="5" height="4" rx="1" fill="currentColor" opacity=".5"/>
      <rect x="28" y="24" width="5" height="4" rx="1" fill="currentColor" opacity=".5"/>
      <rect x="37" y="24" width="9" height="4" rx="1" fill="currentColor" opacity=".5"/>
      <rect x="16" y="32" width="24" height="4" rx="1" fill="currentColor" opacity=".3"/>
    </svg>
  ),
  'Câmeras e Drones': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="6" y="16" width="44" height="28" rx="4" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <circle cx="28" cy="30" r="9" stroke="currentColor" strokeWidth="2" fill="none" opacity=".6"/>
      <circle cx="28" cy="30" r="4" fill="currentColor" opacity=".4"/>
      <rect x="16" y="10" width="8" height="6" rx="2" fill="currentColor" opacity=".4"/>
    </svg>
  ),
  'Móveis': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="8" y="20" width="40" height="16" rx="3" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <line x1="16" y1="36" x2="16" y2="46" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
      <line x1="40" y1="36" x2="40" y2="46" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
      <rect x="14" y="12" width="28" height="8" rx="2" fill="currentColor" opacity=".3"/>
    </svg>
  ),
  'Utilidades Domésticas': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <path d="M12 44 L16 16 h24 l4 28" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
      <line x1="10" y1="44" x2="46" y2="44" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
      <path d="M22 16 v-4 a6 6 0 0 1 12 0 v4" stroke="currentColor" strokeWidth="2" fill="none" opacity=".5"/>
    </svg>
  ),
  'Beleza e Perfumaria': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="16" y="8" width="24" height="38" rx="12" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <line x1="28" y1="8" x2="28" y2="28" stroke="currentColor" strokeWidth="1.5" opacity=".4"/>
      <circle cx="28" cy="21" r="3" fill="currentColor" opacity=".4"/>
    </svg>
  ),
  'Saúde': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <path d="M28 10 L28 46 M10 28 L46 28" stroke="currentColor" strokeWidth="4" strokeLinecap="round"/>
    </svg>
  ),
  'Esporte e Lazer': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <circle cx="28" cy="28" r="18" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <path d="M10 28 Q18 16 28 28 Q38 40 46 28" stroke="currentColor" strokeWidth="2" fill="none" opacity=".5"/>
    </svg>
  ),
  'Bebês': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <circle cx="28" cy="20" r="10" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <path d="M12 46 C12 36 44 36 44 46" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" fill="none"/>
    </svg>
  ),
  'Brinquedos': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="10" y="22" width="36" height="24" rx="4" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <rect x="18" y="10" width="8" height="12" rx="2" stroke="currentColor" strokeWidth="2" fill="none" opacity=".5"/>
      <rect x="30" y="10" width="8" height="12" rx="2" stroke="currentColor" strokeWidth="2" fill="none" opacity=".5"/>
    </svg>
  ),
  'Livros': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <rect x="10" y="8" width="26" height="40" rx="2" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <line x1="16" y1="18" x2="30" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".5"/>
      <line x1="16" y1="24" x2="30" y2="24" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".5"/>
      <line x1="16" y1="30" x2="24" y2="30" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity=".5"/>
      <line x1="10" y1="8" x2="10" y2="48" stroke="currentColor" strokeWidth="4" strokeLinecap="round" opacity=".3"/>
    </svg>
  ),
  'Casa e Construção': (
    <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
      <path d="M8 28 L28 10 L48 28" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
      <rect x="14" y="28" width="28" height="18" rx="2" stroke="currentColor" strokeWidth="2.5" fill="none"/>
      <rect x="22" y="34" width="12" height="12" rx="1" fill="currentColor" opacity=".3"/>
    </svg>
  ),
}

const defaultIcon = (
  <svg width="32" height="32" viewBox="0 0 56 56" fill="none">
    <rect x="8" y="14" width="40" height="30" rx="4" stroke="currentColor" strokeWidth="2.5" fill="none"/>
    <path d="M20 14 v-4 a8 8 0 0 1 16 0 v4" stroke="currentColor" strokeWidth="2" fill="none" opacity=".5"/>
  </svg>
)

export default function CategoryIcon({ category, size = 32 }) {
  const icon = icons[category] ?? defaultIcon
  return (
    <span style={{ width: size, height: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {icon}
    </span>
  )
}
