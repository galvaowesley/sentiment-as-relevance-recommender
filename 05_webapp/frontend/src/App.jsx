import { Routes, Route, Navigate } from 'react-router-dom'
import Header from './components/Header.jsx'
import Browse from './components/pages/Browse.jsx'
import ProductDetail from './components/pages/ProductDetail.jsx'

export default function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Browse />} />
        <Route path="/browse" element={<Navigate to="/" replace />} />
        <Route path="/product/:productId" element={<ProductDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
