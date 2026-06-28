export default function SentimentBadge({ score }) {
  const pct = Math.round(score * 100)
  const tier = score >= 0.75 ? 'green' : score >= 0.50 ? 'yellow' : 'red'
  return (
    <span className={`sentiment-badge ${tier}`}>
      <span className="sentiment-dot" />
      {pct}% positivo
    </span>
  )
}
