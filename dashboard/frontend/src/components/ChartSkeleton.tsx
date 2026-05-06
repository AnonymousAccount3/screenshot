
export default function ChartSkeleton() {
  return (
    <div
      data-testid="chart-skeleton"
      className="skeleton"
      style={{
        width: 320,
        height: 240,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--text-tertiary)',
        fontSize: 'var(--text-sm)',
      }}
    >
      Loading...
    </div>
  );
}
