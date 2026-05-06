
import { useMemo } from 'react';
import { scaleLinear, scaleBand } from 'd3-scale';
import type { DeltaSummary } from '../types/api';
import { groupDeltasByGroup, computeBoxPlotStats } from '../utils/analytics';

interface GroupComparisonSectionProps {
  deltas: DeltaSummary[];
  groups: Record<string, string>; // sample_id -> group label
  groupColumn: string;
}

const BOX_WIDTH = 30;
const BOX_MARGIN = 10;
const PLOT_HEIGHT = 150;
const PLOT_TOP = 20;
const PLOT_LEFT = 60;
const PLOT_RIGHT = 20;

export default function GroupComparisonSection({
  deltas,
  groups,
  groupColumn,
}: GroupComparisonSectionProps) {
  // Group deltas by group label
  const groupedDeltas = useMemo(
    () => groupDeltasByGroup(deltas, groups),
    [deltas, groups],
  );

  // Get all unique group labels
  const groupLabels = useMemo(
    () => Array.from(groupedDeltas.keys()).sort(),
    [groupedDeltas],
  );

  // Get unique drug keys across all deltas
  const drugKeys = useMemo(() => {
    const set = new Set<string>();
    for (const d of deltas) {
      set.add(d.drug_key);
    }
    return Array.from(set).sort();
  }, [deltas]);

  // Compute box plot stats per (group, drug) pair
  const boxStats = useMemo(() => {
    const stats = new Map<string, ReturnType<typeof computeBoxPlotStats>>();
    for (const [group, gDeltas] of groupedDeltas) {
      for (const drug of drugKeys) {
        const vals = gDeltas
          .filter((d) => d.drug_key === drug)
          .map((d) => d.delta);
        if (vals.length > 0) {
          stats.set(`${group}||${drug}`, computeBoxPlotStats(vals));
        }
      }
    }
    return stats;
  }, [groupedDeltas, drugKeys]);

  // Compute global delta range for consistent y-axis across all plots
  const [globalMin, globalMax] = useMemo(() => {
    if (deltas.length === 0) return [0, 1];
    const vals = deltas.map((d) => d.delta);
    return [Math.min(...vals) - 0.05, Math.max(...vals) + 0.05];
  }, [deltas]);

  // Compute mean delta summary table
  const summaryTable = useMemo(() => {
    const rows: { group: string; drug: string; meanDelta: number; count: number }[] = [];
    for (const [group, gDeltas] of groupedDeltas) {
      for (const drug of drugKeys) {
        const vals = gDeltas.filter((d) => d.drug_key === drug).map((d) => d.delta);
        if (vals.length > 0) {
          rows.push({
            group,
            drug,
            meanDelta: vals.reduce((a, b) => a + b, 0) / vals.length,
            count: vals.length,
          });
        }
      }
    }
    return rows;
  }, [groupedDeltas, drugKeys]);

  if (deltas.length === 0 || groupLabels.length === 0) {
    return null;
  }

  const groupBarWidth = groupLabels.length * (BOX_WIDTH + BOX_MARGIN);
  const plotWidth = PLOT_LEFT + groupBarWidth + PLOT_RIGHT;

  const yScale = scaleLinear()
    .domain([globalMin, globalMax])
    .range([PLOT_HEIGHT + PLOT_TOP, PLOT_TOP]);

  const groupScale = scaleBand<string>()
    .domain(groupLabels)
    .range([PLOT_LEFT, PLOT_LEFT + groupBarWidth])
    .paddingInner(0.2)
    .paddingOuter(0.1);

  return (
    <div data-testid="group-comparison-section">
      <h3 style={{ margin: '0 0 var(--space-3) 0', fontSize: 'var(--text-lg)', fontWeight: 'var(--font-weight-semibold)' }}>
        Group Comparison ({groupColumn})
      </h3>

      {/* Box plots per drug */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
        {drugKeys.map((drug) => (
          <div key={drug} style={{ border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: 'var(--space-2)', boxShadow: 'var(--shadow-xs)' }}>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 'bold', marginBottom: '4px', textAlign: 'center' }}>
              {drug}
            </div>
            <svg width={plotWidth} height={PLOT_HEIGHT + PLOT_TOP + 30}>
              {/* Y-axis line */}
              <line
                x1={PLOT_LEFT}
                y1={PLOT_TOP}
                x2={PLOT_LEFT}
                y2={PLOT_HEIGHT + PLOT_TOP}
                stroke="var(--color-neutral-200)"
              />

              {/* Y-axis ticks */}
              {yScale.ticks(5).map((tick) => (
                <g key={tick}>
                  <line
                    x1={PLOT_LEFT - 4}
                    y1={yScale(tick)}
                    x2={PLOT_LEFT}
                    y2={yScale(tick)}
                    stroke="var(--color-neutral-200)"
                  />
                  <text
                    x={PLOT_LEFT - 6}
                    y={yScale(tick)}
                    textAnchor="end"
                    dominantBaseline="middle"
                    fontSize={9}
                    fill="var(--text-tertiary)"
                  >
                    {tick.toFixed(2)}
                  </text>
                </g>
              ))}

              {/* Box plots for each group */}
              {groupLabels.map((group) => {
                const stats = boxStats.get(`${group}||${drug}`);
                if (!stats) return null;

                const cx = (groupScale(group) ?? 0) + groupScale.bandwidth() / 2;
                const halfWidth = Math.min(groupScale.bandwidth() / 2, BOX_WIDTH / 2);

                return (
                  <g key={group}>
                    {/* Whisker line (min to max) */}
                    <line
                      x1={cx}
                      y1={yScale(stats.min)}
                      x2={cx}
                      y2={yScale(stats.max)}
                      stroke="#555"
                      strokeWidth={1}
                    />

                    {/* Box (Q1 to Q3) */}
                    <rect
                      x={cx - halfWidth}
                      y={yScale(stats.q3)}
                      width={halfWidth * 2}
                      height={Math.max(yScale(stats.q1) - yScale(stats.q3), 1)}
                      fill="#b3d4fc"
                      stroke="#555"
                      strokeWidth={1}
                    />

                    {/* Median line */}
                    <line
                      x1={cx - halfWidth}
                      y1={yScale(stats.median)}
                      x2={cx + halfWidth}
                      y2={yScale(stats.median)}
                      stroke="#d32f2f"
                      strokeWidth={2}
                    />

                    {/* Min whisker cap */}
                    <line
                      x1={cx - halfWidth / 2}
                      y1={yScale(stats.min)}
                      x2={cx + halfWidth / 2}
                      y2={yScale(stats.min)}
                      stroke="#555"
                      strokeWidth={1}
                    />

                    {/* Max whisker cap */}
                    <line
                      x1={cx - halfWidth / 2}
                      y1={yScale(stats.max)}
                      x2={cx + halfWidth / 2}
                      y2={yScale(stats.max)}
                      stroke="#555"
                      strokeWidth={1}
                    />

                    {/* Group label */}
                    <text
                      x={cx}
                      y={PLOT_HEIGHT + PLOT_TOP + 14}
                      textAnchor="middle"
                      fontSize={10}
                      fill="var(--text-primary)"
                    >
                      {group}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        ))}
      </div>

      {/* Summary table */}
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>
                Group
              </th>
              <th>
                Drug
              </th>
              <th style={{ textAlign: 'right' }}>
                Mean {'\u0394'}
              </th>
              <th style={{ textAlign: 'right' }}>
                N
              </th>
            </tr>
          </thead>
          <tbody>
            {summaryTable.map((row, i) => (
              <tr key={i}>
                <td>
                  {row.group}
                </td>
                <td>
                  {row.drug}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {row.meanDelta.toFixed(4)}
                </td>
                <td style={{ textAlign: 'right' }}>
                  {row.count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
