
import { memo, useEffect, useId, useRef } from 'react';
import * as d3 from 'd3';
import type { Prediction, InputDataPoint } from '../types/dose-response';
import { getLogDose } from '../utils/dataTransform';
import { createDoseResponseLine } from '../utils/curveInterpolation';
import { useVivid } from '../contexts/VividContext';
import './DoseResponseChart.css';

interface DoseResponseChartProps {
  drugLabel: string;
  sampleId: string;
  predictions: Prediction[];
  inputPoints: InputDataPoint[];
  subsampledIndices?: Set<number>;
  width?: number;
  height?: number;
  showQueryPoints?: boolean;
  doseRange?: [number, number];
  metrics?: { auc: number; ic50: number | null };
}

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

export default memo(function DoseResponseChart({
  drugLabel,
  sampleId: _sampleId,
  predictions,
  inputPoints,
  subsampledIndices,
  width = 320,
  height = 240,
  showQueryPoints = true,
  doseRange,
  metrics,
}: DoseResponseChartProps) {
  const vivid = useVivid();
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId().replace(/:/g, '_');

  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const g = svg.select<SVGGElement>('.chart-content');

    // Collect all dose values for domain calculation
    const allLogDoses: number[] = [];
    predictions.forEach((p) => allLogDoses.push(getLogDose(p.dose1)));
    inputPoints.forEach((p) => allLogDoses.push(getLogDose(p.dose1)));

    if (allLogDoses.length === 0) return;

    let domainMin: number, domainMax: number;
    if (doseRange) {
      domainMin = doseRange[0];
      domainMax = doseRange[1];
    } else {
      const doseMin = d3.min(allLogDoses) ?? -6;
      const doseMax = d3.max(allLogDoses) ?? 4;
      const domainPad = Math.max((doseMax - doseMin) * 0.1, 0.5);
      domainMin = doseMin - domainPad;
      domainMax = doseMax + domainPad;
    }

    const xScale = d3
      .scaleLinear()
      .domain([domainMin, domainMax])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear().domain([0, 1.05]).range([innerHeight, 0]);

    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5);

    g.select<SVGGElement>('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis);

    g.select<SVGGElement>('.y-axis').call(yAxis);

    // Prepare curve data: sort predictions by dose
    const sortedPreds = [...predictions].sort((a, b) => a.dose1 - b.dose1);
    const curveData: [number, number][] = sortedPreds.map((p) => [
      getLogDose(p.dose1),
      Math.max(0, Math.min(1, p.predicted_viability)),
    ]);

    // Prediction curve
    const lineGen = createDoseResponseLine(xScale, yScale);
    const plotArea = g.select<SVGGElement>('.plot-area');
    plotArea.select<SVGPathElement>('.prediction-curve').attr(
      'd',
      lineGen(curveData) || '',
    );

    // Input data points — split into selected (white) and background (small grey)
    const inputWithVis = inputPoints
      .filter((p) => p.float_value !== undefined)
      .map((p) => ({
        x: getLogDose(p.dose1),
        y: Math.max(0, Math.min(1, p.float_value!)),
        selected: !subsampledIndices || subsampledIndices.has(p._rowIndex),
      }));

    const bgPoints = inputWithVis.filter((d) => !d.selected);
    const fgPoints = inputWithVis.filter((d) => d.selected);

    // Background points (non-selected): small, grey
    plotArea.selectAll<SVGCircleElement, typeof bgPoints[0]>('[data-point-type="input-bg"]')
      .data(bgPoints)
      .join('circle')
      .attr('data-point-type', 'input-bg')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', vivid ? 3 : 2)
      .attr('fill', vivid ? '#888' : '#bbb')
      .attr('stroke', vivid ? '#555' : 'none')
      .attr('stroke-width', vivid ? 0.5 : 0)
      .attr('opacity', vivid ? 0.8 : 0.5);

    // Selected points: white, black edge
    plotArea.selectAll<SVGCircleElement, typeof fgPoints[0]>('[data-point-type="input"]')
      .data(fgPoints)
      .join('circle')
      .attr('data-point-type', 'input')
      .attr('cx', (d) => xScale(d.x))
      .attr('cy', (d) => yScale(d.y))
      .attr('r', vivid ? 6 : 5)
      .attr('fill', vivid ? '#e8e8e8' : 'white')
      .attr('stroke', vivid ? '#1a1a1a' : 'black')
      .attr('stroke-width', vivid ? 2 : 1.5);

    // Query data points (grey, only if showQueryPoints and float_value exists)
    if (showQueryPoints) {
      const queryData = sortedPreds
        .filter((p) => p.float_value !== undefined && p.float_value !== null)
        .map((p) => ({
          x: getLogDose(p.dose1),
          y: Math.max(0, Math.min(1, p.float_value!)),
        }));

      plotArea.selectAll<SVGCircleElement, typeof queryData[0]>('[data-point-type="query"]')
        .data(queryData)
        .join('circle')
        .attr('data-point-type', 'query')
        .attr('cx', (d) => xScale(d.x))
        .attr('cy', (d) => yScale(d.y))
        .attr('r', vivid ? 4 : 3)
        .attr('fill', vivid ? '#666' : '#999');
    } else {
      plotArea.selectAll('[data-point-type="query"]').remove();
    }

    // Raise the prediction curve to render on top of data points
    plotArea.select<SVGPathElement>('.prediction-curve').raise();
  }, [predictions, inputPoints, subsampledIndices, showQueryPoints, innerWidth, innerHeight, doseRange, vivid]);

  return (
    <div className="dose-response-chart-container">
      <div className="chart-title" style={{ textAlign: 'center' }}>
        {drugLabel}
        {_sampleId && <span style={{ color: 'var(--text-secondary)', fontWeight: 'var(--font-weight-normal)', fontSize: '0.9em' }}> — {_sampleId}</span>}
      </div>
      {metrics && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: '6px', fontWeight: 'var(--font-weight-normal)', textAlign: 'center' }}>
          AUC: {metrics.auc.toFixed(3)} | IC50: {metrics.ic50 !== null ? metrics.ic50.toFixed(2) : '—'}
        </div>
      )}
      <svg
        data-testid="dose-response-chart"
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: 'block' }}
        ref={svgRef}
      >
        <defs>
          <clipPath id={`clip-${clipId}`}>
            <rect x={-6} y={-6} width={innerWidth + 12} height={innerHeight + 6} />
          </clipPath>
        </defs>
        <g
          className="chart-content"
          transform={`translate(${MARGIN.left},${MARGIN.top})`}
        >
          <g className="x-axis" />
          <g className="y-axis" />

          <text
            className="x-axis-label"
            x={innerWidth / 2}
            y={innerHeight + 32}
            textAnchor="middle"
            fontSize="11"
          >
            Log₁₀ Dose (µM)
          </text>
          <text
            className="y-axis-label"
            transform={`rotate(-90)`}
            x={-innerHeight / 2}
            y={-36}
            textAnchor="middle"
            fontSize="11"
          >
            Viability
          </text>

          {/* Clipped plot area */}
          <g className="plot-area" clipPath={`url(#clip-${clipId})`}>
            <path
              className="prediction-curve"
              fill="none"
              stroke="var(--color-brand-500)"
              strokeWidth={2}
            />
          </g>
        </g>
      </svg>
    </div>
  );
})
