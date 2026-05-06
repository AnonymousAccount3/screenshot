
import { useEffect, useId, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import type { SampleCurve } from '../types/dose-response';
import { getLogDose } from '../utils/dataTransform';
import { createDoseResponseLine } from '../utils/curveInterpolation';

interface OverlayChartProps {
  drugLabel: string;
  sampleCurves: SampleCurve[];
  width?: number;
  height?: number;
  doseRange?: [number, number];
}

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

export default function OverlayChart({
  drugLabel,
  sampleCurves,
  width = 400,
  height = 300,
  doseRange,
}: OverlayChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId().replace(/:/g, '_');
  const [hoveredSample, setHoveredSample] = useState<string | null>(null);
  const [legendExpanded, setLegendExpanded] = useState(false);

  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  const handleMouseOver = useCallback((sampleId: string) => {
    setHoveredSample(sampleId);
  }, []);

  const handleMouseOut = useCallback(() => {
    setHoveredSample(null);
  }, []);

  useEffect(() => {
    if (!svgRef.current || sampleCurves.length === 0) return;

    const svg = d3.select(svgRef.current);
    const g = svg.select<SVGGElement>('.chart-content');

    // Collect all doses for domain
    const allLogDoses: number[] = [];
    sampleCurves.forEach((sc) =>
      sc.predictions.forEach((p) => allLogDoses.push(getLogDose(p.dose1))),
    );

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

    // Axes
    g.select<SVGGElement>('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5));

    g.select<SVGGElement>('.y-axis').call(d3.axisLeft(yScale).ticks(5));

    // Draw curves
    const lineGen = createDoseResponseLine(xScale, yScale);
    const curvesGroup = g.select<SVGGElement>('.curves');

    const curveSelection = curvesGroup
      .selectAll<SVGPathElement, SampleCurve>('.overlay-curve')
      .data(sampleCurves, (d) => d.sampleId);

    curveSelection
      .join('path')
      .attr('class', 'overlay-curve')
      .attr('fill', 'none')
      .attr('stroke', (d) => d.color)
      .attr('stroke-width', 2)
      .attr('d', (d) => {
        const sorted = [...d.predictions].sort((a, b) => a.dose1 - b.dose1);
        const data: [number, number][] = sorted.map((p) => [
          getLogDose(p.dose1),
          Math.max(0, Math.min(1, p.predicted_viability)),
        ]);
        return lineGen(data) || '';
      });

    // Invisible wider hit areas for easier hovering
    const hitGroup = g.select<SVGGElement>('.hit-areas');
    const hitSelection = hitGroup
      .selectAll<SVGPathElement, SampleCurve>('.hit-area')
      .data(sampleCurves, (d) => d.sampleId);

    hitSelection
      .join('path')
      .attr('class', 'hit-area')
      .attr('fill', 'none')
      .attr('stroke', 'transparent')
      .attr('stroke-width', 12)
      .attr('d', (d) => {
        const sorted = [...d.predictions].sort((a, b) => a.dose1 - b.dose1);
        const data: [number, number][] = sorted.map((p) => [
          getLogDose(p.dose1),
          Math.max(0, Math.min(1, p.predicted_viability)),
        ]);
        return lineGen(data) || '';
      })
      .style('cursor', 'pointer');
  }, [sampleCurves, innerWidth, innerHeight, doseRange]);

  // Update opacity + stroke-width on hover (separate effect to avoid full redraw)
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    svg.selectAll<SVGPathElement, SampleCurve>('.overlay-curve')
      .attr('opacity', (d) => {
        if (!hoveredSample) return 0.8;
        return d.sampleId === hoveredSample ? 1 : 0.15;
      })
      .attr('stroke-width', (d) => {
        if (!hoveredSample) return 2;
        return d.sampleId === hoveredSample ? 3.5 : 1;
      });
  }, [hoveredSample]);

  // Attach hover events via d3 (separate effect)
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    svg.selectAll<SVGPathElement, SampleCurve>('.hit-area')
      .on('mouseover', (_event, d) => handleMouseOver(d.sampleId))
      .on('mouseout', handleMouseOut);

    return () => {
      svg.selectAll('.hit-area').on('mouseover', null).on('mouseout', null);
    };
  }, [sampleCurves, handleMouseOver, handleMouseOut]);

  return (
    <div className="chart-card" style={{ minWidth: 0 }}>
      <div
        style={{
          fontWeight: 600,
          fontSize: '0.95em',
          textAlign: 'center',
          marginBottom: '4px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {drugLabel}
      </div>
      <svg
        data-testid="overlay-chart"
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: 'block' }}
        ref={svgRef}
        onMouseLeave={handleMouseOut}
      >
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
            transform="rotate(-90)"
            x={-innerHeight / 2}
            y={-36}
            textAnchor="middle"
            fontSize="11"
          >
            Viability
          </text>
          <defs>
            <clipPath id={`clip-${clipId}`}>
              <rect x={-6} y={-6} width={innerWidth + 12} height={innerHeight + 6} />
            </clipPath>
          </defs>
          <g clipPath={`url(#clip-${clipId})`}>
            <g className="curves" />
            <g className="hit-areas" />
          </g>
        </g>

        {/* Tooltip label on hover */}
        {hoveredSample && (() => {
          const sc = sampleCurves.find((s) => s.sampleId === hoveredSample);
          return sc ? (
            <text
              x={MARGIN.left + 8}
              y={MARGIN.top + 14}
              fontSize="12"
              fontWeight="600"
              fill={sc.color}
            >
              {sc.sampleId}
            </text>
          ) : null;
        })()}
      </svg>

      {/* Collapsible legend box */}
      <div
        onClick={() => sampleCurves.length > 1 && setLegendExpanded(!legendExpanded)}
        style={{
          border: '1px solid #ddd',
          borderRadius: '4px',
          padding: '4px 8px',
          fontSize: '0.7em',
          cursor: sampleCurves.length > 1 ? 'pointer' : 'default',
          marginTop: '4px',
        }}
      >
        {legendExpanded ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px' }}>
            {sampleCurves.map((sc) => (
              <div
                key={sc.sampleId}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '3px',
                  opacity: hoveredSample && hoveredSample !== sc.sampleId ? 0.3 : 1,
                }}
                onMouseEnter={(e) => { e.stopPropagation(); handleMouseOver(sc.sampleId); }}
                onMouseLeave={handleMouseOut}
              >
                <div style={{ width: 10, height: 3, backgroundColor: sc.color, borderRadius: 1, flexShrink: 0 }} />
                <span style={{ whiteSpace: 'nowrap' }}>{sc.sampleId}</span>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div
              style={{ display: 'flex', alignItems: 'center', gap: '3px' }}
              onMouseEnter={() => handleMouseOver(sampleCurves[0].sampleId)}
              onMouseLeave={handleMouseOut}
            >
              <div style={{ width: 10, height: 3, backgroundColor: sampleCurves[0].color, borderRadius: 1 }} />
              <span style={{ whiteSpace: 'nowrap' }}>{sampleCurves[0].sampleId}</span>
            </div>
            {sampleCurves.length > 1 && (
              <span style={{ color: '#888' }}>+{sampleCurves.length - 1} more</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
