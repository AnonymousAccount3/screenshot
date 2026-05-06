
import { useEffect, useId, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import type { Prediction } from '../types/dose-response';
import { getLogDose } from '../utils/dataTransform';
import { createDoseResponseLine } from '../utils/curveInterpolation';

const CONTROL_COLOR = '#1565c0';
const NON_CONTROL_COLOR = '#c62828';

interface GroupCurve {
  sampleId: string;
  predictions: Prediction[];
  isControl: boolean;
}

interface GroupCompareChartProps {
  drugLabel: string;
  curves: GroupCurve[];
  width?: number;
  height?: number;
  doseRange?: [number, number];
}

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

function computeMedianCurve(
  curves: GroupCurve[],
): [number, number][] {
  if (curves.length === 0) return [];

  // Collect all unique doses
  const doseSet = new Set<number>();
  for (const c of curves) {
    for (const p of c.predictions) doseSet.add(p.dose1);
  }
  const doses = Array.from(doseSet).sort((a, b) => a - b);

  return doses.map((dose) => {
    const viabilities: number[] = [];
    for (const c of curves) {
      const match = c.predictions.find((p) => p.dose1 === dose);
      if (match) viabilities.push(Math.max(0, Math.min(1, match.predicted_viability)));
    }
    viabilities.sort((a, b) => a - b);
    const mid = Math.floor(viabilities.length / 2);
    const median = viabilities.length % 2 === 0
      ? (viabilities[mid - 1] + viabilities[mid]) / 2
      : viabilities[mid];
    return [getLogDose(dose), median] as [number, number];
  });
}

export default function GroupCompareChart({
  drugLabel,
  curves,
  width = 220,
  height = 170,
  doseRange,
}: GroupCompareChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId().replace(/:/g, '_');
  const [hoveredSample, setHoveredSample] = useState<string | null>(null);

  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  const handleMouseOver = useCallback((sampleId: string) => {
    setHoveredSample(sampleId);
  }, []);
  const handleMouseOut = useCallback(() => {
    setHoveredSample(null);
  }, []);

  const controlCurves = curves.filter((c) => c.isControl);
  const nonControlCurves = curves.filter((c) => !c.isControl);

  useEffect(() => {
    if (!svgRef.current || curves.length === 0) return;

    const svg = d3.select(svgRef.current);
    const g = svg.select<SVGGElement>('.chart-content');

    // Dose domain
    const allLogDoses: number[] = [];
    curves.forEach((c) =>
      c.predictions.forEach((p) => allLogDoses.push(getLogDose(p.dose1))),
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

    const xScale = d3.scaleLinear()
      .domain([domainMin, domainMax])
      .range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain([0, 1.05]).range([innerHeight, 0]);

    // Axes
    g.select<SVGGElement>('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(4));
    g.select<SVGGElement>('.y-axis').call(d3.axisLeft(yScale).ticks(4));

    const lineGen = createDoseResponseLine(xScale, yScale);

    function curveData(c: GroupCurve): [number, number][] {
      const sorted = [...c.predictions].sort((a, b) => a.dose1 - b.dose1);
      return sorted.map((p) => [
        getLogDose(p.dose1),
        Math.max(0, Math.min(1, p.predicted_viability)),
      ] as [number, number]);
    }

    // Individual curves
    const membersGroup = g.select<SVGGElement>('.members');
    membersGroup.selectAll('*').remove();

    for (const c of curves) {
      const color = c.isControl ? CONTROL_COLOR : NON_CONTROL_COLOR;
      membersGroup.append('path')
        .datum(c)
        .attr('class', 'member-curve')
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', 1.2)
        .attr('opacity', 0.25)
        .attr('d', lineGen(curveData(c)) || '');
    }

    // Median curves
    const mediansGroup = g.select<SVGGElement>('.medians');
    mediansGroup.selectAll('*').remove();

    const controlMedian = computeMedianCurve(controlCurves);
    const nonControlMedian = computeMedianCurve(nonControlCurves);

    if (controlMedian.length > 0) {
      mediansGroup.append('path')
        .attr('class', 'median-control')
        .attr('fill', 'none')
        .attr('stroke', CONTROL_COLOR)
        .attr('stroke-width', 3)
        .attr('opacity', 1)
        .attr('d', lineGen(controlMedian) || '');
    }
    if (nonControlMedian.length > 0) {
      mediansGroup.append('path')
        .attr('class', 'median-noncontrol')
        .attr('fill', 'none')
        .attr('stroke', NON_CONTROL_COLOR)
        .attr('stroke-width', 3)
        .attr('opacity', 1)
        .attr('d', lineGen(nonControlMedian) || '');
    }

    // Hit areas for hovering individual curves
    const hitGroup = g.select<SVGGElement>('.hit-areas');
    hitGroup.selectAll('*').remove();

    for (const c of curves) {
      hitGroup.append('path')
        .datum(c)
        .attr('class', 'hit-area')
        .attr('fill', 'none')
        .attr('stroke', 'transparent')
        .attr('stroke-width', 10)
        .attr('d', lineGen(curveData(c)) || '')
        .style('cursor', 'pointer')
        .on('mouseover', () => handleMouseOver(c.sampleId))
        .on('mouseout', handleMouseOut);
    }
  }, [curves, controlCurves, nonControlCurves, innerWidth, innerHeight, doseRange, handleMouseOver, handleMouseOut]);

  // Update opacity on hover
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);

    svg.selectAll<SVGPathElement, GroupCurve>('.member-curve')
      .attr('opacity', (d) => {
        if (!hoveredSample) return 0.25;
        return d.sampleId === hoveredSample ? 0.9 : 0.08;
      })
      .attr('stroke-width', (d) => {
        if (!hoveredSample) return 1.2;
        return d.sampleId === hoveredSample ? 2.5 : 1;
      });

    // Dim medians when hovering a specific sample
    svg.selectAll('.median-control, .median-noncontrol')
      .attr('opacity', hoveredSample ? 0.3 : 1);
  }, [hoveredSample]);

  return (
    <div className="chart-card" style={{ minWidth: 0 }}>
      <div style={{ fontWeight: 600, fontSize: '0.85em', textAlign: 'center', marginBottom: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {drugLabel}
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: 'block' }}
        ref={svgRef}
        onMouseLeave={handleMouseOut}
      >
        <g className="chart-content" transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          <g className="x-axis" />
          <g className="y-axis" />
          <text x={innerWidth / 2} y={innerHeight + 32} textAnchor="middle" fontSize="10">
            Log₁₀ Dose (µM)
          </text>
          <text transform="rotate(-90)" x={-innerHeight / 2} y={-36} textAnchor="middle" fontSize="10">
            Viability
          </text>
          <defs>
            <clipPath id={`clip-${clipId}`}>
              <rect x={-6} y={-6} width={innerWidth + 12} height={innerHeight + 6} />
            </clipPath>
          </defs>
          <g clipPath={`url(#clip-${clipId})`}>
            <g className="members" />
            <g className="medians" />
            <g className="hit-areas" />
          </g>
        </g>

        {/* Tooltip on hover */}
        {hoveredSample && (() => {
          const c = curves.find((x) => x.sampleId === hoveredSample);
          if (!c) return null;
          return (
            <text
              x={MARGIN.left + 4}
              y={MARGIN.top + 12}
              fontSize="10"
              fontWeight="600"
              fill={c.isControl ? CONTROL_COLOR : NON_CONTROL_COLOR}
            >
              {c.sampleId} ({c.isControl ? 'control' : 'non-control'})
            </text>
          );
        })()}
      </svg>
    </div>
  );
}
