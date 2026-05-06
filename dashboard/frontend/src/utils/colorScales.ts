
import * as d3 from 'd3';

export type ColorOption = 'drug' | 'viability';

export interface EmbeddingPoint2D {
  x: number;
  y: number;
  sampleId: string;
  drug1: string;
  dose1: number;
  predictedViability?: number;
  avgViability?: number;
  rawIndex: number;
  group?: string;
}

export function createColorScale(
  option: ColorOption,
  data: EmbeddingPoint2D[],
): (point: EmbeddingPoint2D) => string {
  switch (option) {
    case 'drug': {
      const drugs = [...new Set(data.map((d) => d.drug1))];
      const scale = d3.scaleOrdinal(d3.schemeCategory10).domain(drugs);
      return (p) => scale(p.drug1);
    }
    case 'viability': {
      const scale = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, 1]);
      return (p) => scale(p.avgViability ?? p.predictedViability ?? 0.5);
    }
    default:
      return () => '#4a90d9';
  }
}
