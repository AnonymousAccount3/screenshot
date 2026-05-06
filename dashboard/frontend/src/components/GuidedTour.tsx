import { useState } from 'react';

const STEPS = [
  "Upload a drug screening dataset (CSV, Feather, or Excel)",
  "Review detected columns and adjust if needed",
  "Generate monotherapy dose-response queries",
  "Run inference — predictions stream in real-time",
  "Overlay view: all samples on one curve per drug",
  "Switch to Per-Sample view to explore individual responses",
  "Click any chart to zoom into a dose-response curve",
  "Scroll down to the analytics section for hit detection",
  "Click a heatmap cell to jump to that drug's dose-response",
  "Predict a combination response surface for any drug pair",
];

export default function GuidedTour() {
  const [active, setActive] = useState(false);
  const [step, setStep] = useState(0);

  if (!active) {
    return (
      <button
        onClick={() => { setActive(true); setStep(0); }}
        style={{
          position: 'fixed',
          bottom: '16px',
          right: '16px',
          zIndex: 9999,
          padding: '8px 16px',
          borderRadius: '20px',
          border: 'none',
          backgroundColor: 'var(--color-brand-500)',
          color: 'white',
          cursor: 'pointer',
          fontSize: 'var(--text-sm)',
          fontWeight: 'var(--font-weight-semibold)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
        }}
      >
        Start Tour
      </button>
    );
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: 0,
      left: 0,
      right: 0,
      zIndex: 9999,
      backgroundColor: 'rgba(15, 23, 42, 0.92)',
      color: 'white',
      padding: '14px 24px',
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      backdropFilter: 'blur(8px)',
    }}>
      <span style={{
        fontSize: '13px',
        color: 'rgba(255,255,255,0.5)',
        fontVariantNumeric: 'tabular-nums',
        minWidth: '40px',
      }}>
        {step + 1}/{STEPS.length}
      </span>
      <span style={{ flex: 1, fontSize: '16px', fontWeight: 500 }}>
        {STEPS[step]}
      </span>
      <div style={{ display: 'flex', gap: '8px' }}>
        {step > 0 && (
          <button
            onClick={() => setStep(s => s - 1)}
            style={{
              padding: '6px 14px',
              borderRadius: '6px',
              border: '1px solid rgba(255,255,255,0.2)',
              backgroundColor: 'transparent',
              color: 'white',
              cursor: 'pointer',
              fontSize: '13px',
            }}
          >
            Back
          </button>
        )}
        {step < STEPS.length - 1 ? (
          <button
            onClick={() => setStep(s => s + 1)}
            style={{
              padding: '6px 14px',
              borderRadius: '6px',
              border: 'none',
              backgroundColor: 'var(--color-brand-400)',
              color: 'white',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
            }}
          >
            Next
          </button>
        ) : (
          <button
            onClick={() => setActive(false)}
            style={{
              padding: '6px 14px',
              borderRadius: '6px',
              border: 'none',
              backgroundColor: 'var(--color-brand-400)',
              color: 'white',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
            }}
          >
            Done
          </button>
        )}
        <button
          onClick={() => setActive(false)}
          style={{
            padding: '6px 10px',
            borderRadius: '6px',
            border: 'none',
            backgroundColor: 'transparent',
            color: 'rgba(255,255,255,0.4)',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          ×
        </button>
      </div>
    </div>
  );
}
