/* Inline SVG icons. Bundled rather than pulled from an icon package -- the set is
   small, and inlining keeps stroke/colour bound to `currentColor` and the type scale. */

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.75,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
}

export function LogoMark() {
  return (
    <svg {...base} strokeWidth={2}>
      <path d="M4 6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v8a2.5 2.5 0 0 1-2.5 2.5H9l-5 3.5z" />
    </svg>
  )
}

export function CheckShield() {
  return (
    <svg {...base}>
      <path d="M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  )
}

export function Crosshair() {
  return (
    <svg {...base}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 2v3M12 19v3M22 12h-3M5 12H2" />
      <circle cx="12" cy="12" r="2.5" />
    </svg>
  )
}

export function Layers() {
  return (
    <svg {...base}>
      <path d="M12 3l9 5-9 5-9-5z" />
      <path d="M3 13l9 5 9-5" />
    </svg>
  )
}

export function Close() {
  return (
    <svg {...base}>
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  )
}

export function Sun() {
  return (
    <svg {...base}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  )
}

export function Moon() {
  return (
    <svg {...base}>
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z" />
    </svg>
  )
}

export function Check() {
  return (
    <svg {...base} strokeWidth={2.25}>
      <path d="M4 12.5l5 5L20 6.5" />
    </svg>
  )
}

/* Ingest pipeline stage icons, one per step in ingest/pipeline.py. */

export function PageScan() {
  return (
    <svg {...base}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </svg>
  )
}

export function TableGrid() {
  return (
    <svg {...base}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 10h18M3 15h18M9 10v10" />
    </svg>
  )
}

export function Vector() {
  return (
    <svg {...base}>
      <circle cx="6" cy="18" r="2.5" />
      <circle cx="18" cy="6" r="2.5" />
      <path d="M8 16.2 16 7.8" />
    </svg>
  )
}

export function Database() {
  return (
    <svg {...base}>
      <ellipse cx="12" cy="6" rx="7.5" ry="3" />
      <path d="M4.5 6v6c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3V6" />
      <path d="M4.5 12v6c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3v-6" />
    </svg>
  )
}
