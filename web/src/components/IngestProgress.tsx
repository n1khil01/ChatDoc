import { useEffect, useRef, useState } from 'react'
import type { ReactElement } from 'react'
import type { Document } from '../lib/api'
import { Check, Database, PageScan, TableGrid, Vector } from './icons'

/* Live ingest progress for a document still being processed.
 *
 * Every number here is real: the API's `stage` / `stage_current` / `stage_total` columns are
 * written by the ingest worker as ingest/pipeline.py advances (pages chunked, chunks embedded),
 * so this is a readout of actual work, not a timed fake.
 *
 * Smoothing between the client's 700ms polls is left to a CSS width transition on the bar
 * rather than a JS animation loop -- requestAnimationFrame is paused in background tabs and
 * in non-compositing views, which would freeze the bar while the counters kept moving.
 */

type StageId = 'reading' | 'chunking' | 'embedding' | 'indexing'

type StageSpec = {
  id: StageId
  label: string
  icon: () => ReactElement
  /* Share of the overall bar. Weighted by observed cost: embedding dominates on a long
     filing, reading a page count is near-instant. */
  weight: number
  /* Unit noun for the "x of y" readout. */
  unit: string
  pendingHint: string
}

const STAGES: StageSpec[] = [
  { id: 'reading', label: 'Reading the PDF', icon: PageScan, weight: 0.04, unit: 'pages', pendingHint: 'Opening the file' },
  { id: 'chunking', label: 'Finding tables and text', icon: TableGrid, weight: 0.36, unit: 'pages', pendingHint: 'Waiting on page count' },
  { id: 'embedding', label: 'Embedding chunks', icon: Vector, weight: 0.5, unit: 'chunks', pendingHint: 'Waiting on chunks' },
  { id: 'indexing', label: 'Building the search index', icon: Database, weight: 0.1, unit: 'chunks', pendingHint: 'Waiting on embeddings' },
]

const ORDER: Record<string, number> = {
  queued: -1,
  reading: 0,
  chunking: 1,
  embedding: 2,
  indexing: 3,
  done: 4,
}

/** Weighted completion across all stages, 0..1. Finished stages contribute their full
 *  weight; the active stage contributes its own fraction. */
function targetFraction(stage: string | null, current: number, total: number): number {
  const active = ORDER[stage ?? 'queued'] ?? -1
  if (active < 0) return 0
  if (active >= STAGES.length) return 1

  let acc = 0
  for (let i = 0; i < STAGES.length; i++) {
    if (i < active) acc += STAGES[i].weight
    else if (i === active) acc += STAGES[i].weight * (total > 0 ? Math.min(current / total, 1) : 0)
  }
  return Math.min(acc, 1)
}

/** Monotonic wrapper around targetFraction: a stage boundary can briefly report a smaller
 *  fraction (new stage, counter back to 0), and a progress bar that slides backwards looks
 *  broken. Clamping to the high-water mark keeps it honest in the only direction that matters. */
function useMonotonic(fraction: number): number {
  const highWater = useRef(fraction)
  highWater.current = Math.max(highWater.current, fraction)
  return highWater.current
}

/** Seconds since the row first appeared as processing, ticking once a second. */
function useElapsed(): number {
  const startRef = useRef(Date.now())
  const [, force] = useState(0)
  useEffect(() => {
    const iv = setInterval(() => force((n) => n + 1), 1000)
    return () => clearInterval(iv)
  }, [])
  return Math.floor((Date.now() - startRef.current) / 1000)
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`
}

export function IngestProgress({ doc }: { doc: Document }) {
  const activeIndex = ORDER[doc.stage ?? 'queued'] ?? -1
  const fraction = useMonotonic(targetFraction(doc.stage, doc.stage_current, doc.stage_total))
  const elapsed = useElapsed()
  const percent = Math.round(fraction * 100)

  return (
    <div className="ingest">
      <div className="ingest-head">
        <div className="ingest-heading">
          <span className="ingest-spinner" aria-hidden="true" />
          <span className="ingest-title">
            {activeIndex < 0 ? 'Queued' : STAGES[Math.min(activeIndex, STAGES.length - 1)].label}
          </span>
        </div>
        <div className="ingest-meta">
          <span className="ingest-percent">{percent}%</span>
          <span className="ingest-elapsed">{formatElapsed(elapsed)}</span>
        </div>
      </div>

      <div
        className="ingest-bar"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Document processing progress"
      >
        <div className="ingest-bar-fill" style={{ width: `${Math.max(fraction * 100, 2)}%` }} />
      </div>

      <ol className="ingest-stages">
        {STAGES.map((stage, i) => {
          const state = i < activeIndex ? 'done' : i === activeIndex ? 'active' : 'pending'
          const Icon = stage.icon
          const showCount = state === 'active' && doc.stage_total > 0
          return (
            <li key={stage.id} className={`ingest-stage ingest-stage-${state}`}>
              <span className="ingest-stage-icon">{state === 'done' ? <Check /> : <Icon />}</span>
              <span className="ingest-stage-label">{stage.label}</span>
              <span className="ingest-stage-detail">
                {state === 'done'
                  ? 'done'
                  : showCount
                    ? `${doc.stage_current.toLocaleString()} / ${doc.stage_total.toLocaleString()} ${stage.unit}`
                    : state === 'active'
                      ? 'working…'
                      : stage.pendingHint}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
