import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import type { TextItem } from 'pdfjs-dist/types/src/display/api'
import { documentFileUrl } from '../lib/api'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl

export interface HighlightItem {
  // [x0, y0, x1, y1] in PDF point units, top-left origin -- PyMuPDF's convention, not
  // native PDF (bottom-left) coordinates. Used as a fallback when the quote can't be
  // located in the page's text layer.
  bbox: [number, number, number, number] | null
  // The model's literal cited quote (eval.gate_schema.Citation.quote). Preferred over
  // bbox: searched against the page's actual text layer so the highlight lands on the
  // specific sentence the answer relied on, not just the paragraph it came from.
  quote: string
}

export interface HighlightTarget {
  // 1-indexed page as pdf.js expects; the API's page_num is 0-indexed (FinanceBench
  // convention, carried through from ingest/chunker.py) so callers must +1 it.
  page: number
  // Multiple citations can land on the same page (the model cites two different
  // sentences from it) -- one chip in the chat merges them into a single click target,
  // and every item's rects get drawn together here.
  items: HighlightItem[]
}

interface Props {
  documentId: number
  target: HighlightTarget | null
}

interface Rect {
  left: number
  top: number
  width: number
  height: number
}

const PAGE_PADDING_PX = 32 // leaves breathing room so text isn't flush against the scrollbar
const HIGHLIGHT_PAD_PX = 2

function normalize(s: string): string {
  return s.replace(/\s+/g, ' ').trim().toLowerCase()
}

// Locate `quote` inside the page's text layer and return one viewport-space rect per
// line it spans. Text items are concatenated with an offset map back to each item so a
// match in the merged string can be traced back to the PDF-space boxes that produced it.
async function findQuoteRects(
  pdfPage: pdfjsLib.PDFPageProxy,
  viewport: pdfjsLib.PageViewport,
  quote: string,
): Promise<Rect[]> {
  const target = normalize(quote)
  if (!target) return []

  const textContent = await pdfPage.getTextContent()
  let concatenated = ''
  const offsets: { start: number; end: number; item: TextItem }[] = []
  for (const raw of textContent.items) {
    const item = raw as TextItem
    if (typeof item.str !== 'string' || !item.str) continue
    const piece = normalize(item.str) + ' '
    const start = concatenated.length
    concatenated += piece
    offsets.push({ start, end: concatenated.length, item })
  }

  let idx = concatenated.indexOf(target)
  let matchLen = target.length
  if (idx === -1) {
    // Model paraphrasing or whitespace differences can break an exact match -- fall
    // back to just the quote's opening words as an anchor.
    const anchor = target.slice(0, Math.min(60, target.length))
    idx = concatenated.indexOf(anchor)
    matchLen = anchor.length
  }
  if (idx === -1) return []

  const end = idx + matchLen
  const matched = offsets.filter((o) => o.end > idx && o.start < end).map((o) => o.item)
  if (matched.length === 0) return []

  // Merge items on the same line (by native baseline y) into a single rect per line.
  const lineKeyOf = (item: TextItem) => Math.round(item.transform[5])
  const lines = new Map<number, Rect>()
  for (const item of matched) {
    const nx0 = item.transform[4]
    const ny0 = item.transform[5]
    const nx1 = nx0 + item.width
    const ny1 = ny0 + item.height
    const [vx0, vy0] = viewport.convertToViewportPoint(nx0, ny0)
    const [vx1, vy1] = viewport.convertToViewportPoint(nx1, ny1)
    const rect: Rect = {
      left: Math.min(vx0, vx1),
      top: Math.min(vy0, vy1),
      width: Math.abs(vx1 - vx0),
      height: Math.abs(vy1 - vy0),
    }
    const key = lineKeyOf(item)
    const existing = lines.get(key)
    if (!existing) {
      lines.set(key, rect)
      continue
    }
    const left = Math.min(existing.left, rect.left)
    const top = Math.min(existing.top, rect.top)
    const right = Math.max(existing.left + existing.width, rect.left + rect.width)
    const bottom = Math.max(existing.top + existing.height, rect.top + rect.height)
    lines.set(key, { left, top, width: right - left, height: bottom - top })
  }
  return [...lines.values()]
}

function bboxRect(
  bbox: [number, number, number, number],
  pdfPage: pdfjsLib.PDFPageProxy,
  viewport: pdfjsLib.PageViewport,
): Rect {
  // PyMuPDF's bbox is top-left-origin (x0,y0,x1,y1 measured down from the page top).
  // pdf.js's viewport transform expects native PDF bottom-left-origin coordinates, so
  // flip y using the page's un-scaled height before converting.
  const [x0, y0, x1, y1] = bbox
  const pageHeight = pdfPage.view[3] - pdfPage.view[1]
  const [vx0, vy0] = viewport.convertToViewportPoint(x0, pageHeight - y1)
  const [vx1, vy1] = viewport.convertToViewportPoint(x1, pageHeight - y0)
  return {
    left: Math.min(vx0, vx1),
    top: Math.min(vy0, vy1),
    width: Math.abs(vx1 - vx0),
    height: Math.abs(vy1 - vy0),
  }
}

export function PdfViewer({ documentId, target }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pdfRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null)
  const [pageNum, setPageNum] = useState(1)
  const [numPages, setNumPages] = useState<number | null>(null)
  const [wrapWidth, setWrapWidth] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [flash, setFlash] = useState(false)
  const [rects, setRects] = useState<Rect[]>([])
  const renderTaskRef = useRef<ReturnType<pdfjsLib.PDFPageProxy['render']> | null>(null)

  useEffect(() => {
    let cancelled = false
    const loadingTask = pdfjsLib.getDocument({
      url: documentFileUrl(documentId),
      withCredentials: true,
    })
    loadingTask.promise
      .then((pdf) => {
        if (cancelled) return
        pdfRef.current = pdf
        setNumPages(pdf.numPages)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'failed to load PDF')
      })
    return () => {
      cancelled = true
      loadingTask.destroy()
    }
  }, [documentId])

  // Track the viewer's available width so the page always renders to fit -- fixes text
  // being clipped at the left/right margins when the container is narrower than a
  // fixed render scale assumed.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width
      if (width) setWrapWidth(width)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (target) setPageNum(target.page)
  }, [target])

  useEffect(() => {
    const pdf = pdfRef.current
    if (!pdf || !numPages || !wrapWidth) return
    const page = Math.min(Math.max(pageNum, 1), numPages)
    let cancelled = false

    pdf.getPage(page).then(async (pdfPage) => {
      if (cancelled) return
      const baseViewport = pdfPage.getViewport({ scale: 1 })
      const scale = Math.max(0.5, (wrapWidth - PAGE_PADDING_PX) / baseViewport.width)
      const viewport = pdfPage.getViewport({ scale })
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = viewport.width
      canvas.height = viewport.height

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      renderTaskRef.current?.cancel()
      const task = pdfPage.render({ canvasContext: ctx, viewport, canvas })
      renderTaskRef.current = task
      try {
        await task.promise
      } catch {
        return // render cancelled (page changed mid-render) -- expected, ignore
      }
      if (cancelled) return

      if (!target || target.page !== page) {
        setRects([])
        setFlash(false)
        return
      }

      // Each item resolves independently (quote match, falling back to its own bbox) and
      // all of them are drawn together -- two citations landing on the same page should
      // show both highlighted passages at once, not just the first one found.
      const allFound: Rect[] = []
      for (const item of target.items) {
        let itemRects: Rect[] = []
        if (item.quote) {
          try {
            itemRects = await findQuoteRects(pdfPage, viewport, item.quote)
          } catch {
            itemRects = []
          }
        }
        if (cancelled) return
        if (itemRects.length === 0 && item.bbox) {
          itemRects = [bboxRect(item.bbox, pdfPage, viewport)]
        }
        allFound.push(...itemRects)
      }
      if (cancelled) return

      if (allFound.length > 0) {
        setRects(allFound)
        setFlash(false)
      } else {
        setRects([])
        setFlash(true)
        setTimeout(() => setFlash(false), 900)
      }
    })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageNum, numPages, wrapWidth, target])

  if (error) {
    return <div className="pdf-viewer pdf-viewer-error">Could not load PDF: {error}</div>
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer-toolbar">
        <button
          className="btn btn-ghost btn-icon"
          aria-label="Previous page"
          disabled={pageNum <= 1}
          onClick={() => setPageNum((p) => p - 1)}
        >
          &larr;
        </button>
        <span className="mono">
          Page {pageNum} {numPages ? `of ${numPages}` : ''}
        </span>
        <button
          className="btn btn-ghost btn-icon"
          aria-label="Next page"
          disabled={!numPages || pageNum >= numPages}
          onClick={() => setPageNum((p) => p + 1)}
        >
          &rarr;
        </button>
      </div>
      <div className="pdf-viewer-canvas-wrap" ref={wrapRef}>
        <div className={`pdf-page-box${flash ? ' pdf-page-flash' : ''}`}>
          <canvas ref={canvasRef} />
          {rects.map((r, i) => (
            <div
              key={i}
              className="pdf-highlight"
              style={{
                left: r.left - HIGHLIGHT_PAD_PX,
                top: r.top - HIGHLIGHT_PAD_PX,
                width: r.width + HIGHLIGHT_PAD_PX * 2,
                height: r.height + HIGHLIGHT_PAD_PX * 2,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
