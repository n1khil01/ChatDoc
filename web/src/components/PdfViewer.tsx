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
// How far outside the actual viewport a page can be before its canvas starts rasterizing --
// generous enough that fast scrolling stays ahead of a blank-page flash, without rendering
// the entire document at once on a large filing.
const RENDER_LOOKAHEAD_PX = 1200

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
  const items = textContent.items.filter(
    (raw): raw is TextItem => typeof (raw as TextItem).str === 'string' && !!(raw as TextItem).str,
  )
  // getTextContent() returns items in the PDF's internal content-stream order, which for
  // box/textframe-heavy exports (DOCX -> PDF in particular) is often not visual reading
  // order -- a title and the subtitle directly below it can be emitted as unrelated shapes
  // far apart in the stream. Re-sort top-to-bottom, then left-to-right within a line,
  // before concatenating, so the search string matches what a reader actually sees instead
  // of the document's internal object order. PDF space has a bottom-left origin, so a
  // larger transform[5] (y) is higher on the page -- descending y is top-to-bottom.
  items.sort((a, b) => {
    const dy = b.transform[5] - a.transform[5]
    if (Math.abs(dy) > 2) return dy
    return a.transform[4] - b.transform[4]
  })

  let concatenated = ''
  const offsets: { start: number; end: number; item: TextItem }[] = []
  let prev: TextItem | null = null
  for (const item of items) {
    // Some PDF generators (DOCX exports of styled headings in particular, via per-glyph
    // kerning/tracking) emit multiple text items for what's visually one word. Always
    // inserting a space between items -- the previous approach -- splits words like
    // "Midnight" into "Mid night", which never matches the model's clean quote text and
    // silently falls back to the whole chunk's bbox. Only insert a space for an actual
    // word/line boundary: a new line, or a real horizontal gap between items relative to
    // the previous item's own character width (adjacent glyph fragments of one word sit
    // flush or slightly overlapping, not gapped).
    let sep = ''
    if (prev) {
      const sameLine = Math.abs(item.transform[5] - prev.transform[5]) <= 2
      if (!sameLine) {
        sep = ' '
      } else {
        const prevEndX = prev.transform[4] + prev.width
        const gap = item.transform[4] - prevEndX
        const avgCharWidth = prev.width / Math.max(prev.str.length, 1)
        if (gap > avgCharWidth * 0.35) sep = ' '
      }
    }
    const piece = sep + normalize(item.str)
    const start = concatenated.length
    concatenated += piece
    offsets.push({ start, end: concatenated.length, item })
    prev = item
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
    // item.transform is a full affine matrix (translation, scale, and potentially skew/
    // rotation for styled runs like DOCX-exported headings), not just an [x, y] origin --
    // treating transform[4]/[5] as a plain coordinate and adding width/height to it (the
    // previous approach) only produces a correct box when that matrix has zero skew, which
    // isn't guaranteed. Transform all four corners of the glyph's local box (0,0)-(width,
    // height) through the item's own matrix, then through the viewport's, and take the
    // axis-aligned bounds of the results -- correct regardless of any scale/skew in transform.
    const corners: [number, number][] = [
      [0, 0],
      [item.width, 0],
      [0, item.height],
      [item.width, item.height],
    ].map(([lx, ly]) => {
      const [px, py] = pdfjsLib.Util.applyTransform([lx, ly], item.transform)
      return viewport.convertToViewportPoint(px, py)
    })
    const xs = corners.map((c) => c[0])
    const ys = corners.map((c) => c[1])
    const rect: Rect = {
      left: Math.min(...xs),
      top: Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
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
  // The single scrollable element: spans the full viewer width (same edge the toolbar
  // above spans to), so its native scrollbar sits at that fixed outer edge. Pages render
  // at their own natural scaled width and are centered inside it -- any leftover space
  // between a page's edge and the true container edge is just background, not something
  // trimmed away.
  const wrapRef = useRef<HTMLDivElement>(null)
  const pageInputRef = useRef<HTMLInputElement>(null)
  const pdfRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null)
  const [numPages, setNumPages] = useState<number | null>(null)
  const [wrapWidth, setWrapWidth] = useState(0)
  const [error, setError] = useState<string | null>(null)
  // The "current" page, for the toolbar counter and the page-number input -- driven by
  // whichever page is most visible in the scroll container (see the visibility observer
  // below), not by which single page is rendered, since every page is on screen at once now.
  const [pageNum, setPageNum] = useState(1)
  // Free-typing buffer for the page-number field -- kept separate from `pageNum` so a
  // half-typed value ("1" while aiming for "15") doesn't get clobbered by the effect
  // below on every keystroke. Only commits (and clamps) on Enter or blur.
  const [pageInput, setPageInput] = useState('1')
  // Each page's untransformed (scale: 1) PDF-point size, fetched once per document so every
  // page wrapper can be laid out at its correct height immediately -- without this, pages
  // would all be zero-height until their canvas rendered, collapsing the scrollbar and
  // breaking scrollIntoView navigation to anything not yet rendered.
  const [pageDims, setPageDims] = useState<Record<number, { baseWidth: number; baseHeight: number }>>({})
  // The canvas's actual rasterized pixel size, filled in once renderPage() finishes for a
  // page. The page box's displayed size prefers this over its own scale*baseWidth guess --
  // that guess and the canvas's real viewport can drift apart (viewportCacheRef persists a
  // page's viewport across wrapWidth changes until its own re-render effect catches up),
  // which used to leave a sliver of the box's white background exposed past the actual
  // painted page, looking like unexplained padding around the document.
  const [canvasSizes, setCanvasSizes] = useState<Record<number, { width: number; height: number }>>({})

  const [highlightPage, setHighlightPage] = useState<number | null>(null)
  const [highlightRects, setHighlightRects] = useState<Rect[]>([])
  const [flashPage, setFlashPage] = useState<number | null>(null)

  const pageElRef = useRef(new Map<number, HTMLDivElement>())
  const canvasElRef = useRef(new Map<number, HTMLCanvasElement>())
  const pdfPageCacheRef = useRef(new Map<number, pdfjsLib.PDFPageProxy>())
  const viewportCacheRef = useRef(new Map<number, pdfjsLib.PageViewport>())
  const renderedSetRef = useRef(new Set<number>())
  const renderTaskRef = useRef(new Map<number, ReturnType<pdfjsLib.PDFPageProxy['render']>>())
  const renderObserverRef = useRef<IntersectionObserver | null>(null)
  const visibilityObserverRef = useRef<IntersectionObserver | null>(null)
  const visibleRatiosRef = useRef(new Map<number, number>())

  async function getPdfPage(n: number): Promise<pdfjsLib.PDFPageProxy | null> {
    const cached = pdfPageCacheRef.current.get(n)
    if (cached) return cached
    const pdf = pdfRef.current
    if (!pdf) return null
    const page = await pdf.getPage(n)
    pdfPageCacheRef.current.set(n, page)
    return page
  }

  function getScaledViewport(pdfPage: pdfjsLib.PDFPageProxy, n: number): pdfjsLib.PageViewport {
    const cached = viewportCacheRef.current.get(n)
    if (cached) return cached
    const base = pdfPage.getViewport({ scale: 1 })
    const scale = wrapWidth ? Math.max(0.5, (wrapWidth - PAGE_PADDING_PX) / base.width) : 1
    const viewport = pdfPage.getViewport({ scale })
    viewportCacheRef.current.set(n, viewport)
    return viewport
  }

  async function renderPage(n: number): Promise<void> {
    if (renderedSetRef.current.has(n)) return
    const pdfPage = await getPdfPage(n)
    if (!pdfPage) return
    const canvas = canvasElRef.current.get(n)
    if (!canvas) return
    const viewport = getScaledViewport(pdfPage, n)
    // Rounded the same way the placeholder box's inline style rounds its width/height below,
    // so the canvas's actual bitmap resolution and its display box always agree exactly --
    // any mismatch forces the browser to stretch the bitmap, which blurs it.
    canvas.width = Math.round(viewport.width)
    canvas.height = Math.round(viewport.height)
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    renderTaskRef.current.get(n)?.cancel()
    const task = pdfPage.render({ canvasContext: ctx, viewport, canvas })
    renderTaskRef.current.set(n, task)
    try {
      await task.promise
      renderedSetRef.current.add(n)
      setCanvasSizes((prev) => ({ ...prev, [n]: { width: canvas.width, height: canvas.height } }))
    } catch {
      // cancelled -- expected when a resize or fast scroll retriggers before this finishes
    }
  }

  function scrollToPage(n: number) {
    pageElRef.current.get(n)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setPageNum(n) // optimistic; the visibility observer confirms it once the scroll settles
  }

  // Load the document and fetch every page's intrinsic size up front (cheap metadata, not a
  // render) so the scrollable stack has correct height from the start.
  useEffect(() => {
    let cancelled = false
    pdfPageCacheRef.current.clear()
    viewportCacheRef.current.clear()
    renderedSetRef.current.clear()
    pageElRef.current.clear()
    canvasElRef.current.clear()
    visibleRatiosRef.current.clear()
    setNumPages(null)
    setPageDims({})
    setCanvasSizes({})
    setPageNum(1)
    setPageInput('1')
    setHighlightPage(null)
    setHighlightRects([])
    setFlashPage(null)

    const loadingTask = pdfjsLib.getDocument({
      url: documentFileUrl(documentId),
      withCredentials: true,
    })
    loadingTask.promise
      .then((pdf) => {
        if (cancelled) return
        pdfRef.current = pdf
        setNumPages(pdf.numPages)
        for (let n = 1; n <= pdf.numPages; n++) {
          pdf
            .getPage(n)
            .then((page) => {
              if (cancelled) return
              pdfPageCacheRef.current.set(n, page)
              const base = page.getViewport({ scale: 1 })
              setPageDims((prev) => ({ ...prev, [n]: { baseWidth: base.width, baseHeight: base.height } }))
            })
            .catch(() => {})
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'failed to load PDF')
      })
    return () => {
      cancelled = true
      loadingTask.destroy()
    }
  }, [documentId])

  // Track the viewer's available width so pages always render to fit -- fixes text being
  // clipped at the left/right margins when the container is narrower than assumed.
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

  // A resize invalidates every cached scale -- re-render whatever was already on screen at
  // the new size rather than leaving stale, wrongly-scaled canvases up.
  useEffect(() => {
    if (!wrapWidth) return
    viewportCacheRef.current.clear()
    const toRerender = [...renderedSetRef.current]
    renderedSetRef.current.clear()
    // Drop the stale rendered sizes too -- otherwise the box briefly displays at the old
    // scale (sized from a canvas that no longer reflects the current wrapWidth) until each
    // page's re-render finishes and overwrites its entry.
    setCanvasSizes({})
    toRerender.forEach((n) => {
      void renderPage(n)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wrapWidth])

  // Two observers, deliberately separate: one triggers rasterizing a page well before it's
  // actually visible (rootMargin lookahead, so scrolling doesn't outrun rendering); the other
  // tracks which page is *actually* on screen right now (no margin) to drive the counter --
  // conflating them would make the counter jump early, while a page was merely approaching.
  useEffect(() => {
    const root = wrapRef.current
    if (!root) return

    const renderObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          const n = Number((entry.target as HTMLElement).dataset.page)
          if (n) void renderPage(n)
        }
      },
      { root, rootMargin: `${RENDER_LOOKAHEAD_PX}px 0px`, threshold: 0 },
    )

    const visibilityObserver = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const n = Number((entry.target as HTMLElement).dataset.page)
          if (!n) continue
          visibleRatiosRef.current.set(n, entry.isIntersecting ? entry.intersectionRatio : 0)
        }
        let best = 0
        let bestRatio = 0
        for (const [n, ratio] of visibleRatiosRef.current) {
          if (ratio > bestRatio) {
            bestRatio = ratio
            best = n
          }
        }
        if (best) setPageNum(best)
      },
      { root, threshold: [0, 0.25, 0.5, 0.75, 1] },
    )

    renderObserverRef.current = renderObserver
    visibilityObserverRef.current = visibilityObserver
    // Pages that mounted before these observers existed (the very first ones, on initial
    // load) still need to be attached.
    pageElRef.current.forEach((el) => {
      renderObserver.observe(el)
      visibilityObserver.observe(el)
    })

    return () => {
      renderObserver.disconnect()
      visibilityObserver.disconnect()
      renderObserverRef.current = null
      visibilityObserverRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId])

  // Keep the input's displayed value in sync whenever the page changes from elsewhere
  // (arrows, scrolling, a citation click) -- but not while the field itself has focus, so
  // committing a typed page number doesn't get immediately overwritten by this same effect.
  useEffect(() => {
    if (document.activeElement !== pageInputRef.current) setPageInput(String(pageNum))
  }, [pageNum])

  function commitPageInput() {
    const parsed = Number(pageInput)
    if (Number.isInteger(parsed) && parsed >= 1 && parsed <= (numPages ?? parsed)) {
      scrollToPage(parsed)
    } else {
      setPageInput(String(pageNum)) // invalid entry -- snap back rather than leave it stuck
    }
  }

  // A citation click: force that page to render even if it's off-screen and hasn't reached
  // the render observer yet, scroll it into view, then resolve every cited item's rects.
  useEffect(() => {
    if (!target) {
      setHighlightPage(null)
      setHighlightRects([])
      setFlashPage(null)
      return
    }
    let cancelled = false
    ;(async () => {
      const pdfPage = await getPdfPage(target.page)
      if (cancelled || !pdfPage) return
      // Compute the viewport directly rather than trusting renderPage's internal cache
      // population -- renderPage can return early (canvas ref not yet attached, a race
      // with the scroll-driven render observer already rendering this page) without ever
      // calling getScaledViewport, which silently left this effect with no viewport and
      // no fallback: it just returned, showing neither a highlight nor the flash cue.
      const viewport = getScaledViewport(pdfPage, target.page)
      void renderPage(target.page)
      scrollToPage(target.page)

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
        setHighlightPage(target.page)
        setHighlightRects(allFound)
        setFlashPage(null)
      } else {
        setHighlightPage(null)
        setHighlightRects([])
        setFlashPage(target.page)
        setTimeout(() => setFlashPage(null), 900)
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target])

  if (error) {
    return <div className="pdf-viewer pdf-viewer-error">Could not load PDF: {error}</div>
  }

  return (
    <div className="pdf-viewer">
      <div className="pdf-viewer-toolbar">
        <button
          className="btn btn-ghost btn-icon"
          aria-label="Previous page"
          disabled={!numPages}
          onClick={() => scrollToPage(pageNum <= 1 ? (numPages ?? pageNum) : pageNum - 1)}
        >
          &larr;
        </button>
        <span className="mono pdf-page-field">
          Page{' '}
          <input
            ref={pageInputRef}
            className="pdf-page-input"
            type="text"
            inputMode="numeric"
            aria-label="Page number"
            // The fixed 2.5ch CSS width clips a 2-digit number the moment a document has
            // more than 9 pages -- size to whichever is wider, what's actually typed or the
            // total page count, so the box never has to cut off a digit. Paired with
            // box-sizing: content-box on .pdf-page-input so this ch value sizes the digits
            // themselves, with the input's own padding/border added on top rather than
            // eating into it.
            style={{ width: `${Math.max(pageInput.length, String(numPages ?? 1).length)}ch` }}
            value={pageInput}
            onChange={(e) => setPageInput(e.target.value.replace(/[^0-9]/g, ''))}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                commitPageInput()
                pageInputRef.current?.blur()
              } else if (e.key === 'Escape') {
                setPageInput(String(pageNum))
                pageInputRef.current?.blur()
              }
            }}
            onBlur={commitPageInput}
          />{' '}
          {numPages ? `of ${numPages}` : ''}
        </span>
        <button
          className="btn btn-ghost btn-icon"
          aria-label="Next page"
          disabled={!numPages}
          onClick={() => scrollToPage(numPages && pageNum >= numPages ? 1 : pageNum + 1)}
        >
          &rarr;
        </button>
      </div>
      <div className="pdf-viewer-canvas-wrap" ref={wrapRef}>
        {Array.from({ length: numPages ?? 0 }, (_, i) => i + 1).map((n) => {
          const dims = pageDims[n]
          const scale = wrapWidth && dims ? Math.max(0.5, (wrapWidth - PAGE_PADDING_PX) / dims.baseWidth) : 1
          // Before this page's own dims have arrived, fall back to the container width as a
          // best-guess placeholder height so the scrollbar doesn't jitter as pages resolve.
          // Rounded the same way renderPage rounds the canvas's actual bitmap size, so once
          // rendered the box and the canvas agree on pixel dimensions exactly -- any mismatch
          // there forces a fractional CSS stretch on the canvas, which blurs it.
          const guessedWidth = dims ? Math.round(dims.baseWidth * scale) : Math.max(wrapWidth - PAGE_PADDING_PX, 300)
          const guessedHeight = dims ? Math.round(dims.baseHeight * scale) : Math.round(guessedWidth * 1.3)
          // Once the canvas has actually rendered, trust its real pixel size over the guess --
          // they can drift apart (see canvasSizes above), and a box bigger than its canvas
          // exposes a strip of the box's own white background past the document's edge.
          const rendered = canvasSizes[n]
          const width = rendered?.width ?? guessedWidth
          const height = rendered?.height ?? guessedHeight

          return (
            <div
              key={n}
              data-page={n}
              ref={(el) => {
                if (el) {
                  pageElRef.current.set(n, el)
                  renderObserverRef.current?.observe(el)
                  visibilityObserverRef.current?.observe(el)
                } else {
                  pageElRef.current.delete(n)
                }
              }}
              className={`pdf-page-box${flashPage === n ? ' pdf-page-flash' : ''}`}
              style={{ width, height }}
            >
              <canvas
                ref={(el) => {
                  if (el) canvasElRef.current.set(n, el)
                  else canvasElRef.current.delete(n)
                }}
              />
              {highlightPage === n &&
                highlightRects.map((r, i) => (
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
          )
        })}
      </div>
    </div>
  )
}
