import { useRef, useState, type FormEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import * as api from '../lib/api'
import { streamQuery, type AnswerEvent, type Citation, type RefusalEvent } from '../lib/queryStream'
import { PdfViewer, type HighlightTarget } from '../components/PdfViewer'

type TurnState =
  | { kind: 'generating'; question: string; charsGenerated: number }
  | { kind: 'answer'; question: string; answer: AnswerEvent }
  | { kind: 'refusal'; question: string; refusal: RefusalEvent }
  | { kind: 'error'; question: string; message: string }

type CitationGroup = { page: number | null; citations: Citation[] }

/** Citations come back in the order the model cited them in the answer text, not page
 * order -- e.g. p.2, p.15, p.3 -- and the same page can be cited more than once for two
 * different sentences. Group by page (sorted, so the chips read left-to-right through the
 * document) rather than rendering one chip per citation, so p.2 shows up once instead of
 * twice; clicking it highlights every citation's passage on that page together. Citations
 * with no page (page: null) each stay their own group -- there's no page to merge on. */
function groupByPage(citations: Citation[]): CitationGroup[] {
  const sorted = [...citations].sort((a, b) => {
    if (a.page === null) return b.page === null ? 0 : 1
    if (b.page === null) return -1
    return a.page - b.page
  })

  const groups: CitationGroup[] = []
  for (const c of sorted) {
    const last = groups[groups.length - 1]
    if (last && last.page !== null && last.page === c.page) {
      last.citations.push(c)
    } else {
      groups.push({ page: c.page, citations: [c] })
    }
  }
  return groups
}

export function ChatPage() {
  const { documentId } = useParams<{ documentId: string }>()
  const docId = Number(documentId)
  const [question, setQuestion] = useState('')
  const [turns, setTurns] = useState<TurnState[]>([])
  const [highlight, setHighlight] = useState<HighlightTarget | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  function onCitationClick(group: CitationGroup) {
    if (group.page === null) return
    // API's page_num is 0-indexed (FinanceBench convention); pdf.js pages are 1-indexed.
    setHighlight({
      page: group.page + 1,
      items: group.citations.map((c) => ({ bbox: c.bbox, quote: c.quote })),
    })
  }

  const documentQuery = useQuery({
    queryKey: ['documents', docId],
    queryFn: () => api.getDocument(docId),
    enabled: Number.isFinite(docId),
  })

  const generating = turns.length > 0 && turns[turns.length - 1].kind === 'generating'

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    const q = question.trim()
    if (!q || generating) return
    setQuestion('')

    const index = turns.length
    setTurns((prev) => [...prev, { kind: 'generating', question: q, charsGenerated: 0 }])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      for await (const evt of streamQuery(docId, q, controller.signal)) {
        if (evt.event === 'delta') {
          setTurns((prev) => {
            const next = [...prev]
            const turn = next[index]
            if (turn.kind === 'generating') next[index] = { ...turn, charsGenerated: evt.data.chars }
            return next
          })
        } else if (evt.event === 'answer') {
          setTurns((prev) => {
            const next = [...prev]
            next[index] = { kind: 'answer', question: q, answer: evt.data }
            return next
          })
        } else if (evt.event === 'refusal') {
          setTurns((prev) => {
            const next = [...prev]
            next[index] = { kind: 'refusal', question: q, refusal: evt.data }
            return next
          })
        } else if (evt.event === 'error') {
          setTurns((prev) => {
            const next = [...prev]
            next[index] = { kind: 'error', question: q, message: evt.data.message }
            return next
          })
        }
      }
    } catch (err) {
      if (controller.signal.aborted) return
      setTurns((prev) => {
        const next = [...prev]
        next[index] = {
          kind: 'error',
          question: q,
          message: err instanceof Error ? err.message : 'request failed',
        }
        return next
      })
    } finally {
      abortRef.current = null
    }
  }

  function onCancel() {
    abortRef.current?.abort()
    setTurns((prev) => {
      const next = [...prev]
      const last = next[next.length - 1]
      if (last?.kind === 'generating') {
        next[next.length - 1] = { kind: 'error', question: last.question, message: 'cancelled' }
      }
      return next
    })
  }

  return (
    <div className="chat-page">
      <header className="page-header">
        <Link to="/documents">&larr; Documents</Link>
        <h1>{documentQuery.data?.display_name ?? documentQuery.data?.doc_name ?? 'Chat'}</h1>
      </header>

      <div className="chat-layout">
        <div className="chat-column">
          <div className="chat-turns">
            {turns.map((turn, i) => (
              <div key={i} className="chat-turn">
                <p className="chat-question">{turn.question}</p>
                {turn.kind === 'generating' && (
                  <p className="chat-generating">
                    Generating<span className="cursor">▍</span>
                  </p>
                )}
                {turn.kind === 'answer' && (
                  <div className="chat-answer">
                    <p>
                      {turn.answer.value !== null
                        ? `${turn.answer.value} ${turn.answer.unit ?? ''} (${turn.answer.scale ?? ''})`
                        : turn.answer.answer_text}
                    </p>
                    <div className="citations">
                      {groupByPage(turn.answer.citations).map((g) => (
                        <button
                          key={g.citations.map((c) => c.chunk_id).join('-')}
                          type="button"
                          className="citation-chip"
                          title={g.citations.map((c) => c.quote).join('\n\n')}
                          onClick={() => onCitationClick(g)}
                        >
                          p.{g.page !== null ? g.page + 1 : '?'}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                {turn.kind === 'refusal' && (
                  <div className="chat-refusal">
                    <p className="refusal-message">{turn.refusal.message}</p>
                    <p className="muted">
                      Searched {turn.refusal.searched.length} excerpts, none sufficient.
                    </p>
                  </div>
                )}
                {turn.kind === 'error' && <p className="form-error">{turn.message}</p>}
              </div>
            ))}
          </div>

          <form onSubmit={onSubmit} className="chat-input-row">
            <input
              className="input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about this document…"
              aria-label="Ask a question about this document"
              disabled={generating}
            />
            {generating ? (
              <button type="button" className="btn btn-secondary" onClick={onCancel}>
                Cancel
              </button>
            ) : (
              <button type="submit" className="btn btn-primary">
                Ask
              </button>
            )}
          </form>
        </div>

        <div className="viewer-column">
          {Number.isFinite(docId) && <PdfViewer documentId={docId} target={highlight} />}
        </div>
      </div>
    </div>
  )
}
