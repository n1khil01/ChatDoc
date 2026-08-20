// Manual SSE parsing over fetch's ReadableStream, because EventSource can't send a POST
// body, custom headers (CSRF), or credentials the way this API needs (api/routes_query.py).
//
// The event payload types below are hand-maintained, not generated -- /query returns a
// StreamingResponse with no response_model, so FastAPI's OpenAPI schema (api-types.ts)
// has no way to describe SSE event shapes. Keep these in sync with the `_sse(...)` calls
// in api/routes_query.py by hand.

import { API_URL } from './api'

function readCookie(name: string): string | null {
  const match = document.cookie.match(new RegExp(`(?:^|; )${name}=([^;]*)`))
  return match ? decodeURIComponent(match[1]) : null
}

export interface RetrievalEvent {
  searched: { chunk_id: number; page: number }[]
}

export interface DeltaEvent {
  chars: number
}

export interface Citation {
  chunk_id: number
  page: number | null
  quote: string
  // [x0, y0, x1, y1] in PDF point units, top-left origin (PyMuPDF convention) --
  // null for prose chunks, which don't carry a table bbox.
  bbox: [number, number, number, number] | null
}

export interface AnswerEvent {
  value: number | null
  unit: string | null
  scale: string | null
  answer_text: string | null
  citations: Citation[]
}

export interface RefusalEvent {
  message: string
  reason: string | null
  searched: { chunk_id: number; page: number }[]
}

export interface ErrorEvent {
  message: string
}

export type QueryEvent =
  | { event: 'retrieval'; data: RetrievalEvent }
  | { event: 'delta'; data: DeltaEvent }
  | { event: 'answer'; data: AnswerEvent }
  | { event: 'refusal'; data: RefusalEvent }
  | { event: 'error'; data: ErrorEvent }
  | { event: 'done'; data: Record<string, never> }

export async function* streamQuery(
  documentId: number,
  question: string,
  signal: AbortSignal,
): AsyncGenerator<QueryEvent> {
  const csrf = readCookie('chatdoc_csrf')
  const headers = new Headers({ 'Content-Type': 'application/json' })
  if (csrf) headers.set('x-csrf-token', csrf)

  const res = await fetch(`${API_URL}/query`, {
    method: 'POST',
    headers,
    credentials: 'include',
    signal,
    body: JSON.stringify({ document_id: documentId, question }),
  })

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `query failed: ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) return
    buffer += decoder.decode(value, { stream: true })

    let sepIndex: number
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)

      if (rawEvent.startsWith(':')) continue // heartbeat / padding comment

      let eventName = 'message'
      let data = ''
      for (const line of rawEvent.split('\n')) {
        if (line.startsWith('event: ')) eventName = line.slice(7)
        else if (line.startsWith('data: ')) data = line.slice(6)
      }
      if (!data) continue

      yield { event: eventName, data: JSON.parse(data) } as QueryEvent
    }
  }
}
