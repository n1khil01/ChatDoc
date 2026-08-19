/**
 * Browser-measured p95 time-to-first-token, warm (PROJECT_PLAN.md §6, §7 Phase 3).
 *
 * Streaming and "blocking" are measured from the *same* /query call, which is a fair
 * comparison rather than a second implementation: a blocking API shows nothing until
 * generation is fully done, so its time-to-first-token is definitionally equal to its
 * total response time. `fullMs` below is exactly that number; `ttftMs` is what the SSE
 * client this project actually ships perceives instead.
 *
 * Not a CI test: it registers a throwaway user, uploads a real PDF, waits for real
 * ingestion, and fires real Gemini calls (a warm-up run plus N measured runs) against
 * whatever GEMINI_API_KEY the API process is holding. Run it manually:
 *
 *   npm run test:perf
 *
 * Requires the API (port 8000) and the Vite dev server (port 5173) already running --
 * this harness does not start either (see playwright.config.ts).
 */

import { test, expect } from '@playwright/test'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

const API_URL = process.env.VITE_API_URL ?? 'http://localhost:8000'
const PDF_PATH =
  process.env.TTFT_PDF ?? resolve(__dirname, '../../eval/pdfs/ULTABEAUTY_2023Q4_EARNINGS.pdf')
const N_RUNS = Number(process.env.TTFT_RUNS ?? 8)
const INGEST_TIMEOUT_MS = 90_000

// Varied enough that the model can't just serve a cached identical completion for
// every run, without needing per-document knowledge -- these are generic enough to at
// least reach generation (a refusal still exercises the same streaming path).
const QUESTIONS = [
  'What was the total revenue reported?',
  'What was the net income for the period?',
  'What were the total operating expenses?',
  'What was reported for cash and cash equivalents?',
  'What was the diluted earnings per share?',
  'What were the total assets?',
  'What was the gross margin?',
  'What was reported for total liabilities?',
]

interface QueryTiming {
  ttftMs: number | null
  fullMs: number
  outcome: 'answer' | 'refusal' | 'error' | 'incomplete'
}

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)
  return sorted[Math.max(0, idx)]
}

test('p95 time-to-first-token vs. full-response time, warm', async ({ page, context }) => {
  const email = `ttft-${Date.now()}@example.com`
  const password = 'correct horse battery staple'

  const registerRes = await context.request.post(`${API_URL}/auth/register`, {
    data: { email, password },
  })
  expect(registerRes.ok(), await registerRes.text()).toBeTruthy()

  const csrf = (await context.cookies()).find((c) => c.name === 'chatdoc_csrf')?.value
  expect(csrf, 'register did not issue a CSRF cookie').toBeTruthy()

  const uploadRes = await context.request.post(`${API_URL}/documents`, {
    multipart: {
      file: {
        name: 'ttft-fixture.pdf',
        mimeType: 'application/pdf',
        buffer: readFileSync(PDF_PATH),
      },
    },
    headers: { 'x-csrf-token': csrf! },
  })
  expect(uploadRes.ok(), await uploadRes.text()).toBeTruthy()
  const documentId = (await uploadRes.json()).id as number

  const deadline = Date.now() + INGEST_TIMEOUT_MS
  let status = 'processing'
  while (Date.now() < deadline) {
    const doc = await (await context.request.get(`${API_URL}/documents/${documentId}`)).json()
    status = doc.status
    if (status !== 'processing') break
    await new Promise((r) => setTimeout(r, 1500))
  }
  expect(status, 'document did not finish ingesting in time').toBe('ready')

  // A real browser page, not Node fetch -- performance.now() runs against the actual
  // browser network stack, matching "browser-measured" in PROJECT_PLAN.md §6.
  await page.goto('/')

  async function runQuery(question: string): Promise<QueryTiming> {
    return page.evaluate(
      async ({ apiUrl, documentId, csrf, question }) => {
        const t0 = performance.now()
        const res = await fetch(`${apiUrl}/query`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json', 'x-csrf-token': csrf } as HeadersInit,
          body: JSON.stringify({ document_id: documentId, question }),
        })
        const reader = res.body!.getReader()
        const decoder = new TextDecoder()
        let buf = ''
        let ttftMs: number | null = null
        let outcome: QueryTiming['outcome'] = 'incomplete'

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buf += decoder.decode(value, { stream: true })

          if (ttftMs === null && buf.includes('event: delta')) {
            ttftMs = performance.now() - t0
          }
          if (outcome === 'incomplete') {
            if (buf.includes('event: answer')) outcome = 'answer'
            else if (buf.includes('event: refusal')) outcome = 'refusal'
            else if (buf.includes('event: error')) outcome = 'error'
          }
          if (buf.includes('event: done')) break
        }

        return { ttftMs, fullMs: performance.now() - t0, outcome }
      },
      { apiUrl: API_URL, documentId, csrf, question },
    )
  }

  // Warm-up call: excluded from the measured set so cold model/connection variance
  // (this project's own honestly-reported cold-start problem, just for Gemini instead
  // of Render) doesn't pollute a number that's supposed to be "warm".
  await runQuery(QUESTIONS[0])

  const runs: QueryTiming[] = []
  for (let i = 0; i < N_RUNS; i++) {
    runs.push(await runQuery(QUESTIONS[i % QUESTIONS.length]))
  }

  const ttftValues = runs.map((r) => r.ttftMs).filter((v): v is number => v !== null)
  const fullValues = runs.map((r) => r.fullMs)

  const summary = {
    documentId,
    nRuns: N_RUNS,
    ttftP95Ms: percentile(ttftValues, 0.95),
    fullResponseP95Ms: percentile(fullValues, 0.95),
    runs,
  }

  writeFileSync(resolve(__dirname, 'ttft-results.json'), JSON.stringify(summary, null, 2))
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(summary, null, 2))

  expect(ttftValues.length, 'no run produced a delta event -- check the API is reachable').toBeGreaterThan(0)
})
