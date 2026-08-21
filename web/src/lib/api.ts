// Thin fetch wrapper over the ChatDoc API. Cookie session auth (credentials: 'include')
// plus the double-submit CSRF cookie the backend issues on register/login (api/csrf.py).
//
// Request/response shapes come from api-types.ts, generated from the live FastAPI
// OpenAPI schema (`npm run gen:types`, after `uv run python scripts/export_openapi.py`)
// rather than hand-written here -- a field renamed or removed on the backend breaks
// the build instead of silently drifting.

import type { components } from './api-types'

export const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

// The CSRF cookie lives on the API's origin (Render), not the frontend's (Vercel) --
// two unrelated origins, so document.cookie on this page can never see it, CSRF cookie
// attributes notwithstanding (api/csrf.py has the full writeup). The API hands the same
// value back via the `x-csrf-token` *response* header on register/login/me instead;
// this module-level var is the frontend's only copy of it, refreshed on every response
// that carries the header and cleared on logout.
let csrfToken: string | null = null

function captureCsrfToken(res: Response): void {
  const token = res.headers.get('x-csrf-token')
  if (token) csrfToken = token
}

// Exposed for callers that can't go through request()/uploadDocument() below -- e.g.
// queryStream.ts, which hand-rolls its own fetch because EventSource can't send a POST
// body, custom headers, or credentials the way SSE querying needs here.
export function getCsrfToken(): string | null {
  return csrfToken
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? 'GET').toUpperCase()
  const headers = new Headers(init?.headers)
  const isMutating = method !== 'GET' && method !== 'HEAD'

  if (isMutating && csrfToken) headers.set('x-csrf-token', csrfToken)

  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    method,
    headers,
    credentials: 'include',
  })
  captureCsrfToken(res)

  if (!res.ok) {
    let message = res.statusText
    try {
      const body = await res.json()
      message = body.detail ?? message
    } catch {
      // non-JSON error body, keep statusText
    }
    throw new ApiError(res.status, message)
  }

  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

// Render free tier spins the API down after 15 minutes idle (~60s cold wake) and Neon
// suspends after 5. `me()` alone can't distinguish "waking up" from "any other slow
// request", so AuthContext pings this separately with its own short-timeout race to
// decide when to show the "waking the server" state (PROJECT_PLAN.md §7 Phase 3).
export function healthz(): Promise<{ status: string }> {
  return request('/healthz')
}

export type User = components['schemas']['UserResponse']

export function register(email: string, password: string) {
  return request<User>('/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password } satisfies components['schemas']['RegisterRequest']),
  })
}

export function login(email: string, password: string) {
  return request<User>('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password } satisfies components['schemas']['LoginRequest']),
  })
}

export async function logout(): Promise<void> {
  try {
    await request<void>('/auth/logout', { method: 'POST' })
  } finally {
    csrfToken = null
  }
}

export function me() {
  return request<User>('/auth/me')
}

export type Document = components['schemas']['DocumentResponse']
export type DocumentStatus = Document['status']

export function listDocuments() {
  return request<Document[]>('/documents')
}

export function getDocument(id: number) {
  return request<Document>(`/documents/${id}`)
}

export function deleteDocument(id: number) {
  return request<void>(`/documents/${id}`, { method: 'DELETE' })
}

export function documentFileUrl(id: number) {
  return `${API_URL}/documents/${id}/file`
}

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData()
  form.append('file', file)
  const headers = new Headers()
  if (csrfToken) headers.set('x-csrf-token', csrfToken)

  const res = await fetch(`${API_URL}/documents`, {
    method: 'POST',
    body: form,
    headers,
    credentials: 'include',
  })
  captureCsrfToken(res)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail ?? res.statusText)
  }
  return res.json()
}
