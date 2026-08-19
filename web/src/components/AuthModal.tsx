import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useAuth } from '../context/AuthContext'
import { ApiError } from '../lib/api'
import { Close } from './icons'

export type AuthMode = 'login' | 'register'

interface Props {
  mode: AuthMode
  onModeChange: (mode: AuthMode) => void
  onClose: () => void
  onSuccess: () => void
}

const COPY = {
  login: {
    title: 'Welcome back',
    subtitle: 'Log in to your filings and saved conversations.',
    submit: 'Log in',
    pending: 'Logging in…',
    switchPrompt: "Don't have an account?",
    switchAction: 'Create one',
  },
  register: {
    title: 'Create your account',
    subtitle: 'Upload a filing and start asking questions in under a minute.',
    submit: 'Create account',
    pending: 'Creating account…',
    switchPrompt: 'Already have an account?',
    switchAction: 'Log in',
  },
} as const

export function AuthModal({ mode, onModeChange, onClose, onSuccess }: Props) {
  const { login, register } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const dialogRef = useRef<HTMLDivElement>(null)
  const emailRef = useRef<HTMLInputElement>(null)
  const titleId = useId()
  const errorId = useId()
  const copy = COPY[mode]

  // Restore focus to whatever opened the modal once it closes -- otherwise keyboard
  // users get dumped back at the top of the document.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    emailRef.current?.focus()
    return () => opener?.focus?.()
  }, [])

  // Lock background scroll while open.
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previous
    }
  }, [])

  // Escape closes; Tab cycles within the dialog rather than escaping into the page
  // behind the scrim.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
        return
      }
      if (e.key !== 'Tab') return

      const focusables = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button, input, a[href], [tabindex]:not([tabindex="-1"])',
      )
      if (!focusables || focusables.length === 0) return
      const first = focusables[0]
      const last = focusables[focusables.length - 1]

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', onKeyDown, true)
    return () => document.removeEventListener('keydown', onKeyDown, true)
  }, [onClose])

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSubmitting(true)
    try {
      if (mode === 'login') await login(email, password)
      else await register(email, password)
      onSuccess()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : `${copy.submit} failed`)
    } finally {
      setSubmitting(false)
    }
  }

  function switchMode() {
    setError(null)
    onModeChange(mode === 'login' ? 'register' : 'login')
  }

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        ref={dialogRef}
      >
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
          <Close />
        </button>

        <h2 className="modal-title" id={titleId}>
          {copy.title}
        </h2>
        <p className="modal-subtitle">{copy.subtitle}</p>

        <form className="modal-form" onSubmit={onSubmit} noValidate>
          <div className="field">
            <label className="field-label" htmlFor={`${titleId}-email`}>
              Email
            </label>
            <input
              id={`${titleId}-email`}
              ref={emailRef}
              className="input"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              aria-invalid={error ? true : undefined}
              aria-describedby={error ? errorId : undefined}
              required
            />
          </div>

          <div className="field">
            <label className="field-label" htmlFor={`${titleId}-password`}>
              Password
            </label>
            <input
              id={`${titleId}-password`}
              className="input"
              type="password"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              minLength={mode === 'register' ? 8 : undefined}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              aria-invalid={error ? true : undefined}
              aria-describedby={error ? errorId : undefined}
              required
            />
            {mode === 'register' && <span className="muted">At least 8 characters.</span>}
          </div>

          {error && (
            <p className="form-error" id={errorId} role="alert">
              {error}
            </p>
          )}

          <button type="submit" className="btn btn-primary btn-lg" disabled={submitting}>
            {submitting && <span className="spinner" />}
            {submitting ? copy.pending : copy.submit}
          </button>
        </form>

        <p className="modal-switch">
          {copy.switchPrompt}{' '}
          <button type="button" onClick={switchMode}>
            {copy.switchAction}
          </button>
        </p>
      </div>
    </div>
  )
}
