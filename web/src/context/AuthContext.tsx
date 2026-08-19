import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import * as api from '../lib/api'

interface AuthContextValue {
  user: api.User | null
  loading: boolean
  waking: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

// A cold Render instance takes up to ~60s to answer at all, so a normal fetch spinner
// would just look frozen. Only flip into the "waking up" state if /healthz hasn't
// resolved within WAKE_THRESHOLD_MS -- a warm server clears this before it ever shows.
const WAKE_THRESHOLD_MS = 1200

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<api.User | null>(null)
  const [loading, setLoading] = useState(true)
  const [waking, setWaking] = useState(false)

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    let settled = false
    const timer = setTimeout(() => {
      if (!settled) setWaking(true)
    }, WAKE_THRESHOLD_MS)

    api
      .healthz()
      .catch(() => {
        // Swallow: this ping only drives the "waking up" banner. If the server never
        // comes up at all, the real request (me(), login, etc.) surfaces that error.
      })
      .finally(() => {
        settled = true
        clearTimeout(timer)
        setWaking(false)
      })

    return () => {
      settled = true
      clearTimeout(timer)
    }
  }, [])

  const value: AuthContextValue = {
    user,
    loading,
    waking,
    login: async (email, password) => {
      setUser(await api.login(email, password))
    },
    register: async (email, password) => {
      setUser(await api.register(email, password))
    },
    logout: async () => {
      try {
        await api.logout()
      } finally {
        // Always clear local state, even if the server call 403s (e.g. a stale/missing
        // CSRF cookie) -- otherwise a failed request leaves the UI stuck "logged in"
        // with no way to leave that state from the client.
        setUser(null)
      }
    },
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
