import { useAuth } from '../context/AuthContext'

/** Honest cold-start state for Render's free-tier spin-down (PROJECT_PLAN.md §7 Phase 3):
 * shown whenever the initial /healthz ping is taking long enough to suggest the API
 * container is waking up, rather than leaving the UI looking merely slow or frozen. */
export function WakingBanner() {
  const { waking } = useAuth()
  if (!waking) return null

  return (
    <div className="waking-banner" role="status">
      <span className="spinner" aria-hidden="true" />
      Waking the server — this can take up to 60 seconds on the first request.
    </div>
  )
}
