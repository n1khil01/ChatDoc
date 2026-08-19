import { useEffect, useState } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { AuthModal, type AuthMode } from '../components/AuthModal'
import { AnimatedAnswerCard } from '../components/AnimatedAnswerCard'
import { ThemeToggle } from '../components/ThemeToggle'
import { ArrowRight, CheckShield, Crosshair, Layers, LogoMark } from '../components/icons'

interface Props {
  /** /login and /register deep-link straight into the modal. */
  initialAuth?: AuthMode
}

export function LandingPage({ initialAuth }: Props) {
  const { user, loading, logout } = useAuth()
  const navigate = useNavigate()
  const [authMode, setAuthMode] = useState<AuthMode | null>(initialAuth ?? null)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  // Only bounce to the app when arriving via a deep auth link (/login, /register) --
  // an already-authenticated visit to plain "/" should still show the landing page
  // (e.g. clicking the logo from inside the app), not loop straight back.
  if (!loading && user && initialAuth) return <Navigate to="/documents" replace />

  return (
    <div className="landing">
      <div className="ambient" aria-hidden="true" />

      <nav className="site-nav" data-scrolled={scrolled}>
        <span className="brand">
          <span className="brand-mark">
            <LogoMark />
          </span>
          ChatDoc
        </span>

        <div className="nav-actions">
          <ThemeToggle />
          {user ? (
            <button type="button" className="btn btn-secondary" onClick={() => logout()}>
              Log out
            </button>
          ) : (
            <button type="button" className="btn btn-primary" onClick={() => setAuthMode('login')}>
              Log in
            </button>
          )}
        </div>
      </nav>

      <main className="hero">
        <div>
          <h1 className="hero-title rise rise-1">
            Ask your questions. <em>Verify every word.</em>
          </h1>

          <p className="hero-lede rise rise-2">
            ChatDoc answers questions about your PDFs with citations that click through to
            the exact page, and refuses when the document does not contain the answer.
          </p>

          <div className="hero-cta rise rise-3">
            <button
              type="button"
              className={`btn btn-primary btn-lg${user ? ' btn-lg-arrow' : ''}`}
              onClick={() => (user ? navigate('/documents') : setAuthMode('register'))}
            >
              {user ? (
                <>
                  Go to documents
                  <ArrowRight />
                </>
              ) : (
                'Get started'
              )}
            </button>
            {!user && <span className="hero-cta-note">Free, no card required</span>}
          </div>
        </div>

        {/* The product's argument, shown rather than described: a verified answer and a
            refusal, side by side. */}
        <section className="thesis rise rise-4" aria-label="How ChatDoc answers">
          <AnimatedAnswerCard />

          <article className="card demo-card demo-card-refusal">
            <header className="demo-head">
              <span className="demo-label demo-refusal-label">Refusal</span>
            </header>
            <p className="demo-question">“What was the crypto impairment charge?”</p>
            <p className="demo-refusal-text">
              This file does not contain the necessary information to answer the question.
            </p>
            <div className="demo-searched">
              <span>Searched 10 excerpts</span>
            </div>
          </article>
        </section>

        <section className="features" aria-label="Capabilities">
          <div className="feature">
            <span className="feature-icon">
              <Crosshair />
            </span>
            <h3 className="feature-title">Citations you can check</h3>
            <p className="feature-body">
              Click any citation to jump to the exact page and highlight the sentence the answer
              came from. No hunting through 200 pages to confirm a number.
            </p>
          </div>

          <div className="feature">
            <span className="feature-icon">
              <CheckShield />
            </span>
            <h3 className="feature-title">Refusal as a feature</h3>
            <p className="feature-body">
              Schema-enforced sufficiency plus numeric provenance: every figure and operand must
              appear in the chunk it cites, or the answer never reaches you.
            </p>
          </div>

          <div className="feature">
            <span className="feature-icon">
              <Layers />
            </span>
            <h3 className="feature-title">Built for financial tables</h3>
            <p className="feature-body">
              Table-aware ingestion keeps rows attached to their headers and scale footnotes, then
              fuses dense and keyword retrieval so exact line items still match.
            </p>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="site-footer-inner">
          <span>ChatDoc, a portfolio project on grounded document QA.</span>
          <span>Eval set derived from FinanceBench (CC BY-NC 4.0).</span>
        </div>
      </footer>

      {authMode && (
        <AuthModal
          mode={authMode}
          onModeChange={setAuthMode}
          onClose={() => setAuthMode(null)}
          onSuccess={() => navigate('/documents')}
        />
      )}
    </div>
  )
}
