import type { ReactNode } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ThemeProvider } from './context/ThemeContext'
import { LandingPage } from './routes/LandingPage'
import { DocumentsPage } from './routes/DocumentsPage'
import { ChatPage } from './routes/ChatPage'
import { WakingBanner } from './components/WakingBanner'

const queryClient = new QueryClient()

function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth()
  if (loading) return <div className="page-loading">Loading…</div>
  // Land on the plain landing page, never the login modal -- covers both an explicit
  // logout and a session expiring mid-use. Sending someone straight into a login form
  // right after they clicked "Log out" reads as broken, not as a redirect.
  if (!user) return <Navigate to="/" replace />
  return <>{children}</>
}

function AppRoutes() {
  return (
    <>
      <WakingBanner />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        {/* Deep links into the auth modal; RequireAuth redirects here on 401. */}
        <Route path="/login" element={<LandingPage initialAuth="login" />} />
        <Route path="/register" element={<LandingPage initialAuth="register" />} />
        <Route
          path="/documents"
          element={
            <RequireAuth>
              <DocumentsPage />
            </RequireAuth>
          }
        />
        <Route
          path="/documents/:documentId/chat"
          element={
            <RequireAuth>
              <ChatPage />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>
          <AuthProvider>
            <AppRoutes />
          </AuthProvider>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App
