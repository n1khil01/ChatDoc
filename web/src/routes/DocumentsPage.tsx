import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import * as api from '../lib/api'
import { ApiError } from '../lib/api'
import { useAuth } from '../context/AuthContext'
import { ThemeToggle } from '../components/ThemeToggle'
import { IngestProgress } from '../components/IngestProgress'
import { Chat, LogoMark, Trash, Upload } from '../components/icons'

export function DocumentsPage() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const documentsQuery = useQuery({
    queryKey: ['documents'],
    queryFn: api.listDocuments,
    // Poll while anything is still processing so the stage checklist advances live. 700ms is
    // fast enough that the per-stage counters visibly move (the worker writes progress at most
    // every 350ms) without the request rate mattering for a single-user dev box.
    refetchInterval: (query) =>
      query.state.data?.some((d) => d.status === 'processing') ? 700 : false,
  })

  const uploadMutation = useMutation({
    mutationFn: api.uploadDocument,
    onSuccess: () => {
      setUploadError(null)
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (err) => {
      setUploadError(err instanceof ApiError ? err.message : 'upload failed')
    },
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteDocument,
    // Optimistic removal: the row disappears the moment Delete is clicked instead of after a
    // DELETE round-trip *plus* a full refetch of the list to confirm it's gone -- the client
    // already knows the answer, waiting on the server to repeat it back is pure latency.
    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: ['documents'] })
      const previous = queryClient.getQueryData<api.Document[]>(['documents'])
      queryClient.setQueryData<api.Document[]>(['documents'], (docs) =>
        docs?.filter((d) => d.id !== id),
      )
      return { previous }
    },
    onError: (_err, _id, context) => {
      // Roll back if the delete actually failed server-side, so a real error doesn't leave
      // the document silently missing.
      if (context?.previous) queryClient.setQueryData(['documents'], context.previous)
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['documents'] }),
  })

  function onFileChange() {
    const file = fileInputRef.current?.files?.[0]
    if (file) uploadMutation.mutate(file)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="page">
      <header className="page-header">
        <Link to="/" className="brand">
          <span className="brand-mark">
            <LogoMark />
          </span>
          ChatDoc
        </Link>
        <div className="header-user">
          <span className="muted">{user?.email}</span>
          <ThemeToggle />
          <button className="btn btn-secondary" onClick={() => logout()}>
            Log out
          </button>
        </div>
      </header>

      <section className="upload-panel">
        <label className="upload-button">
          <Upload />
          {uploadMutation.isPending ? 'Uploading…' : 'Upload a PDF'}
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={onFileChange}
            disabled={uploadMutation.isPending}
            hidden
          />
        </label>
        {uploadError && <p className="form-error">{uploadError}</p>}
      </section>

      <section>
        {documentsQuery.isLoading && <p className="muted">Loading documents…</p>}
        {documentsQuery.data?.length === 0 && (
          <div className="empty-state">No documents yet. Upload a filing to get started.</div>
        )}
        <ul className="document-list">
          {documentsQuery.data?.map((doc) => {
            const isReady = doc.status === 'ready'
            const openChat = () => {
              if (isReady) navigate(`/documents/${doc.id}/chat`)
            }
            return (
              <li
                key={doc.id}
                className={`document-row${doc.status === 'processing' ? ' document-row-processing' : ''}${isReady ? ' document-row-clickable' : ''}`}
                onClick={openChat}
                role={isReady ? 'button' : undefined}
                tabIndex={isReady ? 0 : undefined}
                onKeyDown={(e) => {
                  if (isReady && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault()
                    openChat()
                  }
                }}
              >
                <div className="document-main">
                  <div className="document-info">
                    <span className="document-name">{doc.display_name ?? doc.doc_name}</span>
                    {/* While processing, the stage panel below is the status indicator -- a
                        duplicate "processing" badge next to it is just noise. */}
                    {doc.status !== 'processing' && (
                      <span className={`badge badge-dot badge-${doc.status}`}>{doc.status}</span>
                    )}
                    {doc.status === 'ready' && <span className="muted">{doc.page_count} pages</span>}
                    {doc.status === 'failed' && <span className="form-error">{doc.error}</span>}
                  </div>
                  <div className="document-actions">
                    {isReady && (
                      <Link
                        className="btn btn-secondary btn-icon"
                        to={`/documents/${doc.id}/chat`}
                        onClick={(e) => e.stopPropagation()}
                        aria-label="Ask a question about this document"
                      >
                        <Chat />
                      </Link>
                    )}
                    <button
                      className="btn btn-danger-ghost btn-icon"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteMutation.mutate(doc.id)
                      }}
                      aria-label="Delete document"
                    >
                      <Trash />
                    </button>
                  </div>
                </div>
                {doc.status === 'processing' && <IngestProgress doc={doc} />}
              </li>
            )
          })}
        </ul>
      </section>
    </div>
  )
}
