import { demoSuffix } from '../demo'
import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import { ArrowLeft, ExternalLink, Zap } from 'lucide-react'
import { fetchDocument } from '../api/client'
import { useStore } from '../store'
import type { DocumentResponse } from '../types'

export function DocumentPage() {
  const { citation } = useParams<{ citation: string }>()
  const navigate = useNavigate()
  const { setQuery } = useStore()
  const [doc, setDoc] = useState<DocumentResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!citation) return
    setLoading(true)
    fetchDocument(decodeURIComponent(citation))
      .then(setDoc)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [citation])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-slate-400">
        <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  if (error || !doc) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400 gap-3">
        <p className="text-sm">Document not found: {citation}</p>
        <button
          onClick={() => navigate(-1)}
          className="text-blue-500 hover:underline text-sm"
        >
          ← Back
        </button>
      </div>
    )
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <button
          onClick={() => navigate(-1)}
          className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800 transition-colors"
        >
          <ArrowLeft size={15} />
          Back
        </button>
        {doc.external_url && (
          <a
            href={doc.external_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-sm text-blue-500 hover:text-blue-700"
          >
            <ExternalLink size={14} />
            Open in {doc.source === 'pubmed' ? 'PubMed' : 'PMC'}
          </a>
        )}
      </div>

      {/* Document title + meta */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <span className={clsx(
            'text-xs font-semibold px-2 py-1 rounded-full',
            doc.source === 'pubmed' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700',
          )}>
            {doc.citation}
          </span>
          <span className="text-xs text-slate-400">{doc.total_chunks} chunks</span>
        </div>
        <h1 className="text-xl font-bold text-slate-800 leading-snug">{doc.title}</h1>
      </div>

      {/* Chunks */}
      <div className="space-y-4">
        {doc.chunks.map((chunk, i) => (
          <div
            key={chunk.chunk_id}
            id={`docchunk-${chunk.chunk_idx}`}
            className="border border-slate-200 rounded-xl p-4 bg-white"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-500">
                Chunk {i + 1} / {doc.total_chunks}
                {chunk.section && ` · ${chunk.section}`}
              </span>
            </div>
            <p className="text-sm text-slate-700 leading-relaxed">{chunk.text}</p>
          </div>
        ))}
      </div>

      {/* Quick-ask button */}
      <div className="mt-8 border-t border-slate-200 pt-6">
        <button
          onClick={() => {
            setQuery(`Based on ${doc.citation}: ${doc.title} — `)
            navigate('/' + demoSuffix)
          }}
          className="flex items-center gap-2 rounded-xl bg-slate-800 hover:bg-slate-900
                     text-white px-5 py-2.5 text-sm font-semibold transition-colors"
        >
          <Zap size={14} />
          Ask a question based on this document
        </button>
      </div>
    </div>
  )
}
