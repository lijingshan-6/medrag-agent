import { demoSuffix, isGuidedDemo } from '../demo'
import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchSearch } from '../api/client'
import type { ChunkOut, SearchResponse } from '../types'

// ── SVG icons ─────────────────────────────────────────────────────────────
function I({ size = 16, sw = 1.6, children, style }: {
  size?: number; sw?: number; children: React.ReactNode; style?: React.CSSProperties
}) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={sw} strokeLinecap="round"
      strokeLinejoin="round" aria-hidden="true" style={style}>
      {children}
    </svg>
  )
}
const IconSearch   = (p: { size?: number; sw?: number; style?: React.CSSProperties }) =>
  <I {...p}><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></I>
const IconExternal = (p: { size?: number; sw?: number }) =>
  <I {...p}><path d="M15 3h6v6M10 14 21 3M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></I>
const IconFilter   = (p: { size?: number; sw?: number; style?: React.CSSProperties }) =>
  <I {...p}><path d="M3 5h18l-7 9v6l-4-2v-4L3 5Z"/></I>

const CITE_VARS = ['--c0','--c1','--c2','--c3','--c4','--c5','--c6','--c7']

const SOURCE_FACETS = [
  { id: 'all',    label: 'All sources' },
  { id: 'pubmed', label: 'PubMed' },
  { id: 'pmc',    label: 'PMC Open Access' },
]

const PIPELINE_OPTIONS = [
  { id: 'p2', label: 'Hybrid',   hint: 'Dense + sparse' },
  { id: 'p3', label: 'Reranked', hint: 'BGE-reranker' },
]

// ── Facet group ────────────────────────────────────────────────────────────
function FacetGroup({ title, options, value, onChange }: {
  title: string; options: { id: string; label: string }[]; value: string; onChange: (v: string) => void
}) {
  return (
    <div style={{ marginBottom: 22 }}>
      <div className="vm-eyebrow" style={{ marginBottom: 10 }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {options.map((o) => (
          <FacetButton key={o.id} option={o} active={value === o.id} onClick={() => onChange(o.id)} />
        ))}
      </div>
    </div>
  )
}

function FacetButton({ option, active, onClick }: {
  option: { id: string; label: string }; active: boolean; onClick: () => void
}) {
  const [hovered, setHovered] = React.useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '7px 8px', borderRadius: 5,
        background: active ? 'var(--accent-soft)' : hovered ? 'var(--panel-2)' : 'transparent',
        border: 'none',
        fontSize: 12.5,
        color: active ? 'var(--accent-ink)' : 'var(--ink-soft)',
        fontWeight: active ? 600 : 500,
        textAlign: 'left', transition: 'background 120ms',
      }}
    >
      {option.label}
    </button>
  )
}

// ── Result row ─────────────────────────────────────────────────────────────
function ResultRow({ chunk, idx, onClick }: { chunk: ChunkOut; idx: number; onClick: () => void }) {
  const colorVar = CITE_VARS[idx % CITE_VARS.length]
  const score = chunk.score ?? 0
  const [hovered, setHovered] = React.useState(false)
  const ext = chunk as ChunkOut & { authors?: string; journal?: string; year?: number }

  return (
    <article
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: 'grid',
        gridTemplateColumns: '40px minmax(0, 1fr) 90px',
        gap: 18,
        padding: '18px 4px',
        borderBottom: '1px solid var(--rule-soft)',
        cursor: 'pointer',
        alignItems: 'baseline',
        background: hovered ? 'var(--panel-2)' : 'transparent',
        transition: 'background 120ms',
        borderRadius: hovered ? 4 : 0,
      }}
    >
      {/* Number */}
      <div style={{
        fontFamily: 'var(--serif)', fontStyle: 'italic',
        fontSize: 28, lineHeight: 1,
        color: `var(${colorVar})`,
        letterSpacing: '-0.02em',
        textAlign: 'right',
      }}>
        {String(idx + 1).padStart(2, '0')}
      </div>

      {/* Body */}
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, flexWrap: 'wrap' }}>
          <span className="vm-mono" style={{ fontSize: 10.5, color: `var(${colorVar})`, fontWeight: 600 }}>
            {chunk.citation}
          </span>
          <span style={{
            display: 'inline-flex', padding: '1px 5px', borderRadius: 3,
            fontSize: 9.5, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
            border: '1px solid var(--rule)', color: 'var(--muted)',
            fontFamily: 'var(--mono)',
          }}>
            {chunk.source === 'pubmed' ? 'PubMed' : 'PMC'}
          </span>
          {(ext.journal || ext.year) && (
            <span style={{ fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: 12, color: 'var(--muted)' }}>
              {ext.journal}{ext.year ? ` · ${ext.year}` : ''}
            </span>
          )}
        </div>
        <h3 style={{
          margin: '0 0 6px', fontSize: 15, fontWeight: 600,
          color: 'var(--ink)', letterSpacing: '-0.005em', lineHeight: 1.35,
        }}>
          {chunk.title}
        </h3>
        <p style={{
          margin: 0, fontSize: 13, color: 'var(--muted)',
          fontFamily: 'var(--serif)', lineHeight: 1.55,
          letterSpacing: '-0.003em',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}>
          {chunk.text}
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 8, fontSize: 11, color: 'var(--muted)' }}>
          {ext.authors && (
            <span style={{ fontFamily: 'var(--serif)', fontStyle: 'italic' }}>{ext.authors}</span>
          )}
          {chunk.external_url && (
            <a
              href={chunk.external_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: 'var(--accent)', textDecoration: 'none' }}
            >
              <IconExternal size={11} sw={2} />
              Open
            </a>
          )}
        </div>
      </div>

      {/* Score */}
      <div style={{ textAlign: 'right' }}>
        <div className="vm-mono" style={{ fontSize: 14, color: 'var(--ink-soft)', fontWeight: 600 }}>
          {isGuidedDemo ? '—' : score.toFixed(3)}
        </div>
        <div className="vm-eyebrow" style={{ fontSize: 9, marginTop: 2 }}>{isGuidedDemo ? 'not measured' : 'relevance'}</div>
        <div style={{ marginTop: 6, height: 2, background: 'var(--rule-soft)', borderRadius: 1, overflow: 'hidden' }}>
          <div style={{ width: `${Math.min(100, score * 100)}%`, height: '100%', background: `var(${colorVar})` }} />
        </div>
      </div>
    </article>
  )
}

// ── Pipeline button ────────────────────────────────────────────────────────
function PipelineBtn({ option, active, onClick }: {
  option: { id: string; label: string; hint: string }; active: boolean; onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      title={option.hint}
      style={{
        flex: 1, padding: '8px 0',
        fontSize: 11.5, fontWeight: 600, letterSpacing: '0.01em',
        background: active ? 'var(--ink)' : 'var(--panel)',
        color: active ? 'var(--canvas)' : 'var(--ink-soft)',
        border: `1px solid ${active ? 'var(--ink)' : 'var(--rule)'}`,
        borderRadius: 5,
        transition: 'all 120ms',
      }}
    >
      {option.label}
    </button>
  )
}

// ── ExplorerPage ───────────────────────────────────────────────────────────
export function ExplorerPage() {
  const [q, setQ] = useState('')
  const [k, setK] = useState(10)
  const [pipeline, setPipeline] = useState<'p2' | 'p3'>('p2')
  const [source, setSource] = useState('all')
  const [results, setResults] = useState<SearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  async function handleSearch() {
    if (!q.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await fetchSearch(q, k, pipeline)
      setResults(data)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(`Search failed: ${msg}`)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') handleSearch()
  }

  const displayChunks = results
    ? (source === 'all' ? results.chunks : results.chunks.filter((c) => c.source === source))
    : []

  return (
    <div className="vm-explorer" style={{ height: '100%', display: 'flex', overflow: 'hidden' }}>
      {/* Sidebar */}
      <aside style={{
        width: 240, flexShrink: 0,
        borderRight: '1px solid var(--rule)',
        padding: '22px 18px',
        overflowY: 'auto',
        background: 'var(--panel-2)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 18 }}>
          <IconFilter size={13} sw={2} style={{ color: 'var(--ink-soft)' }} />
          <span className="vm-eyebrow">Filters</span>
        </div>

        <FacetGroup title="Source" options={SOURCE_FACETS} value={source} onChange={setSource} />

        <div className="vm-eyebrow" style={{ marginBottom: 10 }}>{isGuidedDemo ? "Fixture text search" : "Retrieval"}</div>
        <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
          {!isGuidedDemo && PIPELINE_OPTIONS.map((p) => (
            <PipelineBtn key={p.id} option={p} active={pipeline === p.id}
              onClick={() => setPipeline(p.id as 'p2' | 'p3')} />
          ))}
        </div>

        <div className="vm-eyebrow" style={{ marginBottom: 8 }}>Results</div>
        <input
          type="range" min={5} max={20} step={1} value={k}
          onChange={(e) => setK(Number(e.target.value))}
          style={{ width: '100%', accentColor: 'var(--accent)' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <span className="vm-mono" style={{ fontSize: 10, color: 'var(--faint)' }}>5</span>
          <span className="vm-mono" style={{ fontSize: 11, color: 'var(--ink-soft)', fontWeight: 600 }}>k = {k}</span>
          <span className="vm-mono" style={{ fontSize: 10, color: 'var(--faint)' }}>20</span>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Search bar */}
        <div style={{
          padding: '20px 32px',
          borderBottom: '1px solid var(--rule)',
          flexShrink: 0,
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'var(--panel)',
            border: '1px solid var(--rule)',
            borderRadius: 10,
            padding: '12px 16px',
          }}>
            <IconSearch size={16} sw={2} style={{ color: 'var(--faint)' }} />
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={handleKeyDown}
              style={{
                flex: 1, minWidth: 0, border: 'none', outline: 'none', background: 'transparent',
                fontFamily: 'var(--serif)', fontSize: 17, color: 'var(--ink)',
                letterSpacing: '-0.005em',
              }}
              placeholder="Search the corpus by keyword, MeSH term, or natural language"
            />
            <button
              onClick={handleSearch}
              disabled={loading || !q.trim()}
              style={{
                padding: '6px 14px', borderRadius: 6, border: 'none',
                background: !q.trim() ? 'var(--rule)' : 'var(--ink)',
                color: !q.trim() ? 'var(--faint)' : 'var(--canvas)',
                fontSize: 12, fontWeight: 600,
                cursor: !q.trim() ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
        </div>

        {/* Results */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 32px 60px' }}>
          {error && (
            <div style={{
              marginBottom: 16, padding: '12px 16px', borderRadius: 8,
              border: '1px solid var(--error)', background: 'var(--error-soft)',
              fontSize: 13, color: 'var(--error)',
            }}>
              {error}
            </div>
          )}

          {results && (
            <>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 14, marginBottom: 10 }}>
                <span style={{ fontFamily: 'var(--serif)', fontSize: 20, color: 'var(--ink)', letterSpacing: '-0.01em' }}>
                  <span style={{ fontStyle: 'italic' }}>{displayChunks.length}</span> passages
                </span>
                <span className="vm-mono" style={{ fontSize: 11, color: 'var(--faint)' }}>
                  {isGuidedDemo ? 'fixed examples · no measured latency' : `${results.latency_ms.toFixed(0)}ms · ${results.pipeline} · ${source}`}
                </span>
              </div>
              <div>
                {displayChunks.map((c, i) => (
                  <ResultRow
                    key={c.chunk_id}
                    chunk={c}
                    idx={i}
                    onClick={() => navigate(`/document/${encodeURIComponent(c.citation)}${demoSuffix}`)}
                  />
                ))}
              </div>
            </>
          )}

          {!results && !loading && !error && (
            <div style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              justifyContent: 'center', height: '100%',
            }}>
              <IconSearch size={32} sw={1.4} style={{ opacity: 0.3, marginBottom: 14, color: 'var(--faint)' }} />
              <p style={{
                margin: 0, fontFamily: 'var(--serif)', fontStyle: 'italic',
                fontSize: 18, color: 'var(--muted)',
              }}>
                Search the corpus directly
              </p>
              <p style={{ margin: '8px 0 0', fontSize: 12, color: 'var(--faint)' }}>
                Keyword, MeSH terms, or natural language
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
