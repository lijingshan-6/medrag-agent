import React, { useState } from 'react'
import { useStore } from '../store'
import { fetchChunk } from '../api/client'
import type { ChunkOut, ChunkContextResponse } from '../types'

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
const IconExternal  = (p: { size?: number; sw?: number; style?: React.CSSProperties }) =>
  <I {...p}><path d="M15 3h6v6M10 14 21 3M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></I>
const IconChevDown  = (p: { size?: number; sw?: number }) => <I {...p}><path d="m6 9 6 6 6-6"/></I>
const IconChevUp    = (p: { size?: number; sw?: number }) => <I {...p}><path d="m6 15 6-6 6 6"/></I>
const IconBook      = (p: { size?: number; sw?: number; style?: React.CSSProperties }) =>
  <I {...p}><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5v-17Z"/><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/></I>

const CITE_VARS = ['--c0','--c1','--c2','--c3','--c4','--c5','--c6','--c7']

// ── Score bar ─────────────────────────────────────────────────────────────
function ScoreBar({ score, colorVar }: { score: number | null; colorVar: string }) {
  if (score == null) return null
  const pct = Math.min(1, Math.max(0, score)) * 100
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span className="vm-mono" style={{ fontSize: 10, color: 'var(--muted)', minWidth: 28 }}>
        {score.toFixed(2)}
      </span>
      <div style={{ flex: 1, height: 2, background: 'var(--rule-soft)', borderRadius: 1, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: `var(${colorVar})` }} />
      </div>
    </div>
  )
}

// ── Highlighted text ──────────────────────────────────────────────────────
function HighlightedText({ text, ranges }: { text: string; ranges: [number, number][] }) {
  if (!ranges || !ranges.length) return <>{text}</>
  const sorted = [...ranges].sort((a, b) => a[0] - b[0])
  const out: React.ReactNode[] = []
  let cur = 0
  for (const [a, b] of sorted) {
    if (a > cur) out.push(<span key={cur}>{text.slice(cur, a)}</span>)
    out.push(<mark key={`m${a}`} className="vm-highlight">{text.slice(a, b)}</mark>)
    cur = b
  }
  if (cur < text.length) out.push(<span key={cur}>{text.slice(cur)}</span>)
  return <>{out}</>
}

// ── Context expander ──────────────────────────────────────────────────────
function ContextExpander({ chunkId }: { chunkId: string }) {
  const [open, setOpen] = useState(false)
  const [ctx, setCtx] = useState<ChunkContextResponse | null>(null)
  const [loading, setLoading] = useState(false)

  async function toggle(e: React.MouseEvent) {
    e.stopPropagation()
    if (!open && !ctx) {
      setLoading(true)
      try {
        const data = await fetchChunk(chunkId, 1)
        setCtx(data)
      } catch { /* ignore */ }
      finally { setLoading(false) }
    }
    setOpen((v) => !v)
  }

  return (
    <div>
      <button
        onClick={toggle}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          background: 'transparent', border: 'none', padding: 0,
          color: 'var(--muted)', fontSize: 11, fontWeight: 500,
        }}
      >
        {open ? <IconChevUp size={11} sw={2} /> : <IconChevDown size={11} sw={2} />}
        {loading ? 'Loading…' : open ? 'Hide context' : 'View context'}
      </button>
      {open && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }} className="vm-fadeup">
          {ctx?.prev_chunk && (
            <ContextSnippet label="Preceding" text={ctx.prev_chunk.text.slice(0, 300)} />
          )}
          {ctx?.next_chunk && (
            <ContextSnippet label="Following" text={ctx.next_chunk.text.slice(0, 300)} />
          )}
          {!ctx && open && (
            <ContextSnippet label="Context" text="Context unavailable for this source." />
          )}
        </div>
      )}
    </div>
  )
}

function ContextSnippet({ label, text }: { label: string; text: string }) {
  return (
    <div style={{
      padding: '8px 10px',
      borderLeft: '2px solid var(--rule)',
      background: 'var(--panel-2)',
      borderRadius: '0 4px 4px 0',
    }}>
      <div className="vm-eyebrow" style={{ fontSize: 9, marginBottom: 4 }}>{label}</div>
      <p style={{
        margin: 0, fontSize: 11.5, lineHeight: 1.55,
        color: 'var(--muted)', fontStyle: 'italic', fontFamily: 'var(--serif)',
      }}>
        {text}
      </p>
    </div>
  )
}

// ── Evidence card ─────────────────────────────────────────────────────────
function EvidenceCard({
  chunk, idx, isSelected, onSelect,
}: {
  chunk: ChunkOut; idx: number; isSelected: boolean; onSelect: (id: string) => void
}) {
  const colorVar = CITE_VARS[idx % CITE_VARS.length]
  const ref = React.useRef<HTMLElement>(null)

  React.useEffect(() => {
    if (isSelected && ref.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [isSelected])

  const ext = chunk as ChunkOut & { authors?: string; journal?: string; year?: number }

  return (
    <article
      ref={ref}
      id={`chunk-${chunk.chunk_id}`}
      onClick={() => onSelect(chunk.chunk_id)}
      className="vm-fadeup"
      style={{
        position: 'relative',
        padding: '18px 18px 16px 22px',
        background: 'var(--panel)',
        border: `1px solid ${isSelected ? `var(${colorVar})` : 'var(--rule)'}`,
        borderLeft: `3px solid var(${colorVar})`,
        borderRadius: 6,
        cursor: 'pointer',
        transition: 'border-color 160ms',
      }}
    >
      {/* Index numeral */}
      <div style={{
        position: 'absolute', top: 12, right: 16,
        fontFamily: 'var(--serif)', fontStyle: 'italic',
        fontSize: 24, lineHeight: 1,
        color: `var(${colorVar})`,
        opacity: 0.85,
        letterSpacing: '-0.02em',
      }}>
        {idx + 1}
      </div>

      {/* Citation + source pill */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4, paddingRight: 30 }}>
        <span className="vm-mono" style={{ fontSize: 10.5, fontWeight: 600, color: `var(${colorVar})`, letterSpacing: '0.02em' }}>
          {chunk.citation}
        </span>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 4,
          padding: '1px 6px', borderRadius: 3,
          fontSize: 9.5, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase',
          border: '1px solid var(--rule)', color: 'var(--muted)',
          fontFamily: 'var(--mono)',
        }}>
          {chunk.source === 'pubmed' ? 'PubMed' : 'PMC'}
        </span>
      </div>

      {/* Title */}
      <h3 style={{
        margin: '4px 0 4px 0',
        fontSize: 13.5, lineHeight: 1.35, fontWeight: 600,
        color: 'var(--ink)', letterSpacing: '-0.005em',
        paddingRight: 30,
      }}>
        {chunk.title}
      </h3>

      {/* Authors / journal / year / section */}
      {(ext.authors || ext.journal || ext.year || chunk.section) && (
        <div style={{
          fontSize: 11, color: 'var(--muted)', marginBottom: 8,
          fontFamily: 'var(--serif)', fontStyle: 'italic',
          letterSpacing: '-0.005em',
        }}>
          {ext.authors && <>{ext.authors} · </>}
          {(ext.journal || ext.year) && (
            <span style={{ fontStyle: 'normal', fontFamily: 'var(--sans)', fontSize: 10.5 }}>
              {ext.journal}{ext.year ? ` ${ext.year}` : ''}
            </span>
          )}
          {chunk.section && (
            <> · <span className="vm-mono" style={{
              fontStyle: 'normal', fontSize: 9.5, color: 'var(--faint)',
              textTransform: 'uppercase', letterSpacing: '0.1em',
            }}>{chunk.section}</span></>
          )}
        </div>
      )}

      {/* Score bar */}
      <div style={{ marginBottom: 10 }}>
        <ScoreBar score={chunk.score ?? null} colorVar={colorVar} />
      </div>

      {/* Quote block */}
      <blockquote style={{ margin: 0, paddingLeft: 10, borderLeft: `2px solid var(${colorVar})`, opacity: 0.85 }}>
        <p style={{
          margin: 0, fontSize: 12.5, lineHeight: 1.55,
          color: 'var(--ink-soft)',
          fontFamily: 'var(--serif)',
          letterSpacing: '-0.003em',
        }}>
          <HighlightedText text={chunk.text} ranges={chunk.highlight_ranges ?? []} />
        </p>
      </blockquote>

      {/* Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 12 }}>
        <ContextExpander chunkId={chunk.chunk_id} />
        {chunk.external_url && (
          <a
            href={chunk.external_url}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 4,
              color: 'var(--muted)', fontSize: 11, fontWeight: 500,
              textDecoration: 'none',
            }}
          >
            <IconExternal size={11} sw={2} />
            Open
          </a>
        )}
      </div>
    </article>
  )
}

// ── EvidencePanel ─────────────────────────────────────────────────────────
export function EvidencePanel() {
  const { result, liveChunks, selectedChunkId, setSelectedChunkId, isStreaming } = useStore()
  const rawChunks = result?.chunks?.length ? result.chunks : liveChunks
  const chunks = Array.from(new Map(rawChunks.map(c => [c.chunk_id, c])).values())
  const citationOrder = [...(result?.citations ?? [])]
  for (const chunk of chunks) if (!citationOrder.includes(chunk.citation)) citationOrder.push(chunk.citation)

  if (!chunks.length && !isStreaming) {
    return (
      <aside style={{
        height: '100%',
        borderLeft: '1px solid var(--rule)',
        background: 'var(--panel-2)',
        padding: '24px 22px',
      }}>
        <div className="vm-eyebrow" style={{ marginBottom: 14 }}>Evidence</div>
        <div style={{
          padding: '40px 8px', textAlign: 'center',
          color: 'var(--faint)', fontSize: 13,
          fontFamily: 'var(--serif)', fontStyle: 'italic',
        }}>
          <IconBook size={22} style={{ opacity: 0.4, marginBottom: 10 }} />
          <p style={{ margin: 0 }}>Retrieved passages and their sources will appear here.</p>
        </div>
      </aside>
    )
  }

  return (
    <aside style={{
      height: '100%', overflowY: 'auto',
      borderLeft: '1px solid var(--rule)',
      background: 'var(--panel-2)',
    }}>
      <div style={{
        padding: '18px 22px 12px',
        borderBottom: '1px solid var(--rule-soft)',
        position: 'sticky', top: 0,
        background: 'var(--panel-2)',
        backdropFilter: 'blur(8px)',
        zIndex: 1,
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span className="vm-eyebrow">Evidence</span>
          <span className="vm-mono" style={{ fontSize: 10, color: 'var(--faint)' }}>
            {chunks.length} passages · {citationOrder.length} sources
          </span>
        </div>
        <div style={{ marginTop: 4, fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--serif)', fontStyle: 'italic' }}>
          Click a passage to highlight its citation in the answer.
        </div>
      </div>

      <div style={{ padding: '16px 18px 32px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {chunks.map((chunk) => (
          <EvidenceCard
            key={chunk.chunk_id}
            chunk={chunk}
            idx={citationOrder.indexOf(chunk.citation)}
            isSelected={selectedChunkId === chunk.chunk_id}
            onSelect={(id) => setSelectedChunkId(selectedChunkId === id ? null : id)}
          />
        ))}
        {isStreaming && (
          <div style={{
            padding: '12px 14px', borderRadius: 6, border: '1px dashed var(--rule)',
            color: 'var(--faint)', fontSize: 12, fontFamily: 'var(--serif)', fontStyle: 'italic',
            textAlign: 'center',
          }} className="vm-pulse">
            Retrieving more passages…
          </div>
        )}
      </div>
    </aside>
  )
}
