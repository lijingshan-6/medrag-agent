import { useCallback } from 'react'
import { AgentTimeline } from '../components/AgentTimeline'
import { AnswerPanel } from '../components/AnswerPanel'
import { EvidencePanel } from '../components/EvidencePanel'
import { QueryInput } from '../components/QueryInput'
import { useAgentStream } from '../hooks/useAgentStream'
import { useStore } from '../store'

const SUGGESTED_QUERIES = [
  'What kinds of data does the fastMRI knee dataset provide?',
  'How does fastMRI+ extend fastMRI for imaging research?',
  'Does this evidence establish which treatment is best for an individual patient?',
]

export function AnswerPage() {
  const { setQuery, setSelectedChunkId, activeQuery, result } = useStore()
  const { send } = useAgentStream()

  const handleCiteClick = useCallback((citation: string) => {
    const chunks = result?.chunks ?? []
    const idx = chunks.findIndex((c) => c.citation === citation)
    if (idx >= 0) {
      const id = chunks[idx].chunk_id
      setSelectedChunkId(id)
      document.getElementById(`chunk-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result, setSelectedChunkId])

  const handlePickQuery = useCallback((q: string) => {
    setQuery(q)
    send(q)
  }, [setQuery, send])

  return (
    <div className="vm-answer-page" style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div className="vm-answer-grid" style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '320px minmax(0, 1fr) 370px',
        overflow: 'hidden',
      }}>
        {/* Left: Reasoning trace */}
        <AgentTimeline />

        {/* Center: Answer */}
        <div style={{ overflow: 'hidden', borderRight: '1px solid var(--rule)' }}>
          <AnswerPanel
            query={activeQuery}
            suggestedQueries={SUGGESTED_QUERIES}
            onCiteClick={handleCiteClick}
            onPickQuery={handlePickQuery}
          />
        </div>

        {/* Right: Evidence */}
        <EvidencePanel />
      </div>

      {/* Bottom: Composer */}
      <QueryInput />
    </div>
  )
}
