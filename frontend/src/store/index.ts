import { create } from 'zustand'
import type { ChunkOut } from '../types'
import type { AnswerOut, TimelineNode } from '../types/ws'

export interface AppState {
  // Session
  threadId: string
  pipeline: 'p2' | 'p3'
  setThreadId: (id: string) => void
  setPipeline: (p: 'p2' | 'p3') => void

  // Query
  query: string
  setQuery: (q: string) => void

  // Active query (snapshot at send time — used for question echo + thread title)
  activeQuery: string
  setActiveQuery: (q: string) => void

  // Streaming state
  isStreaming: boolean
  setStreaming: (v: boolean) => void

  // Timeline
  timeline: TimelineNode[]
  setTimeline: (nodes: TimelineNode[]) => void
  updateNode: (name: string, patch: Partial<TimelineNode>) => void
  pushNode: (node: TimelineNode) => void

  // Live chunks arriving during retrieval
  liveChunks: ChunkOut[]
  pushLiveChunk: (chunk: ChunkOut) => void
  clearLiveChunks: () => void

  // Final result
  result: AnswerOut | null
  setResult: (r: AnswerOut | null) => void

  // Selected chunk (for EvidencePanel highlight)
  selectedChunkId: string | null
  setSelectedChunkId: (id: string | null) => void

  // Error message
  errorMessage: string | null
  setErrorMessage: (msg: string | null) => void
}

export const useStore = create<AppState>((set) => ({
  threadId: `session-${Date.now()}`,
  pipeline: 'p2',
  setThreadId: (id) => set({ threadId: id }),
  setPipeline: (p) => set({ pipeline: p }),

  query: '',
  setQuery: (q) => set({ query: q }),

  activeQuery: '',
  setActiveQuery: (q) => set({ activeQuery: q }),

  isStreaming: false,
  setStreaming: (v) => set({ isStreaming: v }),

  timeline: [],
  setTimeline: (nodes) => set({ timeline: nodes }),
  updateNode: (name, patch) =>
    set((s) => ({
      timeline: s.timeline.map((n, i) => {
        if (i !== s.timeline.findLastIndex(node => node.name === name && node.status === "running")) return n
        const elapsed_ms =
          patch.status === 'done' && n.timestamp
            ? Date.now() - n.timestamp
            : n.elapsed_ms
        return { ...n, ...patch, elapsed_ms }
      }),
    })),
  pushNode: (node) => set((s) => ({ timeline: [...s.timeline, node] })),

  liveChunks: [],
  pushLiveChunk: (chunk) => set((s) => ({ liveChunks: [...s.liveChunks, chunk] })),
  clearLiveChunks: () => set({ liveChunks: [] }),

  result: null,
  setResult: (r) => set({ result: r }),

  selectedChunkId: null,
  setSelectedChunkId: (id) => set({ selectedChunkId: id }),

  errorMessage: null,
  setErrorMessage: (msg) => set({ errorMessage: msg }),
}))
