import test from 'node:test'
import assert from 'node:assert/strict'
import { connectStream, cancelStream } from '../src/api/streamConnection.js'

class Socket {
  send() {}
  close() { this.closed = true }
  event(event) { this.onmessage({ data: JSON.stringify(event) }) }
}

test('replaced and cancelled requests cannot update the active result', () => {
  const events = []
  const handlers = { onEvent: (e) => events.push(e), onError: (e) => events.push(e) }
  const old = connectStream('ws://fixture', {}, handlers, Socket)
  const current = connectStream('ws://fixture', {}, handlers, Socket)
  old.event({ event: 'done' })
  old.onclose()
  assert.equal(old.closed, true)
  assert.deepEqual(events, [])
  current.event({ event: 'done' })
  assert.equal(events.length, 1)
  cancelStream()
  current.event({ event: 'done' })
  assert.equal(events.length, 1)
})

test('unexpected close yields one failure and error prevents later success', () => {
  const events = []
  const handlers = { onEvent: (e) => events.push(e.event), onError: () => events.push('connection-error') }
  const socket = connectStream('ws://fixture', {}, handlers, Socket)
  socket.onclose()
  assert.deepEqual(events, ['connection-error'])
  const next = connectStream('ws://fixture', {}, handlers, Socket)
  next.event({ event: 'error' })
  next.event({ event: 'done' })
  next.onclose()
  assert.deepEqual(events, ['connection-error', 'error'])
  cancelStream()
})
