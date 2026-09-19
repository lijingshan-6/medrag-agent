// One active connection across all hook instances (composer and suggestions).
let active = null

export function cancelStream() {
  if (!active) return
  const previous = active
  active = null
  clearTimeout(previous.timer)
  previous.socket.close()
}

export function connectStream(url, request, handlers, Socket = WebSocket) {
  cancelStream()
  const socket = new Socket(url)
  const connection = { socket, finished: false, timer: null }
  active = connection
  function fail(message) {
    if (active !== connection || connection.finished) return
    connection.finished = true
    clearTimeout(connection.timer)
    handlers.onError(message)
    socket.close()
  }
  connection.timer = setTimeout(() => fail('The answer timed out. Please try again.'), 310_000)
  socket.onopen = () => {
    if (active === connection) socket.send(JSON.stringify(request))
  }
  socket.onmessage = (message) => {
    if (active !== connection || connection.finished) return
    let event
    try { event = JSON.parse(message.data) } catch { fail('The server sent an invalid response.'); return }
    if (event.event === 'done' || event.event === 'error') {
      connection.finished = true
      clearTimeout(connection.timer)
    }
    handlers.onEvent(event)
  }
  socket.onerror = () => fail('Cannot connect to the backend. Check that the API is running.')
  socket.onclose = () => {
    if (active !== connection) return
    if (!connection.finished) fail('The connection closed before an answer was ready. Please try again.')
    clearTimeout(connection.timer)
    active = null
  }
  return socket
}
