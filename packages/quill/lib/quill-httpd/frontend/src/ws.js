// WebSocket client with reconnection and message dispatch.

export class WsClient {
  constructor(store) {
    this.store = store;
    this.ws = null;
    this.reconnectDelay = 1000;
    this._pendingCompletions = new Map();
    this._pendingTypeAt = new Map();
    this._pendingDiagnostics = new Map();
    this._requestCounter = 0;
    this._sourceDebounceTimers = new Map();
  }

  connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws`;
    this.ws = new WebSocket(url);
    this.ws.onopen = () => {
      const wasDisconnected = this.reconnectDelay > 1000;
      this.reconnectDelay = 1000;
      this.store.setConnectionStatus('connected');
      if (wasDisconnected) {
        this.store.emit('reconnected');
      }
    };
    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.debug('[ws]', msg.type, msg.cell_id || '');
        this._onMessage(msg);
      } catch (err) {
        console.error('[ws] message error:', err, event.data.slice(0, 200));
      }
    };
    this.ws.onclose = () => {
      this.ws = null;
      this.store.setConnectionStatus('disconnected');
      setTimeout(() => this.reconnect(), this.reconnectDelay);
    };
    this.ws.onerror = () => {
      if (this.ws) this.ws.close();
    };
  }

  reconnect() {
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
    this.connect();
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  _onMessage(msg) {
    switch (msg.type) {
      case 'notebook':
        this.store.loadNotebook(msg);
        break;
      case 'cell_status':
        this.store.setCellStatus(msg.cell_id, msg.status);
        break;
      case 'cell_output':
        this.store.appendOutput(msg.cell_id, msg.output);
        break;
      case 'cell_updated': {
        // Detect execution completion: cell was running/queued, now idle
        const oldCell = this.store.findCell(msg.cell_id);
        const wasExecuting = oldCell && (oldCell.status === 'running' || oldCell.status === 'queued');
        this.store.updateCell(msg.cell_id, msg.cell);
        if (wasExecuting && msg.cell.status === 'idle') {
          const hasError = msg.cell.outputs && msg.cell.outputs.some(o => o.kind === 'error');
          this.store.finishExecution(msg.cell_id, !hasError);
        }
        break;
      }
      case 'cell_inserted':
        this.store.insertCell(msg.pos, msg.cell);
        break;
      case 'cell_deleted':
        this.store.deleteCell(msg.cell_id);
        break;
      case 'cell_moved':
        this.store.moveCell(msg.cell_id, msg.pos);
        break;
      case 'completions': {
        const resolve = this._pendingCompletions.get(msg.request_id);
        if (resolve) {
          this._pendingCompletions.delete(msg.request_id);
          resolve(msg.items);
        }
        break;
      }
      case 'type_at': {
        const resolve = this._pendingTypeAt.get(msg.request_id);
        if (resolve) {
          this._pendingTypeAt.delete(msg.request_id);
          resolve(msg);
        }
        break;
      }
      case 'diagnostics': {
        const resolve = this._pendingDiagnostics.get(msg.request_id);
        if (resolve) {
          this._pendingDiagnostics.delete(msg.request_id);
          resolve(msg);
        }
        break;
      }
      case 'saved':
        this.store.emit('saved');
        break;
      case 'undo_redo':
        this.store.setUndoRedo(msg.can_undo, msg.can_redo);
        break;
      case 'error':
        this.store.emit('error', { message: msg.message });
        break;
    }
  }

  // --- Commands ---

  updateSource(cellId, source) {
    // Debounce: wait 150ms after last keystroke
    const existing = this._sourceDebounceTimers.get(cellId);
    if (existing) clearTimeout(existing);
    this._sourceDebounceTimers.set(cellId, setTimeout(() => {
      this._sourceDebounceTimers.delete(cellId);
      this.send({ type: 'update_source', cell_id: cellId, source });
    }, 150));
  }

  /** Cancel a pending debounced source update (caller sends explicitly). */
  cancelPendingSource(cellId) {
    const existing = this._sourceDebounceTimers.get(cellId);
    if (existing) {
      clearTimeout(existing);
      this._sourceDebounceTimers.delete(cellId);
    }
  }

  checkpoint() { this.send({ type: 'checkpoint' }); }

  executeCell(cellId) {
    this.cancelPendingSource(cellId);
    const cell = this.store.findCell(cellId);
    if (cell) {
      this.send({ type: 'update_source', cell_id: cellId, source: cell.source });
    }
    this.send({ type: 'execute_cell', cell_id: cellId });
  }

  executeCells(cellIds) { this.send({ type: 'execute_cells', cell_ids: cellIds }); }
  executeAll() { this.send({ type: 'execute_all' }); }
  interrupt() { this.send({ type: 'interrupt' }); }

  insertCell(pos, kind) { this.send({ type: 'insert_cell', pos, kind }); }
  deleteCell(cellId) { this.send({ type: 'delete_cell', cell_id: cellId }); }
  moveCell(cellId, pos) { this.send({ type: 'move_cell', cell_id: cellId, pos }); }
  setCellKind(cellId, kind) { this.send({ type: 'set_cell_kind', cell_id: cellId, kind }); }

  clearOutputs(cellId) { this.send({ type: 'clear_outputs', cell_id: cellId }); }
  clearAllOutputs() { this.send({ type: 'clear_all_outputs' }); }

  save() { this.send({ type: 'save' }); }
  undo() { this.send({ type: 'undo' }); }
  redo() { this.send({ type: 'redo' }); }

  complete(code, pos) {
    const requestId = `req_${++this._requestCounter}`;
    return new Promise((resolve) => {
      this._pendingCompletions.set(requestId, resolve);
      this.send({ type: 'complete', request_id: requestId, code, pos });
      setTimeout(() => {
        if (this._pendingCompletions.has(requestId)) {
          this._pendingCompletions.delete(requestId);
          resolve([]);
        }
      }, 3000);
    });
  }

  typeAt(code, pos) {
    const requestId = `req_${++this._requestCounter}`;
    return new Promise((resolve) => {
      this._pendingTypeAt.set(requestId, resolve);
      this.send({ type: 'type_at', request_id: requestId, code, pos });
      setTimeout(() => {
        if (this._pendingTypeAt.has(requestId)) {
          this._pendingTypeAt.delete(requestId);
          resolve(null);
        }
      }, 3000);
    });
  }

  diagnostics(code) {
    const requestId = `req_${++this._requestCounter}`;
    return new Promise((resolve) => {
      this._pendingDiagnostics.set(requestId, resolve);
      this.send({ type: 'diagnostics', request_id: requestId, code });
      setTimeout(() => {
        if (this._pendingDiagnostics.has(requestId)) {
          this._pendingDiagnostics.delete(requestId);
          resolve({ items: [] });
        }
      }, 3000);
    });
  }
}
