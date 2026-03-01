// State container with event emitter for the notebook.

export class Store {
  constructor() {
    this.cells = [];
    this.focusedCellId = null;
    this.kernelStatus = 'connecting';
    this.canUndo = false;
    this.canRedo = false;
    this.loaded = false;
    this._listeners = new Map();
  }

  on(event, fn) {
    if (!this._listeners.has(event)) this._listeners.set(event, []);
    this._listeners.get(event).push(fn);
  }

  off(event, fn) {
    const list = this._listeners.get(event);
    if (list) {
      const idx = list.indexOf(fn);
      if (idx !== -1) list.splice(idx, 1);
    }
  }

  emit(event, data) {
    const list = this._listeners.get(event);
    if (list) list.forEach(fn => fn(data));
  }

  // --- Mutations ---

  loadNotebook(data) {
    this.cells = data.cells;
    this.canUndo = data.can_undo;
    this.canRedo = data.can_redo;
    this.loaded = true;
    if (!this.focusedCellId || !this.cells.find(c => c.id === this.focusedCellId)) {
      this.focusedCellId = this.cells.length > 0 ? this.cells[0].id : null;
    }
    this.emit('notebook:loaded', this.cells);
  }

  findCell(id) {
    return this.cells.find(c => c.id === id);
  }

  findCellIndex(id) {
    return this.cells.findIndex(c => c.id === id);
  }

  setCellStatus(cellId, status) {
    const cell = this.findCell(cellId);
    if (cell) {
      cell.status = status;
      this.emit('cell:status', { cellId, status });
    }
  }

  finishExecution(cellId, success) {
    const cell = this.findCell(cellId);
    if (cell) {
      cell.lastRunSuccess = success;
      cell.status = 'idle';
      this.emit('cell:execution-done', { cellId, success });
    }
  }

  appendOutput(cellId, output) {
    const cell = this.findCell(cellId);
    if (cell) {
      if (!cell.outputs) cell.outputs = [];
      cell.outputs.push(output);
      this.emit('cell:output', { cellId, output });
    }
  }

  clearOutputs(cellId) {
    const cell = this.findCell(cellId);
    if (cell && cell.outputs) {
      cell.outputs = [];
      this.emit('cell:outputs-cleared', { cellId });
    }
  }

  updateCell(cellId, cellData) {
    const idx = this.findCellIndex(cellId);
    if (idx !== -1) {
      // Preserve lastRunSuccess from the old cell
      const oldCell = this.cells[idx];
      if (oldCell.lastRunSuccess !== undefined) {
        cellData.lastRunSuccess = oldCell.lastRunSuccess;
      }
      this.cells[idx] = cellData;
      this.emit('cell:updated', { cellId, cell: cellData });
    }
  }

  insertCell(pos, cell) {
    this.cells.splice(pos, 0, cell);
    this.emit('cell:inserted', { pos, cell });
  }

  deleteCell(cellId) {
    const idx = this.findCellIndex(cellId);
    if (idx !== -1) {
      this.cells.splice(idx, 1);
      this.emit('cell:deleted', { cellId });
      // Update focus
      if (this.focusedCellId === cellId) {
        if (this.cells.length > 0) {
          const newIdx = Math.min(idx, this.cells.length - 1);
          this.setFocus(this.cells[newIdx].id);
        } else {
          this.focusedCellId = null;
        }
      }
    }
  }

  moveCell(cellId, pos) {
    const oldIdx = this.findCellIndex(cellId);
    if (oldIdx !== -1) {
      const [cell] = this.cells.splice(oldIdx, 1);
      this.cells.splice(pos, 0, cell);
      this.emit('cell:moved', { cellId, pos });
    }
  }

  setUndoRedo(canUndo, canRedo) {
    this.canUndo = canUndo;
    this.canRedo = canRedo;
    this.emit('undo-redo:changed', { canUndo, canRedo });
  }

  setConnectionStatus(status) {
    this.kernelStatus = status;
    this.emit('connection:changed', { status });
  }

  setFocus(cellId) {
    const prev = this.focusedCellId;
    this.focusedCellId = cellId;
    this.emit('focus:changed', { cellId, prevCellId: prev });
  }

  focusNext() {
    const idx = this.findCellIndex(this.focusedCellId);
    if (idx < this.cells.length - 1) {
      this.setFocus(this.cells[idx + 1].id);
    }
  }

  focusPrev() {
    const idx = this.findCellIndex(this.focusedCellId);
    if (idx > 0) {
      this.setFocus(this.cells[idx - 1].id);
    }
  }
}
