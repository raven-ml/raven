// Cell renderer for code and text cells.

import { createEditor } from './editor.js';
import { appendOutputToContainer, renderOutput } from './output.js';

const editors = new Map(); // cellId -> EditorView

export function getEditor(cellId) {
  return editors.get(cellId);
}

export function createCellElement(cell, wsClient, store) {
  const el = document.createElement('div');
  el.className = `cell cell-${cell.kind}`;
  el.dataset.cellId = cell.id;
  el.dataset.status = cell.status || 'idle';

  if (cell.kind === 'code') {
    el.appendChild(createCodeCell(cell, wsClient, store));
  } else {
    el.appendChild(createTextCell(cell, wsClient, store));
  }

  // Focus on click — any click on the cell sets notebook-level focus
  el.addEventListener('click', (e) => {
    if (!e.target.closest('button')) {
      store.setFocus(cell.id);
    }
  });

  return el;
}

function createCodeCell(cell, wsClient, store) {
  const wrapper = document.createElement('div');
  wrapper.className = 'cell-wrapper';

  // Gutter
  const gutter = document.createElement('div');
  gutter.className = 'cell-gutter';
  const numSpan = document.createElement('span');
  numSpan.className = 'cell-number';
  numSpan.textContent = cell.execution_count > 0 ? `[${cell.execution_count}]` : '[ ]';
  gutter.appendChild(numSpan);
  const statusIcon = document.createElement('span');
  statusIcon.className = 'cell-status-icon';
  gutter.appendChild(statusIcon);
  wrapper.appendChild(gutter);

  // Content
  const content = document.createElement('div');
  content.className = 'cell-content';

  // Editor container
  const editorContainer = document.createElement('div');
  editorContainer.className = 'cell-editor';
  content.appendChild(editorContainer);

  // Mount CodeMirror
  const view = createEditor(editorContainer, cell.source, {
    onChange: (source) => {
      // Update local cell source
      cell.source = source;
      wsClient.updateSource(cell.id, source);
    },
    onExecute: () => {
      wsClient.executeCell(cell.id);
    },
    onExecuteAndMoveNext: () => {
      wsClient.executeCell(cell.id);
      store.focusNext();
    },
    onEscape: () => {
      editorContainer.querySelector('.cm-content')?.blur();
    },
    wsClient,
  });
  editors.set(cell.id, view);

  // Actions
  const actions = document.createElement('div');
  actions.className = 'cell-actions';
  actions.innerHTML = `
    <button data-action="run" title="Evaluate cell (Ctrl+Enter)">Run</button>
    <button data-action="delete" title="Delete cell (d)">Delete</button>
    <button data-action="move-up" title="Move cell up (Shift+K)">\u2191</button>
    <button data-action="move-down" title="Move cell down (Shift+J)">\u2193</button>
    <button data-action="toggle-type" title="Convert to text cell (m)">\u21C4 Text</button>
  `;
  actions.addEventListener('click', (e) => {
    const action = e.target.dataset.action;
    if (!action) return;
    const idx = store.findCellIndex(cell.id);
    switch (action) {
      case 'run': wsClient.executeCell(cell.id); break;
      case 'delete': wsClient.deleteCell(cell.id); break;
      case 'move-up': if (idx > 0) wsClient.moveCell(cell.id, idx - 1); break;
      case 'move-down': if (idx < store.cells.length - 1) wsClient.moveCell(cell.id, idx + 1); break;
      case 'toggle-type': wsClient.setCellKind(cell.id, 'text'); break;
    }
  });
  content.appendChild(actions);

  // Outputs
  const outputsContainer = document.createElement('div');
  outputsContainer.className = 'cell-outputs';
  if (cell.outputs) {
    for (const output of cell.outputs) {
      const el = renderOutput(output);
      if (el) outputsContainer.appendChild(el);
    }
  }
  content.appendChild(outputsContainer);

  wrapper.appendChild(content);
  return wrapper;
}

function createTextCell(cell, wsClient, store) {
  const wrapper = document.createElement('div');
  wrapper.className = 'cell-wrapper';

  const content = document.createElement('div');
  content.className = 'cell-content';

  // Rendered markdown view
  const markdownView = document.createElement('div');
  markdownView.className = 'cell-markdown';
  markdownView.innerHTML = cell.rendered_html || '<p class="cell-markdown-empty">Empty text cell \u2014 click to edit</p>';
  content.appendChild(markdownView);

  // Editor container (hidden by default)
  const editorContainer = document.createElement('div');
  editorContainer.className = 'cell-editor';
  editorContainer.style.display = 'none';
  content.appendChild(editorContainer);

  let editorView = null;

  // Double-click to edit
  markdownView.addEventListener('dblclick', () => {
    enterEditMode();
  });

  function enterEditMode() {
    markdownView.style.display = 'none';
    editorContainer.style.display = 'block';
    if (!editorView) {
      editorView = createEditor(editorContainer, cell.source, {
        onChange: (source) => {
          cell.source = source;
          wsClient.updateSource(cell.id, source);
        },
        onExecute: () => exitEditMode(),
        onExecuteAndMoveNext: () => { exitEditMode(); store.focusNext(); },
        onEscape: () => exitEditMode(),
        wsClient: null, // No autocomplete for markdown
      });
      editors.set(cell.id, editorView);
    }
    editorView.focus();
  }

  function exitEditMode() {
    editorContainer.style.display = 'none';
    markdownView.style.display = 'block';
    // Source was already sent via debounced updateSource
    // The server will send back cell_updated with fresh rendered_html
    wsClient.checkpoint();
  }

  // Actions
  const actions = document.createElement('div');
  actions.className = 'cell-actions';
  actions.innerHTML = `
    <button data-action="edit" title="Edit text (Enter)">Edit</button>
    <button data-action="delete" title="Delete cell (d)">Delete</button>
    <button data-action="move-up" title="Move cell up (Shift+K)">\u2191</button>
    <button data-action="move-down" title="Move cell down (Shift+J)">\u2193</button>
    <button data-action="toggle-type" title="Convert to code cell (m)">\u21C4 Code</button>
  `;
  actions.addEventListener('click', (e) => {
    const action = e.target.dataset.action;
    if (!action) return;
    const idx = store.findCellIndex(cell.id);
    switch (action) {
      case 'edit': enterEditMode(); break;
      case 'delete': wsClient.deleteCell(cell.id); break;
      case 'move-up': if (idx > 0) wsClient.moveCell(cell.id, idx - 1); break;
      case 'move-down': if (idx < store.cells.length - 1) wsClient.moveCell(cell.id, idx + 1); break;
      case 'toggle-type': wsClient.setCellKind(cell.id, 'code'); break;
    }
  });
  content.appendChild(actions);

  wrapper.appendChild(content);
  return wrapper;
}

export function destroyCell(cellId) {
  const view = editors.get(cellId);
  if (view) {
    view.destroy();
    editors.delete(cellId);
  }
}

export function updateCellStatus(cellId, status) {
  const el = document.querySelector(`[data-cell-id="${cellId}"]`);
  if (el) {
    el.dataset.status = status;
    // Clear previous result indicator when starting new execution
    if (status === 'running' || status === 'queued') {
      const icon = el.querySelector('.cell-status-icon');
      if (icon) delete icon.dataset.result;
    }
  }
}

export function clearCellOutputs(cellId) {
  const el = document.querySelector(`[data-cell-id="${cellId}"] .cell-outputs`);
  if (el) el.innerHTML = '';
}
