// Keyboard shortcut handler for global notebook navigation.

export function initShortcuts(store, wsClient) {
  document.addEventListener('keydown', (e) => {
    // Close shortcuts dialog on Escape
    const dialog = document.getElementById('shortcuts-dialog');
    if (e.key === 'Escape' && dialog && !dialog.hidden) {
      e.preventDefault();
      dialog.hidden = true;
      return;
    }

    // Skip if inside a CodeMirror editor
    if (e.target.closest('.cm-content')) return;
    // Skip if inside an input/textarea
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const ctrl = e.ctrlKey || e.metaKey;

    // Ctrl/Cmd shortcuts
    if (ctrl) {
      switch (e.key) {
        case 's':
          e.preventDefault();
          wsClient.save();
          return;
        case 'z':
          e.preventDefault();
          if (e.shiftKey) wsClient.redo();
          else wsClient.undo();
          return;
        case 'Enter':
          e.preventDefault();
          if (e.shiftKey) wsClient.executeAll();
          else if (store.focusedCellId) wsClient.executeCell(store.focusedCellId);
          return;
      }
    }

    // Navigation and cell management (no modifier)
    if (!ctrl && !e.altKey) {
      switch (e.key) {
        case '?':
          e.preventDefault();
          if (window._quillToggleShortcuts) window._quillToggleShortcuts();
          return;
        case 'j':
        case 'ArrowDown':
          e.preventDefault();
          store.focusNext();
          return;
        case 'k':
        case 'ArrowUp':
          e.preventDefault();
          store.focusPrev();
          return;
        case 'J':
          e.preventDefault();
          {
            const idx = store.findCellIndex(store.focusedCellId);
            if (idx >= 0 && idx < store.cells.length - 1) {
              wsClient.moveCell(store.focusedCellId, idx + 1);
            }
          }
          return;
        case 'K':
          e.preventDefault();
          {
            const idx = store.findCellIndex(store.focusedCellId);
            if (idx > 0) {
              wsClient.moveCell(store.focusedCellId, idx - 1);
            }
          }
          return;
        case 'Enter':
          e.preventDefault();
          if (e.shiftKey) {
            // Shift+Enter: execute and move next
            if (store.focusedCellId) {
              wsClient.executeCell(store.focusedCellId);
              store.focusNext();
            }
          } else {
            // Enter: focus the editor in the focused cell
            if (store.focusedCellId) {
              const el = document.querySelector(`[data-cell-id="${store.focusedCellId}"] .cm-content`);
              if (el) el.focus();
              else {
                // For text cells, trigger edit mode
                const markdown = document.querySelector(`[data-cell-id="${store.focusedCellId}"] .cell-markdown`);
                if (markdown) markdown.dispatchEvent(new Event('dblclick'));
              }
            }
          }
          return;
        case 'a':
          e.preventDefault();
          {
            const idx = store.focusedCellId ? store.findCellIndex(store.focusedCellId) + 1 : store.cells.length;
            wsClient.insertCell(idx, 'code');
          }
          return;
        case 't':
          e.preventDefault();
          {
            const idx = store.focusedCellId ? store.findCellIndex(store.focusedCellId) + 1 : store.cells.length;
            wsClient.insertCell(idx, 'text');
          }
          return;
        case 'd':
          e.preventDefault();
          if (store.focusedCellId) wsClient.deleteCell(store.focusedCellId);
          return;
        case 'm':
          e.preventDefault();
          if (store.focusedCellId) {
            const cell = store.findCell(store.focusedCellId);
            if (cell) {
              wsClient.setCellKind(store.focusedCellId, cell.kind === 'code' ? 'text' : 'code');
            }
          }
          return;
        case 'c':
          e.preventDefault();
          if (store.focusedCellId) {
            const cell = store.findCell(store.focusedCellId);
            if (cell && cell.kind === 'code') {
              wsClient.clearOutputs(store.focusedCellId);
            }
          }
          return;
        case 'z':
          e.preventDefault();
          if (store.focusedCellId) {
            const cell = store.findCell(store.focusedCellId);
            if (cell) {
              const attrs = cell.attrs || {};
              wsClient.setCellAttrs(store.focusedCellId, { ...attrs, collapsed: !attrs.collapsed });
            }
          }
          return;
        case 'Z':
          e.preventDefault();
          if (store.focusedCellId) {
            const cell = store.findCell(store.focusedCellId);
            if (cell && cell.kind === 'code') {
              const attrs = cell.attrs || {};
              wsClient.setCellAttrs(store.focusedCellId, { ...attrs, hide_source: !attrs.hide_source });
            }
          }
          return;
      }
    }
  });
}
