// Entry point: initialize WebSocket, store, and mount notebook.

import { Store } from './store.js';
import { WsClient } from './ws.js';
import { NotebookRenderer } from './notebook.js';
import { initShortcuts } from './shortcuts.js';
import '../css/notebook.css';

const store = new Store();
const wsClient = new WsClient(store);

// Mount notebook renderer
const container = document.getElementById('notebook');
const renderer = new NotebookRenderer(container, store, wsClient);

// --- Toast notification system ---

const toastContainer = document.getElementById('toast-container');

function showToast(message, kind = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  // Trigger reflow so the animation starts from the initial state
  el.offsetHeight;
  el.classList.add('toast-visible');
  setTimeout(() => {
    el.classList.remove('toast-visible');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
    // Fallback removal if transitionend doesn't fire
    setTimeout(() => el.remove(), 500);
  }, 2500);
}

// --- Event wiring ---

// Save feedback
store.on('saved', () => showToast('Notebook saved', 'success'));

// Error surfacing
store.on('error', ({ message }) => showToast(message, 'error'));

// Reconnection feedback
store.on('reconnected', () => showToast('Reconnected', 'success'));

// Delete hint
store.on('cell:deleted', () => showToast('Cell deleted \u2014 Ctrl+Z to undo', 'info'));

// Connection status indicator
const statusDot = document.querySelector('#kernel-status .status-dot');
const statusText = document.querySelector('#kernel-status .status-text');

store.on('connection:changed', ({ status }) => {
  statusDot.dataset.status = status;
  statusText.textContent = status === 'connected' ? 'Connected'
    : status === 'disconnected' ? 'Reconnecting\u2026'
    : 'Connecting\u2026';
});

// Connection banner
const connectionBanner = document.getElementById('connection-banner');
store.on('connection:changed', ({ status }) => {
  if (status === 'disconnected') {
    connectionBanner.hidden = false;
    // Trigger reflow for animation
    connectionBanner.offsetHeight;
    connectionBanner.classList.add('visible');
  } else if (status === 'connected') {
    connectionBanner.classList.remove('visible');
    connectionBanner.addEventListener('transitionend', () => {
      connectionBanner.hidden = true;
    }, { once: true });
    // Fallback
    setTimeout(() => { connectionBanner.hidden = true; }, 500);
  }
});

// --- Toolbar buttons ---

document.getElementById('btn-run-all').addEventListener('click', () => {
  wsClient.executeAll();
});
document.getElementById('btn-interrupt').addEventListener('click', () => {
  wsClient.interrupt();
});
document.getElementById('btn-clear-all').addEventListener('click', () => {
  wsClient.clearAllOutputs();
});
document.getElementById('btn-save').addEventListener('click', () => {
  wsClient.save();
});

// --- Shortcuts dialog ---

const shortcutsDialog = document.getElementById('shortcuts-dialog');
const shortcutsClose = document.getElementById('shortcuts-close');

function toggleShortcutsDialog() {
  shortcutsDialog.hidden = !shortcutsDialog.hidden;
}

document.getElementById('btn-help').addEventListener('click', toggleShortcutsDialog);
shortcutsClose.addEventListener('click', toggleShortcutsDialog);
shortcutsDialog.addEventListener('click', (e) => {
  // Close on backdrop click
  if (e.target === shortcutsDialog) toggleShortcutsDialog();
});

// Export for shortcuts.js
window._quillToggleShortcuts = toggleShortcutsDialog;

// Init keyboard shortcuts
initShortcuts(store, wsClient);

// Connect WebSocket
wsClient.connect();
