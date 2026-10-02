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

// Click outside cells to unfocus
document.addEventListener('mousedown', (e) => {
  if (!e.target.closest('.cell') && !e.target.closest('.toolbar') && !e.target.closest('.dialog-backdrop')) {
    store.clearFocus();
  }
});

// Init keyboard shortcuts
initShortcuts(store, wsClient);

// --- Sidebar (directory mode) ---

async function initSidebar() {
  try {
    const res = await fetch('/api/notebooks');
    if (!res.ok) return; // Single-file mode, no notebooks
    const chapters = await res.json();
    if (!chapters || chapters.length === 0) return;

    const sidebar = document.getElementById('sidebar');
    const layout = document.getElementById('layout');
    sidebar.hidden = false;
    layout.classList.add('has-sidebar');

    const currentPath = location.pathname;

    for (const entry of chapters) {
      if (entry.type === 'section') {
        const part = document.createElement('div');
        part.className = 'sidebar-part';
        part.textContent = entry.title;
        sidebar.appendChild(part);
      } else if (entry.type === 'separator') {
        const sep = document.createElement('div');
        sep.className = 'sidebar-separator';
        sidebar.appendChild(sep);
      } else if (entry.type === 'notebook') {
        const link = document.createElement('a');
        link.className = 'sidebar-chapter';
        link.href = entry.url;
        link.textContent = entry.title;
        // Highlight active notebook
        if (currentPath === entry.url || currentPath === entry.url.replace(/\/$/, '')) {
          link.classList.add('active');
        }
        sidebar.appendChild(link);
      }
    }

    // When visiting '/', highlight the first notebook
    if (currentPath === '/' && !sidebar.querySelector('.sidebar-chapter.active')) {
      const first = sidebar.querySelector('.sidebar-chapter');
      if (first) first.classList.add('active');
    }

    // Find active notebook and compute prev/next
    const chapterEntries = chapters.filter(ch => ch.type === 'notebook');
    let activeIdx = chapterEntries.findIndex(
      ch => currentPath === ch.url || currentPath === ch.url.replace(/\/$/, '')
    );
    if (activeIdx < 0 && currentPath === '/') activeIdx = 0;

    if (activeIdx >= 0) {
      wsClient.chapterPath = chapterEntries[activeIdx].path;

      const prev = activeIdx > 0 ? chapterEntries[activeIdx - 1] : null;
      const next = activeIdx < chapterEntries.length - 1 ? chapterEntries[activeIdx + 1] : null;

      // Expose for keyboard shortcuts
      window._quillChapterNav = {
        prev: prev ? prev.url : null,
        next: next ? next.url : null,
      };

      // Render prev/next navigation after the notebook container
      // (placed outside #notebook so renderAll() doesn't destroy it)
      const nav = document.createElement('div');
      nav.className = 'chapter-nav';

      if (prev) {
        const prevLink = document.createElement('a');
        prevLink.className = 'chapter-nav-link chapter-nav-prev';
        prevLink.href = prev.url;
        prevLink.innerHTML = `<span class="chapter-nav-dir">\u2190 Previous</span><span class="chapter-nav-title">${prev.title}</span>`;
        nav.appendChild(prevLink);
      } else {
        nav.appendChild(document.createElement('span'));
      }

      if (next) {
        const nextLink = document.createElement('a');
        nextLink.className = 'chapter-nav-link chapter-nav-next';
        nextLink.href = next.url;
        nextLink.innerHTML = `<span class="chapter-nav-dir">Next \u2192</span><span class="chapter-nav-title">${next.title}</span>`;
        nav.appendChild(nextLink);
      }

      // Store the nav element globally so the notebook renderer can
      // re-append it after renderAll() clears the container.
      window._quillChapterNavEl = nav;
      container.appendChild(nav);
    }
  } catch {
    // No sidebar in single-file mode
  }
}

// Initialize sidebar (if directory mode), then connect WebSocket
initSidebar().then(() => wsClient.connect());
