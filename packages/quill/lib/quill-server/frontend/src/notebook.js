// Notebook renderer: manages cells, sections, and dividers.

import { createCellElement, destroyCell, updateCellStatus, clearCellOutputs } from './cell.js';
import { appendOutputToContainer } from './output.js';

export class NotebookRenderer {
  constructor(container, store, wsClient) {
    this.container = container;
    this.store = store;
    this.wsClient = wsClient;
    this._showSkeleton();
    this._bindEvents();
  }

  _showSkeleton() {
    this.container.innerHTML = '';
    const skeleton = document.createElement('div');
    skeleton.className = 'skeleton';
    for (let i = 0; i < 4; i++) {
      const block = document.createElement('div');
      block.className = 'skeleton-cell';
      block.style.animationDelay = `${i * 0.1}s`;
      // Vary heights to suggest different cell types
      if (i === 0) block.classList.add('skeleton-cell-short');
      skeleton.appendChild(block);
    }
    this.container.appendChild(skeleton);
  }

  _bindEvents() {
    this.store.on('notebook:loaded', () => this.renderAll());
    this.store.on('cell:status', ({ cellId, status }) => {
      updateCellStatus(cellId, status);
      if (status === 'running') clearCellOutputs(cellId);
    });
    this.store.on('cell:output', ({ cellId, output }) => {
      const container = document.querySelector(`[data-cell-id="${cellId}"] .cell-outputs`);
      if (container) appendOutputToContainer(container, output);
    });
    this.store.on('cell:outputs-cleared', ({ cellId }) => {
      clearCellOutputs(cellId);
    });
    this.store.on('cell:updated', ({ cellId, cell }) => {
      this._replaceCell(cellId, cell);
    });
    this.store.on('cell:inserted', ({ pos, cell }) => {
      this._insertCellAt(pos, cell);
    });
    this.store.on('cell:deleted', ({ cellId }) => {
      this._removeCell(cellId);
    });
    this.store.on('cell:moved', () => {
      // Re-render all on move for simplicity
      this.renderAll();
    });
    this.store.on('cell:execution-done', ({ cellId, success }) => {
      const el = document.querySelector(`[data-cell-id="${cellId}"]`);
      if (el) {
        el.dataset.status = 'idle';
        // Flash animation for completion feedback
        const cls = success ? 'flash-success' : 'flash-error';
        el.classList.add(cls);
        el.addEventListener('animationend', () => el.classList.remove(cls), { once: true });
        // Update gutter indicator
        const icon = el.querySelector('.cell-status-icon');
        if (icon) icon.dataset.result = success ? 'success' : 'error';
        // Update execution count
        const numSpan = el.querySelector('.cell-number');
        const cell = this.store.findCell(cellId);
        if (numSpan && cell && cell.execution_count > 0) {
          numSpan.textContent = `[${cell.execution_count}]`;
        }
      }
    });
    this.store.on('focus:changed', ({ cellId, prevCellId }) => {
      if (prevCellId) {
        const prev = document.querySelector(`[data-cell-id="${prevCellId}"]`);
        if (prev) prev.classList.remove('focused');
      }
      if (cellId) {
        const curr = document.querySelector(`[data-cell-id="${cellId}"]`);
        if (curr) {
          curr.classList.add('focused');
          curr.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
      }
    });
  }

  renderAll() {
    // Destroy existing editors
    this.container.querySelectorAll('[data-cell-id]').forEach(el => {
      destroyCell(el.dataset.cellId);
    });
    this.container.innerHTML = '';

    // Empty state
    if (this.store.cells.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty-state';
      empty.innerHTML = `
        <div class="empty-state-icon">&laquo; &raquo;</div>
        <p>No cells yet.</p>
        <p class="empty-state-hint">
          Press <kbd>a</kbd> to add a code cell, or <kbd>t</kbd> for text.
        </p>
      `;
      this.container.appendChild(empty);
      return;
    }

    // Group cells into sections
    const sections = this._groupIntoSections(this.store.cells);

    for (const section of sections) {
      const sectionEl = this._createSection(section);
      this.container.appendChild(sectionEl);
    }

    // Add final divider
    this.container.appendChild(this._createDivider(this.store.cells.length));

    // Re-append chapter navigation if present (book mode)
    if (window._quillChapterNavEl) {
      this.container.appendChild(window._quillChapterNavEl);
    }

    // Apply focus
    if (this.store.focusedCellId) {
      const el = document.querySelector(`[data-cell-id="${this.store.focusedCellId}"]`);
      if (el) el.classList.add('focused');
    }
  }

  _splitRenderedHtml(html) {
    // Split server-rendered HTML by h1/h2 headings, returning an array of
    // HTML fragments.  Each fragment starts at a heading and runs until the
    // next heading (or the end).  Any content before the first heading is
    // returned as the first element with index -1.
    if (!html) return [];
    const container = document.createElement('div');
    container.innerHTML = html;
    const parts = [];
    let buf = [];
    let seenHeading = false;
    for (const node of Array.from(container.childNodes)) {
      const tag = node.nodeName;
      if (tag === 'H1' || tag === 'H2') {
        if (buf.length > 0) {
          parts.push(buf.map(n => n.outerHTML || n.textContent).join(''));
        }
        buf = [node];
        seenHeading = true;
      } else {
        if (!seenHeading && buf.length === 0 && parts.length === 0) {
          // Content before first heading
          buf.push(node);
        } else {
          buf.push(node);
        }
      }
    }
    if (buf.length > 0) {
      parts.push(buf.map(n => n.outerHTML || n.textContent).join(''));
    }
    return parts;
  }

  _groupIntoSections(cells) {
    const sections = [];
    let current = { name: null, cells: [] };
    const headingRe = /^#{1,2}\s+(.+)/gm;

    for (const cell of cells) {
      if (cell.kind === 'text' && cell.source) {
        // Find all headings and their positions in this cell
        const headings = [];
        let m;
        while ((m = headingRe.exec(cell.source)) !== null) {
          headings.push({ name: m[1].trim(), index: m.index });
        }
        headingRe.lastIndex = 0;

        if (headings.length <= 1) {
          // Zero or one heading — original behavior
          const match = headings[0];
          if (match && current.cells.length > 0) {
            sections.push(current);
            current = { name: match.name, cells: [] };
          } else if (match && current.cells.length === 0) {
            current.name = match.name;
          }
          current.cells.push(cell);
        } else {
          // Multiple headings — split cell into virtual sub-cells
          // Also split the rendered HTML so each virtual cell gets its own fragment
          const htmlParts = this._splitRenderedHtml(cell.rendered_html);
          // htmlParts[0] may be content before first heading
          let htmlIdx = 0;
          const hasPreHeadingContent = cell.source.slice(0, headings[0].index).trim().length > 0;

          for (let i = 0; i < headings.length; i++) {
            const start = headings[i].index;
            const end = i + 1 < headings.length ? headings[i + 1].index : cell.source.length;
            const source = cell.source.slice(start, end).trim();

            // Text before the first heading stays in the current section
            if (i === 0 && start > 0) {
              const before = cell.source.slice(0, start).trim();
              if (before) {
                const preHtml = hasPreHeadingContent && htmlParts.length > 0 ? htmlParts[htmlIdx++] : null;
                current.cells.push({
                  ...cell, source: before, rendered_html: preHtml,
                  id: cell.id + '_v0', _virtual: true
                });
              }
            }

            if (current.cells.length > 0) {
              sections.push(current);
            }
            current = { name: headings[i].name, cells: [] };
            const partHtml = htmlIdx < htmlParts.length ? htmlParts[htmlIdx++] : null;
            current.cells.push({
              ...cell, source, rendered_html: partHtml,
              id: cell.id + '_v' + (i + 1), _virtual: true
            });
          }
        }
      } else {
        current.cells.push(cell);
      }
    }
    if (current.cells.length > 0 || sections.length === 0) {
      sections.push(current);
    }
    return sections;
  }

  _createSection(section) {
    const sectionEl = document.createElement('section');
    sectionEl.className = 'notebook-section';

    if (section.name) {
      const header = document.createElement('div');
      header.className = 'section-header';
      const toggle = document.createElement('span');
      toggle.className = 'section-toggle';
      toggle.textContent = '\u25BE'; // ▾
      header.appendChild(toggle);
      const title = document.createElement('h2');
      title.textContent = section.name;
      header.appendChild(title);
      header.addEventListener('click', () => {
        const collapsed = sectionEl.dataset.collapsed === 'true';
        sectionEl.dataset.collapsed = collapsed ? 'false' : 'true';
        toggle.textContent = collapsed ? '\u25BE' : '\u25B8'; // ▾ or ▸
      });
      sectionEl.appendChild(header);
    }

    const body = document.createElement('div');
    body.className = 'section-body';

    for (const cell of section.cells) {
      const idx = this.store.findCellIndex(cell.id);
      body.appendChild(this._createDivider(idx));
      body.appendChild(createCellElement(cell, this.wsClient, this.store));
    }

    sectionEl.appendChild(body);
    return sectionEl;
  }

  _createDivider(pos) {
    const div = document.createElement('div');
    div.className = 'cell-divider';
    div.dataset.dividerPos = pos;
    div.innerHTML = `
      <span class="divider-line"></span>
      <span class="divider-buttons">
        <button data-action="add-code" title="Add code cell">+ Code</button>
        <button data-action="add-text" title="Add text cell">+ Text</button>
      </span>
      <span class="divider-line"></span>
    `;
    div.addEventListener('click', (e) => {
      const action = e.target.dataset.action;
      if (!action) return;
      // Compute actual position from DOM order to avoid stale closures
      const dividers = this.container.querySelectorAll('.cell-divider');
      let actualPos = 0;
      for (const d of dividers) {
        if (d === div) break;
        actualPos++;
      }
      if (action === 'add-code') this.wsClient.insertCell(actualPos, 'code');
      else if (action === 'add-text') this.wsClient.insertCell(actualPos, 'text');
    });
    return div;
  }

  _replaceCell(cellId, cellData) {
    const el = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (!el) return;
    destroyCell(cellId);
    const newEl = createCellElement(cellData, this.wsClient, this.store);
    if (cellId === this.store.focusedCellId) newEl.classList.add('focused');
    el.replaceWith(newEl);
  }

  _insertCellAt(pos, cell) {
    // Re-render all for simplicity on insert
    this.renderAll();
    this.store.setFocus(cell.id);
  }

  _removeCell(cellId) {
    const el = document.querySelector(`[data-cell-id="${cellId}"]`);
    if (el) {
      destroyCell(cellId);
      // Also remove the preceding divider
      const prev = el.previousElementSibling;
      if (prev && prev.classList.contains('cell-divider')) prev.remove();
      el.remove();
    }
    // Show empty state if no cells left
    if (this.store.cells.length === 0) {
      this.renderAll();
    }
  }
}
