// Enhancements only: every page is complete without this file.

document.documentElement.classList.add('js');

document.addEventListener('DOMContentLoaded', () => {
  // A copy button after each [data-copy] element. The attribute's value is
  // what is copied; when it is empty, the element's text is.
  for (const el of document.querySelectorAll('[data-copy]')) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy';
    button.textContent = 'copy';
    const status = document.createElement('span');
    status.className = 'sr';
    status.setAttribute('role', 'status');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(el.dataset.copy || el.innerText);
        button.textContent = 'copied';
        status.textContent = 'Copied to clipboard';
      } catch {
        button.textContent = 'copy failed';
        status.textContent = 'Could not copy';
      }
      setTimeout(() => { button.textContent = 'copy'; status.textContent = ''; }, 1600);
    });
    el.after(button, status);
  }

  // A copy button in the caption of each code figure that offers one: it
  // copies the figure's code.
  for (const figure of document.querySelectorAll('figure[data-copy-code]')) {
    const pre = figure.querySelector('pre');
    const caption = figure.querySelector('figcaption');
    if (!pre || !caption) continue;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy';
    button.textContent = 'copy';
    button.setAttribute('aria-label', 'copy code');
    const status = document.createElement('span');
    status.className = 'sr';
    status.setAttribute('role', 'status');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        button.textContent = 'copied';
        status.textContent = 'Copied to clipboard';
      } catch {
        button.textContent = 'copy failed';
        status.textContent = 'Could not copy';
      }
      setTimeout(() => { button.textContent = 'copy'; status.textContent = ''; }, 1600);
    });
    caption.append(button, status);
  }

  // A sidebar taller than the window scrolls on its own: start it with the
  // current page in view.
  for (const nav of document.querySelectorAll('.docs > .sidebar')) {
    const current = nav.querySelector('[aria-current="page"]');
    if (!current || nav.scrollHeight <= nav.clientHeight) continue;
    const offset = current.getBoundingClientRect().top - nav.getBoundingClientRect().top + nav.scrollTop;
    nav.scrollTop = Math.max(0, offset - nav.clientHeight / 3);
  }

  // The index, A to Z: the query filters it. Matches show ranked, best
  // first: a name that is the query, then one that starts with it, then
  // one that contains it, then any entry that mentions it.
  const index = document.querySelector('[data-index]');
  const field = document.getElementById('sq');
  if (index && field) {
    const status = document.getElementById('sq-status');
    const results = document.getElementById('az-results');
    const none = document.getElementById('az-none');
    const items = [...index.querySelectorAll('li[data-k]')].map((li) => ({ li, n: li.dataset.n, k: li.dataset.k }));
    const score = (it, words) => {
      let s = 0;
      for (const w of words) {
        if (!it.k.includes(w)) return -1;
        s += it.n === w ? 0 : it.n.startsWith(w) ? 1 : it.n.includes(w) ? 2 : 3;
      }
      return s;
    };
    const run = () => {
      const q = field.value.trim().toLowerCase();
      const words = q.split(/\s+/).filter(Boolean);
      const url = new URL(location.href);
      if (q) url.searchParams.set('q', field.value.trim()); else url.searchParams.delete('q');
      history.replaceState(null, '', url);
      if (!words.length) {
        index.hidden = false;
        results.hidden = true;
        none.hidden = true;
        status.textContent = `${items.length} entries`;
        return;
      }
      const found = items
        .map((it) => ({ it, s: score(it, words) }))
        .filter((r) => r.s >= 0)
        .sort((a, b) => a.s - b.s || a.it.n.length - b.it.n.length);
      index.hidden = true;
      results.replaceChildren(...found.slice(0, 100).map((r) => r.it.li.cloneNode(true)));
      results.hidden = found.length === 0;
      none.hidden = found.length > 0;
      status.textContent = found.length > 100 ? `${found.length} matches, the first 100 shown` : `${found.length} ${found.length === 1 ? 'match' : 'matches'}`;
    };
    field.value = new URLSearchParams(location.search).get('q') || '';
    let pending = 0;
    field.addEventListener('input', () => { cancelAnimationFrame(pending); pending = requestAnimationFrame(run); });
    run();
  }

  // Tabs, following the WAI-ARIA tabs pattern. Each .panel's heading
  // becomes its tab; the headings stay as they are when this doesn't run.
  for (const tabs of document.querySelectorAll('.tabs')) {
    const panels = [...tabs.querySelectorAll('.panel')];
    const list = document.createElement('div');
    list.setAttribute('role', 'tablist');
    list.setAttribute('aria-label', tabs.getAttribute('aria-label'));
    const buttons = panels.map((panel) => {
      const heading = panel.querySelector('.panel-h');
      const tab = document.createElement('button');
      tab.type = 'button';
      tab.id = panel.id + '-tab';
      tab.setAttribute('role', 'tab');
      tab.setAttribute('aria-controls', panel.id);
      tab.innerHTML = heading.innerHTML;
      heading.hidden = true;
      panel.setAttribute('role', 'tabpanel');
      panel.setAttribute('aria-labelledby', tab.id);
      list.append(tab);
      return tab;
    });
    const select = (i, focus) => {
      buttons.forEach((tab, j) => {
        tab.setAttribute('aria-selected', String(i === j));
        tab.tabIndex = i === j ? 0 : -1;
        panels[j].classList.toggle('off', i !== j);
      });
      if (focus) buttons[i].focus();
    };
    buttons.forEach((tab, i) => tab.addEventListener('click', () => select(i, false)));
    list.addEventListener('keydown', (e) => {
      const i = buttons.indexOf(document.activeElement);
      const last = buttons.length - 1;
      const next = { ArrowRight: i === last ? 0 : i + 1, ArrowLeft: i === 0 ? last : i - 1, Home: 0, End: last }[e.key];
      if (next === undefined) return;
      e.preventDefault();
      select(next, true);
    });
    tabs.prepend(list);
    select(0, false);
  }
});
