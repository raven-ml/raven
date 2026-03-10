(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Default book theme — embedded HTML template and CSS stylesheet.

   Aesthetic: editorial/scholarly with warm paper tones. Designed for long-form
   technical reading with code and figures. System fonts for performance and
   offline use. *)

let style_css =
  {css|:root {
  --bg: #f9f7f3;
  --text: #2d2d2d;
  --text-muted: #6b6560;
  --heading: #1a1a1a;
  --accent: #daa550;
  --accent-hover: #c4933e;
  --link: #b8862e;
  --link-hover: #daa550;
  --border: #d8d2c8;
  --rule: #c8c0b4;
  --sidebar-bg: #1c2128;
  --sidebar-text: #b0b4bc;
  --sidebar-heading: #d8dce4;
  --sidebar-active: #daa550;
  --sidebar-hover: #d0d4dc;
  --sidebar-border: #2c333b;
  --code-bg: #efece6;
  --code-border: #ddd7cd;
  --code-text: #3c3428;
  --output-bg: #e6e1d8;
  --output-border: #d0c9bc;
  --error-bg: #fcecea;
  --error-border: #e4b4ac;
  --error-text: #a43828;
  --stderr-text: #887020;
  --content-width: 48rem;
  --sidebar-width: 17rem;
}

[data-theme="dark"] {
  --bg: #1a1a22;
  --text: #c8ccd4;
  --text-muted: #8a8e96;
  --heading: #e0e4ec;
  --accent: #d4a060;
  --accent-hover: #e0b878;
  --link: #c0946c;
  --link-hover: #d4a878;
  --border: #2c2c38;
  --rule: #3c3c48;
  --sidebar-bg: #141418;
  --sidebar-text: #8a8e96;
  --sidebar-heading: #c8ccd4;
  --sidebar-active: #e4b868;
  --sidebar-hover: #c0c4cc;
  --sidebar-border: #22222c;
  --code-bg: #22222c;
  --code-border: #2c2c38;
  --code-text: #c8ccd4;
  --output-bg: #1e1e28;
  --output-border: #2c2c38;
  --error-bg: #2c1a1a;
  --error-border: #5c2828;
  --error-text: #e87060;
  --stderr-text: #c8a848;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html {
  font-size: 17px;
  -webkit-text-size-adjust: 100%;
  text-size-adjust: 100%;
}

body {
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", Palatino,
    "Book Antiqua", serif;
  color: var(--text);
  background: var(--bg);
  line-height: 1.68;
}

/* ─── Sidebar ─── */

.sidebar {
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  width: var(--sidebar-width);
  background: var(--sidebar-bg);
  color: var(--sidebar-text);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  z-index: 100;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.82rem;
  line-height: 1.45;
  transition: transform 0.25s ease;
}

.sidebar-scrollable {
  overflow-y: auto;
  flex: 1;
  padding: 1.5rem 0 2rem;
  scrollbar-width: thin;
  scrollbar-color: #3c444e transparent;
}

.sidebar-header {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--sidebar-border);
  margin-bottom: 1rem;
}

.sidebar-header a {
  color: var(--sidebar-heading);
  text-decoration: none;
  font-weight: 700;
  font-size: 0.92rem;
  letter-spacing: 0.01em;
}

.toc-part {
  padding: 0.9rem 1.25rem 0.35rem;
  color: var(--sidebar-heading);
  font-weight: 600;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.toc-chapter {
  display: block;
  padding: 0.3rem 1.25rem 0.3rem 1.5rem;
  color: var(--sidebar-text);
  text-decoration: none;
  border-left: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.toc-chapter:hover {
  color: var(--sidebar-hover);
}

.toc-chapter.active {
  color: var(--sidebar-active);
  border-left-color: var(--sidebar-active);
  font-weight: 600;
}

.toc-separator {
  height: 1px;
  background: var(--sidebar-border);
  margin: 0.8rem 1.25rem;
}

.menu-btn {
  display: none;
}

/* ─── Main content ─── */

main {
  margin-left: var(--sidebar-width);
  min-height: 100vh;
}

article {
  max-width: var(--content-width);
  width: 100%;
  margin: 0 auto;
  padding: 2.5rem 2rem 3rem;
}

/* ─── Typography ─── */

h1, h2, h3, h4, h5, h6 {
  color: var(--heading);
  line-height: 1.25;
  margin-top: 2.4rem;
  margin-bottom: 0.7rem;
  position: relative;
}

h1 { font-size: 2rem; margin-top: 0; margin-bottom: 1rem; }
h2 { font-size: 1.45rem; padding-bottom: 0.3rem; border-bottom: 1px solid var(--border); }
h3 { font-size: 1.2rem; }
h4 { font-size: 1.05rem; }

p { margin-bottom: 1rem; }

a { color: var(--link); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:hover { color: var(--link-hover); }

strong { font-weight: 700; }
em { font-style: italic; }

blockquote {
  border-left: 3px solid var(--rule);
  padding: 0.15rem 0 0.15rem 1.2rem;
  margin: 1.2rem 0;
  color: var(--text-muted);
}

blockquote > p:last-child { margin-bottom: 0; }

hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2rem 0;
}

ul, ol { padding-left: 1.6rem; margin-bottom: 1rem; }
li { margin-bottom: 0.25rem; }
li > p { margin-bottom: 0.4rem; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 1.2rem 0;
  font-size: 0.92rem;
}

th, td {
  border: 1px solid var(--border);
  padding: 0.45rem 0.7rem;
  text-align: left;
}

th {
  background: var(--code-bg);
  font-weight: 600;
}

/* ─── Heading anchors ─── */

.heading-anchor {
  color: var(--text-muted);
  text-decoration: none;
  font-weight: 400;
  opacity: 0;
  margin-left: 0.3em;
  font-size: 0.85em;
  transition: opacity 0.15s;
}

h1:hover .heading-anchor,
h2:hover .heading-anchor,
h3:hover .heading-anchor,
h4:hover .heading-anchor,
h5:hover .heading-anchor,
h6:hover .heading-anchor {
  opacity: 0.5;
}

.heading-anchor:hover {
  opacity: 1 !important;
  color: var(--accent);
}

/* ─── Edit link ─── */

.edit-link {
  text-align: right;
  margin-bottom: 0.5rem;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.8rem;
}

.edit-link a {
  color: var(--text-muted);
  text-decoration: none;
  transition: color 0.15s;
}

.edit-link a:hover {
  color: var(--accent);
}

/* ─── Code ─── */

code {
  font-family: ui-monospace, "Cascadia Code", "Source Code Pro", Menlo,
    Consolas, monospace;
  font-size: 0.88em;
}

:not(pre) > code {
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 3px;
  padding: 0.12em 0.35em;
}

.code-cell {
  margin: 1.2rem 0;
  position: relative;
}

.code-cell > pre {
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 4px;
  padding: 0.85rem 1rem;
  overflow-x: auto;
  line-height: 1.5;
  tab-size: 2;
}

.code-cell > pre > code {
  font-size: 0.84rem;
  color: var(--code-text);
}

/* ─── Copy button ─── */

.copy-btn {
  position: absolute;
  top: 0.4rem;
  right: 0.4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1.8rem;
  height: 1.8rem;
  border: 1px solid var(--code-border);
  border-radius: 4px;
  background: var(--code-bg);
  color: var(--text-muted);
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s, color 0.15s, background 0.15s;
  z-index: 2;
}

.code-cell:hover .copy-btn { opacity: 1; }
.copy-btn:hover { color: var(--text); background: var(--bg); }

.copy-btn .check-icon { display: none; }
.copy-btn.copied .copy-icon { display: none; }
.copy-btn.copied .check-icon { display: block; }
.copy-btn.copied { color: #5a7a40; }

/* ─── Outputs ─── */

.output {
  background: var(--output-bg);
  border: 1px solid var(--output-border);
  border-top: none;
  border-radius: 0 0 4px 4px;
  padding: 0.7rem 1rem;
  overflow-x: auto;
  font-family: ui-monospace, "Cascadia Code", "Source Code Pro", Menlo,
    Consolas, monospace;
  font-size: 0.82rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}

.output + .output { border-top: 1px dashed var(--output-border); border-radius: 0; }
.output:first-of-type { margin-top: -1px; }
.output:last-of-type { margin-bottom: 0; }

.code-cell > pre + .output {
  border-radius: 0 0 4px 4px;
}

.code-cell > pre:has(+ .output) {
  border-radius: 4px 4px 0 0;
  border-bottom: none;
}

.output.stderr { color: var(--stderr-text); }

.output.error {
  background: var(--error-bg);
  border-color: var(--error-border);
  color: var(--error-text);
}

.output img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0.3rem 0;
  border-radius: 3px;
}

/* Collapsed cells */
details.collapsed {
  margin: 1.2rem 0;
}

details.collapsed > summary {
  cursor: pointer;
  padding: 0.4rem 0.7rem;
  background: var(--code-bg);
  border: 1px solid var(--code-border);
  border-radius: 4px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.82rem;
  color: var(--text-muted);
  list-style: none;
}

details.collapsed > summary::before {
  content: "\25B8 ";
}

details.collapsed[open] > summary {
  border-radius: 4px 4px 0 0;
  border-bottom: none;
}

details.collapsed[open] > summary::before {
  content: "\25BE ";
}

/* ─── Images & Figures ─── */

article img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

figure {
  margin: 1.5rem 0;
  text-align: center;
}

figure img { margin: 0 auto; }

figcaption, blockquote:has(> strong:first-child) {
  font-size: 0.9rem;
  color: var(--text-muted);
  margin-top: 0.5rem;
}

/* ─── On-page TOC ─── */

.page-toc {
  display: none;
}

@media (min-width: 1200px) {
  main {
    display: grid;
    grid-template-columns: 1fr 14rem;
    grid-template-rows: 1fr auto;
  }

  article {
    grid-column: 1;
    grid-row: 1;
  }

  .page-toc {
    display: block;
    grid-column: 2;
    grid-row: 1;
    position: sticky;
    top: 1.5rem;
    align-self: start;
    max-height: calc(100vh - 3rem);
    overflow-y: auto;
    padding: 2.5rem 1rem 2rem 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 0.78rem;
    line-height: 1.5;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }

  .page-nav {
    grid-column: 1 / -1;
    grid-row: 2;
  }
}

.page-toc-title {
  font-weight: 600;
  color: var(--heading);
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

.page-toc ul {
  list-style: none;
  padding: 0;
  margin: 0;
  border-left: 1px solid var(--border);
}

.page-toc li {
  margin: 0;
}

.page-toc li a {
  display: block;
  padding: 0.2rem 0 0.2rem 0.75rem;
  color: var(--text-muted);
  text-decoration: none;
  transition: color 0.15s;
}

.page-toc li a:hover {
  color: var(--accent);
}

.page-toc li.toc-h3 a {
  padding-left: 1.5rem;
}

@media print {
  .page-toc { display: none; }
}

/* ─── Page navigation ─── */

.page-nav {
  max-width: var(--content-width);
  width: 100%;
  margin: 0 auto;
  padding: 1.5rem 2rem 2.5rem;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  border-top: 1px solid var(--border);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.88rem;
}

.page-nav a {
  text-decoration: none;
  color: var(--accent);
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  transition: color 0.15s;
}

.page-nav a:hover { color: var(--accent-hover); }

.page-nav .nav-dir {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}

.page-nav .nav-title { font-weight: 600; }
.nav-next { text-align: right; margin-left: auto; }

/* ─── Syntax highlighting (light theme) ─── */

.hljs-keyword, .hljs-built_in { color: #8b5a30; font-weight: 600; }
.hljs-type, .hljs-title.class_ { color: #3d6880; }
.hljs-string, .hljs-doctag { color: #5a7a40; }
.hljs-number, .hljs-literal { color: #a06030; }
.hljs-comment { color: #8a8480; font-style: italic; }
.hljs-meta, .hljs-preprocessor { color: #7a6a58; }
.hljs-symbol, .hljs-attr { color: #6a5a80; }
.hljs-variable, .hljs-template-variable { color: #805a50; }
.hljs-function .hljs-title, .hljs-title.function_ { color: #6a5030; }
.hljs-operator { color: #6a6058; }

/* ─── Syntax highlighting (dark theme) ─── */

[data-theme="dark"] .hljs-keyword,
[data-theme="dark"] .hljs-built_in { color: #d4a060; }
[data-theme="dark"] .hljs-type,
[data-theme="dark"] .hljs-title.class_ { color: #6cb0d0; }
[data-theme="dark"] .hljs-string,
[data-theme="dark"] .hljs-doctag { color: #8cb870; }
[data-theme="dark"] .hljs-number,
[data-theme="dark"] .hljs-literal { color: #d0884c; }
[data-theme="dark"] .hljs-comment { color: #6a6e76; }
[data-theme="dark"] .hljs-meta,
[data-theme="dark"] .hljs-preprocessor { color: #9a8e7c; }
[data-theme="dark"] .hljs-symbol,
[data-theme="dark"] .hljs-attr { color: #a08cc0; }
[data-theme="dark"] .hljs-variable,
[data-theme="dark"] .hljs-template-variable { color: #c09080; }
[data-theme="dark"] .hljs-function .hljs-title,
[data-theme="dark"] .hljs-title.function_ { color: #c0a060; }
[data-theme="dark"] .hljs-operator { color: #9090a0; }

/* ─── Theme toggle ─── */

.theme-toggle {
  position: fixed;
  top: 0.75rem;
  right: 0.75rem;
  z-index: 200;
  width: 2.2rem;
  height: 2.2rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg);
  color: var(--text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.15s, background 0.15s;
}

.theme-toggle:hover { color: var(--text); }

.theme-toggle .icon-sun,
.theme-toggle .icon-moon { width: 18px; height: 18px; }

.theme-toggle .icon-moon { display: none; }
[data-theme="dark"] .theme-toggle .icon-sun { display: none; }
[data-theme="dark"] .theme-toggle .icon-moon { display: block; }

/* ─── Sidebar search ─── */

.sidebar-search {
  padding: 0.6rem 1.25rem 0.8rem;
}

.sidebar-search input {
  width: 100%;
  padding: 0.38rem 0.6rem;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--sidebar-border);
  border-radius: 4px;
  color: var(--sidebar-text);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.82rem;
  outline: none;
  transition: border-color 0.15s;
}

.sidebar-search input::placeholder {
  color: rgba(176, 180, 188, 0.5);
}

.sidebar-search input:focus {
  border-color: var(--sidebar-active);
}

.search-results {
  padding: 0 0 0.5rem;
}

.search-results a {
  display: block;
  padding: 0.32rem 1.25rem 0.32rem 1.6rem;
  color: var(--sidebar-text);
  text-decoration: none;
  font-size: 0.82rem;
  border-left: 2px solid transparent;
  transition: color 0.15s;
}

.search-results a:hover {
  color: var(--sidebar-hover);
}

.search-results .no-results {
  padding: 0.32rem 1.25rem 0.32rem 1.6rem;
  color: var(--sidebar-text);
  font-size: 0.82rem;
  opacity: 0.6;
}

/* ─── Sidebar footer ─── */

.sidebar-footer {
  padding: 0.6rem 1.25rem;
  border-top: 1px solid var(--sidebar-border);
  font-size: 0.78rem;
}

.sidebar-footer a {
  color: var(--sidebar-text);
  text-decoration: none;
  opacity: 0.7;
  transition: opacity 0.15s;
}

.sidebar-footer a:hover {
  opacity: 1;
}

/* ─── Print page ─── */

.print-chapter {
  margin-bottom: 3rem;
  page-break-after: always;
}

.print-chapter:last-child {
  page-break-after: avoid;
}

.print-header {
  max-width: var(--content-width);
  margin: 0 auto;
  padding: 2.5rem 2rem 1rem;
  text-align: center;
}

.print-header h1 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

.print-btn {
  display: inline-block;
  padding: 0.5rem 1.2rem;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 4px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 0.88rem;
  cursor: pointer;
  text-decoration: none;
  transition: background 0.15s;
}

.print-btn:hover {
  background: var(--accent-hover);
}

@media print {
  .print-btn { display: none; }
}

/* ─── Responsive ─── */

@media (max-width: 768px) {
  :root { --sidebar-width: 16rem; }

  html { font-size: 16px; }

  .sidebar {
    transform: translateX(-100%);
  }

  .sidebar.open {
    transform: translateX(0);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.25);
  }

  .menu-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    position: fixed;
    top: 0.75rem;
    left: 0.75rem;
    z-index: 200;
    width: 2.5rem;
    height: 2.5rem;
    border: none;
    border-radius: 4px;
    background: var(--bg);
    color: var(--text);
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
  }

  main { margin-left: 0; }

  article {
    padding: 3.5rem 1.2rem 2rem;
  }

  .page-nav { padding: 1.2rem 1.2rem 2rem; }
}

@media print {
  .sidebar, .menu-btn, .page-nav, .theme-toggle, .copy-btn { display: none; }
  main { margin-left: 0; }
  article { max-width: none; padding: 0; }
  body { background: white; font-size: 11pt; }
  a { color: inherit; text-decoration: none; }
  .code-cell > pre, .output { break-inside: avoid; }
  .heading-anchor { display: none; }
}
|css}

let template_html =
  {html|<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{chapter_title}} — {{book_title}}</title>
<link rel="stylesheet" href="{{root_path}}style.css">
<script>
(function() {
  var t = localStorage.getItem('quill-theme');
  if (t === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();
</script>
</head>
<body>
<button class="menu-btn" aria-label="Menu">
<svg viewBox="0 0 24 24" width="22" height="22"><path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="2" stroke-linecap="round" fill="none"/></svg>
</button>

<button class="theme-toggle" aria-label="Toggle theme">
<svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
<svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
</button>

<nav class="sidebar">
<div class="sidebar-scrollable">
<div class="sidebar-header">
<a href="{{root_path}}index.html">{{book_title}}</a>
</div>
<div class="sidebar-search">
<input type="text" id="search-input" placeholder="Search..." aria-label="Search">
</div>
<div id="search-results" class="search-results" hidden></div>
<div id="sidebar-toc">
{{toc}}
</div>
</div>
<div class="sidebar-footer">
<a href="{{root_path}}print.html">Print version</a>
</div>
</nav>

<main>
<article>
{{edit_link}}
{{content}}
</article>
{{page_toc}}
<nav class="page-nav">
{{prev_nav}}
{{next_nav}}
</nav>
</main>
<script>
document.querySelector('.menu-btn').addEventListener('click', function() {
  document.querySelector('.sidebar').classList.toggle('open');
});
document.querySelectorAll('.sidebar a').forEach(function(a) {
  a.addEventListener('click', function() {
    document.querySelector('.sidebar').classList.remove('open');
  });
});
document.querySelector('.theme-toggle').addEventListener('click', function() {
  var html = document.documentElement;
  var isDark = html.getAttribute('data-theme') === 'dark';
  if (isDark) {
    html.removeAttribute('data-theme');
    localStorage.setItem('quill-theme', 'light');
  } else {
    html.setAttribute('data-theme', 'dark');
    localStorage.setItem('quill-theme', 'dark');
  }
});
document.querySelectorAll('.copy-btn').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var code = btn.parentElement.querySelector('pre code');
    if (code) {
      navigator.clipboard.writeText(code.textContent).then(function() {
        btn.classList.add('copied');
        setTimeout(function() { btn.classList.remove('copied'); }, 1500);
      });
    }
  });
});
</script>
<script>
(function() {
  var searchIndex = null;
  var searchInput = document.getElementById('search-input');
  var searchResults = document.getElementById('search-results');
  var sidebarToc = document.getElementById('sidebar-toc');
  fetch('{{root_path}}searchindex.json')
    .then(function(r) { return r.json(); })
    .then(function(data) { searchIndex = data; });
  searchInput.addEventListener('input', function() {
    var q = searchInput.value.trim().toLowerCase();
    if (!q || !searchIndex) {
      searchResults.hidden = true;
      sidebarToc.hidden = false;
      searchResults.innerHTML = '';
      return;
    }
    var matches = [];
    for (var i = 0; i < searchIndex.length && matches.length < 20; i++) {
      var entry = searchIndex[i];
      if (entry.title.toLowerCase().indexOf(q) !== -1 ||
          entry.body.toLowerCase().indexOf(q) !== -1) {
        matches.push(entry);
      }
    }
    sidebarToc.hidden = true;
    searchResults.hidden = false;
    if (matches.length === 0) {
      searchResults.innerHTML = '<div class="no-results">No results found</div>';
    } else {
      var html = '';
      for (var j = 0; j < matches.length; j++) {
        html += '<a href="{{root_path}}' + matches[j].url + '">' +
          matches[j].title.replace(/</g, '&lt;') + '</a>';
      }
      searchResults.innerHTML = html;
    }
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === '/' && document.activeElement !== searchInput &&
        document.activeElement.tagName !== 'INPUT' &&
        document.activeElement.tagName !== 'TEXTAREA') {
      e.preventDefault();
      searchInput.focus();
    }
  });
})();
</script>
{{live_reload_script}}
</body>
</html>|html}

let print_template_html =
  {html|<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{book_title}} — Print</title>
<link rel="stylesheet" href="style.css">
<script>
(function() {
  var t = localStorage.getItem('quill-theme');
  if (t === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();
</script>
</head>
<body>
<div class="print-header">
<h1>{{book_title}}</h1>
<button class="print-btn" onclick="window.print()">Print this page</button>
</div>
<main style="margin-left:0">
{{chapters}}
</main>
</body>
</html>|html}
