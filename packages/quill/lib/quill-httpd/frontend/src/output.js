// Output renderer for cell execution results.

export function renderOutput(output) {
  switch (output.kind) {
    case 'stdout': return renderStdout(output.text);
    case 'stderr': return renderStderr(output.text);
    case 'error': return renderError(output.text);
    case 'display': return renderDisplay(output.mime, output.data);
    default: return null;
  }
}

function renderStdout(text) {
  const pre = document.createElement('pre');
  pre.className = 'output output-stdout';
  pre.textContent = text;
  return pre;
}

function renderStderr(text) {
  const pre = document.createElement('pre');
  pre.className = 'output output-stderr';
  pre.textContent = text;
  return pre;
}

function renderError(text) {
  const div = document.createElement('div');
  div.className = 'output output-error';
  const pre = document.createElement('pre');
  pre.textContent = text;
  div.appendChild(pre);
  return div;
}

function renderDisplay(mime, data) {
  const div = document.createElement('div');
  div.className = 'output output-display';

  if (mime === 'text/plain') {
    const pre = document.createElement('pre');
    pre.textContent = data;
    div.appendChild(pre);
  } else if (mime === 'text/html') {
    const iframe = document.createElement('iframe');
    iframe.sandbox = 'allow-scripts';
    iframe.srcdoc = data;
    iframe.style.width = '100%';
    iframe.style.border = 'none';
    iframe.onload = () => {
      try {
        iframe.style.height = iframe.contentDocument.body.scrollHeight + 'px';
      } catch { /* cross-origin */ }
    };
    div.appendChild(iframe);
  } else if (mime.startsWith('image/')) {
    const img = document.createElement('img');
    if (mime === 'image/svg+xml') {
      img.src = 'data:image/svg+xml;base64,' + btoa(data);
    } else {
      img.src = 'data:' + mime + ';base64,' + data;
    }
    img.style.maxWidth = '100%';
    div.appendChild(img);
  } else if (mime === 'application/json') {
    const pre = document.createElement('pre');
    try {
      pre.textContent = JSON.stringify(JSON.parse(data), null, 2);
    } catch {
      pre.textContent = data;
    }
    div.appendChild(pre);
  } else {
    const pre = document.createElement('pre');
    pre.textContent = data;
    div.appendChild(pre);
  }

  return div;
}

/** Append an output to a cell's output container, coalescing consecutive stdout. */
export function appendOutputToContainer(container, output) {
  // Coalesce consecutive stdout without creating a DOM element
  if (output.kind === 'stdout' && container.lastElementChild &&
      container.lastElementChild.classList.contains('output-stdout')) {
    container.lastElementChild.textContent += output.text;
    return;
  }
  const el = renderOutput(output);
  if (el) container.appendChild(el);
}
