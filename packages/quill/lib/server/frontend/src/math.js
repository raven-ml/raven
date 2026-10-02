// Math equation rendering for text cells.

import renderMathInElement from 'katex/contrib/auto-render';

/**
 * Render all math delimiters in the given element.
 * cmarkit outputs \(...\) for inline and \[...\] for display math,
 * which are the default delimiters for renderMathInElement.
 */
export function renderMath(element) {
  renderMathInElement(element, {
    output: 'mathml',
    throwOnError: false,
  });
}
