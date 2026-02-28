// CodeMirror 6 editor setup for OCaml code cells.

import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection, dropCursor, highlightSpecialChars } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { StreamLanguage, bracketMatching, indentOnInput } from '@codemirror/language';
import { oCaml } from '@codemirror/legacy-modes/mode/mllike';
import { closeBrackets } from '@codemirror/autocomplete';
import { autocompletion } from '@codemirror/autocomplete';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { highlightSelectionMatches } from '@codemirror/search';
import { tags } from '@lezer/highlight';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';

// --- Theme ---

const quillTheme = EditorView.theme({
  '&': {
    fontSize: '14px',
    backgroundColor: 'transparent',
  },
  '.cm-content': {
    fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', 'Consolas', monospace",
    caretColor: '#daa550',
    padding: '8px 0',
  },
  '.cm-cursor, .cm-dropCursor': {
    borderLeftColor: '#daa550',
  },
  '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': {
    backgroundColor: '#3a3a50',
  },
  '.cm-gutters': {
    backgroundColor: 'transparent',
    color: '#646870',
    border: 'none',
    paddingLeft: '4px',
  },
  '.cm-activeLineGutter': {
    backgroundColor: 'transparent',
    color: '#aab0ba',
  },
  '.cm-activeLine': {
    backgroundColor: 'transparent',
  },
  '.cm-line': {
    padding: '0 8px',
  },
  '.cm-tooltip': {
    backgroundColor: '#24242e',
    border: '1px solid #32323a',
    color: '#c8ccd4',
  },
  '.cm-tooltip-autocomplete': {
    '& > ul > li[aria-selected]': {
      backgroundColor: '#3a3a50',
    },
  },
}, { dark: true });

const quillHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: '#c792ea' },
  { tag: tags.operator, color: '#89ddff' },
  { tag: tags.string, color: '#c3e88d' },
  { tag: tags.number, color: '#f78c6c' },
  { tag: tags.bool, color: '#f78c6c' },
  { tag: tags.comment, color: '#646870', fontStyle: 'italic' },
  { tag: tags.typeName, color: '#ffcb6b' },
  { tag: tags.definition(tags.variableName), color: '#82aaff' },
  { tag: tags.variableName, color: '#c8ccd4' },
  { tag: tags.function(tags.variableName), color: '#82aaff' },
  { tag: tags.propertyName, color: '#c8ccd4' },
  { tag: tags.meta, color: '#daa550' },
  { tag: tags.punctuation, color: '#89ddff' },
]);

// --- Completion source ---

function makeCompletionSource(wsClient) {
  return async (context) => {
    const trigger = context.matchBefore(/[\w.]+$/);
    if (!trigger && !context.explicit) return null;

    const code = context.state.doc.toString();
    const pos = context.pos;

    try {
      const items = await wsClient.complete(code, pos);
      if (!items || items.length === 0) return null;
      return {
        from: trigger ? trigger.from : context.pos,
        options: items.map(label => ({ label, type: 'variable' })),
      };
    } catch {
      return null;
    }
  };
}

// --- Editor creation ---

export function createEditor(container, source, options) {
  const { onChange, onExecute, onExecuteAndMoveNext, onEscape, wsClient } = options;

  const extensions = [
    StreamLanguage.define(oCaml).extension,
    lineNumbers(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    highlightSpecialChars(),
    bracketMatching(),
    closeBrackets(),
    indentOnInput(),
    history(),
    drawSelection(),
    dropCursor(),
    highlightSelectionMatches(),
    quillTheme,
    syntaxHighlighting(quillHighlightStyle),
    EditorState.tabSize.of(2),
    EditorView.updateListener.of(update => {
      if (update.docChanged) {
        onChange(update.state.doc.toString());
      }
    }),
    keymap.of([
      { key: 'Ctrl-Enter', run: () => { onExecute(); return true; } },
      { key: 'Cmd-Enter', run: () => { onExecute(); return true; } },
      { key: 'Shift-Enter', run: () => { onExecuteAndMoveNext(); return true; } },
      { key: 'Escape', run: () => { onEscape(); return true; } },
      ...defaultKeymap,
      ...historyKeymap,
    ]),
  ];

  if (wsClient) {
    extensions.push(
      autocompletion({
        override: [makeCompletionSource(wsClient)],
        activateOnTyping: true,
      })
    );
  }

  const view = new EditorView({
    parent: container,
    doc: source,
    extensions,
  });

  return view;
}
