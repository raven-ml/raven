# Reviewing quill

## Design

- A branch for one frontend (terminal, web, batch) for what the session or
  the document should state.

## Semantics

References: CommonMark for the notebook's Markdown.

Failure here is a notebook changed where the user didn't change it, or a
cell run against state it should not see.

- A document's cells, ids and outputs through read, edit and write.
- Cells against one session, after a cell that raised.

## Skip

`lib/server/frontend/dist/`, `lib/server/frontend/package-lock.json`.
