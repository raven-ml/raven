# Quill Developer Guide

## Architecture

Quill is a writing-first interactive document environment blending markdown editing with live OCaml code execution. It consists of a web-based editor, Dream web server, and markdown processing pipeline.

### Core Components

- **[src/editor/](src/editor/)**: Browser-based editor (OCaml → js_of_ocaml)
- **[src/server/](src/server/)**: Dream web server for serving and executing documents
- **[src/markdown/](src/markdown/)**: Markdown parsing and AST manipulation
- **[src/top/](src/top/)**: OCaml toplevel integration for code execution
- **[src/cli/](src/cli/)**: CLI for serve/exec commands

### Key Design Principles

1. **Writing-first**: Markdown is primary, code cells fit naturally into text
2. **Browser-based editor**: Rich editing without heavy frameworks
3. **Persistent sessions**: OCaml toplevel state maintained across cells
4. **Live rendering**: Markdown renders as you type
5. **Lightweight**: Minimal dependencies, fast startup

## Architecture Overview

```
┌─────────────┐
│   Browser   │
│   (Editor)  │ ← js_of_ocaml compiled OCaml
└─────────────┘
      ↕ HTTP/WebSocket
┌─────────────┐
│ Dream Server│
│  (Backend)  │
└─────────────┘
      ↕
┌─────────────┐
│ OCaml Top   │ ← Executes code cells
│  (Session)  │
└─────────────┘
      ↕
┌─────────────┐
│  Markdown   │ ← Document storage
│    Files    │
└─────────────┘
```

## Editor (Browser)

### Model-View-Update Pattern

Elm architecture for predictable state management:

```ocaml
type model = {
  document : Quill_markdown.document;
  (* Editor state *)
}

type msg =
  | Insert_char of char
  | Delete_backward
  | Move_cursor of direction
  | Execute_cell
  (* ... *)

let update msg model =
  match msg with
  | Insert_char c -> {model with document = insert_char model.document c}
  | Execute_cell -> (* Send to server, update with result *)
  | ...
```

### Document Model

Document = list of blocks (paragraphs, code cells, etc.):

```ocaml
type block =
  | Paragraph of inline list
  | Code_block of { lang : string; content : string; output : string option }
  | Heading of { level : int; content : inline list }
  | ...

type inline =
  | Text of string
  | Code of string
  | Emphasis of inline list
  | ...

type document = block list
```

### Event Handling

DOM events → messages → update:

```ocaml
let on_keydown event model =
  match key_of_event event with
  | "Enter" -> Some Insert_newline
  | "Backspace" -> Some Delete_backward
  | _ when is_printable event -> Some (Insert_char (char_of_event event))
  | _ -> None

(* In view *)
let view model =
  div [on_keydown (fun e -> update (on_keydown e model))] [...]
```

### Rendering

Virtual DOM pattern:

```ocaml
let render_block = function
  | Paragraph inlines -> p [] (List.map render_inline inlines)
  | Code_block {lang; content; output} ->
      div [class_ "code-cell"] [
        pre [] [code [class_ ("language-" ^ lang)] [text content]];
        (match output with
         | Some out -> div [class_ "output"] [text out]
         | None -> empty);
      ]
  | ...
```

## Server (Backend)

### Dream Routes

```ocaml
let routes base_dir =
  Dream.router [
    Dream.get "/" (serve_directory_index base_dir);
    Dream.get "/document/:filename" serve_document_editor;
    Dream.get "/content/:filename" (serve_document_content base_dir);
    Dream.post "/execute" handle_execute;
    Dream.get "/static/**" serve_static;
  ]
```

### Code Execution

Execute OCaml code in persistent toplevel:

```ocaml
let execute_code session code =
  (* 1. Send code to toplevel *)
  Toploop.eval session code;

  (* 2. Capture output *)
  let stdout = capture_stdout () in
  let result = capture_result () in

  (* 3. Return output *)
  {stdout; result; error = None}
```

**Session management:**
- One toplevel per document
- State persists across cell executions
- Variables defined in cell 1 available in cell 2

### Markdown Processing

Parse markdown → AST → Modify → Render:

```ocaml
let process_markdown content =
  (* 1. Parse markdown to AST *)
  let ast = Quill_markdown.parse content in

  (* 2. Extract code cells *)
  let code_cells = extract_code_cells ast in

  (* 3. Execute cells *)
  let results = List.map execute_code code_cells in

  (* 4. Update AST with results *)
  let updated_ast = update_outputs ast results in

  (* 5. Render back to markdown *)
  Quill_markdown.to_string updated_ast
```

## Markdown Processing

### Parsing

Transform markdown text → structured AST:

```ocaml
let parse_block line =
  match line with
  | "" -> Empty_line
  | s when starts_with "# " s -> Heading {level = 1; content = parse_inline (String.drop 2 s)}
  | s when starts_with "```" s -> start_code_block (String.drop 3 s)
  | _ -> parse_paragraph line
```

### Code Cell Format

Code cells with optional output:

```markdown
\```ocaml
let x = 42
\```

<!-- quill-output -->
\```
val x : int = 42
\```
<!-- /quill-output -->
```

**Execution:**
1. Extract code from fenced block
2. Execute in toplevel
3. Insert output as comment-wrapped block

### Inline Rendering

Support for inline code, emphasis, links:

```ocaml
let parse_inline text =
  (* Regex-based parsing *)
  match text with
  | ... when has_code_backticks -> Code (extract_code text)
  | ... when has_emphasis -> Emphasis (parse_inline (extract_emphasis text))
  | _ -> Text text
```

## Development Workflow

### Building and Testing

```bash
# Build quill
dune build quill/

# Run server in dev mode
dune exec quill serve example/

# Run tests
dune build quill/test/test_quill.exe && _build/default/quill/test/test_quill.exe
```

### Testing the Editor

Browser-based testing:

1. Start dev server: `dune exec quill serve example/`
2. Open browser: `http://localhost:8080`
3. Test interactions manually
4. Check browser console for errors

### Testing Markdown Processing

```ocaml
let test_code_cell_parsing () =
  let md = "```ocaml\nlet x = 1\n```" in
  let doc = Quill_markdown.parse md in
  match doc with
  | [Code_block {lang = "ocaml"; content = "let x = 1"; _}] -> ()
  | _ -> Alcotest.fail "Expected code block"
```

### Testing Execution

```ocaml
let test_toplevel_execution () =
  let session = Top.create () in
  let result = Top.eval session "let x = 42" in
  Alcotest.(check string) "result" "val x : int = 42" result.stdout
```

## Adding Features

### New Block Type

1. Add to AST:

```ocaml
type block =
  | ...
  | Table of { headers : string list; rows : string list list }
```

2. Add parser:

```ocaml
let parse_table lines =
  (* Parse markdown table syntax *)
  ...
```

3. Add renderer:

```ocaml
let render_block = function
  | ...
  | Table {headers; rows} ->
      table [] [
        thead [] [tr [] (List.map (fun h -> th [] [text h]) headers)];
        tbody [] (List.map render_row rows);
      ]
```

### New Editor Command

1. Add message:

```ocaml
type msg =
  | ...
  | Toggle_bold
```

2. Add update handler:

```ocaml
let update msg model =
  match msg with
  | ...
  | Toggle_bold ->
      {model with document = wrap_selection_with model.document "**"}
```

3. Add keybinding:

```ocaml
let on_keydown event =
  match key_of_event event with
  | ...
  | "b" when ctrl_key event -> Some Toggle_bold
  | ...
```

### New Execution Mode

Add execution backend:

```ocaml
let execute_python session code =
  (* Integrate Python REPL *)
  ...

let execute_code lang session code =
  match lang with
  | "ocaml" -> execute_ocaml session code
  | "python" -> execute_python session code
  | _ -> {error = Some "Unsupported language"}
```

## Common Pitfalls

### js_of_ocaml DOM

Browser APIs differ from native OCaml:

```ocaml
(* Wrong: direct DOM manipulation *)
let el = get_element_by_id "myid" in
el##innerHTML := "new content"

(* Correct: use virtual DOM *)
let view model =
  div [id "myid"] [text (get_content model)]
```

### State Synchronization

Editor state must sync with server:

```ocaml
(* Wrong: client-side only *)
let model = {model with document = updated_doc}

(* Correct: persist to server *)
let send_update updated_doc =
  fetch "/save" ~body:(serialize updated_doc) >>= fun _ ->
  {model with document = updated_doc}
```

### Toplevel State

Toplevel state is mutable:

```ocaml
(* Wrong: create new session per execution *)
let execute code =
  let session = Top.create () in
  Top.eval session code

(* Correct: reuse session *)
let session = Top.create () in
let execute code = Top.eval session code
```

### Markdown Escaping

Code blocks need escaping:

```ocaml
(* Wrong: raw backticks in code *)
let md = "```\nlet code = `42`\n```"

(* Correct: escape or use different delimiter *)
let md = "````\nlet code = `42`\n````"
```

## Performance

- **Virtual DOM diffing**: Only update changed elements
- **Lazy rendering**: Render visible blocks only
- **Debounced saves**: Batch document updates
- **Session pooling**: Reuse toplevel sessions

## Code Style

- **Editor**: Elm architecture (model-view-update)
- **Server**: Dream handlers with labeled arguments
- **Errors**: Log to browser console (client), Dream logs (server)
- **Documentation**: JSDoc-style for editor, ocamldoc for server

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- Dream documentation for server framework
- js_of_ocaml documentation for browser compilation
