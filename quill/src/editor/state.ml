module Int_set = Set.Make (Int)

type load_state =
  | Idle
  | Loading of { path : string }
  | Load_failed of { path : string; error : string }

type caret = { block_id : int; inline_id : int option; offset : int }

type selection =
  | No_selection
  | Caret of caret
  | Range of { anchor : caret; focus : caret }

type config = { history_limit : int; auto_normalize : bool }
type snapshot = { document : Document.t; selection : selection }
type history = { past : snapshot list; future : snapshot list; capacity : int }

type t = {
  document : Document.t;
  selection : selection;
  load_state : load_state;
  running_blocks : Int_set.t;
  history : history;
  config : config;
}

let default_config = { history_limit = 100; auto_normalize = true }
let empty_history capacity = { past = []; future = []; capacity }

let create ?(config = default_config) ?(document = Document.empty)
    ?(selection = No_selection) () =
  {
    document;
    selection;
    load_state = Idle;
    running_blocks = Int_set.empty;
    history = empty_history config.history_limit;
    config;
  }

let init = create ()

let with_document ?config ?selection document =
  create ?config ?selection ~document ()

let set_document state document = { state with document }
let set_load_state state load_state = { state with load_state }
let set_selection state selection = { state with selection }
let clear_selection state = { state with selection = No_selection }

let mark_block_running state block_id =
  let running_blocks = Int_set.add block_id state.running_blocks in
  { state with running_blocks }

let mark_block_idle state block_id =
  let running_blocks = Int_set.remove block_id state.running_blocks in
  { state with running_blocks }

let is_block_running state block_id = Int_set.mem block_id state.running_blocks
let snapshot state = { document = state.document; selection = state.selection }

let trim_history capacity lst =
  let rec aux acc count = function
    | [] -> List.rev acc
    | _ when count >= capacity -> List.rev acc
    | x :: xs -> aux (x :: acc) (count + 1) xs
  in
  aux [] 0 lst

let push_history state =
  let snap = snapshot state in
  let history =
    let past =
      trim_history state.history.capacity (snap :: state.history.past)
    in
    { state.history with past; future = [] }
  in
  { state with history }

let normalize_if_enabled state document =
  if state.config.auto_normalize then Document.normalize_blanklines document
  else document

let restore snap ~config ~history =
  {
    document = snap.document;
    selection = snap.selection;
    load_state = Idle;
    running_blocks = Int_set.empty;
    history;
    config;
  }

let record_document_change ?selection state document =
  let selection = Option.value selection ~default:state.selection in
  let state = push_history state in
  let document = normalize_if_enabled state document in
  { state with document; selection }

let has_undo state = state.history.past <> []
let has_redo state = state.history.future <> []

let undo state =
  match state.history.past with
  | [] -> None
  | snap :: past ->
      let future = snapshot state :: state.history.future in
      let history = { state.history with past; future } in
      let state =
        {
          state with
          document = snap.document;
          selection = snap.selection;
          history;
        }
      in
      Some state

let redo state =
  match state.history.future with
  | [] -> None
  | snap :: future ->
      let past =
        trim_history state.history.capacity
          (snapshot state :: state.history.past)
      in
      let history = { state.history with past; future } in
      let state =
        {
          state with
          document = snap.document;
          selection = snap.selection;
          history;
        }
      in
      Some state

let selection_blocks state =
  match state.selection with
  | No_selection -> []
  | Caret caret -> [ caret.block_id ]
  | Range { anchor; focus } ->
      Document.block_ids_between state.document ~start_id:anchor.block_id
        ~end_id:focus.block_id
