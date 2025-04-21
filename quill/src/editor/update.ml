open Model

type msg =
  | Focus_inline of int * int (* user clicked/moved caret into a run *)
  | Set_document of block list

let log fmt =
  Printf.ksprintf
    (fun s ->
      Js_of_ocaml.Console.console##log (Js_of_ocaml.Js.string ("[update] " ^ s)))
    fmt

let update (m : model) (message : msg) : model =
  match message with
  | Focus_inline (block, run_j) ->
      log "Focus_inline: block=%d, run_j=%d" block run_j;
      { m with focused = Some (block, run_j) }
  | Set_document docs ->
      log "Set_document: %d blocks" (List.length docs);
      { m with document = docs }
