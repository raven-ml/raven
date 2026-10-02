(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let basic_tests =
  [
    test "create session" (fun () ->
        let doc = Doc.of_cells [ Cell.text "hello" ] in
        let s = Session.create doc in
        equal int 1 (Doc.length (Session.doc s)));
    test "update source" (fun () ->
        let c = Cell.text "old" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "new" s in
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some c -> equal string "new" (Cell.source c)
        | None -> fail "cell not found");
    test "insert cell" (fun () ->
        let doc = Doc.of_cells [ Cell.text "a" ] in
        let s = Session.create doc in
        let new_cell = Cell.text "b" in
        let s = Session.insert_cell ~pos:1 new_cell s in
        equal int 2 (Doc.length (Session.doc s)));
    test "remove cell" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.text "b" in
        let doc = Doc.of_cells [ c1; c2 ] in
        let s = Session.create doc in
        let s = Session.remove_cell (Cell.id c1) s in
        equal int 1 (Doc.length (Session.doc s)));
    test "move cell" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.text "b" in
        let c3 = Cell.text "c" in
        let doc = Doc.of_cells [ c1; c2; c3 ] in
        let s = Session.create doc in
        let s = Session.move_cell (Cell.id c3) ~pos:0 s in
        match Doc.nth 0 (Session.doc s) with
        | Some c -> equal string "c" (Cell.source c)
        | None -> fail "expected Some");
    test "set cell kind" (fun () ->
        let c = Cell.text "code here" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.set_cell_kind (Cell.id c) `Code s in
        match Doc.nth 0 (Session.doc s) with
        | Some (Cell.Code _) -> ()
        | _ -> fail "expected Code cell");
    test "clear outputs" (fun () ->
        let c = Cell.code "x" |> Cell.set_outputs [ Cell.Stdout "out" ] in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.clear_outputs (Cell.id c) s in
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some (Cell.Code { outputs; _ }) -> equal int 0 (List.length outputs)
        | _ -> fail "expected Code cell");
    test "clear all outputs" (fun () ->
        let c1 = Cell.code "x" |> Cell.set_outputs [ Cell.Stdout "out1" ] in
        let c2 = Cell.code "y" |> Cell.set_outputs [ Cell.Stdout "out2" ] in
        let doc = Doc.of_cells [ c1; c2 ] in
        let s = Session.create doc in
        let s = Session.clear_all_outputs s in
        List.iter
          (fun cell ->
            match cell with
            | Cell.Code { outputs; _ } -> equal int 0 (List.length outputs)
            | _ -> ())
          (Doc.cells (Session.doc s)));
  ]

let execution_state_tests =
  [
    test "mark running" (fun () ->
        let c = Cell.code "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.mark_running (Cell.id c) s in
        match Session.cell_status (Cell.id c) s with
        | Session.Running -> ()
        | _ -> fail "expected Running");
    test "mark queued" (fun () ->
        let c = Cell.code "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.mark_queued (Cell.id c) s in
        match Session.cell_status (Cell.id c) s with
        | Session.Queued -> ()
        | _ -> fail "expected Queued");
    test "apply output and finish" (fun () ->
        let c = Cell.code "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.mark_running (Cell.id c) s in
        let s = Session.apply_output (Cell.id c) (Cell.Stdout "val x = 1") s in
        let s =
          Session.apply_output (Cell.id c) (Cell.Stderr "more output") s
        in
        let s = Session.finish_execution (Cell.id c) ~success:true s in
        (match Session.cell_status (Cell.id c) s with
        | Session.Idle -> ()
        | _ -> fail "expected Idle after finish");
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some (Cell.Code { outputs; execution_count; _ }) ->
            equal int 2 (List.length outputs);
            equal int 1 execution_count
        | _ -> fail "expected Code cell with outputs");
    test "default status is idle" (fun () ->
        let c = Cell.code "x" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        match Session.cell_status (Cell.id c) s with
        | Session.Idle -> ()
        | _ -> fail "expected Idle");
  ]

let undo_redo_tests =
  [
    test "update_source does not push history" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "changed" s in
        is_false ~msg:"no undo without checkpoint" (Session.can_undo s));
    test "checkpoint enables undo" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        is_false ~msg:"no undo initially" (Session.can_undo s);
        let s = Session.update_source (Cell.id c) "changed" s in
        let s = Session.checkpoint s in
        is_true ~msg:"can undo after checkpoint" (Session.can_undo s);
        let s = Session.undo s in
        (match Doc.find (Cell.id c) (Session.doc s) with
        | Some c -> equal string "original" (Cell.source c)
        | None -> fail "cell not found");
        is_true ~msg:"can redo" (Session.can_redo s));
    test "redo after undo" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "changed" s in
        let s = Session.checkpoint s in
        let s = Session.undo s in
        let s = Session.redo s in
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some c -> equal string "changed" (Cell.source c)
        | None -> fail "cell not found");
    test "structural ops auto-checkpoint" (fun () ->
        let c = Cell.text "a" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        is_false ~msg:"no undo initially" (Session.can_undo s);
        let s = Session.insert_cell ~pos:1 (Cell.text "b") s in
        is_true ~msg:"can undo after insert" (Session.can_undo s);
        let s = Session.undo s in
        equal int 1 (Doc.length (Session.doc s)));
    test "checkpoint is noop when unchanged" (fun () ->
        let doc = Doc.of_cells [ Cell.text "a" ] in
        let s = Session.create doc in
        let s = Session.checkpoint s in
        is_false ~msg:"no undo after noop checkpoint" (Session.can_undo s));
    test "undo on empty history is noop" (fun () ->
        let doc = Doc.of_cells [ Cell.text "a" ] in
        let s = Session.create doc in
        let s2 = Session.undo s in
        equal int (Doc.length (Session.doc s)) (Doc.length (Session.doc s2)));
    test "reload clears history" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "changed" s in
        let s = Session.checkpoint s in
        is_true ~msg:"can undo before reload" (Session.can_undo s);
        let new_doc = Doc.of_cells [ Cell.text "reloaded" ] in
        let s = Session.reload new_doc s in
        is_false ~msg:"no undo after reload" (Session.can_undo s);
        equal int 1 (Doc.length (Session.doc s)));
  ]

(* Displays with ids *)

let pp_output ppf = function
  | Cell.Stdout s -> Format.fprintf ppf "Stdout %S" s
  | Cell.Stderr s -> Format.fprintf ppf "Stderr %S" s
  | Cell.Error s -> Format.fprintf ppf "Error %S" s
  | Cell.Display { mime; id; data } ->
      Format.fprintf ppf "Display {mime = %S; id = %s; data = %S}" mime
        (match id with
        | None -> "None"
        | Some id -> Printf.sprintf "Some %S" id)
        data

let output = Testable.make ~pp:pp_output ~equal:( = )
let svg ?id data = Cell.Display { mime = "image/svg+xml"; id; data }

(* [outputs s] is the outputs of each cell of [s], in order. *)
let outputs s =
  List.map
    (function Cell.Code { outputs; _ } -> outputs | Cell.Text _ -> [])
    (Doc.cells (Session.doc s))

(* [session n] is a session over [n] empty code cells, and their ids. *)
let session n =
  let cells = List.init n (fun i -> Cell.code (string_of_int i)) in
  (Session.create (Doc.of_cells cells), List.map Cell.id cells)

let apply outs s ids =
  List.fold_left
    (fun s (i, o) -> Session.apply_output (List.nth ids i) o s)
    s outs

(* [model n outs] is the outputs of [n] cells after [outs], each a cell index
   and an output: a display with an id replaces the session's display with that
   id, in whichever cell holds it, and every other output is appended to its
   cell. *)
let model n outs =
  let cells = Array.make n [] in
  let holds id =
    List.exists (function
      | Cell.Display { id = Some d; _ } -> d = id
      | _ -> false)
  in
  List.iter
    (fun (i, o) ->
      match o with
      | Cell.Display { id = Some id; _ } -> (
          let same = function
            | Cell.Display { id = Some d; _ } -> d = id
            | _ -> false
          in
          match
            List.find_opt (fun j -> holds id cells.(j)) (List.init n Fun.id)
          with
          | Some j ->
              cells.(j) <- List.map (fun x -> if same x then o else x) cells.(j)
          | None -> cells.(i) <- cells.(i) @ [ o ])
      | _ -> cells.(i) <- cells.(i) @ [ o ])
    outs;
  Array.to_list cells

(* Plain outputs are errors, so that no two coalesce as standard output does.
   Each output's data is its position in the sequence. *)
let gen_outputs =
  let gen = Gen.(list (pair (int_range 0 2) (int_range 0 3))) in
  Gen.map
    (List.mapi (fun k (i, kind) ->
         let data = string_of_int k in
         let o =
           match kind with
           | 0 -> Cell.Error data
           | 1 -> svg data
           | 2 -> svg ~id:"a" data
           | _ -> svg ~id:"b" data
         in
         (i, o)))
    gen
  |> Gen.with_pp (fun ppf outs ->
      Format.pp_print_list
        (fun ppf (i, o) -> Format.fprintf ppf "(%d, %a)" i pp_output o)
        ppf outs)

let display_tests =
  [
    test "a display with an id replaces it in its own cell" (fun () ->
        let s, ids = session 1 in
        let s =
          apply
            [ (0, svg ~id:"a" "1"); (0, Cell.Error "e"); (0, svg ~id:"a" "2") ]
            s ids
        in
        equal
          (list (list output))
          [ [ svg ~id:"a" "2"; Cell.Error "e" ] ]
          (outputs s));
    test "a display with an id replaces it in another cell" (fun () ->
        let s, ids = session 2 in
        let s = apply [ (0, svg ~id:"a" "1"); (1, svg ~id:"a" "2") ] s ids in
        equal (list (list output)) [ [ svg ~id:"a" "2" ]; [] ] (outputs s));
    test "a display without an id never replaces" (fun () ->
        let s, ids = session 2 in
        let s = apply [ (0, svg "1"); (1, svg "1") ] s ids in
        equal (list (list output)) [ [ svg "1" ]; [ svg "1" ] ] (outputs s));
    test "a display's id is free again once its cell is cleared" (fun () ->
        let s, ids = session 2 in
        let s = apply [ (0, svg ~id:"a" "1") ] s ids in
        let s = Session.clear_outputs (List.nth ids 0) s in
        let s = apply [ (1, svg ~id:"a" "2") ] s ids in
        equal (list (list output)) [ []; [ svg ~id:"a" "2" ] ] (outputs s));
    prop "outputs follow the display ids" gen_outputs (fun outs ->
        let replaced =
          List.exists
            (fun (i, o) ->
              match o with
              | Cell.Display { id = Some id; _ } ->
                  List.exists
                    (fun (j, o') ->
                      j <> i
                      &&
                      match o' with
                      | Cell.Display { id = Some id'; _ } -> id = id'
                      | _ -> false)
                    outs
              | _ -> false)
            outs
        in
        cover "an id shows in two cells" replaced;
        let s, ids = session 3 in
        equal (list (list output)) (model 3 outs) (outputs (apply outs s ids)));
  ]

let () =
  exit
    (run "Session"
       [
         group "Basic" basic_tests;
         group "Execution state" execution_state_tests;
         group "Undo/Redo" undo_redo_tests;
         group "Displays with ids" display_tests;
       ])
