(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

(* A kernel that records the code it is asked to run. *)
let recording_kernel ran ~on_event:_ =
  {
    Kernel.execute = (fun ~cell_id:_ ~code -> ran := code :: !ran);
    interrupt = ignore;
    complete = (fun ~code:_ ~pos:_ -> []);
    type_at = None;
    diagnostics = None;
    is_complete = None;
    status = (fun () -> Kernel.Idle);
    shutdown = ignore;
  }

let execution_count c =
  match c with Cell.Code { execution_count; _ } -> execution_count | _ -> -1

let tests =
  [
    test "runs OCaml cells and no others" (fun () ->
        let ran = ref [] in
        let doc =
          Doc.of_cells
            [
              Cell.text "# Title";
              Cell.code "let x = 1";
              Cell.code ~language:"bash" "quill run notebook.md";
              Cell.code ~language:"" "unlabelled";
              Cell.code "let y = x + 1";
            ]
        in
        let doc = Eval.run ~create_kernel:(recording_kernel ran) doc in
        equal (list string) [ "let x = 1"; "let y = x + 1" ] (List.rev !ran);
        equal (list int) [ -1; 1; 0; 0; 1 ]
          (List.map execution_count (Doc.cells doc)));
  ]

let () = exit (run "Eval" [ group "Languages" tests ])
