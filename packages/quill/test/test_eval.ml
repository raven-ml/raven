(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

(* An OCaml kernel that records the code it is asked to run and finishes each
   run. *)
let recording_kernel ran ~on_event =
  Kernel.make ~language:"ocaml" ~on_event
    ~execute:(fun ~cell_id ~code ->
      ran := code :: !ran;
      on_event (Kernel.Finished { cell_id; success = true }))
    ~interrupt:ignore
    ~complete:(fun ~code:_ ~pos:_ -> [])
    ~status:(fun () -> Kernel.Idle)
    ~shutdown:ignore ()

let execution_count c =
  match c with Cell.Code { execution_count; _ } -> execution_count | _ -> -1

let outputs c = match c with Cell.Code { outputs; _ } -> outputs | _ -> []

let pp_output ppf = function
  | Cell.Stdout s -> Format.fprintf ppf "Stdout %S" s
  | Cell.Stderr s -> Format.fprintf ppf "Stderr %S" s
  | Cell.Error s -> Format.fprintf ppf "Error %S" s
  | Cell.Display { mime; data; _ } ->
      Format.fprintf ppf "Display (%S, %S)" mime data

let output = Testable.make ~pp:pp_output ~equal:( = )

let tests =
  [
    test "runs the kernel's cells in order and refuses the others" (fun () ->
        let ran = ref [] in
        let doc =
          Doc.of_cells
            [
              Cell.text "# Title";
              Cell.code ~language:"ocaml" "let x = 1";
              Cell.code ~language:"bash" "quill run notebook.md";
              Cell.code ~language:"python" "print(1)";
              Cell.code ~language:"ocaml" "let y = x + 1";
            ]
        in
        let doc = Eval.run ~create_kernel:(recording_kernel ran) doc in
        equal (list string) [ "let x = 1"; "let y = x + 1" ] (List.rev !ran);
        equal
          (list (list output))
          [
            [];
            [];
            [
              Cell.Error
                "no kernel runs bash code: this notebook's kernel runs ocaml";
            ];
            [
              Cell.Error
                "no kernel runs python code: this notebook's kernel runs ocaml";
            ];
            [];
          ]
          (List.map outputs (Doc.cells doc));
        equal (list int) [ -1; 1; 1; 1; 1 ]
          (List.map execution_count (Doc.cells doc)));
  ]

let () = exit (run "Eval" [ group "Languages" tests ])
