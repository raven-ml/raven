(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let pp_event ppf = function
  | Kernel.Output { cell_id; output = Cell.Stdout s } ->
      Format.fprintf ppf "Output (%S, Stdout %S)" cell_id s
  | Kernel.Output { cell_id; output = Cell.Error s } ->
      Format.fprintf ppf "Output (%S, Error %S)" cell_id s
  | Kernel.Output { cell_id; _ } -> Format.fprintf ppf "Output (%S, _)" cell_id
  | Kernel.Finished { cell_id; success } ->
      Format.fprintf ppf "Finished (%S, %b)" cell_id success
  | Kernel.Status_changed _ -> Format.fprintf ppf "Status_changed"

let event = Testable.make ~pp:pp_event ~equal:( = )
let item = Kernel.{ label = "x"; kind = Value; detail = "int" }
let info = Kernel.{ typ = "int"; doc = None; from_pos = 0; to_pos = 1 }

let diagnostic =
  Kernel.{ from_pos = 0; to_pos = 1; severity = Error; message = "bad" }

(* An OCaml kernel whose backend prints the code it runs, answers every query
   and records the events it reports. *)
let kernel () =
  let events = ref [] in
  let on_event e = events := e :: !events in
  let k =
    Kernel.make ~language:"ocaml" ~on_event
      ~execute:(fun ~cell_id ~code ->
        on_event (Output { cell_id; output = Stdout code });
        on_event (Finished { cell_id; success = true }))
      ~interrupt:ignore
      ~complete:(fun ~code:_ ~pos:_ -> [ item ])
      ~type_at:(fun ~code:_ ~pos:_ -> Some info)
      ~diagnostics:(fun ~code:_ -> [ diagnostic ])
      ~is_complete:(fun _ -> false)
      ~status:(fun () -> Kernel.Idle)
      ~shutdown:ignore ()
  in
  (k, fun () -> List.rev !events)

let python = Cell.code ~id:"p" ~language:"python" "print(1)"
let ocaml = Cell.code ~id:"o" ~language:"ocaml" "let x = 1"

let execute_tests =
  [
    test "a cell in the kernel's language runs on its backend" (fun () ->
        let k, events = kernel () in
        Kernel.execute k ocaml;
        equal (list event)
          [
            Output { cell_id = "o"; output = Stdout "let x = 1" };
            Finished { cell_id = "o"; success = true };
          ]
          (events ()));
    test "a cell in another language is refused" (fun () ->
        let k, events = kernel () in
        Kernel.execute k python;
        equal (list event)
          [
            Output
              {
                cell_id = "p";
                output =
                  Error
                    "no kernel runs python code: this notebook's kernel runs \
                     ocaml";
              };
            Finished { cell_id = "p"; success = false };
          ]
          (events ()));
    test "a text cell is not run" (fun () ->
        let k, _ = kernel () in
        raises (Invalid_argument "Quill.Kernel.execute: a text cell") (fun () ->
            Kernel.execute k (Cell.text "# Title")));
  ]

let completion =
  Testable.make
    ~pp:(fun ppf (i : Kernel.completion_item) ->
      Format.fprintf ppf "%s : %s" i.label i.detail)
    ~equal:( = )

let type_info =
  Testable.make
    ~pp:(fun ppf (i : Kernel.type_info) ->
      Format.fprintf ppf "%s at %d-%d" i.typ i.from_pos i.to_pos)
    ~equal:( = )

let diagnostic_t =
  Testable.make
    ~pp:(fun ppf (d : Kernel.diagnostic) ->
      Format.pp_print_string ppf d.message)
    ~equal:( = )

let query_tests =
  [
    cases "queries reach the backend for the kernel's language only"
      ~id:(fun (name, _, _) -> name)
      [
        ("ocaml", ocaml, true);
        ("python", python, false);
        ("text", Cell.text "let x = 1", false);
      ]
      (fun (_, cell, runs) ->
        let k, _ = kernel () in
        equal (list completion)
          (if runs then [ item ] else [])
          (Kernel.complete k cell ~pos:0);
        equal (option type_info)
          (if runs then Some info else None)
          (Kernel.type_at k cell ~pos:0);
        equal (list diagnostic_t)
          (if runs then [ diagnostic ] else [])
          (Kernel.diagnostics k cell);
        equal bool (not runs) (Kernel.is_complete k cell));
    test "a kernel's language is a word" (fun () ->
        raises (Invalid_argument "Quill.Kernel.make: \"a b\" is not a language")
          (fun () ->
            Kernel.make ~language:"a b" ~on_event:ignore
              ~execute:(fun ~cell_id:_ ~code:_ -> ())
              ~interrupt:ignore
              ~complete:(fun ~code:_ ~pos:_ -> [])
              ~status:(fun () -> Kernel.Idle)
              ~shutdown:ignore ()));
    test "a backend without queries answers none" (fun () ->
        let k =
          Kernel.make ~language:"ocaml" ~on_event:ignore
            ~execute:(fun ~cell_id:_ ~code:_ -> ())
            ~interrupt:ignore
            ~complete:(fun ~code:_ ~pos:_ -> [])
            ~status:(fun () -> Kernel.Idle)
            ~shutdown:ignore ()
        in
        equal string "ocaml" (Kernel.language k);
        equal (option type_info) None (Kernel.type_at k ocaml ~pos:0);
        equal (list diagnostic_t) [] (Kernel.diagnostics k ocaml);
        equal bool true (Kernel.is_complete k ocaml));
  ]

let () =
  exit
    (run "Kernel"
       [ group "Execute" execute_tests; group "Queries" query_tests ])
