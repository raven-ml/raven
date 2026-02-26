(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let run ~create_kernel doc =
  let outputs = ref [] in
  let on_event = function
    | Kernel.Output { output; cell_id = _ } -> outputs := output :: !outputs
    | _ -> ()
  in
  let (kernel : Kernel.t) = create_kernel ~on_event in
  let doc = ref doc in
  List.iter
    (fun cell ->
      match cell with
      | Cell.Code { id; source; language = _; outputs = _; execution_count = _ }
        ->
          outputs := [];
          kernel.execute ~cell_id:id ~code:source;
          let cell_outputs = List.rev !outputs in
          doc :=
            Doc.update id
              (fun c ->
                Cell.set_outputs cell_outputs (Cell.increment_execution_count c))
              !doc
      | Cell.Text _ -> ())
    (Doc.cells !doc);
  kernel.shutdown ();
  !doc
