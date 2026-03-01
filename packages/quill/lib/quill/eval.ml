(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let run ~create_kernel doc =
  let doc = ref doc in
  let on_event = function
    | Kernel.Output { cell_id; output } ->
        doc := Doc.update cell_id (Cell.append_output output) !doc
    | _ -> ()
  in
  let (kernel : Kernel.t) = create_kernel ~on_event in
  List.iter
    (fun cell ->
      match cell with
      | Cell.Code { id; source; _ } ->
          kernel.execute ~cell_id:id ~code:source;
          doc := Doc.update id Cell.increment_execution_count !doc
      | Cell.Text _ -> ())
    (Doc.cells !doc);
  kernel.shutdown ();
  !doc
