(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let run ~create_kernel doc =
  let doc = ref doc in
  let cur_id = ref "" in
  let on_event = function
    | Kernel.Output { output; cell_id = _ } ->
        doc := Doc.update !cur_id (Cell.append_output output) !doc
    | _ -> ()
  in
  let (kernel : Kernel.t) = create_kernel ~on_event in
  List.iter
    (fun cell ->
      match cell with
      | Cell.Code { id; source; language = _; outputs = _; execution_count = _ }
        ->
          cur_id := id;
          kernel.execute ~cell_id:id ~code:source;
          doc := Doc.update id Cell.increment_execution_count !doc
      | Cell.Text _ -> ())
    (Doc.cells !doc);
  kernel.shutdown ();
  !doc
