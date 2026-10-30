(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let run ~create_kernel doc =
  let doc = ref doc in
  let on_event = function
    | Kernel.Output { cell_id; output } ->
        doc := Doc.add_output cell_id output !doc
    | _ -> ()
  in
  let kernel = create_kernel ~on_event in
  List.iter
    (fun cell ->
      match cell with
      | Cell.Code { id; _ } ->
          Kernel.execute kernel cell;
          doc := Doc.update id Cell.increment_execution_count !doc
      | Cell.Text _ -> ())
    (Doc.cells !doc);
  Kernel.shutdown kernel;
  !doc
