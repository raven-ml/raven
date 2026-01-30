(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type block_status = [ `Idle | `Running ]

type selection_status =
  [ `None | `Caret | `Range_single | `Range_start | `Range_end | `Range_middle ]

type block = {
  id : int;
  focused : bool;
  status : block_status;
  selection : selection_status;
  content : Document.block_content;
}

type document = block list

type diff =
  | Added of block
  | Removed of block
  | Updated of { before : block; after : block }

val of_state : State.t -> document
val diff : document -> document -> diff list
