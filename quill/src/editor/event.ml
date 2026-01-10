(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type code_execution_status = [ `Success | `Error ]

type code_execution_result = {
  output : string;
  error : string option;
  status : code_execution_status;
}

type t =
  | Document_loaded of { path : string; document : Document.t }
  | Document_load_failed of { path : string; error : string }
  | Document_saved of { path : string }
  | Document_save_failed of { path : string; error : string }
  | Code_execution_completed of {
      block_id : int;
      result : code_execution_result;
    }
  | Clipboard_content_received of { text : string }
  | Clipboard_operation_failed of { error : string }
