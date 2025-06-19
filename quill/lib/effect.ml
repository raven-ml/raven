type t =
  | Execute_code of {
      block_id : Document.block_id;
      code : string;
      language : string option;
      callback : Execution.execution_result -> Command.t;
    }
  | Save_document
  | Load_document of string
