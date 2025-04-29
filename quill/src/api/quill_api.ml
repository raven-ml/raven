type code_execution_request = { code : string }
[@@deriving yojson { strict = false }]

type code_execution_result = {
  output : string;
  error : string option;
  status : [ `Success | `Error ];
}
[@@deriving yojson]
