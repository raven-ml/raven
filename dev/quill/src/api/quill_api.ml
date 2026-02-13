(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type code_execution_request = { code : string }

let code_execution_request_jsont =
  Jsont.Object.map (fun code -> { code })
  |> Jsont.Object.mem "code" Jsont.string ~enc:(fun r -> r.code)
  |> Jsont.Object.finish

type code_execution_result = {
  output : string;
  error : string option;
  status : [ `Success | `Error ];
}

let status_jsont =
  Jsont.enum [ ("Success", `Success); ("Error", `Error) ]

let code_execution_result_jsont =
  Jsont.Object.map (fun output error status -> { output; error; status })
  |> Jsont.Object.mem "output" Jsont.string ~enc:(fun r -> r.output)
  |> Jsont.Object.opt_mem "error" Jsont.string ~enc:(fun r -> r.error)
  |> Jsont.Object.mem "status" status_jsont ~enc:(fun r -> r.status)
  |> Jsont.Object.finish
