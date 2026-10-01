(* Execution tests of Tolk.Search: kernels searched on the host, timed by the
   engine, compute what their unoptimised programs compute. *)

open Windtrap
open Tolk

let host = Cstyle.clang (Tolk_engine.target Nx_device.host)
let devices = Tolk_engine.device [ ("CPU", Nx_device.host) ]
let measure ~cold ~vars prg = Tolk_engine.measure ~cold ~vars ~devices "CPU" prg

let with_info f k =
  match Ops.arg k with
  | Kernel info -> Ops.replace k ~arg:(Kernel (f info))
  | _ -> failf "%a is no kernel" Ops.pp k

let applied_opts prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel info -> info.applied_opts
  | _ -> failf "the program %a has no kernel" Ops.pp prg

let run k prg =
  Run.on_host ~vars:(Kernel_opts.variables k) prg (Kernel_opts.inputs k)

let values = list (pair int (array Dtypes.value))

let searched name =
  let k = Golden.sink (name ^ ".golden") in
  let unoptimised =
    Codegen.to_program
      (with_info (fun i -> { i with opts_to_apply = Some [] }) k)
      host
  in
  let prg =
    Helpers.context
      [ B (Helpers.cachelevel, 0) ]
      (fun () ->
        Codegen.to_program
          ~beam:(Search.beam_search ~measure)
          (with_info (fun i -> { i with beam = 2 }) k)
          host)
  in
  is_true ~msg:"optimised" (applied_opts prg <> []);
  equal values (run k unoptimised) (run k prg)

let () =
  exit
    (Windtrap.run "Tolk.Search on the host"
       [
         cases ~tags:[ "slow" ] ~name:Fun.id
           "a searched kernel computes what its unoptimised kernel computes"
           [ "symbolic"; "add_small"; "sum_rows"; "variable_rows"; "pad_7x7" ]
           searched;
       ])
