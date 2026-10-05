(* Execution tests of Tolk.Search: kernels searched on the host, timed by the
   engine, compute what their unoptimised programs compute. *)

open Windtrap
open Tolk

let host = Cstyle.clang (Tolk_engine.target Nx_device.host)
let devices = Tolk_engine.device [ ("CPU", Nx_device.host) ]

let applied_opts prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel info -> info.applied_opts
  | _ -> failf "the program %a has no kernel" Ops.pp prg

(* The engine's linking and timing of a kernel's programs on the host, on the
   slots of the first one timed, with the samples of an optimised program scaled
   down a thousandfold, so that a search progresses past its kernel whatever the
   host's noise: when a search stops is the Search suite's. *)
let timing () =
  let slots = ref None in
  let link prg =
    let scale = if applied_opts prg = [] then 1. else 1e-3 in
    (Tolk_engine.link_program ~devices "CPU" prg, scale)
  in
  let time ~vars (s, scale) =
    if Option.is_none !slots then slots := Some (Tolk_engine.slots s);
    Tolk_engine.time ~vars s (Option.get !slots) *. scale
  in
  (link, time)

let with_info f k =
  match Ops.arg k with
  | Kernel info -> Ops.replace k ~arg:(Kernel (f info))
  | _ -> failf "%a is no kernel" Ops.pp k

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
  (* A search times anew: a result kept by an earlier run would hide what this
     one finds. *)
  let link, time = timing () in
  let prg =
    Setting.context
      [ B (Setting.ignore_beam_cache, true) ]
      (fun () ->
        Codegen.to_program
          ~beam:(Search.beam_search ~link ~time)
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
