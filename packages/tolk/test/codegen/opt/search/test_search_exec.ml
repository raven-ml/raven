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
  let link call =
    let prg = Ops.body call in
    let scale = if applied_opts prg = [] then 1. else 1e-3 in
    (Tolk_engine.link_call ~devices call, scale)
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
          ~beam:
            (Search.beam_search ~link ~time ~clock:(fun (s, _) ->
                 Tolk_engine.clock s))
          (with_info (fun i -> { i with beam = 2 }) k)
          host)
  in
  is_true ~msg:"optimised" (applied_opts prg <> []);
  equal values (run k unoptimised) (run k prg)

let scalar_arguments =
  cases ~name:string_of_int
    "a search runs candidates with named and positional scalars" [ 1; 4 ]
    (fun slot ->
      let z n = `Int (Bigint.of_int n) in
      let named = Ops.variable ~dtype:Int32 "n" (z 3) (z 9) in
      let positional =
        Call.param ~addrspace:(Some Alu) ~vmin_vmax:(z (-7), z (-1)) slot Int32
      in
      let output = Call.param ~shape:[ Ops.Int 4 ] 0 Float32 in
      let r = Ops.range (Int 4) [ 0 ] in
      let value = Ops.cast (Ops.add named positional) Float32 in
      let ast =
        Ops.sink
          ~kernel:(Ops.kernel_info ~beam:2 ())
          [ Ops.end_ (Ops.store (Ops.index output [ r ]) value) [ r ] ]
      in
      let measured = ref 0 in
      let link call = Tolk_engine.link_call ~devices call in
      let time ~vars s =
        let slots = Tolk_engine.slots s in
        Tolk_engine.run ~vars s slots;
        equal (array Dtypes.value)
          (Array.make 4 (`Float 2.))
          (Run.values Float32 (List.hd slots.(0)));
        incr measured;
        1.
      in
      let prg =
        Setting.context
          [ B (Setting.ignore_beam_cache, true) ]
          (fun () ->
            Codegen.to_program
              ~beam:(Search.beam_search ~link ~time ~clock:Tolk_engine.clock)
              ast host)
      in
      greater int ~than:0 !measured;
      (* The chosen program still reads its argument on every call. *)
      List.iter
        (fun scalar ->
          let args =
            List.init (slot + 1) (fun i ->
                if i = slot then Ops.const ~dtype:Int32 (z scalar)
                else
                  Call.param ~shape:[ Int 4 ] ~device:(Single "CPU") i Float32)
          in
          let s = Tolk_engine.link_call ~devices (Ops.call prg args) in
          let slots = Tolk_engine.slots s in
          Tolk_engine.run ~vars:[ ("n", 3) ] s slots;
          equal (array Dtypes.value)
            (Array.make 4 (`Float (float_of_int (3 + scalar))))
            (Run.values Float32 (List.hd slots.(0))))
        [ -7; -1 ])

let () =
  exit
    (Windtrap.run "Tolk.Search on the host"
       [
         scalar_arguments;
         cases ~tags:[ "slow" ] ~name:Fun.id
           "a searched kernel computes what its unoptimised kernel computes"
           [ "symbolic"; "add_small"; "sum_rows"; "variable_rows"; "pad_7x7" ]
           searched;
       ])
