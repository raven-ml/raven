(* Execution tests of Tolk.Search: kernels searched on the host, timed by the
   engine, compute what their unoptimised programs compute. *)

open Windtrap
open Tolk

let host = Cstyle.clang (Tolk_engine.target Rig.host)
let devices _ = Rig.host

let applied_opts prg =
  match Ops.arg (Ops.nth prg 0) with
  | Kernel info -> info.applied_opts
  | _ -> failf "the program %a has no kernel" Ops.pp prg

(* The engine's linking and timing of a kernel's programs on the host, on the
   slots of the first one timed. *)
let timing () =
  let slots = ref None in
  let link call = Tolk_engine.link_call ~devices call in
  let time ~vars s =
    if Option.is_none !slots then slots := Some (Tolk_engine.slots s);
    Tolk_engine.time ~vars s (Option.get !slots)
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
            (Search.beam_search ~link ~time)
          (with_info (fun i -> { i with beam = 2 }) k)
          host)
  in
  is_true ~msg:"optimised" (applied_opts prg <> []);
  equal values (run k unoptimised) (run k prg)

let scalar_arguments =
  cases "a search runs candidates with named and positional scalars"
    ~id:string_of_int [ 1; 4 ] (fun slot ->
      let z n = `Int (Bigint.of_int n) in
      let named = Ops.variable ~dtype:Int32 "n" (z 3) (z 9) in
      let positional =
        Shape.param ~addrspace:Alu ~vmin_vmax:(z (-7), z (-1)) slot Int32
      in
      let output = Shape.param ~shape:[ Ops.Int 4 ] 0 Float32 in
      let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
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
          (Run.values Float32 (Option.get slots.(0)));
        incr measured;
        1.
      in
      let prg =
        Setting.context
          [ B (Setting.ignore_beam_cache, true) ]
          (fun () ->
            Codegen.to_program
              ~beam:(Search.beam_search ~link ~time)
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
                  Shape.param ~shape:[ Int 4 ] ~device:"CPU" i Float32)
          in
          let s = Tolk_engine.link_call ~devices (Ops.call prg args) in
          let slots = Tolk_engine.slots s in
          Tolk_engine.run ~vars:[ ("n", 3) ] s slots;
          equal (array Dtypes.value)
            (Array.make 4 (`Float (float_of_int (3 + scalar))))
            (Run.values Float32 (Option.get slots.(0))))
        [ -7; -1 ])

(* [out[r, i]] sums [j] over [j < x[i]]: a loop whose trips are what the
   storage of [x] holds. Its [rows] axis reads no bound, so a search would have
   candidates that upcast it. *)
let loaded_trips ?beam () =
  let rows = 4 and width = 64 in
  let x = Shape.placeholder ~slot:1 [ width ] Uint8
  and out = Shape.placeholder ~slot:0 [ rows * width ] Float32 in
  let r = Ops.range ~axis_type:Weak (Int rows) [ 0 ]
  and i = Ops.range ~axis_type:Weak (Int width) [ 1 ] in
  let bound = Ops.cast (Ops.load (Ops.index x [ i ]) []) Int32 in
  let j = Ops.range ~axis_type:Weak ~dtype:Int32 (Sym bound) [ 2 ] in
  let sum = Ops.reduce (Ops.cast j Float32) Op.Add [ j ] in
  let at = Ops.index out [ Ops.O.((r * Ops.int width) + i) ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"loaded_trips" ?beam ())
    [ Ops.end_ (Ops.store at sum) [ r; i ] ]

let opt_names prg = List.map (Format.asprintf "%a" Opt.pp) (applied_opts prg)

let memory_bound =
  test
    "a kernel whose loop's bound loads from memory is not searched, and takes \
     the heuristic's optimisations"
    (fun () ->
      let measured = ref 0 in
      let link call = Tolk_engine.link_call ~devices call in
      let time ~vars:_ _ =
        incr measured;
        1.
      in
      let searched =
        Setting.context
          [ B (Setting.ignore_beam_cache, true) ]
          (fun () ->
            Codegen.to_program
              ~beam:(Search.beam_search ~link ~time)
              (loaded_trips ~beam:2 ()) host)
      in
      equal int ~msg:"programs timed" 0 !measured;
      let hand = opt_names (Codegen.to_program (loaded_trips ()) host) in
      greater int ~msg:"the heuristic's optimisations" ~than:0
        (List.length hand);
      equal (list string) hand (opt_names searched))

let () =
  exit
    (Windtrap.run "Tolk.Search on the host"
       [
         scalar_arguments;
         memory_bound;
         slow
         @@ cases
              "a searched kernel computes what its unoptimised kernel computes"
              ~id:Fun.id
              [
                "symbolic"; "add_small"; "sum_rows"; "variable_rows"; "pad_7x7";
              ]
              searched;
       ])
