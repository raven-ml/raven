open Windtrap
open Tolk_next

let plain f = Helpers.context [ B (Helpers.no_color, true) ] f

let is_batch c =
  match Ops.arg (Ops.without_after c) with
  | Call { aux = Some _; _ } -> true
  | _ -> false

(* Recorded cases *)

let host_target =
  {
    Helpers.Target.device = "CPU";
    renderer = "";
    arch = "x86_64,x86-64";
    interface = "";
    indices = "";
  }

(* The host, and CUDA on sm_89, which reaches the host's memory. *)
let recorded_devices = function
  | "CPU" -> { Hcq2.target = host_target; queues = None }
  | _ ->
      {
        Hcq2.target = { host_target with device = "CUDA"; arch = "sm_89" };
        queues =
          Some (Ops_cuda.queues ~host:"CPU" ~reaches:(fun d -> d = "CPU"));
      }

(* A kernel's profile key is its program's BLAKE2 digest, where tinygrad's is a
   SHA-256 (DIVERGENCES D12): recorded graphs are compared without them. *)
let without_profile_keys u =
  let unkeyed c =
    match Ops.arg c with
    | Call ({ aux = Some info; _ } as ci) ->
        let kernels =
          List.map
            (fun (k : Ops.hcq_kernel) -> { k with profile_key = None })
            info.kernels
        in
        Some
          ( c,
            Ops.replace
              ~arg:(Call { ci with aux = Some { info with kernels } })
              c )
    | _ -> None
  in
  Ops.substitute u (List.filter_map unkeyed (Ops.toposort u))

let host_sources linear =
  String.concat ""
    (List.map
       (fun b ->
         match Ops.arg (Ops.nth (Ops.nth (Ops.without_after b) 0) 2) with
         | String src -> src
         | _ -> fail "a compiled host program holds its source")
       (List.filter is_batch (Ops.src linear)))

(* Each case with whether it profiles. *)
let cases =
  [
    ("chain", false);
    ("chain_profile", true);
    ("variable", false);
    ("copy_in", false);
    ("copy_in_profile", true);
    ("host_split", false);
  ]

let recorded =
  group "recorded cases"
    (List.map
       (fun (case, profile) ->
         let compiled =
           lazy
             (plain (fun () ->
                  Hcq2.compile_linear ~profile ~devices:recorded_devices
                    (Golden.sink (case ^ "_prepared.golden"))))
         in
         group case
           [
             test (case ^ "_compiled.golden") (fun () ->
                 let golden = Golden.sink (case ^ "_compiled.golden") in
                 let same u = Graph.to_string (without_profile_keys u) in
                 equal text (same golden)
                   (same
                      (Uops.placeholders_like golden
                         (Uops.binaries_as_sources (Lazy.force compiled)))));
             Golden.text (case ^ "_host.golden") (fun () ->
                 host_sources (Lazy.force compiled));
           ])
       cases)

(* DIVERGENCES D36: a function's address is a word *)

(* A renderer whose binary is its source's bytes: batching compiles nothing. *)
let uncompiled =
  Renderer.with_compiler (Renderer.Compiler.v Fun.id) (Cstyle.clang host_target)

(* The call of the kernel adding one to [inp] into [out], compiled. *)
let adds out inp =
  let device = Option.get (Ops.device out) in
  let param slot = Ops.param ~shape:[ Int 4 ] ~device slot Float32 in
  let i = Ops.range (Int 4) [ 0 ] in
  let x = Ops.load (Ops.index (param 1) [ i ]) [] in
  let st =
    Ops.store
      (Ops.index (param 0) [ i ])
      (Ops.add x (Ops.float ~dtype:Float32 1.))
  in
  let kernel =
    Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ i ] ]
  in
  Ops.call (Codegen.to_program kernel uncompiled) [ out; inp ]

let storage d = Ops.new_buffer (Single d) 4 Float32

let function_words =
  group "function words (D36)"
    [
      test
        "a batch over two devices reads a kernel's function from a word of each"
        (fun () ->
          let calls =
            List.map
              (fun d -> adds (storage d) (storage d))
              [ "CUDA"; "CUDA:1" ]
          in
          let compiled =
            plain (fun () ->
                Hcq2.compile_linear ~devices:recorded_devices
                  (Ops.v Linear ~src:calls))
          in
          let words =
            List.filter
              (fun u ->
                match Ops.tag u with
                | Some (Tuple (String "function" :: _)) -> true
                | _ -> false)
              (Ops.toposort compiled)
          in
          let placement u =
            match Ops.device u with
            | Some (Single d) | Some (Multi [ d ]) -> d
            | _ -> fail "a function's word is on one device"
          in
          equal (list string) [ "CUDA"; "CUDA:1" ]
            (List.sort_uniq String.compare (List.map placement words)));
    ]

let () = exit (run "Tolk_next.Ops_cuda" [ recorded; function_words ])
