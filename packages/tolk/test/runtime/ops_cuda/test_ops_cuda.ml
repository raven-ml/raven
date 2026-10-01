open Windtrap
open Tolk

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
         (* Each test compiles afresh: a mutant must not be answered from an
            earlier compilation. *)
         let compiled () =
           plain (fun () ->
               Hcq2.compile_linear ~profile ~devices:recorded_devices
                 (Golden.sink (case ^ "_prepared.golden")))
         in
         group case
           [
             test (case ^ "_compiled.golden") (fun () ->
                 let golden = Golden.sink (case ^ "_compiled.golden") in
                 let same u = Graph.to_string (without_profile_keys u) in
                 equal text (same golden)
                   (same
                      (Uops.placeholders_like golden
                         (Uops.binaries_as_sources (compiled ())))));
             Golden.text (case ^ "_host.golden") (fun () ->
                 host_sources (compiled ()));
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

(* Whether [s] holds [sub]. *)
let contains s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* A range of three trips around a launch on CUDA, each trip on its own window
   of four floats. *)
let ranged () =
  let r = Ops.range (Int 3) [ 7 ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let buf () = Ops.new_buffer (Single "CUDA") 12 Float32 in
  Ops.end_ (adds (window (buf ())) (window (buf ()))) [ r ]

(* The kernel adding one, alone. *)
let once () = adds (storage "CUDA") (storage "CUDA")

let loops =
  group "loops (D30)"
    [
      test "a range is a loop of the host program around its launches"
        (fun () ->
          let src =
            host_sources
              (plain (fun () ->
                   Hcq2.compile_linear ~devices:recorded_devices
                     (Ops.v Linear ~src:[ ranged () ])))
          in
          (* The launch's extra words are five 64-bit words a trip. *)
          is_true ~msg:"a loop of three trips" (contains src "< 3; Lidx");
          is_true ~msg:"each trip's extra words" (contains src "*40)))"));
      test "a loop's trip is its own launches' words, after the launches before"
        (fun () ->
          let src =
            host_sources
              (plain (fun () ->
                   Hcq2.compile_linear ~devices:recorded_devices
                     (Ops.v Linear ~src:[ once (); ranged () ])))
          in
          (* Each trip's extra words follow the launch's before the loop. *)
          is_true ~msg:"each trip's extra words" (contains src "*40)+40)"));
      test "a range's addresses are integers, profiled or not" (fun () ->
          List.iter
            (fun profile ->
              let src =
                host_sources
                  (plain (fun () ->
                       Hcq2.compile_linear ~profile ~devices:recorded_devices
                         (Ops.v Linear ~src:[ ranged () ])))
              in
              is_false ~msg:"float" (contains src "float"))
            [ false; true ]);
    ]

let () = exit (run "Tolk.Ops_cuda" [ recorded; function_words; loops ])
