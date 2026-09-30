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

(* The host, and METAL of the GPU family [arch], with or without a residency
   set. *)
let recorded_devices ~arch ~residency_set = function
  | "CPU" -> { Hcq2.target = host_target; queues = None }
  | _ ->
      {
        Hcq2.target = { host_target with device = "METAL"; arch };
        queues = Some (Ops_metal.queues ~host:"CPU" ~arch ~residency_set);
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

(* Each case with its GPU family, residency set and profile. *)
let cases =
  [
    ("chain", "Apple9", true, false);
    ("chain_apple7", "Apple7", true, false);
    ("chain_mac2", "Mac2", true, false);
    ("chain_no_residency_set", "Apple9", false, false);
    ("chain_profile", "Apple9", true, true);
    ("one_profile", "Apple9", true, true);
    ("variable", "Apple9", true, false);
    ("host_split", "Apple9", true, false);
  ]

let recorded =
  group "recorded cases"
    (List.map
       (fun (case, arch, residency_set, profile) ->
         let compiled =
           lazy
             (plain (fun () ->
                  Hcq2.compile_linear ~profile
                    ~devices:(recorded_devices ~arch ~residency_set)
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

(* Loops (DIVERGENCES D30) *)

let uncompiled =
  Renderer.with_compiler (Renderer.Compiler.v Fun.id) (Cstyle.clang host_target)

(* A range of three trips around a kernel on METAL adding one to each window of
   four floats. *)
let ranged () =
  let r = Ops.range (Int 3) [ Ops.unique_num () ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let param slot =
    Ops.param ~shape:[ Int 4 ] ~device:(Single "METAL") slot Float32
  in
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
  let buf () = Ops.new_buffer (Single "METAL") 12 Float32 in
  Ops.end_
    (Ops.call
       (Codegen.to_program kernel uncompiled)
       [ window (buf ()); window (buf ()) ])
    [ r ]

(* Whether [s] holds [sub]. *)
let contains s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* A kernel on METAL of a variable's quarter, rounded up, of threads: its launch
   size is an expression of the variable, which the host program computes on
   each run, as the kernel's compilation committed it. *)
let quarter_sized () =
  let n = Ops.variable "n" (`Int Z.one) (`Int (Z.of_int 1024)) in
  let size = Ops.O.((n + Ops.int 3) // Ops.int 4) in
  let param slot =
    Ops.param ~shape:[ Int 256 ] ~device:(Single "METAL") slot Float32
  in
  let i = Ops.range (Sym size) [ 0 ] in
  let x = Ops.load (Ops.index (param 1) [ i ]) [] in
  let st =
    Ops.store
      (Ops.index (param 0) [ i ])
      (Ops.add x (Ops.float ~dtype:Float32 1.))
  in
  let kernel =
    Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ i ] ]
  in
  let metal =
    Renderer.with_compiler
      (Renderer.Compiler.v Fun.id)
      (Cstyle.metal { host_target with device = "METAL"; arch = "Apple9" })
  in
  let buf () = Ops.new_buffer (Single "METAL") 256 Float32 in
  Ops.call
    (Codegen.to_program kernel metal)
    [ buf (); buf (); Ops.bind n (`Int (Z.of_int 100)) ]

let sizes =
  group "launch sizes"
    [
      test "a launch size of an expression is computed in integers" (fun () ->
          let src =
            host_sources
              (plain (fun () ->
                   Hcq2.compile_linear
                     ~devices:
                       (recorded_devices ~arch:"Apple9" ~residency_set:true)
                     (Ops.v Linear ~src:[ quarter_sized () ])))
          in
          is_false ~msg:"float" (contains src "float"));
    ]

let loops =
  group "loops (D30)"
    [
      test "a range's addresses are integers, profiled or not" (fun () ->
          List.iter
            (fun (arch, residency_set, profile) ->
              let src =
                host_sources
                  (plain (fun () ->
                       Hcq2.compile_linear ~profile
                         ~devices:(recorded_devices ~arch ~residency_set)
                         (Ops.v Linear ~src:[ ranged () ])))
              in
              is_false ~msg:"float" (contains src "float"))
            [
              ("Apple9", true, false);
              ("Apple7", false, false);
              ("Apple9", true, true);
            ]);
    ]

let () = exit (run "Tolk_next.Ops_metal" [ recorded; loops; sizes ])
