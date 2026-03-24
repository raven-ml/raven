(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Unit tests for Gpudims.

   Every _check_grouped_dims assertion is covered. The Z3 bijectivity proof
   is replaced by exhaustive enumeration (all test products <= 131072). *)

open Windtrap
open Tolk
open Tolk_ir
module K = Kernel
module D = Dtype
module C = Const
module Ak = Axis_kind

(* Helpers *)

let idx n = K.const (C.int D.index n)
let global_fptr = D.ptr_of D.float32 ~addrspace:Global ~size:(-1)

let kernel_info ?(dont_use_locals = false) () =
  {
    K.name = "";
    axis_kinds = [];
    dont_use_locals;
    applied_opts = [];
    opts_to_apply = None;
    estimates = None;
  }

let wrap_sink ?(ki = kernel_info ()) srcs = K.sink ~kernel_info:ki srcs

(* Expression evaluator *)

(* Evaluate a Kernel expression tree by substituting integer values for
   SPECIAL nodes.  Only handles the node types produced by
   get_grouped_dims: Const, Special, Add, Mul, Idiv, Mod. *)
let rec eval_expr (env : Special_dim.t -> int) (node : K.t) : int =
  match K.view node with
  | K.Const { value; _ } ->
      (match C.view value with C.Int n -> Int64.to_int n | _ -> failwith "eval_expr: not int")
  | K.Special { dim; _ } -> env dim
  | K.Binary { op = `Add; lhs; rhs; _ } -> eval_expr env lhs + eval_expr env rhs
  | K.Binary { op = `Mul; lhs; rhs; _ } -> eval_expr env lhs * eval_expr env rhs
  | K.Binary { op = `Idiv; lhs; rhs; _ } -> eval_expr env lhs / eval_expr env rhs
  | K.Binary { op = `Mod; lhs; rhs; _ } -> eval_expr env lhs mod eval_expr env rhs
  | _ -> failwith "eval_expr: unexpected node kind"

(* SPECIAL collection *)

(* Collect unique SPECIAL (dim, size) pairs from a list of index expressions,
   sorted by dim.  Deduplication is by Special_dim equality. *)
let collect_specials idxs =
  let all =
    List.concat_map
      (fun root ->
        List.filter_map
          (fun n ->
            match K.view n with
            | K.Special { dim; size; _ } -> Some (dim, K.const_to_int size)
            | _ -> None)
          (K.toposort root))
      idxs
  in
  let deduped =
    List.fold_left
      (fun acc (dim, size) ->
        if List.exists (fun (d, _) -> Special_dim.equal d dim) acc then acc
        else (dim, size) :: acc)
      [] all
  in
  List.sort (fun (a, _) (b, _) -> Special_dim.compare a b) deduped

let special_sizes idxs = List.map snd (collect_specials idxs)

(* Bijectivity verifier *)

(* Exhaustive check: for every valid combination of SPECIAL values, compute the
   flat multi-dimensional index into the original dims and verify that the
   mapping is a bijection (every flat index in [0, total) is hit exactly
   once). *)
let verify_bijectivity idxs (dims : int array) =
  let n = Array.length dims in
  let total = Array.fold_left ( * ) 1 dims in
  let specials = collect_specials idxs in
  let spec_dims = List.map fst specials in
  let spec_sizes = List.map snd specials in
  let seen = Array.make total false in
  (* suffix products: prod(dims[i+1..n-1]) *)
  let suffix = Array.make (n + 1) 1 in
  for i = n - 1 downto 0 do
    suffix.(i) <- suffix.(i + 1) * dims.(i)
  done;
  (* Enumerate all SPECIAL value combinations *)
  let rec enumerate vals remaining =
    match remaining with
    | [] ->
        let bindings = List.combine spec_dims (List.rev vals) in
        let env dim =
          let rec find = function
            | (d, v) :: _ when Special_dim.equal d dim -> v
            | _ :: rest -> find rest
            | [] -> failwith "eval_expr: unknown SPECIAL dim"
          in
          find bindings
        in
        let flat = ref 0 in
        List.iteri
          (fun i idx_expr ->
            flat := !flat + (eval_expr env idx_expr * suffix.(i + 1)))
          idxs;
        if !flat < 0 || !flat >= total then
          failwith
            (Printf.sprintf "bijectivity: flat=%d out of bounds [0,%d)" !flat total);
        if seen.(!flat) then
          failwith (Printf.sprintf "bijectivity: flat=%d already seen (not injective)" !flat);
        seen.(!flat) <- true
    | size :: rest ->
        for v = 0 to size - 1 do
          enumerate (v :: vals) rest
        done
  in
  enumerate [] spec_sizes;
  Array.iteri
    (fun i b ->
      if not b then
        failwith (Printf.sprintf "bijectivity: flat=%d never hit (not surjective)" i))
    seen

(* Unified check (mirrors _check_grouped_dims) *)

(* Calls get_grouped_dims, asserts len(idxs)==len(dims), asserts SPECIAL
   sizes match expected, and verifies bijectivity. *)
let check_grouped_dims ?(assert_same_length = true) prefix dims max_sizes reverse expected_sizes =
  let kt_dims = Array.map (fun n -> idx n) dims in
  let idxs = Gpudims.get_grouped_dims prefix kt_dims max_sizes ~reverse in
  equal int (List.length idxs) (Array.length dims)
    ~msg:"idxs length should equal dims length";
  let sizes = special_sizes idxs in
  if assert_same_length then begin
    let num_specials = List.length (collect_specials idxs) in
    equal int num_specials (min num_specials (Array.length dims))
      ~msg:"unique SPECIAL count should not exceed dims count"
  end;
  equal (list int) sizes expected_sizes ~msg:"SPECIAL sizes";
  verify_bijectivity idxs dims

(* Group 1: no-op *)

let noop_tests =
  group "no-op"
    [
      test "single dim fits" (fun () ->
          check_grouped_dims "gidx" [| 2 |] (Some [ 16; 16; 16 ]) false [ 2 ]);
      test "two dims fit" (fun () ->
          check_grouped_dims "gidx" [| 2; 3 |] (Some [ 16; 16; 16 ]) false [ 2; 3 ]);
    ]

(* Group 2: reverse *)

let reverse_tests =
  group "reverse"
    [
      test "reverse two dims" (fun () ->
          check_grouped_dims "gidx" [| 2; 3 |] (Some [ 16; 16; 16 ]) true [ 3; 2 ]);
      test "three dims not reversed" (fun () ->
          check_grouped_dims "gidx" [| 2; 3; 4 |] (Some [ 16; 16; 16 ]) false
            [ 2; 3; 4 ]);
    ]

(* Group 3: splitting (len(dims)==len(max)) *)

let split_same_len_tests =
  group "splitting same-length"
    [
      test "(64,3,4) / (16,16,16)" (fun () ->
          check_grouped_dims "gidx" [| 64; 3; 4 |] (Some [ 16; 16; 16 ]) false
            [ 16; 12; 4 ]);
      test "(64,3,4) / (16,4,16)" (fun () ->
          check_grouped_dims "gidx" [| 64; 3; 4 |] (Some [ 16; 4; 16 ]) false
            [ 16; 3; 16 ]);
      test "(64,3,4) reversed / (16,16,16)" (fun () ->
          check_grouped_dims "gidx" [| 64; 3; 4 |] (Some [ 16; 16; 16 ]) true
            [ 16; 3; 16 ]);
      test "(128,3,4) / (16,4,256)" (fun () ->
          check_grouped_dims "gidx" [| 128; 3; 4 |] (Some [ 16; 4; 256 ]) false
            [ 16; 3; 32 ]);
      test "(4,4,512) / (16,4,256)" (fun () ->
          check_grouped_dims "gidx" [| 4; 4; 512 |] (Some [ 16; 4; 256 ]) false
            [ 8; 4; 256 ]);
      test "(5,12,7) / (8,4,16)" (fun () ->
          check_grouped_dims "gidx" [| 5; 12; 7 |] (Some [ 8; 4; 16 ]) false
            [ 10; 3; 14 ]);
    ]

(* Group 4: grouping preferred *)

let grouping_preferred_tests =
  group "grouping preferred"
    [
      test "(512,4,2) / (8192,2,2)" (fun () ->
          check_grouped_dims "gidx" [| 512; 4; 2 |] (Some [ 8192; 2; 2 ]) false
            [ 2048; 2 ]);
    ]

(* Group 5: expansion (len(dims) < len(max)) *)

let expansion_tests =
  group "expansion"
    [
      test "(128,) -> (16,8)" (fun () ->
          check_grouped_dims ~assert_same_length:false "gidx" [| 128 |]
            (Some [ 16; 16; 256 ]) false [ 16; 8 ]);
      test "(65536,) -> (16,16,256)" (fun () ->
          check_grouped_dims ~assert_same_length:false "gidx" [| 65536 |]
            (Some [ 16; 16; 256 ]) false [ 16; 16; 256 ]);
      test "(65536,2) -> (32768,4)" (fun () ->
          check_grouped_dims ~assert_same_length:false "gidx" [| 65536; 2 |]
            (Some [ 65535; 65535; 65535 ]) false [ 32768; 4 ]);
      test "(121,) -> (11,11) sqrt factor" (fun () ->
          check_grouped_dims ~assert_same_length:false "gidx" [| 121 |]
            (Some [ 12; 12; 12 ]) false [ 11; 11 ]);
      test "(128,128) -> (16,16,64)" (fun () ->
          check_grouped_dims ~assert_same_length:false "gidx" [| 128; 128 |]
            (Some [ 16; 16; 256 ]) false [ 16; 16; 64 ]);
    ]

(* Group 6: contraction (len(dims) > len(max)) *)

let contraction_tests =
  group "contraction"
    [
      test "(2,3,4,5) / (16,16,16)" (fun () ->
          check_grouped_dims "gidx" [| 2; 3; 4; 5 |] (Some [ 16; 16; 16 ]) false
            [ 6; 4; 5 ]);
      test "(2,3,4,5) / (32,16,16) reversed" (fun () ->
          check_grouped_dims "gidx" [| 2; 3; 4; 5 |] (Some [ 32; 16; 16 ]) true
            [ 20; 3; 2 ]);
      test "(2,3,4,5) / (4,16,16) left too small" (fun () ->
          check_grouped_dims "gidx" [| 2; 3; 4; 5 |] (Some [ 4; 16; 16 ]) false
            [ 2; 12; 5 ]);
      test "(2,3,4,5) / (16,16,16) reversed" (fun () ->
          check_grouped_dims "gidx" [| 2; 3; 4; 5 |] (Some [ 16; 16; 16 ]) true
            [ 5; 12; 2 ]);
    ]

(* Group 7: error cases *)

let is_failure = function Failure _ -> true | _ -> false

let error_tests =
  group "errors"
    [
      test "prime dim 23 unfactorable" (fun () ->
          raises_match is_failure (fun () ->
              ignore
                (Gpudims.get_grouped_dims "gidx" (Array.map idx [| 23 |]) (Some [ 16; 16; 16 ])
                   ~reverse:false)));
      test "unfactorable (128,3,4) / (16,2,2)" (fun () ->
          raises_match is_failure (fun () ->
              ignore
                (Gpudims.get_grouped_dims "gidx" (Array.map idx [| 128; 3; 4 |])
                   (Some [ 16; 2; 2 ]) ~reverse:false)));
      test "too many dims (2,3,4,5,6)" (fun () ->
          raises_match is_failure (fun () ->
              ignore
                (Gpudims.get_grouped_dims "gidx" (Array.map idx [| 2; 3; 4; 5; 6 |])
                   (Some [ 16; 16; 16 ]) ~reverse:false)));
    ]

(* Group 8: direct-mapped SPECIAL *)

(* When (2,3,4,5) contracts to 3 SPECIAL dims (6,4,5), unmerged dims (4,5)
   map directly to SPECIAL ops. *)

let direct_special_tests =
  group "direct-mapped SPECIAL"
    [
      test "unmerged dims are bare SPECIAL" (fun () ->
          let idxs =
            Gpudims.get_grouped_dims "gidx" (Array.map idx [| 2; 3; 4; 5 |])
              (Some [ 16; 16; 16 ]) ~reverse:false
          in
          (match K.view (List.nth idxs 2) with
          | K.Special _ -> ()
          | _ -> fail "expected SPECIAL for idxs[2]");
          (match K.view (List.nth idxs 3) with
          | K.Special _ -> ()
          | _ -> fail "expected SPECIAL for idxs[3]"));
    ]

(* Group 9: max_sizes=None passthrough *)

let none_passthrough_tests =
  group "max_sizes=None"
    [
      test "three dims passthrough" (fun () ->
          check_grouped_dims "gidx" [| 2; 3; 4 |] None false [ 2; 3; 4 ]);
      test "single dim passthrough" (fun () ->
          check_grouped_dims "gidx" [| 100 |] None false [ 100 ]);
    ]

(* Group 10: integration via pm_add_gpudims *)

let gpu_renderer
    ?(global_max = Some [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ])
    ?(local_max = Some [ 0x8FFFFFFF; 0x8FFFFFFF; 0x8FFFFFFF ]) () =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:true ~has_shared:true
    ~shared_max:32768 ~global_max ~local_max ~render:(fun ?name:_ _ -> "") ()

let thread_renderer () =
  Renderer.make ~name:"thread" ~device:"CPU" ~has_local:false ~has_shared:false
    ~shared_max:0 ~has_threads:true
    ~global_max:(Some [ 8; 0; 0 ])
    ~render:(fun ?name:_ _ -> "") ()

(* Build a simple kernel: load from data0[range_sum], store to data0[range_sum]. *)
let make_global_kernel ?(ki = kernel_info ()) ranges =
  let p = K.param ~idx:0 ~dtype:global_fptr in
  let open K.O in
  let combined =
    List.fold_left (fun acc r -> acc + r) (List.hd ranges) (List.tl ranges)
  in
  let index_node = K.index ~ptr:p ~idxs:[ combined ] () in
  let ld = K.load ~src:index_node () in
  let store_idx = K.index ~ptr:p ~idxs:[ combined ] () in
  let st = K.store ~dst:store_idx ~value:ld ~ranges in
  K.sink ~kernel_info:ki [ st ]

let find_specials root =
  List.filter
    (fun n -> match K.view n with K.Special _ -> true | _ -> false)
    (K.toposort root)

let find_ranges root =
  List.filter K.is_range (K.toposort root)

let find_define_vars root =
  List.filter
    (fun n -> match K.view n with K.Define_var _ -> true | _ -> false)
    (K.toposort root)

let integration_tests =
  group "pm_add_gpudims"
    [
      test "replaces global ranges with SPECIAL" (fun () ->
          let r0 =
            K.range ~size:(idx 32) ~axis:0 ~kind:Ak.Global ~dtype:D.index ()
          in
          let r1 =
            K.range ~size:(idx 16) ~axis:1 ~kind:Ak.Global ~dtype:D.index ()
          in
          let sink = make_global_kernel [ r0; r1 ] in
          let ren = gpu_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          equal int (List.length (find_ranges result)) 0
            ~msg:"no ranges after pass";
          is_true (List.length (find_specials result) > 0)
            ~msg:"SPECIAL nodes present");
      test "replaces global+local ranges" (fun () ->
          let g0 =
            K.range ~size:(idx 32) ~axis:0 ~kind:Ak.Global ~dtype:D.index ()
          in
          let l0 =
            K.range ~size:(idx 8) ~axis:1 ~kind:Ak.Local ~dtype:D.index ()
          in
          let sink = make_global_kernel [ g0; l0 ] in
          let ren = gpu_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          let specials = find_specials result in
          let has_gid =
            List.exists
              (fun n ->
                match K.view n with
                | K.Special { dim = Special_dim.Group_id _; _ } -> true
                | _ -> false)
              specials
          in
          let has_lid =
            List.exists
              (fun n ->
                match K.view n with
                | K.Special { dim = Special_dim.Local_id _; _ } -> true
                | _ -> false)
              specials
          in
          is_true has_gid ~msg:"has Group_id SPECIAL";
          is_true has_lid ~msg:"has Local_id SPECIAL");
      test "no-op when no GPU ranges" (fun () ->
          let r =
            K.range ~size:(idx 4) ~axis:0 ~kind:Ak.Reduce ~dtype:D.index ()
          in
          let p = K.param ~idx:0 ~dtype:global_fptr in
          let index_node = K.index ~ptr:p ~idxs:[ r ] () in
          let ld = K.load ~src:index_node () in
          let end_node = K.end_ ~value:ld ~ranges:[ r ] () in
          let sink = wrap_sink [ end_node ] in
          let ren = gpu_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          equal int (List.length (find_specials result)) 0
            ~msg:"no SPECIALs for reduce-only kernel");
      test "no-op when SPECIAL already present" (fun () ->
          let s =
            K.special ~dim:(Special_dim.Group_id 0) ~size:(idx 32)
              ~dtype:D.int32 ()
          in
          let p = K.param ~idx:0 ~dtype:global_fptr in
          let index_node = K.index ~ptr:p ~idxs:[ s ] () in
          let ld = K.load ~src:index_node () in
          let store_idx = K.index ~ptr:p ~idxs:[ s ] () in
          let st = K.store ~dst:store_idx ~value:ld ~ranges:[] in
          let sink = wrap_sink [ st ] in
          let ren = gpu_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          let specials_before = find_specials sink in
          let specials_after = find_specials result in
          equal int (List.length specials_before) (List.length specials_after)
            ~msg:"same SPECIAL count (idempotent)");
      test "threaded renderer uses core_id" (fun () ->
          let r0 =
            K.range ~size:(idx 4) ~axis:0 ~kind:Ak.Global ~dtype:D.index ()
          in
          let sink = make_global_kernel [ r0 ] in
          let ren = thread_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          let dvars = find_define_vars result in
          is_true (List.length dvars > 0) ~msg:"has Define_var";
          (match K.view (List.hd dvars) with
          | K.Define_var { name; _ } ->
              equal string name "core_id" ~msg:"variable named core_id"
          | _ -> fail "expected Define_var"));
    ]

(* Group 11: missing-locals gating *)

(* When a STORE writes to a global address and the INDEX does not depend on a
   local range, the missing local range should be gated with an Invalid mask
   (l0 == 0 ? value : Invalid). *)

let missing_locals_tests =
  group "missing-locals gating"
    [
      test "missing local range gets gated with Invalid" (fun () ->
          let g0 =
            K.range ~size:(idx 32) ~axis:0 ~kind:Ak.Global ~dtype:D.index ()
          in
          let l0 =
            K.range ~size:(idx 8) ~axis:1 ~kind:Ak.Local ~dtype:D.index ()
          in
          let p = K.param ~idx:0 ~dtype:global_fptr in
          (* Load using both ranges *)
          let load_idx = K.index ~ptr:p ~idxs:[ K.O.(g0 + l0) ] () in
          let loaded = K.load ~src:load_idx () in
          (* End local loop (local reduction) *)
          let reduced = K.end_ ~value:loaded ~ranges:[ l0 ] () in
          (* Store using ONLY the global range in the index *)
          let store_idx = K.index ~ptr:p ~idxs:[ g0 ] () in
          let st = K.store ~dst:store_idx ~value:reduced ~ranges:[ g0 ] in
          let ki = kernel_info () in
          let sink = K.sink ~kernel_info:ki [ st ] in
          let ren = gpu_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          (* Should have Invalid_index node (the mask gating) *)
          let has_invalid =
            List.exists
              (fun n ->
                match K.view n with K.Invalid_index _ -> true | _ -> false)
              (K.toposort result)
          in
          is_true has_invalid
            ~msg:"has Invalid_index node from missing-locals gating";
          (* Should have a Ternary Where (the condition) *)
          let has_where =
            List.exists
              (fun n ->
                match K.view n with
                | K.Ternary { op = `Where; _ } -> true
                | _ -> false)
              (K.toposort result)
          in
          is_true has_where ~msg:"has Where node for gating");
    ]

(* Group 12: dont_use_locals *)

let dont_use_locals_tests =
  group "dont_use_locals"
    [
      test "uses idx prefix with no local SPECIALs" (fun () ->
          let g0 =
            K.range ~size:(idx 32) ~axis:0 ~kind:Ak.Global ~dtype:D.index ()
          in
          let g1 =
            K.range ~size:(idx 16) ~axis:1 ~kind:Ak.Global ~dtype:D.index ()
          in
          let ki = kernel_info ~dont_use_locals:true () in
          let sink = make_global_kernel ~ki [ g0; g1 ] in
          let ren = gpu_renderer () in
          let result = Gpudims.pm_add_gpudims ren sink in
          let specials = find_specials result in
          is_true (List.length specials > 0) ~msg:"has SPECIAL nodes";
          (* All specials should be Global_idx, not Group_id or Local_id *)
          let all_global_idx =
            List.for_all
              (fun n ->
                match K.view n with
                | K.Special { dim = Special_dim.Global_idx _; _ } -> true
                | _ -> false)
              specials
          in
          is_true all_global_idx ~msg:"all SPECIAL nodes are Global_idx";
          (* No local SPECIALs *)
          let has_local =
            List.exists
              (fun n ->
                match K.view n with
                | K.Special { dim = Special_dim.Local_id _; _ } -> true
                | _ -> false)
              specials
          in
          is_true (not has_local) ~msg:"no Local_id SPECIALs");
    ]

(* Entry point *)

let () =
  run "Codegen.Gpudims"
    [
      noop_tests;
      reverse_tests;
      split_same_len_tests;
      grouping_preferred_tests;
      expansion_tests;
      contraction_tests;
      error_tests;
      direct_special_tests;
      none_passthrough_tests;
      integration_tests;
      missing_locals_tests;
      dont_use_locals_tests;
    ]
