(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir

let strf = Printf.sprintf

let get_reduce_axes (tc : Renderer.tensor_core) =
  let _, _, k = tc.dims in
  List.init (int_of_float (log (float_of_int k) /. log 2.0)) (fun i -> (i, 2))

let get_upcast_axes (tc : Renderer.tensor_core) =
  List.filter (fun opt -> opt.[0] = 'u') tc.opts

let get_local_axes (tc : Renderer.tensor_core) =
  List.filter (fun opt -> opt.[0] = 'l') tc.opts

let base_shape_str (tc : Renderer.tensor_core) =
  let u_cnt = ref 0 and l_cnt = ref 0 in
  let labels =
    List.map (fun opt ->
      let c = opt.[0] in
      let cnt = if c = 'u' then u_cnt else l_cnt in
      let label = strf "%c%d" c !cnt in
      incr cnt; label
    ) tc.opts
  in
  labels @ List.map (fun (i, _) -> strf "r%d" i) (get_reduce_axes tc)

let base_upcast_axes (tc : Renderer.tensor_core) =
  let reduce_labels = List.map (fun (i, _) -> strf "r%d" i) (get_reduce_axes tc) in
  let upcast_labels = List.mapi (fun i _ -> strf "u%d" i) (get_upcast_axes tc) in
  List.rev (reduce_labels @ upcast_labels)

let remaps (tc : Renderer.tensor_core) =
  let local_n = List.length (get_local_axes tc) in
  let upcast_n = List.length (get_upcast_axes tc) in
  let reduce_n = List.length (get_reduce_axes tc) in
  let fwd_st =
    List.init local_n (fun i -> strf "l%d" i)
    @ List.init upcast_n (fun i -> strf "u%d" i)
    @ List.init reduce_n (fun i -> strf "r%d" i)
  in
  let (s0_l, s0_u, s0_r), (s1_l, s1_u, s1_r) = tc.swizzle in
  let make_remap flat =
    let tbl = Hashtbl.create (List.length fwd_st) in
    List.iter2 (fun k v -> Hashtbl.replace tbl k v) fwd_st flat;
    tbl
  in
  (make_remap (s0_l @ s0_u @ s0_r), make_remap (s1_l @ s1_u @ s1_r))

let permutes_for_shape_str (tc : Renderer.tensor_core) shape_str =
  let remap0, remap1 = remaps tc in
  let compute_perm remap =
    List.mapi (fun i ss ->
      match Hashtbl.find_opt remap ss with
      | Some mapped ->
          let rec find j = function
            | [] -> failwith (strf "permutes_for_shape_str: %S not in shape_str" mapped)
            | x :: _ when x = mapped -> j
            | _ :: rest -> find (j + 1) rest
          in
          find 0 shape_str
      | None -> i
    ) shape_str
  in
  (compute_perm remap0, compute_perm remap1)

let dtype_name = function
  | Dtype.Float16 -> "half"
  | Dtype.Bfloat16 -> "__bf16"
  | Dtype.Float32 -> "float"
  | Dtype.Fp8e4m3 -> "float8_e4m3"
  | Dtype.Fp8e5m2 -> "float8_e5m2"
  | s -> Dtype.scalar_to_string s

let to_string (tc : Renderer.tensor_core) =
  let n, m, k = tc.dims in
  strf "WMMA_%d_%d_%d_%s_%s" n m k (dtype_name tc.dtype_in) (dtype_name tc.dtype_out)

(* Validation *)

let pow2 n = 1 lsl n

let validate (tc : Renderer.tensor_core) =
  let local_axes = List.length (get_local_axes tc) in
  let upcast_axes = List.length (get_upcast_axes tc) in
  let reduce_axes = List.length (get_reduce_axes tc) in
  let n, m, _ = tc.dims in
  let a_ept, b_ept, c_ept = tc.elements_per_thread in
  let check cond msg = if not cond then failwith msg in
  check (n * m = pow2 (local_axes + upcast_axes))
    (strf "N(%d) x M(%d) != local(%d) x upcast(%d) with opts"
       n m (pow2 local_axes) (pow2 upcast_axes));
  check (pow2 local_axes = tc.threads)
    (strf "%d threads but found %d in opts" tc.threads (pow2 local_axes));
  check (pow2 upcast_axes = c_ept)
    (strf "%d elements from C but found %d in opts" c_ept (pow2 upcast_axes));
  let count_dim d = List.length (List.filter (fun o -> o.[1] = d) tc.opts) in
  check (n = pow2 (count_dim '0'))
    (strf "opts wrong on dims[0], %d vs %d" n (pow2 (count_dim '0')));
  check (m = pow2 (count_dim '1'))
    (strf "opts wrong on dims[1], %d vs %d" m (pow2 (count_dim '1')));
  let (s0_l, s0_u, s0_r), (s1_l, s1_u, s1_r) = tc.swizzle in
  let len = List.length in
  check (len s0_l = local_axes && len s1_l = local_axes)
    "local swizzle size is wrong";
  check (len s0_u = upcast_axes && len s1_u = upcast_axes)
    "upcast swizzle size is wrong";
  check (len s0_r = reduce_axes && len s1_r = reduce_axes)
    "reduce swizzle size is wrong";
  let total = local_axes + upcast_axes + reduce_axes in
  let remap0, remap1 = remaps tc in
  check (Hashtbl.length remap0 = total && Hashtbl.length remap1 = total)
    "remaps are the wrong size";
  let u_cnt = ref 0 and l_cnt = ref 0 in
  let zs0 = ref [] and zs1 = ref [] in
  List.iter (fun o ->
    let label = strf "%c%d" o.[0] (if o.[0] = 'u' then !u_cnt else !l_cnt) in
    if o.[1] = '0' then zs0 := label :: !zs0;
    if o.[1] = '1' then zs1 := label :: !zs1;
    if o.[0] = 'u' then incr u_cnt else incr l_cnt
  ) tc.opts;
  let non_local_non_zero zs x = not (List.mem x zs) && x.[0] <> 'l' in
  let upcasted_0 = List.filter (non_local_non_zero !zs0) (s0_u @ s0_r) in
  let upcasted_1 = List.filter (non_local_non_zero !zs1) (s1_u @ s1_r) in
  check (pow2 (len upcasted_0) = a_ept)
    (strf "mismatch in elements_per_thread[0], %d vs %d"
       (pow2 (len upcasted_0)) a_ept);
  check (pow2 (len upcasted_1) = b_ept)
    (strf "mismatch in elements_per_thread[1], %d vs %d"
       (pow2 (len upcasted_1)) b_ept)

(* NVIDIA *)

let cuda_tc_opts = ["u0";"l0";"l0";"l1";"l1";"l1";"u1"]

let cuda_81616 =
  let swizzle =
    ( (["r1";"r2";"l2";"l3";"l4"], ["u1";"r3"], ["l0";"l1";"u0";"r0"]),
      (["r1";"r2";"u0";"l0";"l1"], ["r0";"r3"], ["l2";"l3";"l4";"u1"]) )
  in
  List.map (fun (di, do_) ->
    { Renderer.dims = (8, 16, 16); threads = 32;
      elements_per_thread = (8, 4, 4);
      dtype_in = di; dtype_out = do_;
      opts = cuda_tc_opts; swizzle })
    Dtype.[ (Float16, Float32); (Bfloat16, Float32); (Float16, Float16) ]

let cuda_81632_f8 =
  let swizzle =
    ( (["r2";"r3";"l2";"l3";"l4"], ["u1";"r4"], ["l0";"l1";"u0";"r0";"r1"]),
      (["r2";"r3";"u0";"l0";"l1"], ["r1";"r4"], ["l2";"l3";"l4";"u1";"r0"]) )
  in
  List.map (fun (di, do_) ->
    { Renderer.dims = (8, 16, 32); threads = 32;
      elements_per_thread = (16, 8, 4);
      dtype_in = di; dtype_out = do_;
      opts = cuda_tc_opts; swizzle })
    Dtype.[ (Fp8e4m3, Float32); (Fp8e5m2, Float32) ]

let cuda_8168_f16 =
  let swizzle =
    ( (["r1";"r2";"l2";"l3";"l4"], ["r0";"u1"], ["l0";"l1";"u0"]),
      (["r1";"r2";"u0";"l0";"l1"], ["u1";"r0"], ["l2";"l3";"l4"]) )
  in
  List.map (fun (di, do_) ->
    { Renderer.dims = (8, 16, 8); threads = 32;
      elements_per_thread = (4, 2, 4);
      dtype_in = di; dtype_out = do_;
      opts = cuda_tc_opts; swizzle })
    Dtype.[ (Float16, Float32); (Float16, Float16) ]

let cuda_8168_tf32 =
  let swizzle =
    ( (["r0";"r1";"l2";"l3";"l4"], ["u1";"r2"], ["l0";"l1";"u0"]),
      (["r0";"r1";"u0";"l0";"l1"], ["u1";"r2"], ["l2";"l3";"l4"]) )
  in
  [ { Renderer.dims = (8, 16, 8); threads = 32;
      elements_per_thread = (4, 2, 4);
      dtype_in = Dtype.Float32; dtype_out = Dtype.Float32;
      opts = cuda_tc_opts; swizzle } ]

let cuda_sm75 = cuda_8168_f16
let cuda_sm80 = cuda_81616 @ cuda_8168_f16 @ cuda_8168_tf32
let cuda_sm89 = cuda_sm80 @ cuda_81632_f8

(* AMD *)

let amd_rdna3 =
  List.map (fun (di, do_) ->
    { Renderer.dims = (16, 16, 16); threads = 32;
      elements_per_thread = (16, 16, 8);
      dtype_in = di; dtype_out = do_;
      opts = ["l0";"l0";"l0";"l0";"l1";"u1";"u1";"u1"];
      swizzle =
        ( (["l4";"u0";"u1";"u2";"l0"], ["r1";"r2";"r3"], ["l1";"l2";"l3";"r0"]),
          (["l0";"l1";"l2";"l3";"l4"], ["r1";"r2";"r3"], ["u0";"u1";"u2";"r0"]) ) })
    Dtype.[ (Float16, Float32); (Float16, Float16); (Bfloat16, Float32) ]

let amd_rdna4 =
  List.map (fun (di, do_) ->
    { Renderer.dims = (16, 16, 16); threads = 32;
      elements_per_thread = (8, 8, 8);
      dtype_in = di; dtype_out = do_;
      opts = ["l0";"l0";"l0";"l0";"u1";"u1";"u1";"l1"];
      swizzle =
        ( (["u0";"u1";"u2";"l4";"r2"], ["r0";"r1";"r3"], ["l0";"l1";"l2";"l3"]),
          (["l0";"l1";"l2";"l3";"r2"], ["r0";"r1";"r3"], ["l4";"u0";"u1";"u2"]) ) })
    Dtype.[ (Float16, Float32); (Float16, Float16);
            (Bfloat16, Float32); (Bfloat16, Bfloat16) ]

let amd_cdna_161616 =
  List.map (fun (di, do_) ->
    { Renderer.dims = (16, 16, 16); threads = 64;
      elements_per_thread = (4, 4, 4);
      dtype_in = di; dtype_out = do_;
      opts = ["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"];
      swizzle =
        ( (["u0";"u1";"l4";"l5";"r2";"r3"], ["r0";"r1"], ["l0";"l1";"l2";"l3"]),
          (["l0";"l1";"l2";"l3";"r2";"r3"], ["r0";"r1"], ["l4";"l5";"u0";"u1"]) ) })
    Dtype.[ (Float16, Float32); (Bfloat16, Float32) ]

let amd_cdna_161632 =
  List.map (fun (di, do_) ->
    { Renderer.dims = (16, 16, 32); threads = 64;
      elements_per_thread = (8, 8, 4);
      dtype_in = di; dtype_out = do_;
      opts = ["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"];
      swizzle =
        ( (["u0";"u1";"l4";"l5";"r3";"r4"], ["r0";"r1"], ["l0";"l1";"l2";"l3";"r2"]),
          (["l0";"l1";"l2";"l3";"r3";"r4"], ["r0";"r1"], ["l4";"l5";"u0";"u1";"r2"]) ) })
    Dtype.[ (Fp8e5m2, Float32); (Fp8e4m3, Float32);
            (Float16, Float32); (Bfloat16, Float32) ]

let amd_cdna_1616128 =
  List.map (fun (di, do_) ->
    { Renderer.dims = (16, 16, 128); threads = 64;
      elements_per_thread = (32, 32, 4);
      dtype_in = di; dtype_out = do_;
      opts = ["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"];
      swizzle =
        ( (["u0";"u1";"l4";"l5";"r5";"r6"], ["r0";"r1"], ["l0";"l1";"l2";"l3";"r2";"r3";"r4"]),
          (["l0";"l1";"l2";"l3";"r5";"r6"], ["r0";"r1"], ["l4";"l5";"u0";"u1";"r2";"r3";"r4"]) ) })
    Dtype.[ (Fp8e5m2, Float32); (Fp8e4m3, Float32) ]

let amd_cdna3 =
  List.filteri (fun i _ -> i < 2) amd_cdna_161632 @ amd_cdna_161616

let amd_cdna4 = amd_cdna_1616128 @ amd_cdna_161632 @ amd_cdna_161616

(* Apple Metal *)

let metal =
  List.map (fun (di, do_) ->
    { Renderer.dims = (8, 8, 8); threads = 32;
      elements_per_thread = (2, 2, 2);
      dtype_in = di; dtype_out = do_;
      opts = ["u0";"l0";"l1";"l1";"l0";"l1"];
      swizzle =
        ( (["r1";"l1";"l2";"r2";"l4"], ["r0"], ["u0";"l0";"l3"]),
          (["l0";"r0";"r1";"l3";"r2"], ["u0"], ["l1";"l2";"l4"]) ) })
    Dtype.[ (Float32, Float32); (Float16, Float32);
            (Float16, Float16); (Bfloat16, Float32); (Bfloat16, Bfloat16) ]

(* Apple AMX *)

let amx =
  let sz = 64 / Dtype.itemsize Dtype.float32 in
  [ { Renderer.dims = (sz, sz, 1); threads = 1;
      elements_per_thread = (sz, sz, sz * sz);
      dtype_in = Dtype.Float32; dtype_out = Dtype.Float32;
      opts = ["u0";"u0";"u0";"u0";"u1";"u1";"u1";"u1"];
      swizzle =
        ( ([], ["u0";"u1";"u2";"u3";"u4";"u5";"u6";"u7"], []),
          ([], ["u4";"u5";"u6";"u7";"u0";"u1";"u2";"u3"], []) ) } ]

(* Intel *)

let intel =
  [ { Renderer.dims = (8, 8, 16); threads = 8;
      elements_per_thread = (16, 16, 8);
      dtype_in = Dtype.Float16; dtype_out = Dtype.Float32;
      opts = ["l0";"l0";"l0";"u1";"u1";"u1"];
      swizzle =
        ( (["r1";"r2";"r3"], ["u0";"u1";"u2"], ["l0";"l1";"l2";"r0"]),
          (["l0";"l1";"l2"], ["r1";"r2";"r3"], ["u0";"u1";"u2";"r0"]) ) } ]

let () =
  List.iter (List.iter validate)
    [ cuda_sm75; cuda_sm80; cuda_sm89;
      amd_rdna3; amd_rdna4; amd_cdna3; amd_cdna4;
      metal; amx; intel ]
