(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/opt/tc.py to the tolk_uop IR. *)

open Tolk_uop

let strf = Printf.sprintf
let pow2 n = 1 lsl n
let log2 n = int_of_float (log (float_of_int n) /. log 2.0)
let labels prefix n = List.init n (fun i -> strf "%s%d" prefix i)
let is_power_of_two n = n > 0 && n land (n - 1) = 0

let rec take n = function
  | _ when n <= 0 -> []
  | [] -> []
  | x :: xs -> x :: take (n - 1) xs

(* Numbered labels for opts: each "u" becomes u0,u1,... and each "l"
   becomes l0,l1,... in occurrence order. *)
let numbered_opts opts =
  let u = ref 0 and l = ref 0 in
  List.map (fun o ->
    let c = o.[0] in
    let cnt = if c = 'u' then u else l in
    let s = strf "%c%d" c !cnt in
    incr cnt; s) opts

(* D = A * B + C.  A is (M x K), B is (K x N), C and D are (M x N).
   dims = (N, M, K).  All axes have size 2. *)
type t = {
  dims : int * int * int;
  threads : int;
  elements_per_thread : int * int * int;
  dtype_in : Dtype.t;
  dtype_out : Dtype.t;
  opts : string list;
  swizzle :
    (string list * string list * string list)
    * (string list * string list * string list);
}

let get_reduce_axes (tc : t) =
  let _, _, k = tc.dims in
  List.init (log2 k) (fun i -> (i, 2))

let get_upcast_axes (tc : t) =
  List.filter (fun o -> String.length o > 0 && o.[0] = 'u') tc.opts

let get_local_axes (tc : t) =
  List.filter (fun o -> String.length o > 0 && o.[0] = 'l') tc.opts

(* Shape string before the reduce UNROLL: numbered local/upcast labels
   from opts, then reduce labels. *)
let base_shape_str (tc : t) =
  numbered_opts tc.opts @ labels "r" (List.length (get_reduce_axes tc))

(* Upcast + reduce axis names in reverse order, used to define the UNROLL
   axes after the opts are applied. *)
let base_upcast_axes (tc : t) =
  List.rev
    (labels "r" (List.length (get_reduce_axes tc))
     @ labels "u" (List.length (get_upcast_axes tc)))

(* Build remap tables from the canonical axis order (l0..lN, u0..uN, r0..rN)
   to the swizzled order for operands A and B. *)
let remaps (tc : t) =
  let n_local = List.length (get_local_axes tc) in
  let n_upcast = List.length (get_upcast_axes tc) in
  let n_reduce = List.length (get_reduce_axes tc) in
  let fwd = labels "l" n_local @ labels "u" n_upcast @ labels "r" n_reduce in
  let (s0_l, s0_u, s0_r), (s1_l, s1_u, s1_r) = tc.swizzle in
  let make flat =
    let tbl = Hashtbl.create (List.length fwd) in
    List.iter2 (Hashtbl.replace tbl) fwd flat;
    tbl
  in
  (make (s0_l @ s0_u @ s0_r), make (s1_l @ s1_u @ s1_r))

(* Compute the two permutation vectors (for A and B) that reorder
   shape_str according to the swizzle. *)
let permutes_for_shape_str (tc : t) shape_str =
  let r0, r1 = remaps tc in
  let perm remap =
    List.mapi (fun i ss ->
      match Hashtbl.find_opt remap ss with
      | Some mapped ->
          let rec find j = function
            | [] -> failwith (strf "permutes_for_shape_str: %S not found" mapped)
            | x :: _ when x = mapped -> j
            | _ :: rest -> find (j + 1) rest
          in
          find 0 shape_str
      | None -> i) shape_str
  in
  (perm r0, perm r1)

let dtype_name = function
  | Dtype.Float16 -> "half"
  | Dtype.Bfloat16 -> "__bf16"
  | Dtype.Float32 -> "float"
  | Dtype.Float64 -> "double"
  | Dtype.Fp8e4m3 -> "float8_e4m3"
  | Dtype.Fp8e5m2 -> "float8_e5m2"
  | Dtype.Fp8e4m3fnuz -> "float8_e4m3fnuz"
  | Dtype.Fp8e5m2fnuz -> "float8_e5m2fnuz"
  | dt -> Dtype.to_string dt

let to_string (tc : t) =
  let n, m, k = tc.dims in
  strf "WMMA_%d_%d_%d_%s_%s" n m k (dtype_name tc.dtype_in)
    (dtype_name tc.dtype_out)

let validate (tc : t) =
  let n, m, k = tc.dims in
  let check cond msg = if not cond then failwith msg in
  check (is_power_of_two k) (strf "K dimension must be a positive power of two: %d" k);
  List.iter
    (fun opt ->
      check
        (String.length opt = 2
        && (opt.[0] = 'u' || opt.[0] = 'l')
        && opt.[1] >= '0' && opt.[1] <= '2')
        (strf "malformed tensor-core opt: %S" opt))
    tc.opts;
  let n_local = List.length (get_local_axes tc) in
  let n_upcast = List.length (get_upcast_axes tc) in
  let n_reduce = List.length (get_reduce_axes tc) in
  let a_ept, b_ept, c_ept = tc.elements_per_thread in
  check (is_power_of_two a_ept && is_power_of_two b_ept && is_power_of_two c_ept)
    "elements_per_thread entries must be positive powers of two";
  check (n * m = pow2 (n_local + n_upcast))
    (strf "N(%d) x M(%d) != local(%d) x upcast(%d)"
       n m (pow2 n_local) (pow2 n_upcast));
  check (pow2 n_local = tc.threads)
    (strf "%d threads but found %d locals" tc.threads (pow2 n_local));
  check (pow2 n_upcast = c_ept)
    (strf "%d C elements but found %d upcasts" c_ept (pow2 n_upcast));
  let count_dim d = List.length (List.filter (fun o -> o.[1] = d) tc.opts) in
  check (n = pow2 (count_dim '0'))
    (strf "opts wrong on dims[0]: %d vs %d" n (pow2 (count_dim '0')));
  check (m = pow2 (count_dim '1'))
    (strf "opts wrong on dims[1]: %d vs %d" m (pow2 (count_dim '1')));
  let (s0_l, s0_u, s0_r), (s1_l, s1_u, s1_r) = tc.swizzle in
  let len = List.length in
  let canonical =
    labels "l" n_local @ labels "u" n_upcast @ labels "r" n_reduce
  in
  List.iter
    (fun label ->
      check (List.mem label canonical)
        (strf "unknown tensor-core swizzle label: %S" label))
    (s0_l @ s0_u @ s0_r @ s1_l @ s1_u @ s1_r);
  check (len s0_l = n_local && len s1_l = n_local) "local swizzle size wrong";
  check (len s0_u = n_upcast && len s1_u = n_upcast) "upcast swizzle size wrong";
  check (len s0_r = n_reduce && len s1_r = n_reduce) "reduce swizzle size wrong";
  let sorted_labels = List.sort String.compare in
  let check_bijection name flat =
    check (sorted_labels flat = sorted_labels canonical)
      (strf "%s swizzle is not a bijection" name)
  in
  check_bijection "operand A" (s0_l @ s0_u @ s0_r);
  check_bijection "operand B" (s1_l @ s1_u @ s1_r);
  let total = n_local + n_upcast + n_reduce in
  let r0, r1 = remaps tc in
  check (Hashtbl.length r0 = total && Hashtbl.length r1 = total)
    "remaps wrong size";
  let zs0, zs1 =
    List.fold_left2 (fun (z0, z1) o label ->
      let z0 = if o.[1] = '0' then label :: z0 else z0 in
      let z1 = if o.[1] = '1' then label :: z1 else z1 in
      (z0, z1))
      ([], []) tc.opts (numbered_opts tc.opts)
  in
  let keep zs x = not (List.mem x zs) && x.[0] <> 'l' in
  let upcasted_0 = List.filter (keep zs0) (s0_u @ s0_r) in
  let upcasted_1 = List.filter (keep zs1) (s1_u @ s1_r) in
  check (pow2 (len upcasted_0) = a_ept)
    (strf "elements_per_thread[0] mismatch: %d vs %d"
       (pow2 (len upcasted_0)) a_ept);
  check (pow2 (len upcasted_1) = b_ept)
    (strf "elements_per_thread[1] mismatch: %d vs %d"
       (pow2 (len upcasted_1)) b_ept)

let create ~dims ~threads ~elements_per_thread ~dtype_in ~dtype_out ~opts
    ~swizzle =
  let tc =
    { dims; threads; elements_per_thread; dtype_in; dtype_out; opts; swizzle }
  in
  validate tc;
  tc

(* Tensor core definitions *)

let mk ~dims ~threads ~ept ~opts ~swizzle dtypes =
  List.map (fun (dtype_in, dtype_out) ->
    create ~dims ~threads ~elements_per_thread:ept ~dtype_in ~dtype_out ~opts
      ~swizzle) dtypes

(* NVIDIA *)

let cuda_tc_opts = ["u0";"l0";"l0";"l1";"l1";"l1";"u1"]

let cuda_81616 = mk ~dims:(8,16,16) ~threads:32 ~ept:(8,4,4) ~opts:cuda_tc_opts
  ~swizzle:((["r1";"r2";"l2";"l3";"l4"], ["u1";"r3"], ["l0";"l1";"u0";"r0"]),
            (["r1";"r2";"u0";"l0";"l1"], ["r0";"r3"], ["l2";"l3";"l4";"u1"]))
  Dtype.[(Float16, Float32); (Bfloat16, Float32); (Float16, Float16)]

let cuda_81632_f8 = mk ~dims:(8,16,32) ~threads:32 ~ept:(16,8,4) ~opts:cuda_tc_opts
  ~swizzle:((["r2";"r3";"l2";"l3";"l4"], ["u1";"r4"], ["l0";"l1";"u0";"r0";"r1"]),
            (["r2";"r3";"u0";"l0";"l1"], ["r1";"r4"], ["l2";"l3";"l4";"u1";"r0"]))
  Dtype.[(Fp8e4m3, Float32); (Fp8e5m2, Float32)]

let cuda_8168_f16 = mk ~dims:(8,16,8) ~threads:32 ~ept:(4,2,4) ~opts:cuda_tc_opts
  ~swizzle:((["r1";"r2";"l2";"l3";"l4"], ["r0";"u1"], ["l0";"l1";"u0"]),
            (["r1";"r2";"u0";"l0";"l1"], ["u1";"r0"], ["l2";"l3";"l4"]))
  Dtype.[(Float16, Float32); (Float16, Float16)]

let cuda_8168_tf32 = mk ~dims:(8,16,8) ~threads:32 ~ept:(4,2,4) ~opts:cuda_tc_opts
  ~swizzle:((["r0";"r1";"l2";"l3";"l4"], ["u1";"r2"], ["l0";"l1";"u0"]),
            (["r0";"r1";"u0";"l0";"l1"], ["u1";"r2"], ["l2";"l3";"l4"]))
  Dtype.[(Float32, Float32)]

let cuda_sm75 = cuda_8168_f16
let cuda_sm80 = cuda_81616 @ cuda_8168_f16 @ cuda_8168_tf32
let cuda_sm89 = cuda_sm80 @ cuda_81632_f8

(* AMD *)

let amd_rdna3 = mk ~dims:(16,16,16) ~threads:32 ~ept:(16,16,8)
  ~opts:["l0";"l0";"l0";"l0";"l1";"u1";"u1";"u1"]
  ~swizzle:((["l4";"u0";"u1";"u2";"l0"], ["r1";"r2";"r3"], ["l1";"l2";"l3";"r0"]),
            (["l0";"l1";"l2";"l3";"l4"], ["r1";"r2";"r3"], ["u0";"u1";"u2";"r0"]))
  Dtype.[(Float16, Float32); (Float16, Float16); (Bfloat16, Float32)]

let amd_rdna4 = mk ~dims:(16,16,16) ~threads:32 ~ept:(8,8,8)
  ~opts:["l0";"l0";"l0";"l0";"u1";"u1";"u1";"l1"]
  ~swizzle:((["u0";"u1";"u2";"l4";"r2"], ["r0";"r1";"r3"], ["l0";"l1";"l2";"l3"]),
            (["l0";"l1";"l2";"l3";"r2"], ["r0";"r1";"r3"], ["l4";"u0";"u1";"u2"]))
  Dtype.[(Float16, Float32); (Float16, Float16); (Bfloat16, Float32); (Bfloat16, Bfloat16)]

let amd_cdna_161616 = mk ~dims:(16,16,16) ~threads:64 ~ept:(4,4,4)
  ~opts:["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"]
  ~swizzle:((["u0";"u1";"l4";"l5";"r2";"r3"], ["r0";"r1"], ["l0";"l1";"l2";"l3"]),
            (["l0";"l1";"l2";"l3";"r2";"r3"], ["r0";"r1"], ["l4";"l5";"u0";"u1"]))
  Dtype.[(Float16, Float32); (Bfloat16, Float32)]

let amd_cdna_161632 = mk ~dims:(16,16,32) ~threads:64 ~ept:(8,8,4)
  ~opts:["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"]
  ~swizzle:((["u0";"u1";"l4";"l5";"r3";"r4"], ["r0";"r1"], ["l0";"l1";"l2";"l3";"r2"]),
            (["l0";"l1";"l2";"l3";"r3";"r4"], ["r0";"r1"], ["l4";"l5";"u0";"u1";"r2"]))
  Dtype.[(Fp8e5m2, Float32); (Fp8e4m3, Float32); (Float16, Float32); (Bfloat16, Float32)]

let amd_cdna_1616128 = mk ~dims:(16,16,128) ~threads:64 ~ept:(32,32,4)
  ~opts:["l0";"l0";"l0";"l0";"u1";"u1";"l1";"l1"]
  ~swizzle:((["u0";"u1";"l4";"l5";"r5";"r6"], ["r0";"r1"], ["l0";"l1";"l2";"l3";"r2";"r3";"r4"]),
            (["l0";"l1";"l2";"l3";"r5";"r6"], ["r0";"r1"], ["l4";"l5";"u0";"u1";"r2";"r3";"r4"]))
  Dtype.[(Fp8e5m2, Float32); (Fp8e4m3, Float32)]

let amd_cdna3 = take 2 amd_cdna_161632 @ amd_cdna_161616
let amd_cdna4 = amd_cdna_1616128 @ amd_cdna_161632 @ amd_cdna_161616

(* Apple Metal *)

let metal = mk ~dims:(8,8,8) ~threads:32 ~ept:(2,2,2)
  ~opts:["u0";"l0";"l1";"l1";"l0";"l1"]
  ~swizzle:((["r1";"l1";"l2";"r2";"l4"], ["r0"], ["u0";"l0";"l3"]),
            (["l0";"r0";"r1";"l3";"r2"], ["u0"], ["l1";"l2";"l4"]))
  Dtype.[(Float32, Float32); (Float16, Float32); (Float16, Float16);
         (Bfloat16, Float32); (Bfloat16, Bfloat16)]

(* Validate all definitions at load time. *)
let () =
  List.iter (List.iter validate)
    [cuda_sm75; cuda_sm80; cuda_sm89;
     amd_rdna3; amd_rdna4; amd_cdna3; amd_cdna4;
     metal]
