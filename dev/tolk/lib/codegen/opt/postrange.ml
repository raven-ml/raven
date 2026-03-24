(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

let strf = Printf.sprintf

exception Opt_error of string

let check cond msg = if not cond then raise (Opt_error msg)

type t = {
  mutable ast : K.t;
  ren : Renderer.t;
  mutable applied_opts : K.Opt.t list;
  mutable dont_use_locals : bool;
  opt_range : int ref;
  mutable tensor_core : Renderer.tensor_core option;
}

let ast t = t.ast
let ren t = t.ren
let applied_opts t = t.applied_opts
let tensor_core t = t.tensor_core

let range_size_int rng = K.const_to_int (K.range_size rng)

let size_to_str sz =
  if K.is_const sz then string_of_int (K.const_to_int sz) else "s"

let const_int_or sz default =
  if K.is_const sz then K.const_to_int sz else default

let rec divides (node : K.t) (v : int) : K.t option =
  if v = 1 then Some node
  else if K.is_const node then
    let n = K.const_to_int node in
    if n mod v = 0 then Some (K.O.int_ (n / v)) else None
  else
    match K.view node with
    | Binary { op = `Add; lhs; rhs; _ } -> (
        match divides lhs v, divides rhs v with
        | Some d0, Some d1 -> Some K.O.(d0 + d1)
        | _ -> None)
    | Binary { op = `Mul; lhs; rhs; _ } -> (
        match divides lhs v with
        | Some d0 -> Some K.O.(d0 * rhs)
        | None -> Option.map (fun d1 -> K.O.(lhs * d1)) (divides rhs v))
    | _ -> None

let is_invalid_index node =
  match K.view node with Invalid_index _ -> true | _ -> false

let get_idx node =
  match K.view node with
  | Ternary { op = `Where; b = idx; c = else_; _ }
    when is_invalid_index else_ -> idx
  | _ -> node

let get_valid node =
  match K.view node with
  | Ternary { op = `Where; a = cond; c = else_; _ }
    when is_invalid_index else_ -> cond
  | _ -> K.O.bool_ (not (is_invalid_index node))

let to_function_name s =
  let buf = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      match c with
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> Buffer.add_char buf c
      | _ -> Buffer.add_string buf (strf "%02X" (Char.code c)))
    s;
  Buffer.contents buf

let rngs t =
  let all =
    List.filter
      (fun n ->
        K.is_range n
        && (let sz = K.range_size n in
            not (K.is_const sz) || K.const_to_int sz > 1))
      (K.backward_slice t.ast)
  in
  List.sort
    (fun a b ->
      let cmp = compare (Axis_kind.to_pos (K.range_kind a))
                        (Axis_kind.to_pos (K.range_kind b)) in
      if cmp <> 0 then cmp else compare (K.range_axis a) (K.range_axis b))
    all

let shape_len t = List.length (rngs t)
let full_shape t = List.map K.range_size (rngs t)
let axis_types t = List.map K.range_kind (rngs t)

let shape_str t =
  let tbl = Hashtbl.create 16 in
  List.map
    (fun kind ->
      let n = match Hashtbl.find_opt tbl kind with Some c -> c | None -> 0 in
      Hashtbl.replace tbl kind (n + 1);
      strf "%s%d" (Axis_kind.letter kind) n)
    (axis_types t)

let shape_str_to_axis t nms =
  let ss = shape_str t in
  List.map
    (fun nm ->
      let rec find i = function
        | [] -> failwith (strf "shape_str_to_axis: %s not found" nm)
        | x :: _ when x = nm -> i
        | _ :: rest -> find (i + 1) rest
      in
      find 0 ss)
    nms

let next_axis t =
  let v = !(t.opt_range) in
  t.opt_range := v + 1;
  v

let create ast ren =
  let ki = match K.view ast with
    | Sink { kernel_info = Some ki; _ } -> Some ki
    | _ -> None
  in
  let max_axis =
    List.fold_left (fun acc r -> max acc (K.range_axis r)) 0
      (List.filter K.is_range (K.backward_slice ast))
  in
  { ast; ren;
    applied_opts = (match ki with Some ki -> ki.applied_opts | None -> []);
    dont_use_locals = (match ki with Some ki -> ki.dont_use_locals | None -> false);
    opt_range = ref (max_axis + 1);
    tensor_core = None }

let copy t =
  let ret = create t.ast t.ren in
  ret.dont_use_locals <- t.dont_use_locals;
  ret.applied_opts <- t.applied_opts;
  ret.tensor_core <- t.tensor_core;
  ret

let output_rngs t =
  let srcs = match K.view t.ast with Sink { srcs; _ } -> srcs | _ -> [] in
  let seen = K.Ref_tbl.create 16 in
  List.concat_map
    (fun s ->
      match K.view s with
      | End { ranges; _ } ->
          List.filter
            (fun r ->
              K.is_range r
              && K.range_kind r <> Axis_kind.Reduce
              && not (K.Ref_tbl.mem seen r)
              && (K.Ref_tbl.replace seen r (); true))
            (List.concat_map K.live_ranges ranges)
      | _ -> [])
    srcs

let mem_phys x xs = List.exists (fun y -> y == x) xs

let globalizable_rngs t =
  let ret =
    List.filter (fun r -> K.range_kind r = Axis_kind.Loop) (output_rngs t)
  in
  let bnode_ranges =
    List.filter_map
      (fun n ->
        match K.view n with Bufferize _ -> Some (K.live_ranges n) | _ -> None)
      (K.toposort t.ast)
  in
  List.filter
    (fun r ->
      List.for_all (fun brngs -> List.exists (fun br -> br == r) brngs) bnode_ranges)
    ret

let convert_loop_to_global t =
  if Renderer.has_local t.ren then begin
    let glob_rngs = globalizable_rngs t in
    let replacements =
      List.filter_map
        (fun rng ->
          if mem_phys rng glob_rngs then
            Some (rng, K.range ~size:(K.range_size rng)
                         ~axis:(K.range_axis rng) ~kind:Axis_kind.Global ())
          else None)
        (rngs t)
    in
    if replacements <> [] then t.ast <- K.substitute replacements t.ast
  end

let colors t =
  let out_rngs = output_rngs t in
  let glob_rngs = globalizable_rngs t in
  List.map2
    (fun at r ->
      if t.dont_use_locals && at = Axis_kind.Global then "BLUE"
      else if at = Axis_kind.Loop && not (mem_phys r out_rngs) then "BLACK"
      else if at = Axis_kind.Loop && not (mem_phys r glob_rngs) then "white"
      else Axis_kind.color at)
    (axis_types t) (rngs t)

let colored_shape t =
  String.concat " "
    (List.map2
       (fun rng color -> strf "[%s:%s]" (size_to_str (K.range_size rng)) color)
       (rngs t) (colors t))

let kernel_cnt : (string, int) Hashtbl.t = Hashtbl.create 16

let generate_name t =
  let k_type =
    if List.exists
         (fun n -> match K.view n with Reduce _ -> true | _ -> false)
         (K.backward_slice t.ast)
    then "r" else "E"
  in
  let special_sizes =
    List.map
      (fun s ->
        match K.view s with Special { size; _ } -> size_to_str size | _ -> "?")
      (List.sort
         (fun a b ->
           match K.view a, K.view b with
           | Special { dim = da; _ }, Special { dim = db; _ } ->
               Special_dim.compare da db
           | _ -> 0)
         (List.filter
            (fun n -> match K.view n with Special _ -> true | _ -> false)
            (K.toposort t.ast)))
  in
  let rng_sizes = List.map (fun rng -> size_to_str (K.range_size rng)) (rngs t) in
  let raw_name = k_type ^ "_" ^ String.concat "_" (special_sizes @ rng_sizes) in
  let fn = to_function_name raw_name in
  let cnt =
    1 + (match Hashtbl.find_opt kernel_cnt fn with Some c -> c | None -> 0)
  in
  Hashtbl.replace kernel_cnt fn cnt;
  raw_name ^ (if cnt > 1 then strf "n%d" (cnt - 1) else "")

let get_optimized_ast ?name_override t =
  let name = match name_override with Some n -> n | None -> generate_name t in
  t.ast <- Simplify.pm_flatten_range t.ast;
  let ki =
    { K.name; axis_kinds = []; dont_use_locals = t.dont_use_locals;
      applied_opts = t.applied_opts; opts_to_apply = None;
      estimates = None }
  in
  let result = K.sink ~kernel_info:ki
    (match K.view t.ast with Sink { srcs; _ } -> srcs | _ -> [ t.ast ])
  in
  K.with_tag "1" result

let shift_to ?(top = false) ?input_new_rng ?tags t rng amount new_kind =
  let size = K.range_size rng in
  let old_sz = match divides size amount with
    | Some q -> q
    | None ->
        let sz_str = if K.is_const size then size_to_str size else "symbolic" in
        raise (Opt_error (strf "%d can't divide %s in %s" amount sz_str
                            (colored_shape t)))
  in
  let open K.O in
  let new_rng = match input_new_rng with
    | Some r -> r
    | None -> K.range ~size:(int_ amount) ~axis:(next_axis t) ~kind:new_kind ()
  in
  let replaced_rng =
    K.range ~size:old_sz ~axis:(K.range_axis rng) ~kind:(K.range_kind rng) ()
  in
  let sub_axis =
    if top then new_rng * old_sz + replaced_rng
    else replaced_rng * int_ amount + new_rng
  in
  t.ast <- K.substitute ?tags [ (rng, sub_axis) ] t.ast;
  (replaced_rng, new_rng)

let ranges_of t kinds =
  List.filter (fun r -> List.mem (K.range_kind r) kinds) (rngs t)

let axes_of t kinds =
  List.filter_map
    (fun (i, at) -> if List.mem at kinds then Some i else None)
    (List.mapi (fun i at -> (i, at)) (axis_types t))

let product_of_axes t axes =
  let fs = full_shape t in
  List.fold_left
    (fun acc a -> acc * const_int_or (List.nth fs a) 1)
    1 axes

let upcast_size t =
  product_of_axes t (axes_of t [ Axis_kind.Upcast; Axis_kind.Unroll ])

let const_gt1_dims t kinds =
  let fs = full_shape t in
  List.filter
    (fun i ->
      let s = List.nth fs i in
      K.is_const s && K.const_to_int s > 1)
    (axes_of t kinds)

let upcastable_dims t =
  const_gt1_dims t [ Axis_kind.Global; Axis_kind.Local; Axis_kind.Loop ]

let unrollable_dims t =
  const_gt1_dims t [ Axis_kind.Group_reduce; Axis_kind.Reduce ]

let opt_axis_amount = function
  | K.Opt.Local { axis; amount }
  | K.Opt.Upcast { axis; amount }
  | K.Opt.Unroll { axis; amount }
  | K.Opt.Group { axis; amount }
  | K.Opt.Grouptop { axis; amount }
  | K.Opt.Thread { axis; amount }
  | K.Opt.Padto { axis; amount } -> (axis, amount)
  | K.Opt.Swap { axis; _ } | K.Opt.Tc { axis; _ } -> (axis, 0)
  | K.Opt.Nolocals -> assert false

let nth_or_error lst i msg =
  match List.nth_opt lst i with Some v -> v | None -> raise (Opt_error msg)

let real_axis t opt =
  let axis, _ = opt_axis_amount opt in
  match opt with
  | K.Opt.Tc _ -> -1
  | K.Opt.Nolocals -> assert false
  | K.Opt.Unroll _ ->
      nth_or_error (unrollable_dims t) axis "invalid unroll axis"
  | K.Opt.Group _ | K.Opt.Grouptop _ ->
      nth_or_error (axes_of t [ Axis_kind.Reduce ]) axis "invalid group axis"
  | _ ->
      check (axis < shape_len t)
        (strf "invalid axis on axis=%d shape_len=%d" axis (shape_len t));
      axis

let reduceops t =
  List.filter
    (fun n -> match K.view n with Reduce _ -> true | _ -> false)
    (K.backward_slice t.ast)

(* We return raw Reduce nodes rather than wrapping in a separate REDUCE_AXIS
   form; all uses in postrange only check presence, dtype, or the reduce op. *)
let reduceop t = match reduceops t with [] -> None | r :: _ -> Some r

let bufs t =
  List.rev
    (List.filter
       (fun n -> match K.view n with Index _ -> true | _ -> false)
       (K.toposort t.ast))

let output_shape t =
  List.map2
    (fun s at ->
      match at with
      | Axis_kind.Reduce | Axis_kind.Unroll | Axis_kind.Group_reduce -> K.O.int_ 1
      | _ -> s)
    (full_shape t) (axis_types t)

let upcasted t =
  List.length (axes_of t [ Axis_kind.Upcast; Axis_kind.Unroll ])

let group_for_reduces t =
  List.length (axes_of t [ Axis_kind.Group_reduce ])

let is_unsafe_pad_op n =
  match K.view n with
  | Unary { op = `Recip | `Log2 | `Exp2; _ }
  | Binary { op = `Idiv | `Pow; _ } -> true
  | _ -> false

let has_unsafe_pad_in_slice r =
  is_unsafe_pad_op r || List.exists is_unsafe_pad_op (K.backward_slice r)

let round_up num amt = ((num + amt - 1) / amt) * amt
let scalar_dt (s : Dtype.scalar) : Dtype.t = Dtype.of_scalar s

let allow_tf32 =
  match Sys.getenv_opt "ALLOW_TF32" with
  | Some s -> (try int_of_string s <> 0 with Failure _ -> false)
  | None -> false

let apply_padto ?tags t rng amount =
  check (K.is_const (K.range_size rng)) "only pad const axes";
  let k = K.range_kind rng in
  check (k <> Axis_kind.Upcast && k <> Axis_kind.Unroll) "cannot pad upcasted";
  check (k <> Axis_kind.Thread) "cannot pad thread";
  (match reduceop t with
  | Some r when k = Axis_kind.Group_reduce || k = Axis_kind.Reduce ->
      let err = strf "cannot pad %s" (Format.asprintf "%a" K.pp_view r) in
      (match K.view r with
      | Reduce { op = `Add; _ } -> check (not (has_unsafe_pad_in_slice r)) err
      | _ -> check false err)
  | _ -> ());
  let cur_sz = range_size_int rng in
  let new_sz = round_up cur_sz amount in
  check (cur_sz > new_sz / 4) "pad adds more than quadruple the work";
  let replaced_rng =
    K.range ~size:(K.O.int_ new_sz) ~axis:(K.range_axis rng) ~kind:k ()
  in
  let replaces = ref [ (rng, replaced_rng) ] in
  let valid = K.O.(replaced_rng < int_ cur_sz) in
  List.iter
    (fun b ->
      match K.view b with
      | Index { ptr; idxs; gate; dtype } ->
          let changed = ref false in
          let idxs' =
            List.map
              (fun off ->
                let idx = get_idx off in
                if K.in_backward_slice rng idx then begin
                  changed := true;
                  let combined =
                    K.binary ~op:`And ~lhs:valid ~rhs:(get_valid off)
                  in
                  K.O.where combined idx (K.invalid_index ())
                end
                else off)
              idxs
          in
          if !changed then
            replaces :=
              (b, K.index_raw ~ptr ~idxs:idxs' ?gate ~dtype ()) :: !replaces
      | _ -> ())
    (bufs t);
  t.ast <- K.substitute ?tags !replaces t.ast

let scalar_of_node n =
  match K.dtype n with Some dt -> Dtype.scalar (Dtype.scalar_of dt) | None -> Dtype.Void

let by_axis_desc a b = compare (K.range_axis b) (K.range_axis a)

let apply_tc_opt t ~use_tensor_cores ~axis ~tc_select ~tc_opt =
  let reds = reduceops t in
  if reds = [] then raise (Opt_error "no reduce ops for TensorCore");
  let red = List.hd reds in
  match K.view red with
  | Reduce { op = `Add; src; ranges; _ } ->
      if use_tensor_cores = 0 then None
      else begin
        let mul = match K.view src with Cast { src = inner; _ } -> inner | _ -> src in
        match K.view mul with
        | Binary { op = `Mul; lhs = in0; rhs = in1; _ } ->
            let tensor_cores =
              if tc_select = -1 then Renderer.tensor_cores t.ren
              else
                match List.nth_opt (Renderer.tensor_cores t.ren) tc_select with
                | Some tc -> [ tc ]
                | None -> raise (Opt_error (strf "invalid tensor core choice %d" tc_select))
            in
            let in0_scalar = scalar_of_node in0 in
            let in1_ranges = K.live_ranges in1 in
            let in0_ranges = K.live_ranges in0 in
            let result = ref None in
            let dev = Renderer.device t.ren in
            List.iter
              (fun (tc : Renderer.tensor_core) ->
                if !result <> None then ()
                else if (dev = "CUDA" || dev = "NV")
                        && tc.dtype_in = Dtype.Float32 && not allow_tf32
                then ()
                else if tc.dtype_in = in0_scalar
                        && tc.dtype_in = scalar_of_node in1
                        && tc.dtype_out = scalar_of_node red
                then begin
                  let exclusive rngs other =
                    List.filter
                      (fun r -> K.is_range r && not (mem_phys r other))
                      rngs
                  in
                  let in0_sorted = List.sort by_axis_desc (exclusive in0_ranges in1_ranges) in
                  let in1_sorted = List.sort by_axis_desc (exclusive in1_ranges in0_ranges) in
                  let red_sorted = List.sort by_axis_desc ranges in
                  if in0_sorted <> [] && in1_sorted <> [] && red_sorted <> [] then begin
                    let axis_choices =
                      List.concat_map
                        (fun y ->
                          List.concat_map
                            (fun x -> List.map (fun r -> (y, x, r)) red_sorted)
                            in0_sorted)
                        in1_sorted
                    in
                    if axis < List.length axis_choices then begin
                      let y, x, r = List.nth axis_choices axis in
                      let axes = [| y; x; r |] in
                      let n, m, k = tc.dims in
                      let tc_dims = [| n; m; k |] in
                      let tags = K.Ref_tbl.create 16 in
                      K.Ref_tbl.replace tags red 1;
                      let pad_ok = ref true in
                      for i = 0 to 2 do
                        if !pad_ok then begin
                          let cur_rngs = rngs t in
                          (* Use axis+kind matching (not physical identity)
                             because K.substitute may rebuild the tree. *)
                          let target_axis = K.range_axis axes.(i) in
                          let target_kind = K.range_kind axes.(i) in
                          let idx_of_axis =
                            let rec find j = function
                              | [] -> -1
                              | r :: _ when K.range_axis r = target_axis
                                         && K.range_kind r = target_kind -> j
                              | _ :: rest -> find (j + 1) rest
                            in
                            find 0 cur_rngs
                          in
                          let sz = range_size_int axes.(i) in
                          if sz mod tc_dims.(i) <> 0 then begin
                            if tc_opt < 2 then
                              pad_ok := false
                            else
                              try
                                let rng_node =
                                  List.nth cur_rngs idx_of_axis
                                in
                                apply_padto ~tags t rng_node tc_dims.(i);
                                axes.(i) <- List.nth (rngs t) idx_of_axis
                              with Opt_error _ -> pad_ok := false
                          end
                        end
                      done;
                      if !pad_ok then begin
                        let warp =
                          K.range ~size:(K.O.int_ tc.threads) ~axis:(-1)
                            ~kind:Axis_kind.Warp ()
                        in
                        let ne = ref [] in
                        let cur_warp = ref warp in
                        List.iter
                          (fun opt ->
                            let c = opt.[0] in
                            let idx = Char.code opt.[1] - Char.code '0' in
                            match c with
                            | 'l' ->
                                let replaced, new_range =
                                  shift_to ~tags t axes.(idx) 2 Axis_kind.Local
                                    ~input_new_rng:K.O.(!cur_warp mod int_ 2)
                                in
                                axes.(idx) <- replaced;
                                cur_warp := K.O.(!cur_warp / int_ 2);
                                ne := new_range :: !ne
                            | 'u' ->
                                let replaced, new_range =
                                  shift_to ~tags t axes.(idx) 2 Axis_kind.Upcast
                                in
                                axes.(idx) <- replaced;
                                ne := new_range :: !ne
                            | _ -> failwith (strf "unsupported opt %c in tensor cores" c))
                          tc.opts;
                        List.iter
                          (fun (_, amt) ->
                            let replaced, new_range =
                              shift_to ~tags t axes.(2) amt Axis_kind.Unroll
                            in
                            axes.(2) <- replaced;
                            ne := new_range :: !ne)
                          (Tc.get_reduce_axes tc);
                        if use_tensor_cores <> 2 then begin
                          let tagged =
                            match List.filter
                                    (fun n -> K.Ref_tbl.find_opt tags n = Some 1)
                                    (K.toposort t.ast)
                            with
                            | [ r ] -> r
                            | [] -> raise (Opt_error "TC: tagged reduce lost after shifts")
                            | _ -> raise (Opt_error "TC: multiple tagged nodes after shifts")
                          in
                          let ne_list = List.rev !ne in
                          (* Temporary markers for permutation substitution *)
                          let tne =
                            List.map
                              (fun _ ->
                                K.range ~size:(K.O.int_ 2) ~axis:(next_axis t)
                                  ~kind:Axis_kind.Upcast ())
                              ne_list
                          in
                          let ret = K.substitute (List.combine ne_list tne) tagged in
                          let mul_src = match K.view ret with
                            | Reduce { src = s; _ } ->
                                (match K.view s with Cast { src = inner; _ } -> inner | _ -> s)
                            | _ -> ret
                          in
                          let srcs = match K.view mul_src with
                            | Binary { lhs; rhs; _ } -> [| lhs; rhs |]
                            | _ -> [| mul_src; mul_src |]
                          in
                          let perm0, perm1 =
                            Tc.permutes_for_shape_str tc (Tc.base_shape_str tc)
                          in
                          let perms = [| perm0; perm1 |] in
                          let argsort lst =
                            List.map fst
                              (List.sort (fun (_, a) (_, b) -> compare a b)
                                 (List.mapi (fun i v -> (i, v)) lst))
                          in
                          for i = 0 to 1 do
                            let inv_perm = argsort perms.(i) in
                            let mappings =
                              List.mapi
                                (fun j tne_node ->
                                  (tne_node, List.nth ne_list (List.nth inv_perm j)))
                                tne
                            in
                            srcs.(i) <- K.substitute mappings srcs.(i)
                          done;
                          let tc_reduce_axes =
                            shape_str_to_axis t
                              (List.init (List.length (Tc.get_reduce_axes tc))
                                 (fun i -> strf "r%d" i))
                          in
                          let base_upcast_axes_indices =
                            List.map (fun s -> (s, 2))
                              (shape_str_to_axis t (Tc.base_upcast_axes tc))
                          in
                          let ept0, ept1, ept2 = tc.elements_per_thread in
                          let log2_ept v =
                            int_of_float (log (float_of_int v) /. log 2.0)
                          in
                          let take n lst =
                            let rec aux acc n = function
                              | [] -> List.rev acc
                              | _ when n <= 0 -> List.rev acc
                              | x :: rest -> aux (x :: acc) (n - 1) rest
                            in
                            aux [] n lst
                          in
                          let tc_upcast_axes = [|
                            take (log2_ept ept0) base_upcast_axes_indices;
                            take (log2_ept ept1) base_upcast_axes_indices;
                            take (log2_ept ept2) base_upcast_axes_indices;
                          |] in
                          let cur_rngs = rngs t in
                          let tc_upcast_axes_ranged =
                            Array.map
                              (fun al ->
                                List.map (fun (a, sz) ->
                                  (K.range_axis (List.nth cur_rngs a), sz)) al)
                              tc_upcast_axes
                          in
                          let tc_reduce_axes_ranged =
                            List.map
                              (fun a -> K.range_axis (List.nth cur_rngs a))
                              tc_reduce_axes
                          in
                          let vec_dt src_dt fallback ept =
                            Dtype.vec (match K.dtype src_dt with Some dt -> dt | None -> scalar_dt fallback) ept
                          in
                          let out_dt = Dtype.vec (scalar_dt tc.dtype_out) ept2 in
                          let contract_a =
                            K.contract ~src:srcs.(0)
                              ~axes:tc_upcast_axes_ranged.(0)
                              ~dtype:(vec_dt srcs.(0) tc.dtype_in ept0)
                          in
                          let contract_b =
                            K.contract ~src:srcs.(1)
                              ~axes:tc_upcast_axes_ranged.(1)
                              ~dtype:(vec_dt srcs.(1) tc.dtype_in ept1)
                          in
                          let wmma =
                            K.wmma ~name:(Tc.to_string tc)
                              ~a:contract_a ~b:contract_b
                              ~c:(K.broadcast
                                    (K.const (Const.float (scalar_dt tc.dtype_out) 0.0))
                                    ept2)
                              ~dtype:out_dt ~dims:tc.dims
                              ~dtype_in:tc.dtype_in ~dtype_out:tc.dtype_out
                              ~device:dev ~threads:tc.threads
                              ~upcast_axes:(tc_upcast_axes_ranged.(0),
                                            tc_upcast_axes_ranged.(1),
                                            tc_upcast_axes_ranged.(2))
                              ~reduce_axes:tc_reduce_axes_ranged
                          in
                          let out_scalar_dt = scalar_dt tc.dtype_out in
                          let tc_uop =
                            K.unroll ~src:wmma
                              ~axes:tc_upcast_axes_ranged.(2) ~dtype:out_scalar_dt
                          in
                          let extra_reduces =
                            List.filter
                              (fun x ->
                                K.is_range x
                                && not (List.mem (K.range_axis x) tc_reduce_axes_ranged))
                              (K.toposort (K.sink ranges))
                          in
                          let tc_uop =
                            if extra_reduces <> [] then
                              K.reduce ~op:`Add ~src:tc_uop
                                ~ranges:extra_reduces ~dtype:out_scalar_dt
                            else tc_uop
                          in
                          t.ast <- K.substitute [ (tagged, tc_uop) ] t.ast
                        end;
                        t.tensor_core <- Some tc;
                        result :=
                          Some [| axes.(0); axes.(1); axes.(2) |]
                      end
                    end
                  end
                end)
              tensor_cores;
            !result
        | _ -> None
      end
  | _ -> None

let opt_to_kind = function
  | K.Opt.Local _ -> Some Axis_kind.Local
  | K.Opt.Upcast _ -> Some Axis_kind.Upcast
  | K.Opt.Unroll _ -> Some Axis_kind.Unroll
  | K.Opt.Group _ | K.Opt.Grouptop _ -> Some Axis_kind.Group_reduce
  | K.Opt.Thread _ -> Some Axis_kind.Thread
  | _ -> None

let needs_smem_check = function
  | K.Opt.Group _ | K.Opt.Grouptop _ -> true
  | K.Opt.Nolocals | K.Opt.Padto _ -> false
  | _ -> true

let apply_opt ?(append_opt = true) t opt =
  match opt with
  | K.Opt.Nolocals ->
      check
        (List.for_all
           (fun at ->
             at <> Axis_kind.Warp && at <> Axis_kind.Local
             && at <> Axis_kind.Group_reduce)
           (axis_types t))
        "no locals can't have locals";
      if append_opt then t.applied_opts <- t.applied_opts @ [ opt ];
      t.dont_use_locals <- true;
      None
  | _ ->
      (match opt with
      | K.Opt.Local _ | K.Opt.Group _ | K.Opt.Grouptop _ ->
          check (Renderer.has_local t.ren) "locals needed for opt"
      | _ -> ());
      let _, opt_amount = opt_axis_amount opt in
      let ra = real_axis t opt in
      let cur_rngs = rngs t in
      let rng =
        if ra >= 0 && ra < List.length cur_rngs then List.nth cur_rngs ra
        else K.const_int 0
      in
      let result =
        match opt_to_kind opt with
        | Some new_kind -> begin
            let amt =
              if opt_amount = 0 then range_size_int rng else opt_amount
            in
            if reduceop t <> None
               && (match opt with
                  | K.Opt.Group _ | K.Opt.Grouptop _ -> true
                  | _ -> group_for_reduces t > 0 && needs_smem_check opt)
            then begin
              let upcast_local_sz =
                product_of_axes t
                  (axes_of t [ Axis_kind.Upcast; Axis_kind.Warp;
                               Axis_kind.Local; Axis_kind.Group_reduce ])
              in
              let red = match reduceop t with Some r -> r | None -> assert false in
              let red_dt = match K.dtype red with Some dt -> dt | None -> Dtype.void in
              let smem_sz = amt * upcast_local_sz * Dtype.itemsize red_dt in
              check (smem_sz <= Renderer.shared_max t.ren)
                (strf "exceeds maximum shared memory size: needs %d, max %d"
                   smem_sz (Renderer.shared_max t.ren))
            end;
            (match opt with
            | K.Opt.Group _ | K.Opt.Grouptop _ ->
                List.iter
                  (fun red_node ->
                    check
                      (not (List.exists
                              (fun u ->
                                let k = K.range_kind u in
                                k = Axis_kind.Reduce || k = Axis_kind.Unroll
                                || k = Axis_kind.Group_reduce)
                              (K.live_ranges red_node)))
                      "cannot have a GROUP_REDUCE inside another reduce")
                  (List.filter
                     (fun u ->
                       match K.view u with
                       | Reduce { ranges = rr; _ } ->
                           List.exists (fun r -> r == rng)
                             (List.concat_map K.live_ranges rr)
                       | _ -> false)
                     (K.backward_slice t.ast))
            | _ -> ());
            let k = K.range_kind rng in
            (match opt with
            | K.Opt.Unroll _ ->
                check (amt <= 32) "don't unroll more than 32";
                check (k = Axis_kind.Group_reduce || k = Axis_kind.Reduce)
                  "unroll is for GROUP_REDUCE/REDUCE"
            | K.Opt.Upcast _ ->
                check (Renderer.device t.ren = "DSP" || amt <= 16)
                  "don't upcast more than 16";
                check (k = Axis_kind.Global || k = Axis_kind.Local || k = Axis_kind.Loop)
                  (strf "upcast is for GLOBAL/LOCAL/LOOP, not %s"
                     (Format.asprintf "%a" Axis_kind.pp k))
            | K.Opt.Local _ ->
                check (not t.dont_use_locals) "can't use locals";
                check (k = Axis_kind.Global || k = Axis_kind.Loop) "local is for globals"
            | K.Opt.Thread _ ->
                check (Renderer.has_threads t.ren) "target does not support threads";
                check
                  (match Renderer.global_max t.ren with
                  | Some (gmax :: _) -> amt <= gmax | _ -> true)
                  "too many threads";
                check
                  (not (List.mem Axis_kind.Thread (axis_types t)))
                  "already threaded";
                check (mem_phys rng (globalizable_rngs t))
                  "can't apply range to this dim"
            | K.Opt.Group _ | K.Opt.Grouptop _ ->
                check
                  (List.for_all
                     (fun o -> match o with K.Opt.Tc _ -> false | _ -> true)
                     t.applied_opts)
                  "no grouping with tensor cores";
                check (not t.dont_use_locals) "can't use locals";
                check (k = Axis_kind.Reduce) "group is for reduce"
            | _ -> ());
            let top = match opt with
              | K.Opt.Grouptop _ | K.Opt.Thread _ -> true | _ -> false
            in
            Some (shift_to ~top t rng amt new_kind)
          end
        | None -> (
            match opt with
            | K.Opt.Tc { axis; tc_select; tc_opt; use_tc } ->
                check (t.applied_opts = []) "tensor core opts must be first";
                check (axis >= 0) "tensor core opts must have an axis";
                check (tc_select >= -1
                       && tc_select < List.length (Renderer.tensor_cores t.ren))
                  "tensor core opts must have valid tc_select";
                check (tc_opt >= 0 && tc_opt <= 2)
                  "tensor core opts must have valid tc_opt";
                check (use_tc > 0 && use_tc <= 2)
                  "use_tensor_cores value is not valid";
                let ret =
                  try apply_tc_opt t ~use_tensor_cores:use_tc ~axis ~tc_select ~tc_opt
                  with Opt_error _ -> None
                in
                check (ret <> None) "no tensor core available";
                Option.map (fun arr -> (arr.(0), arr.(1))) ret
            | K.Opt.Padto { amount; _ } ->
                apply_padto t rng amount;
                None
            (* Fresh ranges from K.substitute are distinct objects, so no
               tag-based cycle-breaking is needed for SWAP. *)
            | K.Opt.Swap { with_axis; _ } ->
                let altrng = nth_or_error cur_rngs with_axis "invalid swap axis" in
                check (K.range_kind rng = Axis_kind.Global
                       && K.range_kind altrng = Axis_kind.Global)
                  "swap only for globals";
                let new_rng =
                  K.range ~size:(K.range_size rng)
                    ~axis:(K.range_axis altrng) ~kind:(K.range_kind rng) ()
                in
                let new_altrng =
                  K.range ~size:(K.range_size altrng)
                    ~axis:(K.range_axis rng) ~kind:(K.range_kind altrng) ()
                in
                t.ast <- K.substitute [ (rng, new_rng); (altrng, new_altrng) ] t.ast;
                None
            | _ -> raise (Opt_error "unsupported opt"))
      in
      if append_opt then t.applied_opts <- t.applied_opts @ [ opt ];
      result

(* Returns raw Param nodes; the caller (Pipeline) constructs device buffers,
   keeping Postrange device-agnostic. *)
let bufs_from_ast ast =
  List.sort
    (fun a b ->
      match K.view a, K.view b with
      | Param { idx = ia; _ }, Param { idx = ib; _ } -> compare ia ib
      | _ -> 0)
    (List.filter
       (fun n -> match K.view n with Param _ -> true | _ -> false)
       (K.backward_slice ast))

let has_bufferize ast =
  List.exists
    (fun n -> match K.view n with Bufferize _ -> true | _ -> false)
    (K.backward_slice ast)

(* Try beam_search first, then heuristic (skipping multi-block kernels). *)
let optimize_strategy beam_search hco ast k =
  match beam_search with
  | Some bs -> bs k
  | None ->
      match hco with
      | Some f when applied_opts k = [] && not (has_bufferize ast) -> f k
      | _ -> k

let apply_opts ?beam_search ?hand_coded_optimizations ast ren =
  if K.tag ast <> None then ast
  else
  let ki = match K.view ast with
    | Sink { kernel_info = Some ki; _ } -> Some ki | _ -> None
  in
  let k = create ast ren in
  convert_loop_to_global k;
  let k = match ki with
    | Some ki -> (
        match ki.opts_to_apply with
        | Some opts ->
            List.iter (fun opt -> ignore (apply_opt k opt)) opts; k
        | None ->
            optimize_strategy beam_search hand_coded_optimizations ast k)
    | None ->
        optimize_strategy beam_search hand_coded_optimizations ast k
  in
  let name_override = match ki with
    | Some ki when ki.name <> "" && ki.name <> "test" -> Some ki.name | _ -> None
  in
  get_optimized_ast ?name_override k

(* Image buffer handling: rather than converting buffer params to a special
   image dtype at this stage, upstream scheduling marks image-eligible buffers
   as Param_image in the Kernel IR, and a late pass (codegen/late/images.ml)
   rewrites accesses to OpenCL image intrinsics at Program IR level.
   See also pipeline.ml step 4. *)
