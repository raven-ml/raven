(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Core rangeify algorithm.

   Converts the high-level tensor graph (with movement ops, REDUCE_AXIS, etc.)
   into an indexed representation with explicit RANGE loops, BUFFERIZE nodes,
   and INDEX operations. The main entry point is {!run_rangeify}. *)

open Tolk_ir
module T = Tensor
module D = Dtype
module C = Const
module Ak = Axis_kind

(* Predicates *)

let is_always_contiguous = function
  | T.Contiguous _ | T.Copy _ | T.Buffer _ | T.Buffer_view _
  | T.Const _ | T.Bind _ | T.Device _ | T.Mselect _ | T.Mstack _ | T.Param _
  | T.Define_local _ | T.Call _ ->
      true
  | _ -> false

(** [is_assign_after v] returns [true] when [v] is an [After] node whose
    deps include at least one [Store], i.e. the Store+After pattern that
    encodes buffer assignment. *)
let is_assign_after program = function
  | T.After { deps; _ } ->
      List.exists (fun d ->
        match T.view program d with T.Store _ -> true | _ -> false) deps
  | _ -> false

let is_movement_op = function
  | T.Reshape _ | T.Expand _ | T.Pad _ | T.Shrink _ | T.Permute _ | T.Flip _
    ->
      true
  | _ -> false

let is_elementwise = function
  | T.Unary _ | T.Binary _ | T.Ternary _ | T.Cast _ | T.Bitcast _ -> true
  | _ -> false

let pcontig_var = Device.Context.int ~name:"PCONTIG" ~default:0

(* Small helpers *)

let select_by_axes axes xs =
  List.filteri (fun idx _ -> List.mem idx axes) xs

let and_all b = function
  | [] -> T.const b (C.bool true)
  | [ v ] -> v
  | first :: rest ->
      List.fold_left
        (fun acc v -> T.binary b ~op:`And ~lhs:acc ~rhs:v)
        first rest

let movement_src = function
  | T.Reshape { src; _ } | T.Expand { src; _ } | T.Pad { src; _ }
  | T.Shrink { src; _ } | T.Permute { src; _ } | T.Flip { src; _ } ->
      Some src
  | _ -> None

let map_device = function
  | Some (T.Single d) -> Some (Kernel.Device_single d)
  | Some (T.Multi ds) -> Some (Kernel.Device_multi ds)
  | None -> None

(* Get_idx / get_valid *)

let get_idx program r =
  match T.view program r with
  | Ternary { op = `Where; b = idx; c = else_; _ } -> (
      match T.view program else_ with
      | Invalid_index _ -> idx
      | _ -> r)
  | _ -> r

let get_valid b program r =
  match T.view program r with
  | Ternary { op = `Where; a = valid; c = else_; _ } -> (
      match T.view program else_ with
      | Invalid_index _ -> valid
      | _ -> T.const b (C.bool true))
  | Invalid_index _ -> T.const b (C.bool false)
  | _ -> T.const b (C.bool true)

(* Indexing context *)

type realize_state =
  | Marked
  | Realized of int list

type indexing_context = {
  realize_map : realize_state option array;
  range_map : (T.id list * T.id list) option array;
  mutable range_idx : int;
  range_axes : (T.id, int) Hashtbl.t;
}

let create_context n =
  {
    realize_map = Array.make n None;
    range_map = Array.make n None;
    range_idx = 0;
    range_axes = Hashtbl.create 64;
  }

let new_range b ctx size ~kind =
  if size = 1 then T.const b (C.int D.index 0)
  else begin
    let axis = ctx.range_idx in
    ctx.range_idx <- ctx.range_idx + 1;
    let sz = T.const b (C.int D.index size) in
    let id = T.range b ~size:sz ~axis ~kind () in
    Hashtbl.replace ctx.range_axes id axis;
    id
  end

let range_axis_of ctx id =
  match Hashtbl.find_opt ctx.range_axes id with
  | Some axis -> [ axis ]
  | None -> []

(* Generate realize map *)

(* Three-pass scan deciding which tensor nodes must be materialized into buffers.
   Pass 1 marks Copy/Contiguous and After-with-Store (assign) nodes.
   Pass 2 marks movement-op sources of Copy/Mselect/Mstack whose base is not
   inherently contiguous (they need a buffer to read from).  After-with-Store
   is treated as contiguous for base-checking purposes (the target buffer is
   contiguous).
   Pass 3 handles assign hazards: unmarks Copy/Buffer_view values when the
   target has no hazardous movement ops, and marks values that alias their
   own assignment target (write-after-read). *)
let generate_realize_map program ctx =
  let n = T.length program in
  let mark id =
    if ctx.realize_map.(id) = None then ctx.realize_map.(id) <- Some Marked
  in
  for i = 0 to n - 1 do
    match T.view program i with
    | Copy _ | Contiguous _ -> mark i
    | After { deps; _ } ->
        if List.exists
             (fun d -> match T.view program d with Store _ -> true | _ -> false)
             deps
        then mark i
    | _ -> ()
  done;
  let is_base_contiguous base_id =
    let base_v = T.view program base_id in
    is_always_contiguous base_v || is_assign_after program base_v
  in
  for i = 0 to n - 1 do
    match T.view program i with
    | Copy { src; _ } | Mselect { src; _ } ->
        if not (is_base_contiguous (T.base program src)) then mark src
    | Mstack { srcs; _ } ->
        List.iter
          (fun s ->
            if not (is_base_contiguous (T.base program s)) then mark s)
          srcs
    | _ -> ()
  done;
  for i = 0 to n - 1 do
    match T.view program i with
    | After { src = target; deps; _ }
      when List.exists (fun d ->
        match T.view program d with Store _ -> true | _ -> false) deps ->
        let value =
          List.find_map (fun d ->
            match T.view program d with
            | Store { value; _ } -> Some value | _ -> None) deps
        in
        (match value with
        | Some value ->
            (match T.view program value with
            | Copy _ | Buffer_view _ when ctx.realize_map.(value) <> None ->
                let has_hazardous_movement =
                  List.exists
                    (fun dep ->
                      match T.view program dep with
                      | Shrink _ | Permute _ | Flip _ | Pad _ -> true
                      | _ -> false)
                    (T.backward_slice program target)
                in
                if not has_hazardous_movement then ctx.realize_map.(value) <- None
            | _ -> ());
            let target_base = T.base program target in
            if List.mem target_base (T.backward_slice program value) then
              mark value
        | None -> ())
    | _ -> ()
  done

(* Tensor <-> Kernel conversion for symbolic simplification *)

let rec tensor_to_kernel program id =
  match T.view program id with
  | Const { value; _ } -> Kernel.const value
  | Range { size; axis; sub; kind; dtype } ->
      Kernel.range ~size:(tensor_to_kernel program size) ~axis ~sub ~kind ~dtype
        ()
  | Binary { op; lhs; rhs; _ } ->
      Kernel.binary ~op ~lhs:(tensor_to_kernel program lhs)
        ~rhs:(tensor_to_kernel program rhs)
  | Unary { op; src; _ } ->
      Kernel.unary ~op ~src:(tensor_to_kernel program src)
  | Ternary { op; a; b; c; _ } ->
      Kernel.ternary ~op ~a:(tensor_to_kernel program a)
        ~b:(tensor_to_kernel program b) ~c:(tensor_to_kernel program c)
  | Invalid_index { dtype } -> Kernel.invalid_index ~lanes:(Dtype.count dtype) ()
  | v ->
      invalid_arg
        (Format.asprintf "tensor_to_kernel: unsupported node: %a"
           T.pp_view v)

let rec kernel_to_tensor b k =
  match Kernel.view k with
  | Const { value; _ } -> T.const b value
  | Range { size; axis; sub; kind; dtype } ->
      T.range b ~size:(kernel_to_tensor b size) ~axis ~sub ~kind ~dtype ()
  | Binary { op; lhs; rhs; _ } ->
      T.binary b ~op ~lhs:(kernel_to_tensor b lhs)
        ~rhs:(kernel_to_tensor b rhs)
  | Unary { op; src; _ } -> T.unary b ~op ~src:(kernel_to_tensor b src)
  | Ternary { op; a; b = b_; c; _ } ->
      T.ternary b ~op ~a:(kernel_to_tensor b a)
        ~b:(kernel_to_tensor b b_) ~c:(kernel_to_tensor b c)
  | Invalid_index { dtype } -> T.invalid_index b ~dtype
  | _ ->
      invalid_arg
        (Format.asprintf "kernel_to_tensor: unsupported node: %a"
           Kernel.pp_view k)

let simplify_tensor_expr b program id =
  let k = tensor_to_kernel program id in
  kernel_to_tensor b (Kernel.graph_rewrite Symbolic.symbolic k)

(* Apply_movement_op *)

(* Two-phase index transformation for reshape: first collapses the output-space
   ranges into a single flat index via weighted sum (stride = cumulative product
   of trailing dimensions), then extracts per-input-dimension indices via repeated
   mod/div. This is the standard modular-arithmetic reshape decomposition. *)
let apply_reshape b program in_shape out_shape rngs =
  let rev_out = List.rev out_shape in
  let rev_rngs = List.rev rngs in
  let axes_in =
    let acc = ref 1 in
    List.map2
      (fun s r ->
        let term =
          if !acc = 1 then r
          else
            T.binary b ~op:`Mul ~lhs:r
              ~rhs:(T.const b (C.int D.index !acc))
        in
        acc := !acc * s;
        term)
      rev_out rev_rngs
  in
  let combined =
    match axes_in with
    | [] -> T.const b (C.int D.index 0)
    | [ x ] -> x
    | first :: rest ->
        List.fold_left
          (fun acc term -> T.binary b ~op:`Add ~lhs:acc ~rhs:term)
          first rest
  in
  let rev_in = List.rev in_shape in
  let axes_out =
    let remaining = ref combined in
    List.map
      (fun s ->
        if s = 1 then T.const b (C.int D.index 0)
        else begin
          let divisor = T.const b (C.int D.index s) in
          let r = T.binary b ~op:`Mod ~lhs:!remaining ~rhs:divisor in
          remaining := T.binary b ~op:`Idiv ~lhs:!remaining ~rhs:divisor;
          r
        end)
      rev_in
  in
  let result = List.rev axes_out in
  let program' = T.finish b in
  List.map (fun id -> simplify_tensor_expr b program' id) result

(* Maps output-space index ranges back to input-space coordinates for each
   movement op: Shrink adds offsets, Permute reorders via argsort, Flip
   mirrors against (dim-1), Expand zeros out broadcast axes, Pad inserts
   validity guards and offset adjustments, Reshape delegates to apply_reshape. *)
let apply_movement_op b program ~shapes op rngs =
  match op with
  | T.Shrink { before; _ } -> (
      match T.extract_int_shape program before with
      | Some starts ->
          List.map2
            (fun r ss ->
              if ss = 0 then r
              else T.binary b ~op:`Add ~lhs:r
                     ~rhs:(T.const b (C.int D.index ss)))
            rngs starts
      | None -> rngs)
  | Permute { order; _ } ->
      let n = List.length order in
      let argsorted = Array.make n 0 in
      List.iteri (fun i p -> argsorted.(p) <- i) order;
      List.init n (fun i -> List.nth rngs argsorted.(i))
  | Flip { src; dims; _ } -> (
      match shapes.(src) with
      | Some in_shape ->
          List.map2
            (fun r (s, f) ->
              if not f then r
              else T.binary b ~op:`Sub
                     ~lhs:(T.const b (C.int D.index (s - 1))) ~rhs:r)
            rngs (List.combine in_shape dims)
      | None -> rngs)
  | Expand { src; shape; _ } -> (
      match (shapes.(src), T.extract_int_shape program shape) with
      | Some in_shape, Some out_shape ->
          List.map2
            (fun r (in_s, out_s) ->
              if in_s = out_s then r else T.const b (C.int D.index 0))
            rngs (List.combine in_shape out_shape)
      | _ -> rngs)
  | Pad { src; before; after; _ } -> (
      match
        ( shapes.(src),
          T.extract_int_shape program before,
          T.extract_int_shape program after )
      with
      | Some in_shape, Some pad_before, Some pad_after ->
          List.map2
            (fun r ((sh, s), e) ->
              if s = 0 && e = 0 then r
              else
                let s_node = T.const b (C.int D.index s) in
                let sh_plus_s = T.const b (C.int D.index (sh + s)) in
                let lt_s = T.binary b ~op:`Cmplt ~lhs:r ~rhs:s_node in
                let ge_s =
                  T.binary b ~op:`Cmpeq ~lhs:lt_s
                    ~rhs:(T.const b (C.bool false))
                in
                let lt_end = T.binary b ~op:`Cmplt ~lhs:r ~rhs:sh_plus_s in
                let valid = T.binary b ~op:`And ~lhs:ge_s ~rhs:lt_end in
                let offset = T.binary b ~op:`Sub ~lhs:r ~rhs:s_node in
                T.ternary b ~op:`Where ~a:valid ~b:offset
                  ~c:(T.invalid_index b ~dtype:D.index))
            rngs
            (List.combine (List.combine in_shape pad_before) pad_after)
      | _ -> rngs)
  | Reshape { src; shape; _ } -> (
      match (shapes.(src), T.extract_int_shape program shape) with
      | Some in_shape, Some out_shape ->
          apply_reshape b program in_shape out_shape rngs
      | _ -> rngs)
  | _ -> rngs

(* Run_rangeify *)

(* Backward walk over the tensor graph computing per-node index ranges and
   realize (buffering) decisions. Realized nodes get fresh RANGE variables;
   single-consumer nodes inherit their consumer's ranges directly. When a node
   has multiple consumers, per-axis ranges are compared: matching axes are
   merged (with validity OR), while conflicting axes force a realize boundary
   so each consumer gets independent iteration. The PCONTIG heuristic relaxes
   merging for partially-contiguous access patterns. Ending-range tracking
   propagates through elementwise and reduce nodes to detect axes that must
   be realized due to ordering conflicts (range axis > ending axis). *)
let run_rangeify b program ~shapes =
  let n = T.length program in
  let ctx = create_context n in
  generate_realize_map program ctx;
  let consumers = T.consumer_map program in
  let ending_ranges : T.id list array = Array.make n [] in
  for i = n - 1 downto 0 do
    let v = T.view program i in
    let skip =
      match v with
      | Device _ | Unique _ | Lunique _ | Call _ | Mstack _ | Mselect _ -> true
      | After { deps; _ } ->
          not
            (List.exists
               (fun d ->
                 match T.view program d with Store _ -> true | _ -> false)
               deps)
      | _ -> false
    in
    if not skip then begin
      let is_index_dtype =
        match T.dtype program i with
        | Some dt -> D.scalar dt = D.Index && D.count dt = 1
        | None -> false
      in
      if not is_index_dtype then begin
        ending_ranges.(i) <-
          List.concat_map (fun c -> ending_ranges.(c)) consumers.(i);
        let consumer_rngs =
          List.filter_map
            (fun c ->
              match ctx.range_map.(c) with
              | Some (in_rngs, _) -> Some in_rngs
              | None -> None)
            consumers.(i)
        in
        let got_ranges = ref false in
        let out_rngs = ref (
          if ctx.realize_map.(i) <> None then begin
            let shape =
              match shapes.(i) with Some s -> s | None -> []
            in
            let rngs =
              List.map (fun s -> new_range b ctx s ~kind:Ak.Loop) shape
            in
            ending_ranges.(i) <- [];
            ctx.realize_map.(i) <-
              Some (Realized (List.init (List.length shape) Fun.id));
            got_ranges := true;
            rngs
          end
          else if consumer_rngs = [] then []
          else if List.length consumer_rngs = 1 then begin
            got_ranges := true;
            List.hd consumer_rngs
          end
          else begin
            let shape =
              match shapes.(i) with Some s -> s | None -> []
            in
            let n_axes = List.length shape in
            let pcontig = Device.Context.get pcontig_var in
            let all_rngs =
              List.init n_axes (fun ax ->
                  List.filter_map
                    (fun cr ->
                      if ax < List.length cr then Some (List.nth cr ax)
                      else None)
                    consumer_rngs)
            in
            let program' = T.finish b in
            let rngs_valids =
              List.map
                (fun per_axis ->
                  ( List.map (fun r -> get_idx program' r) per_axis,
                    List.map (fun r -> get_valid b program' r) per_axis ))
                all_rngs
            in
            let all_all_same =
              List.for_all
                (fun (local_rngs, _) ->
                  match local_rngs with
                  | [] -> true
                  | first :: rest -> List.for_all (( = ) first) rest)
                rngs_valids
            in
            let realize_axes = ref [] in
            let rngs =
              List.mapi
                (fun ax (local_rngs, valids) ->
                  let axis_same =
                    match local_rngs with
                    | [] -> true
                    | first :: rest -> List.for_all (( = ) first) rest
                  in
                  if all_all_same || (pcontig > 0 && axis_same) then begin
                    match (local_rngs, valids) with
                    | idx :: _, _ ->
                        (* TODO(F6): apply symbolic simplification here.
                           Currently skipped because simplify_tensor_expr
                           creates builder nodes that conflict with the
                           merge_builder id shift. *)
                        let merged_valid =
                          match valids with
                          | [] -> T.const b (C.bool true)
                          | [ v ] -> v
                          | first :: rest ->
                              List.fold_left
                                (fun acc v ->
                                  T.binary b ~op:`Or ~lhs:acc ~rhs:v)
                                first rest
                        in
                        T.ternary b ~op:`Where ~a:merged_valid ~b:idx
                          ~c:(T.invalid_index b ~dtype:D.index)
                    | [], _ ->
                        new_range b ctx (List.nth shape ax) ~kind:Ak.Loop
                  end
                  else begin
                    realize_axes := ax :: !realize_axes;
                    new_range b ctx (List.nth shape ax) ~kind:Ak.Loop
                  end)
                rngs_valids
            in
            if !realize_axes <> [] then
              ctx.realize_map.(i) <-
                Some (Realized (List.rev !realize_axes));
            got_ranges := true;
            rngs
          end)
        in
        if ending_ranges.(i) <> []
           && (is_elementwise v
              || match v with Reduce_axis _ -> true | _ -> false)
        then begin
          let pcontig = Device.Context.get pcontig_var in
          let prev_realize =
            match ctx.realize_map.(i) with
            | Some (Realized axes) -> axes
            | _ -> []
          in
          let new_realize_axes = ref prev_realize in
          List.iteri
            (fun idx r ->
              if not (List.mem idx !new_realize_axes) then begin
                let should_realize =
                  pcontig <= 1
                  || (let r_ranges = range_axis_of ctx r in
                      let e_axes =
                        List.concat_map
                          (fun e -> range_axis_of ctx e)
                          ending_ranges.(i)
                      in
                      List.exists
                        (fun rr -> List.exists (fun ea -> rr > ea) e_axes)
                        r_ranges)
                in
                if should_realize then
                  new_realize_axes := idx :: !new_realize_axes
              end)
            !out_rngs;
          ending_ranges.(i) <- [];
          if !new_realize_axes <> prev_realize then begin
            let realize_axes = List.rev !new_realize_axes in
            ctx.realize_map.(i) <- Some (Realized realize_axes);
            out_rngs :=
              List.mapi
                (fun idx r ->
                  if List.mem idx realize_axes
                     && not (List.mem idx prev_realize)
                  then begin
                    let shape =
                      match shapes.(i) with Some s -> s | None -> []
                    in
                    if idx < List.length shape then
                      new_range b ctx (List.nth shape idx) ~kind:Ak.Loop
                    else r
                  end
                  else r)
                !out_rngs
          end
        end;
        if !got_ranges || !out_rngs <> [] then begin
          let in_rngs = ref !out_rngs in
          if is_movement_op v then
            in_rngs := apply_movement_op b program ~shapes v !out_rngs;
          (match v with
          | Expand _ when shapes.(i) <> None ->
              let new_ending =
                List.filter_map
                  (fun (ri, ro) ->
                    if ri <> ro && Hashtbl.mem ctx.range_axes ro then Some ro
                    else None)
                  (List.combine !in_rngs !out_rngs)
              in
              ending_ranges.(i) <- ending_ranges.(i) @ new_ending
          | _ -> ());
          (match v with
          | Reduce_axis { src; axes; _ } -> (
              match shapes.(src) with
              | Some src_shape ->
                  in_rngs :=
                    List.mapi
                      (fun idx (r, s) ->
                        if List.mem idx axes then
                          new_range b ctx s ~kind:Ak.Reduce
                        else r)
                      (List.combine !in_rngs src_shape)
              | None -> ())
          | _ -> ());
          ctx.range_map.(i) <- Some (!in_rngs, !out_rngs)
        end
      end
    end
  done;
  let merged, shift = T.merge_builder program b in
  let merged_n = T.length merged in
  for i = 0 to n - 1 do
    match ctx.range_map.(i) with
    | Some (in_rngs, out_rngs) ->
        ctx.range_map.(i) <-
          Some (List.map shift in_rngs, List.map shift out_rngs)
    | None -> ()
  done;
  let extend_opt_array arr new_len =
    let old_len = Array.length arr in
    if new_len > old_len then begin
      let arr' = Array.make new_len None in
      Array.blit arr 0 arr' 0 old_len;
      arr'
    end
    else arr
  in
  let ctx =
    {
      ctx with
      realize_map = extend_opt_array ctx.realize_map merged_n;
      range_map = extend_opt_array ctx.range_map merged_n;
    }
  in
  (ctx, merged)

(* Apply_rangeify_pass *)

(* Manual translation pass that applies the rangeify rewrite rules.

   Walks the OLD program (which includes merged range nodes), uses OLD ids
   for ctx lookups, and builds a new program with remapped ids. *)
let apply_rangeify_pass program ctx ~shapes:_ ~devices =
  let n = T.length program in
  let b = T.create () in
  let remap = Array.make n (-1) in
  let map_ref id = if id >= 0 && id < n then remap.(id) else id in
  (* Pre-emit builder-created nodes (range/const/arith from multi-consumer
     merge) that appear at the end of the program. *)
  for i = 0 to n - 1 do
    if remap.(i) = -1 && ctx.realize_map.(i) = None
       && ctx.range_map.(i) = None
       && (match T.view program i with
          | Range _ | Const _ | Invalid_index _ | Vconst _ | Binary _
          | Unary _ | Ternary _ ->
              true
          | _ -> false)
    then begin
      let children = T.children_of (T.view program i) in
      List.iter
        (fun c ->
          if c >= 0 && c < n && remap.(c) = -1 then
            remap.(c) <- T.emit b (T.map_children map_ref (T.view program c)))
        children;
      remap.(i) <- T.emit b (T.map_children map_ref (T.view program i))
    end
  done;
  (* Build reverse map: new_id -> list of old_ids.
     Maintained incrementally as remap entries are added.
     When looking up a child for realize_map, we pick the first old_id
     that IS in realize_map, avoiding the non-injectivity problem from
     movement-op aliasing. *)
  let rev_remap : (int, int list) Hashtbl.t = Hashtbl.create n in
  let add_rev_remap ~old_id ~new_id =
    if new_id >= 0 then begin
      let prev = match Hashtbl.find_opt rev_remap new_id with
        | Some l -> l | None -> []
      in
      Hashtbl.replace rev_remap new_id (old_id :: prev)
    end
  in
  (* Seed with pre-emitted entries *)
  Array.iteri (fun old_id new_id -> add_rev_remap ~old_id ~new_id) remap;
  for i = 0 to n - 1 do
    if remap.(i) <> -1 then ()
    else begin
    let old_v = T.view program i in
    let mapped_v = T.map_children map_ref old_v in

    (* Rule 1: REDUCE_AXIS -> REDUCE *)
    let node_result =
      match old_v with
      | Reduce_axis { op; axes; dtype; _ } -> (
          match ctx.range_map.(i) with
          | Some (in_rngs, _) ->
              let mapped_src =
                match mapped_v with Reduce_axis { src; _ } -> src | _ -> 0
              in
              Some
                (T.Reduce
                   {
                     src = mapped_src;
                     ranges = List.map map_ref (select_by_axes axes in_rngs);
                     op;
                     dtype;
                   })
          | None -> None)
      | _ -> None
    in
    (* Rule 2: PAD -> WHERE *)
    let node_result =
      match node_result with
      | Some _ -> node_result
      | None -> (
          match old_v with
          | Pad { dtype; _ } -> (
              match ctx.range_map.(i) with
              | Some (in_rngs, _) ->
                  let valids =
                    List.filter_map
                      (fun r ->
                        match T.view program r with
                        | Ternary { op = `Where; a = valid; _ } -> Some valid
                        | _ -> None)
                      in_rngs
                  in
                  let valid =
                    and_all b (List.map map_ref valids)
                  in
                  let mapped_src =
                    match mapped_v with Pad { src; _ } -> src | _ -> 0
                  in
                  Some
                    (T.Ternary
                       {
                         op = `Where;
                         a = valid;
                         b = mapped_src;
                         c = T.const b (C.zero dtype);
                         dtype;
                       })
              | None -> None)
          | _ -> None)
    in

    (* Rule 3: create BUFFERIZE + INDEX for realized sources.
       Fires on ALL nodes (GroupOp.All) including those already transformed by
       Rules 1/2. We operate on the effective view and use rev_remap to find
       old child ids. *)
    let effective_v = match node_result with Some v -> v | None -> mapped_v in
    let result =
      match effective_v with
      | Bufferize _ | Index _ -> node_result
      | _ ->
          let children_new = T.children_of effective_v in
          let changed = ref false in
          let new_children =
            List.map
              (fun new_s ->
                (* Find the old_id for this new child.  rev_remap maps
                   new_id -> old_id list (multiple old nodes may alias
                   to the same new node after movement-op removal).
                   Pick the first old_id that is in realize_map. *)
                let old_candidates =
                  match Hashtbl.find_opt rev_remap new_s with
                  | Some l -> l | None -> []
                in
                let old_s =
                  match List.find_opt
                    (fun o -> o >= 0 && o < n && ctx.realize_map.(o) <> None)
                    old_candidates
                  with
                  | Some o -> o
                  | None -> (
                      match old_candidates with o :: _ -> o | [] -> -1)
                in
                if old_s >= 0 && old_s < n
                   && ctx.realize_map.(old_s) <> None then begin
                  match ctx.realize_map.(old_s) with
                  | Some (Realized axes) -> (
                      match ctx.range_map.(old_s) with
                      | Some (_, out_rngs) ->
                          let closed = select_by_axes axes out_rngs in
                          if match T.view program old_s with
                             | Store _ -> true
                             | _ -> false
                          then begin
                            let range_ids =
                              List.filter
                                (fun r ->
                                  match T.view program r with
                                  | Range _ -> true
                                  | _ -> false)
                                closed
                            in
                            changed := true;
                            T.end_ b ~value:new_s
                              ~ranges:(List.map map_ref range_ids)
                          end
                          else begin
                            let is_copy =
                              match effective_v with Copy _ -> true | _ -> false
                            in
                            let removable =
                              (not is_copy)
                              && not (is_always_contiguous
                                       (T.view program old_s))
                            in
                            let is_local =
                              List.length out_rngs <> List.length axes
                            in
                            let addrspace =
                              if is_local then D.Local else D.Global
                            in
                            let opts : Kernel.bufferize_opts =
                              {
                                device = map_device devices.(old_s);
                                addrspace;
                                removable;
                              }
                            in
                            let dtype =
                              match T.dtype program old_s with
                              | Some d -> d
                              | None -> D.float32
                            in
                            let mapped_closed = List.map map_ref closed in
                            let bufferized =
                              T.bufferize b ~src:new_s ~ranges:mapped_closed
                                ~dtype ~opts
                            in
                            changed := true;
                            match ctx.range_map.(i) with
                            | Some (in_rngs, _) ->
                                let idx_rngs =
                                  select_by_axes axes in_rngs
                                in
                                T.index b ~ptr:bufferized
                                  ~idxs:(List.map map_ref idx_rngs) ~dtype ()
                            | None -> bufferized
                          end
                      | None -> new_s)
                  | _ -> new_s
                end
                else if old_s >= 0 && old_s < n
                   && (match T.view program old_s with
                      | Param _ | Buffer_view _ | Mstack _ | Mselect _
                      | After _ ->
                          true
                      | _ -> false)
                   && ctx.range_map.(i) <> None
                then begin
                  match ctx.range_map.(i) with
                  | Some (in_rngs, _) ->
                      let dtype =
                        match T.dtype program old_s with
                        | Some d -> d
                        | None -> D.index
                      in
                      changed := true;
                      T.index b ~ptr:new_s
                        ~idxs:(List.map map_ref in_rngs) ~dtype ()
                  | None -> new_s
                end
                else new_s)
              children_new
          in
          if !changed then begin
            let tbl = Hashtbl.create 8 in
            List.iter2
              (fun orig_s new_s -> Hashtbl.replace tbl orig_s new_s)
              children_new new_children;
            Some
              (T.map_children
                 (fun id ->
                   match Hashtbl.find_opt tbl id with
                   | Some s -> s
                   | None -> id)
                 effective_v)
          end
          else node_result
    in
    (* Rule 4: remove movement op.
       Alias the remap to the source's remap instead of emitting. *)
    let aliased = ref false in
    let result =
      match result with
      | Some _ -> result
      | None -> (
          match movement_src old_v with
          | Some src ->
              if ctx.range_map.(i) <> None
                 || (match T.view program src with
                    | Index _ -> true
                    | _ -> false)
              then begin
                aliased := true;
                None
              end
              else None
          | None -> None)
    in
    if !aliased then begin
      match movement_src old_v with
      | Some src ->
          let new_id = map_ref src in
          remap.(i) <- new_id;
          add_rev_remap ~old_id:i ~new_id
      | None ->
          let new_id = T.emit b mapped_v in
          remap.(i) <- new_id;
          add_rev_remap ~old_id:i ~new_id
    end
    else begin
      let new_id = T.emit b (match result with Some v -> v | None -> mapped_v) in
      remap.(i) <- new_id;
      add_rev_remap ~old_id:i ~new_id
    end
    end
  done;
  T.finish b
