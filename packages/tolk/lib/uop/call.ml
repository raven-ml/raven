(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module Value = Dtype.Value

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let repr_const c = Format.asprintf "%a" Dtype.pp_const c
let repr_dtype dt = Format.asprintf "%a" Dtype.pp dt

let repr_tuple = function
  | [ x ] -> "(" ^ x ^ ",)"
  | xs -> "(" ^ String.concat ", " xs ^ ")"

let repr_sint = function
  | Int n -> string_of_int n
  | Sym u -> Format.asprintf "%a" pp u

let repr_shape s = repr_tuple (List.map repr_sint s)

(* The first [n] elements of a list, as Python's slice: a short list gives what
   it has. *)
let rec take n l =
  if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: take (n - 1) r

let first op = function
  | s :: _ -> s
  | [] -> invalid_argf "%s needs a source" (Op.name op)

let param_arg_of u =
  match arg u with
  | Param p -> p
  | _ -> invalid_argf "%s has no ParamArg" (Op.name (op u))

let weak_storage dt =
  if List.mem dt Dtype.weaks then
    invalid_argf "a %s cannot be stored" (repr_dtype dt)

let dedup_nodes l =
  Helpers.dedup
    (module struct
      type nonrec t = t

      let equal = ( == )
      let hash = hash
    end)
    l

(* The truth of a condition that must be decidable. *)
let truth = function Sint.Known b -> b | Sint.Cond u -> to_bool u

let ssimplify_sint = function Int n -> Int n | Sym u -> ssimplify u

(* The number of elements of a shape of known sizes. *)
let size_of shape =
  if List.mem 0 shape then 0
  else
    List.fold_left
      (fun acc n ->
        if n <> 0 && abs acc > max_int / abs n then
          invalid_argf "a shape of %s elements is larger than an int"
            (Bigint.to_string
               (List.fold_left (fun z n -> Bigint.mul z (Bigint.of_int n)) Bigint.one shape))
        else acc * n)
      1 shape

let as_value (c : Dtype.const) : Dtype.value =
  match c with
  | `Invalid -> invalid_arg "Invalid is not a number"
  | #Dtype.value as x -> x

(* A constant that takes part in a greatest common divisor: an integer. *)
let integer c =
  match as_value c with
  | `Float x when not (Float.is_integer x) ->
      invalid_argf "%s is not an integer" (repr_const c)
  | v -> Value.to_z v

let marg_shape u =
  match marg u with
  | Reshape s | Expand s -> s
  | _ -> invalid_argf "%s has no shape argument" (Op.name (op u))

let marg_bounds u =
  match marg u with
  | Pad b | Shrink b -> b
  | _ -> invalid_argf "%s has no bounds argument" (Op.name (op u))

(* Several devices *)

let rec axis u =
  match axis_memo u with
  | Some a -> a
  | None ->
      let a = compute_axis u in
      set_axis_memo u a;
      a

and compute_axis u =
  let src0 () = first (op u) (src u) in
  match op u with
  | Op.Copy | Op.Param -> None
  | Op.Unshard -> (
      match arg u with
      | Axes [ a ] -> Some a
      | Axes l ->
          invalid_argf "the value is sharded on several axes, %s"
            (repr_tuple (List.map string_of_int l))
      | _ -> invalid_arg "an unshard needs its axes")
  | op when Op.Set.mem op Op.Set.alu || op = Op.Stack -> (
      let n = ndim u in
      let axes =
        List.filter_map
          (fun x -> Option.map (fun a -> a + n - ndim x) (axis x))
          (src u)
      in
      match List.rev (Helpers.dedup (module Int) axes) with
      | [] -> None
      | last :: _ -> Some last)
  | _ when List.is_empty (src u) -> None
  | op -> (
      let src_axis = axis (src0 ()) in
      match (op, src_axis) with
      | Op.Shrink, Some a ->
          let o, sz = List.nth (marg_bounds u) a in
          if
            Sint.equal o (Int 0) && Sint.equal sz (List.nth (shape (src0 ())) a)
          then Some a
          else None
      | Op.Reduce, a -> (
          match (a, arg u) with
          | None, _ -> None
          | Some a, Reduce { num_axes; _ } ->
              if a < num_axes then None else Some (a - num_axes)
          | _ -> invalid_arg "a reduction needs its argument")
      | Op.Reshape, None -> None
      | Op.Reshape, Some a -> Some (reshape_axis u a)
      | Op.Permute, a -> (
          match (a, marg u) with
          | Some a, Permute order -> List.find_index (Int.equal a) order
          | _ -> None)
      | Op.Expand, a -> (
          match (a, marg u) with
          | Some a, Expand s -> Some (a + List.length s)
          | _ -> None)
      | _, a -> a)

(* The new axis is the last one before which the element count is the count
   before the source's axis, and it must not move elements between shards. *)
and reshape_axis u src_axis =
  let src = first (op u) (src u) in
  let new_shape = marg_shape u in
  let prefix =
    List.rev
      (List.fold_left
         (fun acc s -> Sint.(List.hd acc * s) :: acc)
         [ Int 1 ] new_shape)
  in
  let acc = List.map ssimplify_sint prefix in
  let target = ssimplify_sint (Sint.prod (take src_axis (shape src))) in
  let moved () =
    invalid_argf "a reshape of %s to %s moves elements between shards"
      (repr_shape (shape src))
      (repr_shape (shape u))
  in
  let rec last_index i best = function
    | [] -> best
    | x :: rest ->
        last_index (i + 1) (if Sint.equal x target then Some i else best) rest
  in
  let new_axis =
    match last_index 0 None acc with Some i -> i | None -> moved ()
  in
  let dcount =
    match device u with
    | Some (Multi ds) -> List.length ds
    | _ -> (
        match
          List.find_opt (fun n -> op n = Op.Unshard) (toposort ~calls:Enter src)
        with
        | Some un -> Value.to_int (vmax (nth un 1)) + 1
        | None -> moved ())
  in
  if truth Sint.(List.nth (shape u) new_axis % Int dcount <> Int 0) then
    moved ();
  new_axis

let shard_count u =
  if op u = Op.Unshard then Value.to_int (vmax (nth u 1)) + 1
  else
    match device u with
    | Some (Multi ds) -> List.length ds
    | _ -> invalid_arg "the value is not on several devices"

let bounds u =
  match axis u with
  | None -> invalid_arg "bounds need a sharded value"
  | Some a ->
      let size = List.nth (shape (first (op u) (src u))) a in
      let starts =
        List.rev
          (List.fold_left
             (fun acc _ -> Sint.(List.hd acc + size) :: acc)
             [ Int 0 ]
             (List.init (shard_count u) Fun.id))
      in
      let rec pairs = function
        | x :: (y :: _ as rest) -> (x, y) :: pairs rest
        | _ -> []
      in
      pairs starts

let shard_shape u =
  match device u with
  | Some (Multi _) -> (
      match axis u with
      | Some a ->
          let n = shard_count u in
          List.mapi
            (fun i x -> if i = a then Sint.(x // Int n) else x)
            (shape u)
      | None -> shape u)
  | _ -> shape u

let max_shard_shape u = to_max_shape (shard_shape u)

let unshard ?ranges u axes =
  let ranges =
    match ranges with
    | Some r -> r
    | None -> (
        match device u with
        | Some (Multi ds) ->
            [ range ~axis_type:Axis_type.Device (Int (List.length ds)) [ -1 ] ]
        | _ -> invalid_arg "an unshard needs a value on several devices")
  in
  if
    List.length axes <> List.length ranges
    || List.length (List.sort_uniq Int.compare axes) <> List.length axes
  then invalid_arg "an unshard needs one range per distinct axis";
  let pairs =
    List.stable_sort
      (fun (a, _) (b, _) -> Int.compare a b)
      (List.combine axes ranges)
  in
  v Op.Unshard ~src:(u :: List.map snd pairs) ~arg:(Axes (List.map fst pairs))

let shard_slice u a rng =
  match shape u with
  | [] -> u
  | s ->
      let dcount = Value.to_int (vmax rng) + 1 in
      let size = List.nth s a in
      if truth Sint.(size % Int dcount <> Int 0) then
        invalid_argf "axis %d of size %s does not split over %d devices" a
          (repr_sint size) dcount;
      let sz = Sint.(size // Int dcount) in
      let r = Sym rng in
      shrink u
        (List.mapi
           (fun i x ->
             if i = a then Some (Sint.(r * sz), Sint.((r * sz) + sz))
             else Some (Int 0, x))
           s)

let shard ?axis u devices =
  let copied = copy_to_device u (Multi devices) in
  match axis with
  | None -> copied
  | Some a ->
      let rng =
        range ~axis_type:Axis_type.Device (Int (List.length devices)) [ -1 ]
      in
      unshard (shard_slice copied a rng) [ a ]

(* Storage *)

let empty ?device new_shape dt =
  weak_storage dt;
  let max_shape = to_max_shape new_shape in
  let u =
    v Op.Alloc ~src:(device_range_src device)
      ~arg:
        (Param
           (param_arg ~slot:(unique_num ()) ~size:(size_of max_shape) ?device
              ~bind_on_realize:true dt))
  in
  shrink_to
    (reshape u (List.map (fun n -> Int n) max_shape))
    (List.map Option.some new_shape)

let view_as ?axis u new_shape =
  let max_shape = List.map (fun n -> Int n) (to_max_shape new_shape) in
  let ret = if List.length new_shape > 1 then reshape u max_shape else u in
  let ret =
    if (List.equal Sint.equal) max_shape new_shape then ret
    else shrink_to ret (List.map Option.some new_shape)
  in
  match axis with None -> ret | Some a -> unshard ret [ a ]

let empty_like ?dtype ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  let dtype = match dtype with Some dt -> dt | None -> commit_dtype u in
  match dev with
  | Some (Multi _) when Option.is_some (axis u) ->
      unshard (empty ?device:dev (shard_shape u) dtype) [ Option.get (axis u) ]
  | _ -> empty ?device:dev (shape u) dtype

let clone ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  (match dev with
  | Some d when is_disk_device d ->
      invalid_arg "cannot clone a disk; store into a disk buffer instead"
  | _ -> ());
  let ret = empty_like ?device:dev u in
  let src =
    match (device u, dev) with
    | None, _ -> u
    | Some d, Some d' when equal_device d d' -> u
    | _, Some d' -> copy_to_device u d'
    | _, None -> u
  in
  after ret [ store ret (cast src (dtype ret)) ]

let alloc ?slot ?(addrspace = Dtype.Global) ?device ?axis new_shape dt =
  let slot = match slot with Some s -> s | None -> unique_num () in
  let ret =
    v Op.Alloc ~src:(device_range_src device)
      ~arg:
        (Param
           (param_arg ~slot
              ~size:(size_of (to_max_shape new_shape))
              ~addrspace:(Some addrspace) ?device (Dtype.strong dt)))
  in
  if List.is_empty new_shape then reshape ret []
  else view_as ?axis ret new_shape

let alloc_like ?slot ?addrspace u =
  alloc ?slot ?addrspace
    (List.map (fun n -> Int n) (max_shard_shape u))
    (dtype u)

let placeholder ?slot ?(addrspace = Dtype.Global) ?device ?(volatile = false)
    ?tag new_shape dt =
  let dt = Dtype.strong dt in
  let slot = match slot with Some s -> s | None -> unique_num () in
  let name = match tag with Some (Tag.String s) -> Some s | _ -> None in
  let size = size_of new_shape in
  let ret =
    match addrspace with
    | Dtype.Global ->
        v Op.Param
          ~arg:
            (Param
               (param_arg ~slot ~size ?name ~addrspace:(Some addrspace) ?device
                  ~volatile dt))
    | Dtype.Local | Dtype.Reg ->
        if Option.is_some device then
          invalid_arg "workgroup and register storage has no device";
        v Op.Buffer
          ~arg:
            (Param (param_arg ~slot ~size ?name ~addrspace:(Some addrspace) dt))
    | Dtype.Alu -> invalid_arg "a placeholder cannot be a scalar variable"
  in
  let ret = match tag with Some g -> rtag ~tag:g ret | None -> ret in
  if List.length new_shape > 1 then
    reshape ret (List.map (fun n -> Int n) new_shape)
  else ret

let placeholder_like ?addrspace u slot =
  if not (List.for_all (function Int _ -> true | Sym _ -> false) (shape u))
  then invalid_arg "a placeholder needs a shape of known sizes";
  placeholder ~slot ?addrspace (max_shard_shape u) (dtype u)

let param ?shape:new_shape ?device ?vmin_vmax ?multiple_of ?name
    ?(addrspace = Some Dtype.Global) ?(volatile = false) ?phase ?align slot dt
    =
  weak_storage dt;
  let make size =
    v Op.Param
      ~arg:
        (Param
           (param_arg ?size ?vmin_vmax ?multiple_of ?name ~addrspace ?device
              ~volatile ?phase ?align ~slot dt))
  in
  match new_shape with
  | None | Some [] -> make None
  | Some s -> view_as (make (Some (size_of (to_max_shape s)))) s

(* Phases of storage *)

(* Where a stage's view of a buffer, through movements and bitcasts, starts if
   it can be a contiguous run of the buffer: the buffer's alignment and the byte
   its first element lies at, an index taken through each movement to its
   source's. A run's last element lies as many elements past its first as the
   view has elements less one; a view whose first or last element is padding,
   or whose ends lie otherwise, is no run. *)
type view_start =
  | Start of int * int
  | Unknown (* A size or a bound is symbolic, or the buffer is sharded. *)
  | No_view (* No buffer is viewed, or the view is no run of it. *)

let rec view_start u =
  let exception Stop of view_start in
  let int = function Int n -> n | Sym _ -> raise_notrace (Stop Unknown) in
  let dims u = List.map int (shape u) in
  let flat idx shape =
    List.fold_left2 (fun acc i n -> (acc * n) + i) 0 idx shape
  in
  let unflat k shape =
    fst
      (List.fold_right
         (fun n (idx, k) -> ((k mod n) :: idx, k / n))
         shape ([], k))
  in
  let rec go u idx =
    match (op u, src u) with
    | Op.Buffer, _ ->
        let p = param_arg_of u in
        (p.align, p.phase + (flat idx (dims u) * element_size u))
    | Op.Detach, x :: _ -> go x idx
    | Op.Bitcast, x :: _ ->
        let bytes = flat idx (dims u) * element_size u
        and size = element_size x in
        let align, start = go x (unflat (bytes / size) (dims x)) in
        (align, start + (bytes mod size))
    | o, x :: _ when Op.Set.mem o Op.Set.movement ->
        let src = dims x in
        let idx =
          match marg u with
          | Reshape _ -> unflat (flat idx (dims u)) src
          | Expand added -> List.drop (List.length added) idx
          | Shrink b -> List.map2 (fun i (off, _) -> i + int off) idx b
          | Pad b ->
              List.map2
                (fun (i, n) (off, _) ->
                  let j = i - int off in
                  if j < 0 || j >= n then raise_notrace (Stop No_view) else j)
                (List.combine idx src) b
          | Permute p ->
              let moved = Array.make (List.length p) 0 in
              List.iter2 (fun i a -> moved.(a) <- i) idx p;
              Array.to_list moved
          | Flip f ->
              List.map2
                (fun (i, n) f -> if f then n - 1 - i else i)
                (List.combine idx src) f
        in
        go x idx
    | Op.Stage, x :: _ when view_start x <> No_view ->
        raise_notrace (Stop Unknown)
    | Op.Unshard, _ -> raise_notrace (Stop Unknown)
    | _ -> raise_notrace (Stop No_view)
  in
  match
    let shape = dims u in
    let n = List.fold_left ( * ) 1 shape in
    if n = 0 then raise_notrace (Stop No_view);
    let align, first = go u (List.map (fun _ -> 0) shape) in
    let _, last = go u (List.map (fun d -> d - 1) shape) in
    if last - first <> (n - 1) * element_size u then
      raise_notrace (Stop No_view);
    (align, first)
  with
  | align, start -> Start (align, start mod align)
  | exception Stop s -> s
  | exception Division_by_zero -> No_view

(* The alignment and phase of the storage [u] views: its storage's, moved by
   the bytes a shrink of storage seen whole skips, known modulo fewer bytes when
   the shrink's start is symbolic. Any other view keeps its storage's, as views
   are taken to start aligned, unless a symbolic start moves it; storage the
   graph allocates starts on a boundary. A stage of a view of a buffer is that
   view when the view is contiguous, which scheduling decides
   ([Schedule.contiguous_mops_to_view]), and storage of its own otherwise: its
   phase is what the two agree on. *)
let rec storage_phase u =
  let rec whole v =
    match (op v, src v) with
    | (Op.Buffer | Op.Param | Op.Alloc | Op.Stage), _ -> true
    | (Op.Bitcast | Op.Reshape | Op.After | Op.Mselect), v :: _ -> whole v
    | _ -> false
  in
  let ints l = List.for_all (function Int _ -> true | Sym _ -> false) l in
  match (op u, src u) with
  | _ when on_disk u -> (16, 0)
  | (Op.Buffer | Op.Param | Op.Alloc), _ ->
      let p = param_arg_of u in
      (p.align, p.phase)
  | Op.Shrink, x :: _ -> (
      let align, phase = storage_phase x in
      let bytes = element_size x in
      (* The start moved by [first] bytes and by multiples of [by] bytes: known
         modulo the largest power of two up to [align] that divides [by]. *)
      let moved by first =
        let rec known a =
          if a < align && Bigint.divisible by (Bigint.of_int (2 * a)) then known (2 * a)
          else a
        in
        let a = known 1 in
        (a, (((phase + first) mod a) + a) mod a)
      in
      match marg u with
      | Shrink bounds when whole x && ints (shape x) -> (
          (* The element the shrink starts at, by the row-major strides of [x]'s
             shape: a constant, and multiples of a constant that move. *)
          let offset =
            List.fold_left2
              (fun acc (start, _) size -> Sint.((acc * size) + start))
              (Int 0) bounds (shape x)
          in
          match offset with
          | Int n -> moved Bigint.zero (n * bytes)
          | Sym e ->
              let rest, c = pop_const (simplify e) in
              moved
                (Bigint.mul (const_factor rest) (Bigint.of_int bytes))
                (Bigint.to_int (integer c) * bytes))
      | Shrink bounds when not (ints (List.map fst bounds)) ->
          (* A symbolic start into a view that reorders or pads its storage
             moves by bytes unknown here: only the element's size holds. *)
          moved (Bigint.of_int bytes) 0
      | _ -> (align, phase))
  | (Op.Bitcast | Op.After | Op.Mselect), x :: _ -> storage_phase x
  | o, x :: _ when Op.Set.mem o Op.Set.movement -> storage_phase x
  | Op.Stage, x :: _ when op x = Op.Bitcast || Op.Set.mem (op x) Op.Set.movement
    -> (
      match view_start x with
      | No_view -> (16, 0)
      | Unknown -> (1, 0)
      | Start (align, phase) ->
          (* The largest power of two up to [align] that [phase] is a multiple
             of: storage on a boundary starts there too. *)
          let rec known a = if phase mod a = 0 then a else known (a / 2) in
          (known align, 0))
  | _ -> (16, 0)

let param_like u slot =
  match op u with
  | Op.Param when addrspace u = Some Dtype.Alu ->
      let p = param_arg_of u in
      v Op.Param ~arg:(Param { p with slot; name = None; bound = None })
  | _ -> (
      let a = axis u and align, phase = storage_phase u in
      match (a, device u) with
      | Some a, Some (Multi _ as d) ->
          let ss = shard_shape u in
          view_as ~axis:a
            (v Op.Param
               ~arg:
                 (Param
                    (param_arg ~slot
                       ~size:(size_of (to_max_shape ss))
                       ~device:d ~phase ~align (dtype u))))
            ss
      | _ ->
          param ?shape:(shape_opt u) ?device:(device u) ~phase ~align slot
            (dtype u))

(* Variables *)

let is_bound_var u =
  is_variable u
  && match arg u with Param { bound = Some _; _ } -> true | _ -> false

let bind var x =
  if (not (is_variable var)) || is_bound_var var then
    invalid_argf "only an unbound variable binds, not %s" (Op.name (op var));
  let c = const (x :> Dtype.const) in
  if not (Value.( <= ) (vmin var) (vmin c) && Value.( <= ) (vmax c) (vmax var))
  then
    invalid_argf "%s is out of [%s, %s]" (repr_const x)
      (repr_const (vmin var))
      (repr_const (vmax var));
  let p = param_arg_of var in
  let multiple = Option.value p.multiple_of ~default:1 in
  if Option.is_none (divides c (Bigint.of_int multiple)) then
    invalid_argf "%s is not a multiple of %d" (repr_const x) multiple;
  replace var ~arg:(Param { p with bound = Some x })

let unbound var =
  if not (is_variable var) then
    invalid_argf "%s is not a variable" (Op.name (op var));
  replace var ~arg:(Param { (param_arg_of var) with bound = None }) ~tag:None

let unbind var =
  match (is_bound_var var, arg var) with
  | true, Param { bound = Some x; _ } -> (unbound var, x)
  | _ -> invalid_arg "only a bound variable unbinds"

let unbind_all u =
  let bound =
    List.filter is_bound_var
      (Nodes.to_list (backward_slice_with_self ~calls:Skip u))
  in
  let pairs = List.map (fun x -> (x, unbound x)) bound in
  ( substitute ~calls:Skip ~pass:Once u pairs,
    List.map (fun (x, var) -> (var, snd (unbind x))) pairs )

let variables u =
  let found =
    dedup_nodes
      (List.filter_map
         (fun x ->
           if op x = Op.Param && addrspace x = Some Dtype.Alu then
             Some (if is_variable x then unbound x else x)
           else if
             op x = Op.Range && Axis_type.equal (axis_type x) Axis_type.Device
           then
             Some
               (variable ~dtype:(dtype x) "_device_num" (`Int Bigint.zero)
                  (vmax x))
           else None)
         (Nodes.to_list (backward_slice_with_self ~calls:Skip u)))
  in
  let key x =
    let p = param_arg_of x in
    (Option.value p.name ~default:"", p.slot)
  in
  List.stable_sort (fun a b -> Stdlib.compare (key a) (key b)) found

(* Calls *)

let store_call dst src =
  call (store (param_like dst 0) (param_like src 1)) [ dst; src ]

let pm_resolve_params : (t option array, t) Pattern_matcher.t =
  Pattern_matcher.(
    v
      (fun () -> [
        rule_ctx (Upat.op Op.Param ~name:"p") (fun params m ->
            let slot = (param_arg_of (m "p")).slot in
            if slot >= 0 then params.(slot) else None);
      ]))

let call_with_outputs ?name ?(precompile = false) ?aux ?output_pos values args =
  let n = List.length args + List.length values in
  let default_dev = List.find_map device (values @ args) in
  let pos =
    match output_pos with
    | None -> List.init (List.length values) (fun i -> List.length args + i)
    | Some p -> p
  in
  if
    List.length pos <> List.length values
    || List.length (List.sort_uniq Int.compare pos) <> List.length pos
  then invalid_arg "output_pos needs one distinct position per output";
  let rec ascending = function
    | a :: (b :: _ as r) -> a < b && ascending r
    | _ -> true
  in
  if not (ascending pos) then
    invalid_arg "output_pos must be strictly ascending";
  if not (List.for_all (fun p -> p >= 0 && p < n) pos) then
    invalid_arg "output_pos must be within the argument list";
  let params = Array.make n None in
  let remaining = ref args in
  for i = 0 to n - 1 do
    if not (List.mem i pos) then
      match !remaining with
      | a :: rest ->
          params.(i) <- Some a;
          remaining := rest
      | [] -> ()
  done;
  let mint o p =
    let dev = match device o with Some d -> Some d | None -> default_dev in
    let axis = match device o with Some (Multi _) -> axis o | _ -> None in
    let buf = alloc (shard_shape o) (dtype o) ?device:dev ?axis in
    let resolved =
      List.map
        (function
          | Int k -> Int k
          | Sym s ->
              Sym
                (graph_rewrite ~calls:Skip ~pass:Once ~ctx:params s
                   (After_sources pm_resolve_params)))
        (shard_shape o)
    in
    ( alloc resolved (dtype o)
        ~slot:(param_arg_of (buf_uop buf)).slot
        ?device:dev ?axis,
      param_like buf p )
  in
  let outputs = List.map2 mint values pos in
  let body = sink (List.map2 (fun x (_, p) -> store p x) values outputs) in
  let inputs =
    ref (List.map (fun x -> if precompile then contiguous x else x) args)
  in
  let call_args =
    List.init n (fun i ->
        match List.find_index (Int.equal i) pos with
        | Some k -> fst (List.nth outputs k)
        | None -> (
            match !inputs with
            | x :: rest ->
                inputs := rest;
                x
            | [] -> invalid_arg "too few arguments"))
  in
  let c = call body call_args ?name ~precompile ?aux in
  List.map (fun (r, _) -> after r [ c ]) outputs

let call_with_output ?name ?precompile value args =
  List.hd (call_with_outputs ?name ?precompile [ value ] args)

let custom_kernel args f =
  let placeholders = List.mapi (fun i s -> placeholder_like s i) args in
  let kernel = call (f placeholders) args in
  List.map (fun s -> after s [ kernel ]) args

(* Programs *)

let program_info_of_sink
    ?(target =
      Helpers.Target.
        { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" })
    sink =
  let vars = ref [] and globals = ref [] and outs = ref [] and ins = ref [] in
  let global_size = Array.make 3 (Int 1)
  and local_size = Array.make 3 (Int 1) in
  (match arg sink with
  | Kernel { split = Some s; _ } -> global_size.(0) <- s.iterations
  | _ -> ());
  List.iter
    (fun u ->
      if op u = Op.Param then
        if addrspace u = Some Dtype.Alu then vars := u :: !vars
        else globals := (param_arg_of u).slot :: !globals;
      if op u = Op.Store || op u = Op.Load then begin
        let s0 = nth u 0 in
        let idx =
          if op s0 = Op.Index || op s0 = Op.Shrink then Some s0
          else if op s0 = Op.Cast && (op (nth s0 0)) = Op.Index then
            Some (nth s0 0)
          else None
        in
        match idx with
        | Some idx ->
            let buf = buf_uop (nth idx 0) in
            if op buf = Op.Param then
              let slot = (param_arg_of buf).slot in
              if op u = Op.Store then outs := slot :: !outs
              else ins := slot :: !ins
        | None -> ()
      end;
      if op u = Op.Special then
        match arg u with
        | String name ->
            let axis =
              Char.code name.[String.length name - 1] - Char.code '0'
            in
            let sizes = if name.[0] = 'l' then local_size else global_size in
            sizes.(axis) <- ssimplify (nth u 0)
        | _ -> invalid_arg "a hardware index needs a name")
    (toposort ~calls:Enter sink);
  let sorted l = List.sort_uniq Int.compare l in
  let outs, ins =
    if List.is_empty !outs && List.is_empty !ins then (!globals, !globals)
    else (!outs, !ins)
  in
  let vars =
    List.stable_sort
      (fun a b -> Int.compare (param_arg_of a).slot (param_arg_of b).slot)
      (dedup_nodes (List.rev !vars))
  in
  {
    global_size = Array.to_list global_size;
    local_size = Array.to_list local_size;
    vars;
    globals = sorted !globals;
    outs = sorted outs;
    ins = sorted ins;
    target;
  }

let launch_dims (p : program_info) vars =
  ( List.map (fun s -> sym_infer s vars) p.global_size,
    List.map (fun s -> sym_infer s vars) p.local_size )
