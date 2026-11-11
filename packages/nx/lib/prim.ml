(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Value
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module Shape = Nx_kernel.Shape

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* Values' facts *)

let rec placement : type v s d. (v, s, d) t -> d Devices.placement = function
  | Array { at; _ } | Shards { at; _ } -> at
  | Handle { over; _ } -> placement over
  | Traced { form; _ } -> form.placement

let rec dtype : type v s d. (v, s, d) t -> (v, s) dtype = function
  | Array { a; _ } -> Nx_array.dtype a
  | Shards { arrays; _ } -> Nx_array.dtype (Iarray.get arrays 0)
  | Handle { over; _ } -> dtype over
  | Traced { form; _ } -> form.dtype

(* The layout of [x]'s array, of its first shard, or of its form. *)
let rec own_layout : type v s d. (v, s, d) t -> L.t = function
  | Array { a; _ } -> Nx_array.layout a
  | Shards { arrays; _ } -> Nx_array.layout (Iarray.get arrays 0)
  | Handle { over; _ } -> own_layout over
  | Traced { form; _ } -> form.layout

let rank x = L.rank (own_layout x)

let rec death : type v s d. (v, s, d) t -> death = function
  | Array r -> r.dead
  | Shards r -> r.dead
  | Handle { donor; _ } -> death donor
  | Traced _ -> Live

let rec arrays : type v s d. (v, s, d) t -> (v, s) Nx_array.t iarray =
  function
  | Array { a; _ } -> [| a |]
  | Shards { arrays; _ } -> arrays
  | Handle { over; _ } -> arrays over
  | Traced _ -> invalid_arg "Prim.arrays: a traced value has no arrays"

let of_arrays (type v s d) (p : d Devices.placement) ~shape
    (arrays : (v, s) Nx_array.t iarray) : (v, s, d) t =
  if Iarray.length arrays = 1 then
    Array { at = p; a = Iarray.get arrays 0; dead = Live }
  else Shards { at = p; shape; arrays; dead = Live }

let rec dim : type v s d. (v, s, d) t -> int -> int =
 fun x i ->
  let l = own_layout x in
  if i < 0 || i >= L.rank l then
    invalid_arg
      (Printf.sprintf "Prim.dim: axis %d of a value of rank %d" i (L.rank l));
  match x with
  | Array _ | Traced _ -> L.dim l i
  | Shards { shape; _ } -> shape.(i)
  | Handle { over; _ } -> dim over i

(* [s] as a fresh array. Iarray.to_array copies by a call into the
   runtime's C, which costs more than reading a shape; an array literal
   allocates inline, so a shape of up to four axes is copied into one. *)
let thaw s =
  let r = Iarray.length s in
  let a =
    match r with
    | 0 -> [||]
    | 1 -> [| 0 |]
    | 2 -> [| 0; 0 |]
    | 3 -> [| 0; 0; 0 |]
    | 4 -> [| 0; 0; 0; 0 |]
    | r -> Array.make r 0
  in
  for i = 0 to r - 1 do
    Array.unsafe_set a i (Iarray.unsafe_get s i)
  done;
  a

let rec shape : type v s d. (v, s, d) t -> int array =
 fun x ->
  match x with
  | Array _ | Traced _ -> thaw (L.shape (own_layout x))
  | Shards { shape; _ } -> Array.copy shape
  | Handle { over; _ } -> shape over

(* [x]'s shape as layouts hold it: its own layout's, without a copy, or the
   whole's of a value on several devices. *)
let rec layout_shape : type v s d. (v, s, d) t -> int iarray =
 fun x ->
  match x with
  | Array _ | Traced _ -> L.shape (own_layout x)
  | Shards { shape; _ } -> Iarray.of_array shape
  | Handle { over; _ } -> layout_shape over

let scale_shape (type d) (sc : d scale) =
  match sc with Exponent x -> shape x | Value x -> shape x

let rec has_dims : type v s d. (v, s, d) t -> int array -> int -> bool =
 fun x s i -> i = Array.length s || (dim x i = s.(i) && has_dims x s (i + 1))

let has_shape x s = rank x = Array.length s && has_dims x s 0

let rec layouts_equal la lb i =
  i = L.rank la || (L.dim la i = L.dim lb i && layouts_equal la lb (i + 1))

let rec dims_equal x y i =
  i = rank x || (dim x i = dim y i && dims_equal x y (i + 1))

let same_shape (type v s w r d) (x : (v, s, d) t) (y : (w, r, d) t) =
  match (x, y) with
  | Array { a; _ }, Array { a = b; _ } ->
      let la = Nx_array.layout a and lb = Nx_array.layout b in
      L.rank la = L.rank lb && layouts_equal la lb 0
  | _ -> rank x = rank y && dims_equal x y 0

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let pp_ints ppf s = pp_shape ppf (Iarray.to_array s)
let pp_value ppf x =
  Format.fprintf ppf "%s %a" (D.name (dtype x)) pp_shape (shape x)

let alive ~by x =
  match death x with
  | Live -> ()
  | Died why -> invalid_argf "%s: %a was donated to %s" by pp_value x why

(* Each refusal of [Layout.fits] in its own words: they are rare, so the cheap
   predicate runs first. *)
let check_shape ~by s =
  if not (L.fits s) then begin
    if Iarray.exists (fun d -> d < 0) s then
      invalid_argf "%s: shape %a has a negative extent" by pp_ints s;
    if Iarray.length s > L.max_rank then
      invalid_argf "%s: shape %a has %d axes, more than %d" by pp_ints s
        (Iarray.length s) L.max_rank;
    invalid_argf "%s: the extents of shape %a other than 0 multiply past %d" by
      pp_ints s L.max_numel
  end

(* The C-order layout of shape [s], as the kernels' rules and {!shape} give
   it. *)
let contiguous s = L.contiguous (Iarray.of_array s)

let check_value ~by dt v =
  if not (D.holds dt v) then
    let pp = D.pp_value dt in
    invalid_argf "%s: %a is outside %s's range [%a, %a]" by pp v (D.name dt) pp
      (D.min_value dt) pp (D.max_value dt)

let axis ~by r a what =
  let a' = if a < 0 then a + r else a in
  if a' < 0 || a' >= r then invalid_argf "%s: %d is not an axis of %t" by a what
  else a'

let merge s s' =
  let r = Int.max (Array.length s) (Array.length s') in
  let at s i =
    let k = i - (r - Array.length s) in
    if k < 0 then 1 else s.(k)
  in
  let out = if r = 0 then [||] else Array.make r 1 in
  let rec go i =
    if i = r then Ok out
    else
      let a = at s i and b = at s' i in
      if a = b || b = 1 then (
        out.(i) <- a;
        go (i + 1))
      else if a = 1 then (
        out.(i) <- b;
        go (i + 1))
      else Error (i, a, b)
  in
  go 0

let broadcast_shape ~by s s' =
  match merge s s' with
  | Ok s'' -> s''
  | Error _ ->
      invalid_argf "%s: shapes %a and %a do not broadcast" by pp_shape s
        pp_shape s'

(* The form of a value of [dtype] laid out as [layout] at [placement]: a value
   on several devices, whose arrays are each its window, is laid out as the
   C-contiguous whole. *)
let of_layout (type v s d) (dtype : (v, s) dtype) layout
    (placement : d Devices.placement) : (v, s, d) form =
  if
    Grid.count (Devices.grid placement) > 1
    && not (L.is_contiguous layout && L.offset layout = 0)
  then { dtype; layout = L.contiguous (L.shape layout); placement }
  else { dtype; layout; placement }

(* [x]'s layout, as its form has it, allocating nothing for a value on one
   device. *)
let rec form_layout : type v s d. (v, s, d) t -> L.t =
 fun x ->
  match x with
  | Array { a; _ } -> Nx_array.layout a
  | Shards _ -> L.contiguous (layout_shape x)
  | Handle { over; _ } -> form_layout over
  | Traced { form; _ } -> form.layout

let form (type v s d) (x : (v, s, d) t) : (v, s, d) form =
  match x with
  | Traced { form; _ } -> form
  | Array _ | Shards _ | Handle _ ->
      { dtype = dtype x; layout = form_layout x; placement = placement x }

let expect (type v s) (dt : (v, s) dtype) (Any x : 'd any) : (v, s, 'd) t =
  match D.equal_witness dt (dtype x) with
  | Some Type.Equal -> x
  | None ->
      invalid_argf "Prim.expect: a %s value where %s was expected"
        (D.name (dtype x))
        (D.name dt)

(* Operations *)

type operands = Operands : 'd any list -> operands

(* Transforms and factorisations: their routines and operands *)

let routine_name : type d r. (d, r) linalg -> string = function
  | Cholesky _ -> "Cholesky"
  | Lu _ -> "Lu"
  | Qr _ -> "Qr"
  | Svd_values _ | Svd _ -> "Svd"
  | Eigh_values _ | Eigh _ -> "Eigh"
  | Eig_values _ | Eig _ -> "Eig"
  | Solve_triangular _ -> "Solve_triangular"

(* [l]'s operands: [a], then [b] for a triangular solve. *)
let linalg_operands : type d r. (d, r) linalg -> d any list = function
  | Cholesky { a; _ } -> [ Any a ]
  | Lu a -> [ Any a ]
  | Qr { a; _ } -> [ Any a ]
  | Svd_values { a; _ } -> [ Any a ]
  | Svd { a; _ } -> [ Any a ]
  | Eigh_values { a; _ } -> [ Any a ]
  | Eigh { a; _ } -> [ Any a ]
  | Eig_values { a; _ } -> [ Any a ]
  | Eig { a; _ } -> [ Any a ]
  | Solve_triangular { a; b; _ } -> [ Any a; Any b ]

let fft_axes : type d r. (d, r) fft -> int iarray = function
  | C2c { axes; _ } | R2c { axes; _ } | C2r { axes; _ } -> axes

let fft_operand : type d r. (d, r) fft -> d any = function
  | C2c { x; _ } -> Any x
  | R2c { x; _ } -> Any x
  | C2r { x; _ } -> Any x

(* The map helpers below give their argument itself where [m] returns each
   operand itself, so that an operation no placement moves allocates
   nothing. *)

(* [f] with its operand [x] replaced by [m x]. *)
let map_fft : type d r.
    ('v 's. ('v, 's, d) t -> ('v, 's, d) t) -> (d, r) fft -> (d, r) fft =
 fun m f ->
  match f with
  | C2c r ->
      let x = m r.x in
      if x == r.x then f else C2c { r with x }
  | R2c r ->
      let x = m r.x in
      if x == r.x then f else R2c { r with x }
  | C2r r ->
      let x = m r.x in
      if x == r.x then f else C2r { r with x }

(* [l] with its operand [x] at position [i] replaced by [m i x]. *)
let map_linalg : type d r.
    ('v 's. int -> ('v, 's, d) t -> ('v, 's, d) t) ->
    (d, r) linalg ->
    (d, r) linalg =
 fun m l ->
  match l with
  | Cholesky r ->
      let a = m 0 r.a in
      if a == r.a then l else Cholesky { r with a }
  | Lu a ->
      let a' = m 0 a in
      if a' == a then l else Lu a'
  | Qr r ->
      let a = m 0 r.a in
      if a == r.a then l else Qr { r with a }
  | Svd_values r ->
      let a = m 0 r.a in
      if a == r.a then l else Svd_values { r with a }
  | Svd r ->
      let a = m 0 r.a in
      if a == r.a then l else Svd { r with a }
  | Eigh_values r ->
      let a = m 0 r.a in
      if a == r.a then l else Eigh_values { r with a }
  | Eigh r ->
      let a = m 0 r.a in
      if a == r.a then l else Eigh { r with a }
  | Eig_values r ->
      let a = m 0 r.a in
      if a == r.a then l else Eig_values { r with a }
  | Eig r ->
      let a = m 0 r.a in
      if a == r.a then l else Eig { r with a }
  | Solve_triangular r ->
      let a = m 0 r.a in
      let b = m 1 r.b in
      if a == r.a && b == r.b then l else Solve_triangular { r with a; b }

let name : type r. r prim -> string = function
  | Map _ -> "Map"
  | Reduce _ -> "Reduce"
  | Scan _ -> "Scan"
  | Gather _ -> "Gather"
  | Gather_ranges _ -> "Gather_ranges"
  | Scatter _ -> "Scatter"
  | Sort _ -> "Sort"
  | Unique _ -> "Unique"
  | Fold _ -> "Fold"
  | Fft _ -> "Fft"
  | Linalg _ -> "Linalg"
  | Assemble _ -> "Assemble"
  | Contract _ -> "Contract"
  | Copy _ -> "Copy"
  | Move _ -> "Move"
  | Bitcast _ -> "Bitcast"
  | Place _ -> "Place"
  | Check _ -> "Check"
  | Attention _ -> "Attention"

let load_any (type d) (l : d load) : d any =
  match l with Plain x -> Any x | Padded { x; _ } -> Any x

let is_padded (type d) (l : d load) =
  match l with Plain _ -> false | Padded _ -> true

(* A check's data as operands, in order. *)
let rec data_any : type d e. (d, e) data -> d any list = function
  | [] -> []
  | x :: rest -> Any x :: data_any rest

(* [data] with datum [j] mapped by [m (i + j)]. *)
let rec map_data : type d e.
    ('v 's. int -> ('v, 's, d) t -> ('v, 's, d) t) ->
    int ->
    (d, e) data ->
    (d, e) data =
 fun m i data ->
  match data with
  | [] -> []
  | x :: rest ->
      let x' = m i x in
      let rest' = map_data m (i + 1) rest in
      if x' == x && rest' == rest then data else x' :: rest'

(* A scale's array, and segments' ids. *)
let scale_any : type d. d scale -> d any = function
  | Exponent x -> Any x
  | Value x -> Any x

let ids : 'i Nx_kernel.segments -> 'i = function
  | Rows { ids; _ } | Contracting { ids; _ } -> ids

(* A contraction's operands after [a] and [b], each it has, in order. *)
let contract_rest (type v s d) (init : (v, s, d) t option) a_scale b_scale
    (segments : d index Nx_kernel.segments option) : d any list =
  let some f o : d any list =
    match o with Some x -> List.cons (f x) List.[] | None -> List.[]
  in
  some (fun i -> Any i) init
  @ some scale_any a_scale
  @ some scale_any b_scale
  @ some (fun g -> Any (ids g)) segments

(* Attention's operands: [q], [k], [v], the page table where paged, the
   lengths where given, then the loads. *)
let attention_operands (type v s d) (q : (v, s, d) t) (kv : (v, s, d) kv)
    (lengths : d index option) (loads : d any iarray) : d any list =
  let kv : d any list =
    match kv with
    | Dense { k; v } -> List.[ Any k; Any v ]
    | Paged { k; v; table } -> List.[ Any k; Any v; Any table ]
  in
  let lengths : d any list =
    match lengths with Some l -> List.[ Any l ] | None -> List.[]
  in
  List.cons (Any q) (kv @ lengths @ Iarray.to_list loads)

let operands : type r. r prim -> operands = function
  | Map { loads; _ } -> Operands (Iarray.to_list (Iarray.map load_any loads))
  | Reduce { loads; _ } -> Operands (Iarray.to_list (Iarray.map load_any loads))
  | Scan { loads; carry; _ } ->
      let loads = Iarray.to_list (Iarray.map load_any loads) in
      Operands (loads @ Option.to_list carry)
  | Gather { idx; x; _ } -> Operands [ Any idx; Any x ]
  | Gather_ranges { starts; ends; x; _ } ->
      Operands [ Any starts; Any ends; Any x ]
  | Scatter { idx; updates; into; _ } ->
      Operands [ Any idx; Any updates; Any into ]
  | Sort { x; _ } -> Operands [ Any x ]
  | Unique { keys; _ } -> Operands [ Any keys ]
  | Fold { x; _ } -> Operands [ Any x ]
  | Fft f -> Operands [ fft_operand f ]
  | Linalg l -> Operands (linalg_operands l)
  | Assemble { pieces; _ } -> Operands (List.map (fun (_, x) -> Any x) pieces)
  | Contract { a; b; init; a_scale; b_scale; segments; _ } ->
      Operands (Any a :: Any b :: contract_rest init a_scale b_scale segments)
  | Copy x -> Operands [ Any x ]
  | Move (_, x) -> Operands [ Any x ]
  | Bitcast (_, x) -> Operands [ Any x ]
  | Place (_, x) -> Operands [ Any x ]
  | Check { ok; data; _ } -> Operands (Any ok :: data_any data)
  | Attention { q; kv; lengths; loads; _ } ->
      Operands (attention_operands q kv lengths loads)

(* Whether [r] covers all [d] elements of an axis in order. *)
let fills (r : Nx_array.Move.range) d =
  r.start = 0 && r.count = d && (r.step = 1 || r.count <= 1)

(* Whether [region] covers all of [shape] in order, as a kernel's in-place
   first piece of an assembly must. *)
let whole_region (region : Nx_array.Move.range iarray) shape =
  Iarray.length region = Iarray.length shape
  && Iarray.for_all2 fills region shape

(* Whether every piece of an assembly into [shape] fills its axis [a]: the
   axis then becomes the result's, a window of the result assembling the
   pieces' windows along it. *)
let filled shape pieces a =
  List.for_all
    (fun (region, _) -> fills (Iarray.get region a) (Iarray.get shape a))
    pieces

(* The positions are [operands]' above: a scatter's [into] after [idx] and
   [updates], a contraction's [init] after [a] and [b], an assembly's first
   piece first. *)
let own_index : type r. r prim -> (int list * L.t * D.any) option =
  function
  | Map { layout; outs = dt :: _; loads; _ } ->
      let plain = ref ([] : int list) in
      Iarray.iteri
        (fun i -> function
          | Plain _ -> plain := List.cons i !plain
          | Padded _ -> ())
        loads;
      Some (List.rev !plain, layout, D.Any dt)
  | Scatter { into; _ } ->
      Some ([ 2 ], L.contiguous (layout_shape into), D.Any (dtype into))
  | Contract { out; init = Some i; _ } ->
      Some ([ 2 ], L.contiguous (layout_shape i), D.Any out)
  | Assemble { dtype; shape; pieces = (region, _) :: _; _ }
    when whole_region region shape ->
      Some ([ 0 ], L.contiguous shape, D.Any dtype)
  | Map _ | Reduce _ | Scan _ | Gather _ | Gather_ranges _ | Contract _
  | Sort _ | Unique _ | Fft _ | Linalg _ | Fold _ | Assemble _ | Copy _
  | Move _ | Bitcast _ | Place _ | Check _ | Attention _ ->
      None

(* [l] with its operand mapped by [m i]. *)
let map_load (type d) (m : 'v 's 'e. int -> ('v, 's, 'e) t -> ('v, 's, 'e) t) i
    (l : d load) : d load =
  match l with
  | Plain x ->
      let y = m i x in
      if y == x then l else Plain y
  | Padded p ->
      let y = m i p.x in
      if y == p.x then l else Padded { p with x = y }

(* [loads] with load [j] mapped by [m j], from [i] on: [loads] itself where
   [m] changes none. *)
let rec map_loads : type d.
    ('v 's 'e. int -> ('v, 's, 'e) t -> ('v, 's, 'e) t) ->
    d load iarray ->
    int ->
    d load iarray =
 fun m loads i ->
  if i = Iarray.length loads then loads
  else
    let l = Iarray.get loads i in
    let l' = map_load m i l in
    if l' == l then map_loads m loads (i + 1)
    else
      Iarray.init (Iarray.length loads) (fun j ->
          if j < i then Iarray.get loads j
          else if j = i then l'
          else map_load m j (Iarray.get loads j))

(* [f e] folded over [loads] from the [i]th on. *)
let rec fold_loads : type e a d.
    ('v 's 'c. e -> a -> ('v, 's, 'c) t -> a) ->
    e ->
    a ->
    d load iarray ->
    int ->
    a =
 fun f e acc loads i ->
  if i = Iarray.length loads then acc
  else
    (* Matched in place: [load_any] would box each operand. *)
    match Iarray.get loads i with
    | Plain x -> fold_loads f e (f e acc x) loads (i + 1)
    | Padded { x; _ } -> fold_loads f e (f e acc x) loads (i + 1)

(* [f e] folded over [pieces]' values. *)
let rec fold_pieces : type e a r w q b.
    ('v 's 'c. e -> a -> ('v, 's, 'c) t -> a) ->
    e ->
    a ->
    (r * (w, q, b) t) list ->
    a =
 fun f e acc pieces ->
  match pieces with
  | [] -> acc
  | (_, x) :: rest -> fold_pieces f e (f e acc x) rest

(* [f e] over a scale's array, where there is a scale. *)
let fold_scale : type e a b.
    ('v 's 'd. e -> a -> ('v, 's, 'd) t -> a) -> e -> a -> b scale option -> a
    =
 fun f e acc -> function
  | None -> acc
  | Some (Exponent x) -> f e acc x
  | Some (Value x) -> f e acc x

(* [f e] folded over [op]'s operands in {!operands}' order. [e] carries what
   [f] needs, so that [f] can be closed: where it is, the fold allocates
   nothing for an operation other than a factorisation or a check. *)
let fold : type e a r.
    ('v 's 'd. e -> a -> ('v, 's, 'd) t -> a) -> e -> a -> r prim -> a =
 fun f e acc op ->
  match op with
  | Map { loads; _ } -> fold_loads f e acc loads 0
  | Reduce { loads; _ } -> fold_loads f e acc loads 0
  | Scan { loads; carry; _ } -> (
      let acc = fold_loads f e acc loads 0 in
      match carry with Some (Any c) -> f e acc c | None -> acc)
  | Gather { idx; x; _ } -> f e (f e acc idx) x
  | Gather_ranges { starts; ends; x; _ } -> f e (f e (f e acc starts) ends) x
  | Scatter { idx; updates; into; _ } -> f e (f e (f e acc idx) updates) into
  | Sort { x; _ } -> f e acc x
  | Unique { keys; _ } -> f e acc keys
  | Fold { x; _ } -> f e acc x
  (* Each kind's operand read in place: [fft_operand] would box it. *)
  | Fft (C2c { x; _ }) -> f e acc x
  | Fft (R2c { x; _ }) -> f e acc x
  | Fft (C2r { x; _ }) -> f e acc x
  | Linalg l ->
      List.fold_left (fun acc (Any x) -> f e acc x) acc (linalg_operands l)
  | Assemble { pieces; _ } -> fold_pieces f e acc pieces
  | Contract { a; b; init; a_scale; b_scale; segments; _ } -> (
      let acc = f e (f e acc a) b in
      let acc = match init with Some i -> f e acc i | None -> acc in
      let acc = fold_scale f e (fold_scale f e acc a_scale) b_scale in
      match segments with Some g -> f e acc (ids g) | None -> acc)
  | Copy x -> f e acc x
  | Move (_, x) -> f e acc x
  | Bitcast (_, x) -> f e acc x
  | Place (_, x) -> f e acc x
  | Check { ok; data; _ } ->
      List.fold_left
        (fun acc (Any x) -> f e acc x)
        (f e acc ok) (data_any data)
  | Attention { q; kv; lengths; loads; _ } ->
      let acc = f e acc q in
      let acc =
        match kv with
        | Dense { k; v } -> f e (f e acc k) v
        | Paged { k; v; table } -> f e (f e (f e acc k) v) table
      in
      let acc = match lengths with Some l -> f e acc l | None -> acc in
      Iarray.fold_left (fun acc (Any x) -> f e acc x) acc loads

(* An operand function as [fold]'s [e], unboxed so passing it allocates
   nothing. *)
type iter = { iter : 'v 's 'd. int -> ('v, 's, 'd) t -> unit } [@@unboxed]
type test = { test : 'v 's 'd. ('v, 's, 'd) t -> bool } [@@unboxed]

let iteri : type r. ('v 's 'd. int -> ('v, 's, 'd) t -> unit) -> r prim -> unit
    =
 fun f op ->
  let step e i x =
    e.iter i x;
    i + 1
  in
  ignore (fold step { iter = f } 0 op)

let exists : type r. ('v 's 'd. ('v, 's, 'd) t -> bool) -> r prim -> bool =
 fun f op -> fold (fun e found x -> found || e.test x) { test = f } false op

let lies_at p op = fold (fun p at x -> at && placement x == p) p true op

(* A contraction's optional operands, mapped by [m i] where present: the
   option itself where [m] returns its operand. *)
let present = function Some _ -> 1 | None -> 0

let map_some (type v s d)
    (m : 'v 's 'e. int -> ('v, 's, 'e) t -> ('v, 's, 'e) t) i
    (o : (v, s, d) t option) =
  match o with
  | None -> o
  | Some x ->
      let y = m i x in
      if y == x then o else Some y

let map_scale (type d)
    (m : 'v 's 'e. int -> ('v, 's, 'e) t -> ('v, 's, 'e) t) i
    (o : d scale option) =
  match o with
  | None -> o
  | Some (Exponent x) ->
      let y = m i x in
      if y == x then o else Some (Exponent y)
  | Some (Value x) ->
      let y = m i x in
      if y == x then o else Some (Value y)

let map_segments (type v s d)
    (m : 'v 's 'e. int -> ('v, 's, 'e) t -> ('v, 's, 'e) t) i
    (o : (v, s, d) t Nx_kernel.segments option) =
  match o with
  | None -> o
  | Some g -> (
      let x = ids g in
      let y = m i x in
      if y == x then o
      else
        match g with
        | Rows r -> Some (Rows { r with ids = y })
        | Contracting r -> Some (Contracting { r with ids = y }))

let mapi : type r.
    ('v 's 'd. int -> ('v, 's, 'd) t -> ('v, 's, 'd) t) -> r prim -> r prim =
 fun m op ->
  match op with
  | Map p ->
      let loads = map_loads m p.loads 0 in
      if loads == p.loads then op else Map { p with loads }
  | Reduce p ->
      let loads = map_loads m p.loads 0 in
      if loads == p.loads then op else Reduce { p with loads }
  | Scan p -> (
      let loads = map_loads m p.loads 0 in
      let n = Iarray.length loads in
      match p.carry with
      | Some (Any c) ->
          let c' = m n c in
          if loads == p.loads && c' == c then op
          else Scan { p with loads; carry = Some (Any c') }
      | None -> if loads == p.loads then op else Scan { p with loads })
  | Gather g ->
      let idx = m 0 g.idx in
      let x = m 1 g.x in
      if idx == g.idx && x == g.x then op else Gather { g with idx; x }
  | Gather_ranges g ->
      let starts = m 0 g.starts in
      let ends = m 1 g.ends in
      let x = m 2 g.x in
      if starts == g.starts && ends == g.ends && x == g.x then op
      else Gather_ranges { g with starts; ends; x }
  | Scatter s ->
      let idx = m 0 s.idx in
      let updates = m 1 s.updates in
      let into = m 2 s.into in
      if idx == s.idx && updates == s.updates && into == s.into then op
      else Scatter { s with idx; updates; into }
  | Sort s ->
      let x = m 0 s.x in
      if x == s.x then op else Sort { s with x }
  | Unique u ->
      let keys = m 0 u.keys in
      if keys == u.keys then op else Unique { u with keys }
  | Fold f ->
      let x = m 0 f.x in
      if x == f.x then op else Fold { f with x }
  | Fft f ->
      let f' = map_fft (fun x -> m 0 x) f in
      if f' == f then op else Fft f'
  | Linalg l ->
      let l' = map_linalg m l in
      if l' == l then op else Linalg l'
  | Assemble a ->
      let pieces =
        List.mapi
          (fun i ((r, x) as piece) ->
            let y = m i x in
            if y == x then piece else (r, y))
          a.pieces
      in
      if List.for_all2 ( == ) pieces a.pieces then op
      else Assemble { a with pieces }
  | Contract c ->
      (* The optional operands follow [a] and [b] in order, each at the next
         position where it is present. *)
      let a = m 0 c.a in
      let b = m 1 c.b in
      let init = map_some m 2 c.init in
      let i = 2 + present c.init in
      let a_scale = map_scale m i c.a_scale in
      let i = i + present c.a_scale in
      let b_scale = map_scale m i c.b_scale in
      let i = i + present c.b_scale in
      let segments = map_segments m i c.segments in
      if
        a == c.a && b == c.b && init == c.init && a_scale == c.a_scale
        && b_scale == c.b_scale && segments == c.segments
      then op
      else Contract { c with a; b; init; a_scale; b_scale; segments }
  | Copy x ->
      let y = m 0 x in
      if y == x then op else Copy y
  | Move (mv, x) ->
      let y = m 0 x in
      if y == x then op else Move (mv, y)
  | Bitcast (dt, x) ->
      let y = m 0 x in
      if y == x then op else Bitcast (dt, y)
  | Place (p, x) ->
      let y = m 0 x in
      if y == x then op else Place (p, y)
  | Check c ->
      let ok = m 0 c.ok in
      let data = map_data m 1 c.data in
      if ok == c.ok && data == c.data then op else Check { c with ok; data }
  | Attention a ->
      (* The table, the lengths and the loads follow [q], [k] and [v], each
         at the next position where it is present. *)
      let q = m 0 a.q in
      let kv, i =
        match a.kv with
        | Dense d ->
            let k = m 1 d.k in
            let v = m 2 d.v in
            ((if k == d.k && v == d.v then a.kv else Dense { k; v }), 3)
        | Paged d ->
            let k = m 1 d.k in
            let v = m 2 d.v in
            let table = m 3 d.table in
            ( (if k == d.k && v == d.v && table == d.table then a.kv
               else Paged { k; v; table }),
              4 )
      in
      let lengths = map_some m i a.lengths in
      let i = i + present a.lengths in
      let changed = ref false in
      let loads =
        Iarray.mapi
          (fun j (Any x as l) ->
            let y = m (i + j) x in
            if y == x then l
            else begin
              changed := true;
              Any y
            end)
          a.loads
      in
      let loads = if !changed then loads else a.loads in
      if q == a.q && kv == a.kv && lengths == a.lengths && loads == a.loads
      then op
      else Attention { a with q; kv; lengths; loads }

let map : type r.
    ('v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t) -> r prim -> r prim =
 fun m op -> mapi (fun _ x -> m x) op

(* Kinds *)

(* Each kind's code among the nodes of its arity, and its name in
   messages and printed programs. The codes of an arity are dense from 0,
   [Copy], [Cast] and [Bitcast] taking one operand's first three: they
   index the one-node programs kept below. *)
let unary_kind : P.unary -> int * string = function
  | Neg -> (3, "neg")
  | Recip -> (4, "recip")
  | Abs -> (5, "abs")
  | Sign -> (6, "sign")
  | Sqrt -> (7, "sqrt")
  | Exp -> (8, "exp")
  | Exp2 -> (9, "exp2")
  | Log -> (10, "log")
  | Log2 -> (11, "log2")
  | Log1p -> (12, "log1p")
  | Expm1 -> (13, "expm1")
  | Sin -> (14, "sin")
  | Cos -> (15, "cos")
  | Tan -> (16, "tan")
  | Asin -> (17, "asin")
  | Acos -> (18, "acos")
  | Atan -> (19, "atan")
  | Sinh -> (20, "sinh")
  | Cosh -> (21, "cosh")
  | Tanh -> (22, "tanh")
  | Asinh -> (23, "asinh")
  | Acosh -> (24, "acosh")
  | Atanh -> (25, "atanh")
  | Erf -> (26, "erf")
  | Floor -> (27, "floor")
  | Ceil -> (28, "ceil")
  | Round -> (29, "round")
  | Trunc -> (30, "trunc")

let binary_kind : P.binary -> int * string = function
  | Add -> (0, "add")
  | Sub -> (1, "sub")
  | Mul -> (2, "mul")
  | Fdiv -> (3, "fdiv")
  | Idiv -> (4, "idiv")
  | Mod -> (5, "mod")
  | Pow -> (6, "pow")
  | Atan2 -> (7, "atan2")
  | Maximum -> (8, "maximum")
  | Minimum -> (9, "minimum")
  | And -> (10, "and")
  | Or -> (11, "or")
  | Xor -> (12, "xor")
  | Threefry -> (13, "threefry")

let compare_kind : P.compare -> int * string = function
  | Equal -> (14, "equal")
  | Not_equal -> (15, "not_equal")
  | Less -> (16, "less")
  | Less_equal -> (17, "less_equal")

let kind : P.node -> string = function
  | In _ -> "In"
  | Coord _ -> "Iota"
  | Const _ -> "Fill"
  | Op1 (Copy, _, _) -> "Copy"
  | Op1 (Unary k, _, _) -> String.capitalize_ascii (snd (unary_kind k))
  | Op1 (Cast, _, _) -> "Cast"
  | Op1 (Bitcast, _, _) -> "Bitcast"
  | Op2 (Binary k, _, _) -> String.capitalize_ascii (snd (binary_kind k))
  | Op2 (Compare k, _, _) -> String.capitalize_ascii (snd (compare_kind k))
  | Op3 (Where, _, _, _) -> "Where"
  | Op3 (Fma, _, _, _) -> "Fma"

(* Printing *)

let any_name (D.Any dt) = D.name dt

let hex bits =
  String.concat ""
    (List.map
       (fun c -> Printf.sprintf "%02x" (Char.code c))
       (List.of_seq (String.to_seq bits)))

(* A program's expressions over the operands [x0], [x1], …, of the nodes
   [roots] reads. A node other than a leaf read more than once, by nodes or by
   [roots], is [nK] where read, defined once as [nK = …; ] by [bindings]: the
   text is linear in the program. *)
type printer = {
  bindings : Format.formatter -> unit -> unit;
  expr : Format.formatter -> int -> unit;
}

let printer (p : P.t) roots =
  let reads = Array.make (Iarray.length p.nodes) 0 in
  let read j = reads.(j) <- reads.(j) + 1 in
  for i = 0 to Iarray.length p.nodes - 1 do
    match Iarray.get p.nodes i with
    | In _ | Coord _ | Const _ -> ()
    | Op1 (_, _, j) -> read j
    | Op2 (_, j, l) ->
        read j;
        read l
    | Op3 (_, j, l, n) ->
        read j;
        read l;
        read n
  done;
  List.iter read roots;
  let shared i =
    reads.(i) > 1
    && match Iarray.get p.nodes i with
       | In _ | Coord _ | Const _ -> false
       | _ -> true
  in
  let rec expr ppf i =
    if shared i then Format.fprintf ppf "n%d" i else body ppf i
  and body ppf i =
    match Iarray.get p.nodes i with
    | In k -> Format.fprintf ppf "x%d" k
    | Coord a -> Format.fprintf ppf "coord%d" a
    | Const (dt, bits) -> Format.fprintf ppf "%s:0x%s" (any_name dt) (hex bits)
    | Op1 (Copy, _, j) -> Format.fprintf ppf "copy(%a)" expr j
    | Op1 (Unary k, _, j) ->
        Format.fprintf ppf "%s(%a)" (snd (unary_kind k)) expr j
    | Op1 (Cast, dt, j) -> Format.fprintf ppf "cast_%s(%a)" (any_name dt) expr j
    | Op1 (Bitcast, dt, j) ->
        Format.fprintf ppf "bitcast_%s(%a)" (any_name dt) expr j
    | Op2 (Binary k, j, l) ->
        Format.fprintf ppf "%s(%a, %a)" (snd (binary_kind k)) expr j expr l
    | Op2 (Compare k, j, l) ->
        Format.fprintf ppf "%s(%a, %a)" (snd (compare_kind k)) expr j expr l
    | Op3 (Where, j, l, n) ->
        Format.fprintf ppf "where(%a, %a, %a)" expr j expr l expr n
    | Op3 (Fma, j, l, n) ->
        Format.fprintf ppf "fma(%a, %a, %a)" expr j expr l expr n
  in
  let bindings ppf () =
    for i = 0 to Iarray.length p.nodes - 1 do
      if shared i then Format.fprintf ppf " n%d = %a;" i body i
    done
  in
  { bindings; expr }

let pp_operand ppf (Any x) =
  Format.fprintf ppf "%s %a at %a"
    (D.name (dtype x))
    pp_shape (shape x) Devices.pp_placement (placement x)

let reduction : type d a.
    (d, a) reduction -> Nx_kernel.reduction * int * D.any = function
  | Monoid (m, k, dt) -> (Monoid m, k, D.Any dt)
  | Logsumexp (k, dt) -> (Logsumexp, k, D.Any dt)
  | Moments (k, dt) -> (Moments, k, D.Any dt)
  | Arg (e, k, dt) -> (Arg e, k, D.Any dt)

let reduction_name : Nx_kernel.reduction -> string = function
  | Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Logsumexp -> "Logsumexp"
  | Moments -> "Moments"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"

let rec reductions_list : type d r.
    (d, r) reductions -> (Nx_kernel.reduction * int * D.any) list = function
  | [] -> []
  | r :: rest -> reduction r :: reductions_list rest

let reductions rs = Array.of_list (reductions_list rs)

(* What [r] computes, a constant: it allocates nothing. *)
let reduction_kind : type d a. (d, a) reduction -> Nx_kernel.reduction =
  function
  | Monoid (Sum, _, _) -> Monoid Sum
  | Monoid (Prod, _, _) -> Monoid Prod
  | Monoid (Max, _, _) -> Monoid Max
  | Monoid (Min, _, _) -> Monoid Min
  | Logsumexp _ -> Logsumexp
  | Moments _ -> Moments
  | Arg (Max, _, _) -> Arg Max
  | Arg (Min, _, _) -> Arg Min

(* The program output [r] reduces. *)
let output : type d a. (d, a) reduction -> int = function
  | Monoid (_, k, _) | Logsumexp (k, _) | Moments (k, _) | Arg (_, k, _) -> k

let rec count_reductions : type d r. (d, r) reductions -> int = function
  | [] -> 0
  | _ :: rest -> 1 + count_reductions rest

(* Each of [rs]' kinds into [a], from [i] on. *)
let rec fill_kinds : type d r.
    Nx_kernel.reduction array -> int -> (d, r) reductions -> unit =
 fun a i rs ->
  match rs with
  | [] -> ()
  | r :: rest ->
      a.(i) <- reduction_kind r;
      fill_kinds a (i + 1) rest

let kernel_reduction r = (reduction_kind r, output r)

(* Each of [rs] as a kernel takes it into [a], from [i] on. *)
let rec fill_kernel : type d r.
    (Nx_kernel.reduction * int) array -> int -> (d, r) reductions -> unit =
 fun a i rs ->
  match rs with
  | [] -> ()
  | r :: rest ->
      a.(i) <- kernel_reduction r;
      fill_kernel a (i + 1) rest

(* [rs]' kinds, in order. *)
let kinds rs =
  let a = Array.make (count_reductions rs) Nx_kernel.Moments in
  fill_kinds a 0 rs;
  a

let kernel_reductions rs =
  let a = Array.make (count_reductions rs) (Nx_kernel.Moments, 0) in
  fill_kernel a 0 rs;
  a

let pp_reduced (prog : P.t) (pr : printer) ppf (r, k, _) =
  Format.fprintf ppf "%s %a" (reduction_name r) pr.expr
    (Iarray.get prog.outs k)

let reduced_roots (prog : P.t) rs =
  List.map (fun (_, k, _) -> Iarray.get prog.outs k) rs

let pp : type r. Format.formatter -> r prim -> unit =
 fun ppf op ->
  let (Operands xs) = operands op in
  Format.fprintf ppf "%s" (name op);
  let sep ppf () = Format.pp_print_string ppf "; " in
  (match op with
  | Map { prog; at; _ } ->
      let outs = Iarray.to_list prog.outs in
      let pr = printer prog outs in
      Format.fprintf ppf "%a [%a]" pr.bindings ()
        (Format.pp_print_list ~pp_sep:sep pr.expr)
        outs;
      Option.iter (Format.fprintf ppf " at %a" Devices.pp_placement) at
  | Reduce { prog; axes; reductions; _ } ->
      let rs = reductions_list reductions in
      let pr = printer prog (reduced_roots prog rs) in
      Format.fprintf ppf "%a [%a] over %a" pr.bindings ()
        (Format.pp_print_list ~pp_sep:sep (pp_reduced prog pr))
        rs pp_ints axes
  | Scan { prog; axis; reduction = s; _ } ->
      let r = reduction s in
      let pr = printer prog (reduced_roots prog [ r ]) in
      Format.fprintf ppf "%a [%a] along %d" pr.bindings () (pp_reduced prog pr)
        r axis
  | Gather { axis; _ } -> Format.fprintf ppf " along %d" axis
  | Gather_ranges { size; starts; _ } ->
      Format.fprintf ppf " of %d along %d" size (rank starts - 1)
  | Scatter { combine; unique; axis; _ } ->
      let combine =
        match (combine : Nx_kernel.combine) with
        | Set -> "set"
        | Add -> "add"
        | Max -> "max"
        | Min -> "min"
      in
      Format.fprintf ppf " %s%s along %d" combine
        (if unique then " unique" else "")
        axis
  | Sort { axis; descending; k; _ } ->
      Format.fprintf ppf " along %d%s%s" axis
        (if descending then " descending" else "")
        (match k with Some k -> Printf.sprintf " keeping %d" k | None -> "")
  | Unique { size; _ } -> Format.fprintf ppf " keeping %d" size
  | Fft f ->
      let kind =
        match f with
        | C2c { direction = Forward; _ } -> "C2c forward"
        | C2c { direction = Inverse; _ } -> "C2c inverse"
        | R2c _ -> "R2c"
        | C2r { n; _ } -> Printf.sprintf "C2r of %d points" n
      in
      Format.fprintf ppf " %s along %a" kind pp_ints (fft_axes f)
  | Linalg l -> Format.fprintf ppf " %s" (routine_name l)
  | Fold { shape; _ } -> Format.fprintf ppf " into %a" pp_ints shape
  | Assemble { shape; _ } -> Format.fprintf ppf " into %a" pp_ints shape
  | Attention { prog; scale; _ } ->
      let outs = Iarray.to_list prog.outs in
      let pr = printer prog outs in
      Format.fprintf ppf "%a [%a] at scale %h" pr.bindings ()
        (Format.pp_print_list ~pp_sep:sep pr.expr)
        outs scale
  | Contract _ | Copy _ | Move _ | Bitcast _ | Place _ | Check _ -> ());
  List.iteri (fun i x -> Format.fprintf ppf " (x%d: %a)" i pp_operand x) xs

(* Rules *)

(* [r]'s value, or its error raised naming [by]. *)
let shaped ~by = function Ok s -> s | Error e -> invalid_argf "%s: %s" by e

(* The shape [l] gives its loop, or its refusal raised naming [by]: a padded
   load's fill is an element of its dtype, and its pad fits its operand. *)
let load_shape (type d) ~by (l : d load) =
  match l with
  | Plain x -> shape x
  | Padded { x; fill; pad } ->
      check_value ~by (dtype x) fill;
      shaped ~by (Shape.padded pad (shape x))

(* Whether [x]'s shape is [l]'s, allocating nothing. *)
let has_layout_shape x l =
  let r = L.rank l in
  rank x = r
  &&
  let i = ref 0 in
  while !i < r && dim x !i = L.dim l !i do
    incr i
  done;
  !i = r

(* A loop's arrays as Nx_kernel's rules read them: its loads in place, a
   map's results and a reduction's as arrays of their dtypes, and its
   reductions as the kernels take them. *)
let loads_view =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype =
      (fun a i ->
        match Iarray.get a i with
        | Plain x -> D.Any (dtype x)
        | Padded { x; _ } -> D.Any (dtype x));
  }

let dtype_array = { Nx_kernel.arrays = Array.length; dtype = Array.get }

let reductions_view =
  {
    Nx_kernel.reductions = Array.length;
    kind = (fun (rs : (Nx_kernel.reduction * int) array) i -> fst rs.(i));
    output = (fun (rs : (Nx_kernel.reduction * int) array) i -> snd rs.(i));
  }

let rec outs_count : type d r. (d, r) outs -> int = function
  | [] -> 0
  | _ :: rest -> 1 + outs_count rest

let rec fill_outs : type d r. D.any array -> int -> (d, r) outs -> unit =
 fun a i -> function
  | [] -> ()
  | dt :: rest ->
      a.(i) <- D.Any dt;
      fill_outs a (i + 1) rest

(* The dtypes of [outs], in order. *)
let outs_dtypes outs =
  let a = Array.make (outs_count outs) (D.Any D.Bool) in
  fill_outs a 0 outs;
  a

(* The dtype of result [i] of [r]: an Arg's position, its second, is an
   int64. *)
let result_dtype : type d a. (d, a) reduction -> int -> D.any =
 fun r i ->
  match r with
  | Monoid (_, _, dt) -> D.Any dt
  | Logsumexp (_, dt) -> D.Any dt
  | Moments (_, dt) -> D.Any dt
  | Arg (_, _, dt) -> if i = 0 then D.Any dt else D.Any D.Int64

let width : type d a. (d, a) reduction -> int = function
  | Monoid _ | Logsumexp _ -> 1
  | Moments _ | Arg _ -> 2

let reduction_view = { Nx_kernel.arrays = width; dtype = result_dtype }

let rec results_count : type d r. (d, r) reductions -> int = function
  | [] -> 0
  | r :: rest -> width r + results_count rest

let rec fill_results : type d r. D.any array -> int -> (d, r) reductions -> unit
    =
 fun a i -> function
  | [] -> ()
  | r :: rest ->
      for j = 0 to width r - 1 do
        a.(i + j) <- result_dtype r j
      done;
      fill_results a (i + width r) rest

(* The dtypes of [rs]' results, in order. *)
let results_dtypes rs =
  let a = Array.make (results_count rs) (D.Any D.Bool) in
  fill_results a 0 rs;
  a

(* Raises naming [by] for a refusal of a rule of Nx_kernel. *)
let refused_by ~by = function
  | Some (_, why) -> invalid_argf "%s: %s" by why
  | None -> ()

(* A loop's rule on its layout and its loads' shapes: [layout] is
   C-contiguous, and each load has its shape. [what] names the loop. *)
let check_loads (type d) ~by ~what layout (loads : d load iarray) =
  if not (L.is_contiguous layout && L.offset layout = 0) then
    invalid_argf "%s: a %s's layout %a is not C-contiguous" by what L.pp layout;
  for i = 0 to Iarray.length loads - 1 do
    let l = Iarray.get loads i in
    let fits =
      match l with
      | Plain x -> has_layout_shape x layout
      | Padded _ -> load_shape ~by l = Iarray.to_array (L.shape layout)
    in
    if not fits then
      invalid_argf "%s: load %d has shape %a, the %s %a" by i pp_shape
        (load_shape ~by l) what pp_ints (L.shape layout)
  done

(* A map's rule; its result's layout, C-contiguous of [shape]. A map of no
   loads names where it lies, a placement that cuts no axis its shape lacks,
   and a map with loads lies where they do. *)
let check_map (type d r) ~by layout (prog : P.t) (outs : (d, r) outs)
    (loads : d load iarray) (at : d Devices.placement option) =
  refused_by ~by
    (Nx_kernel.map_refusal prog ~rank:(L.rank layout) ~dsts:dtype_array
       (outs_dtypes outs) ~loads:loads_view loads);
  check_loads ~by ~what:"map" layout loads;
  match at with
  | None when Iarray.length loads = 0 ->
      invalid_argf "%s: a map of no loads names no placement" by
  | Some _ when Iarray.length loads > 0 ->
      invalid_argf "%s: a map of %d loads names a placement" by
        (Iarray.length loads)
  | Some p ->
      ignore (Devices.window ~by p (Iarray.to_array (L.shape layout)) 0)
  | None -> ()

(* The shape of the results of [rs] along [axes] of a loop of [layout]: its
   rule. *)
let reduced_shape ~by layout axes rs =
  shaped ~by (Shape.reduce ~axes (kinds rs) (Iarray.to_array (L.shape layout)))

(* A reduction's rule; its results' shape. *)
let check_reduce (type d r) ~by layout axes prog (rs : (d, r) reductions)
    (loads : d load iarray) =
  refused_by ~by
    (Nx_kernel.reduce_refusal prog ~rank:(L.rank layout)
       ~reductions:reductions_view (kernel_reductions rs) ~dsts:dtype_array
       (results_dtypes rs) ~loads:loads_view loads);
  check_loads ~by ~what:"reduction" layout loads;
  reduced_shape ~by layout axes rs

(* A scan's rule, {!Nx_kernel.scan_refusal}'s, then its loads' shapes and
   its carry's, which {!Nx_kernel.Shape.carry} gives. *)
let check_scan (type d r) ~by layout axis prog (r : (d, r) reduction)
    (loads : d load iarray) (carry : d any option) =
  refused_by ~by
    (Nx_kernel.scan_refusal prog ~rank:(L.rank layout) (reduction_kind r)
       ~output:(output r) ~dsts:reduction_view r
       ?carry:(Option.map (fun (Any c) -> D.Any (dtype c)) carry)
       ~loads:loads_view loads);
  check_loads ~by ~what:"scan" layout loads;
  let y = shaped ~by (Shape.scan ~axis (Iarray.to_array (L.shape layout))) in
  match carry with
  | None -> ()
  | Some (Any c) ->
      if shape c <> shaped ~by (Shape.carry ~axis y) then
        invalid_argf "%s: a carry of shape %a for a scan of shape %a" by
          pp_shape (shape c) pp_shape y

(* [shape] without [axes]. *)
let reduced shape axes =
  Array.of_list
    (List.filteri (fun a _ -> not (Iarray.mem a axes)) (Array.to_list shape))

(* Each operation's rule ({!rule}) and route: where it reads its operands, in
   [operands]'s order, and where its results lie. [route] and [results] take
   placements from it alone. *)

(* Marks in [whole] the axes of a loop of rank [Array.length whole] that a
   padded load by [p] reads whole: those it pads, crops or windows, and those
   its windows append. *)
let mark_padded whole (p : Nx_kernel.pad) =
  let r = Iarray.length p.lo in
  let padded a =
    Iarray.get p.lo a <> 0 || Iarray.get p.hi a <> 0
    || Iarray.get p.interior a <> 0
  in
  Array.iteri (fun a _ -> if a >= r || padded a then whole.(a) <- true) whole;
  Iarray.iter
    (fun (w : Nx_array.Move.window) -> whole.(w.axis) <- true)
    p.windows

let marked whole =
  Iarray.of_list
    (List.filter (fun a -> whole.(a)) (List.init (Array.length whole) Fun.id))

(* The axes of a loop of [layout] that its loads read whole, beside [axes]. *)
let padded_axes (type d) layout ?(axes : int iarray = [||])
    (loads : d load iarray) =
  let whole = Array.make (L.rank layout) false in
  Iarray.iter (fun a -> whole.(a) <- true) axes;
  Iarray.iter
    (fun (l : d load) ->
      match l with Plain _ -> () | Padded { pad; _ } -> mark_padded whole pad)
    loads;
  marked whole

(* An operand of rank [r] whose axis becomes the results' axis of its index, but
   those of [whole], read whole. *)
let reading whole r =
  Array.init r (fun a ->
      if Iarray.mem a whole then Route.Whole else Route.Becomes a)

(* A loop's operands' ranks: its rank, but a padded load's own. A loop of no
   load has the one row of its rank, so that on a split placement each device
   computes its window as over a load of the loop's shape. *)
let load_ranks (type d) layout (loads : d load iarray) =
  if Iarray.length loads = 0 then [| L.rank layout |]
  else
    Array.init (Iarray.length loads) (fun i ->
        match Iarray.get loads i with
        | Plain _ -> L.rank layout
        | Padded { x; _ } -> rank x)

(* Whether a scan by [r] continues from the terms before a window: one of a
   [Sum], [Prod], [Max], [Min] or [Arg], whose kernels take a carry or whose
   results combine exactly. *)
let carries : type d a. (d, a) reduction -> bool = function
  | Monoid ((Sum | Prod | Max | Min), _, _) | Arg _ -> true
  | Logsumexp _ | Moments _ -> false

(* A loop's rule over loads of [ranks]: a load's axis becomes the results' axis
   of its index less the axes reduced before it, but those read whole, those
   its padded loads pad, crop or window and a scan's axis that does not carry
   or has a carry already, and those reduced. A scan's axis that carries is
   [Scanned] where every load reads it plain. *)
let loop_rule : type r. r prim -> int array -> Route.rule =
 fun op ranks ->
  let rows ?scanned axes whole =
    let becomes a : Route.axis =
      if Iarray.mem a whole then Whole
      else if Iarray.mem a axes then Reduced a
      else if scanned = Some a then Scanned a
      else
        Becomes
          (a - Iarray.fold_left (fun n r -> if r < a then n + 1 else n) 0 axes)
    in
    Route.Axes (Array.map (fun r -> Array.init r becomes) ranks)
  in
  let padded (type d) layout (loads : d load iarray) axes =
    if Iarray.exists is_padded loads then padded_axes layout ~axes loads
    else axes
  in
  match op with
  | Map { layout; loads; _ } -> rows [||] (padded layout loads [||])
  | Reduce { layout; axes; loads; _ } -> rows axes (padded layout loads [||])
  | Scan { layout; axis; loads; reduction; carry; _ } ->
      let plain = not (Iarray.exists is_padded loads) in
      (* A loop of no load has no operand to cut: it scans whole. *)
      let loaded = Iarray.length loads > 0 in
      if carries reduction && Option.is_none carry && plain && loaded then
        rows ~scanned:axis [||] [||]
      else rows [||] (padded layout loads [| axis |])
  | Gather _ | Gather_ranges _ | Scatter _ | Sort _ | Unique _ | Fold _
  | Fft _ | Linalg _ | Assemble _ | Contract _ | Copy _ | Move _ | Bitcast _
  | Place _ | Check _ | Attention _ ->
      invalid_arg "Prim.loop_rule: not a loop"

(* A fold's rule: the axes of its operand that the padded load it sums back
   reads whole. *)
let fold_rule pad x =
  let whole = Array.make (rank x) false in
  mark_padded whole pad;
  Route.Axes [| reading (marked whole) (rank x) |]

(* A routine acts along its matrices' last two axes. *)
let linalg_rule l =
  let ranks = List.map (fun (Any x) -> rank x) (linalg_operands l) in
  let r = List.hd ranks in
  Route.Axes (Array.of_list (List.map (reading [| r - 2; r - 1 |]) ranks))

(* A contraction's result has the batch pairs' axes, then [a]'s other axes, then
   [b]'s; [init] has its shape. A contracting pair's two axes share its index as
   their factor. [Rows]' axis of [b]'s groups is the factor after the pairs,
   reduced in [b] alone, [Σ_e [r in group e] a_r · b_e]; its ids follow [a]'s
   segmented axis, so that a cut of the rows cuts them with it, and {!Partial}
   numbers a tile's groups from its first. The result's leading axis of
   [Contracting]'s groups is read whole, and a [Contracting] contraction is
   read whole along its contracting pairs, its ids along their segmented
   axis. *)
let contract_rule ~batch ~contracting ~segments sa sb init sca scb =
  let nb = Iarray.length batch and nc = Iarray.length contracting in
  let lead, groups =
    match (segments : _ Nx_kernel.segments option) with
    | Some (Rows { b_axis; _ }) -> (0, b_axis)
    | Some (Contracting _) -> (1, -1)
    | None -> (0, -1)
  in
  let reduced =
    match segments with
    | Some (Contracting _) -> false
    | Some (Rows _) | None -> true
  in
  let operand pick r first segmented =
    let next = ref first in
    Array.init r (fun x ->
        match Iarray.find_index (fun p -> pick p = x) batch with
        | Some j -> Route.Becomes (lead + j)
        | None -> (
            match Iarray.find_index (fun p -> pick p = x) contracting with
            | Some p -> if reduced then Route.Reduced p else Whole
            | None -> segmented next x))
  in
  let free_a next _ =
    incr next;
    Route.Becomes (!next - 1)
  in
  let free_b next x =
    if x = groups then Route.Reduced nc
    else begin
      incr next;
      Becomes (!next - 1)
    end
  in
  let ra = Array.length sa in
  let a = operand fst ra (lead + nb) free_a in
  let b = operand snd (Array.length sb) (lead + ra - nc) free_b in
  (* A scale's axis reads as the operand's where its extent is the
     operand's. Where a block holds more elements, one that becomes the
     result's regroups into it, and one of a factor, a contracting pair's or
     [b]'s groups', holds blocks of it, so that a cut that keeps whole blocks
     is read where it lies and partials take the blocks each part of a tile
     spans ({!Partial}). One of one block, and every other, is read
     whole. *)
  let scaled row s ss =
    Array.mapi
      (fun d e ->
        match (row.(d) : Route.axis) with
        | (Becomes _ | Reduced _) when e = s.(d) -> row.(d)
        | (Becomes _ | Reduced _) when e <= 1 -> Whole
        | Becomes j -> Regroups { axis = j; extent = s.(d) }
        | Reduced p -> Blocks { factor = p; extent = s.(d) }
        | Whole | Blocks _ | Scanned _ | Indexed _ | Regroups _ -> Whole)
      ss
  in
  let some f o = match o with Some x -> [| f x |] | None -> [||] in
  let reads_init r = Array.init r (fun i -> Route.Becomes i) in
  let ids (g : _ Nx_kernel.segments) =
    let along =
      match g with
      | Rows { a_axis; _ } -> a.(a_axis)
      | Contracting { pair; _ } -> a.(fst (Iarray.get contracting pair))
    in
    Array.init (nb + 1) (fun i ->
        if i < nb then Route.Becomes (lead + i) else along)
  in
  let rows : Route.axis array array list =
    [
      [| a; b |];
      some reads_init init;
      some (scaled a sa) sca;
      some (scaled b sb) scb;
      some ids segments;
    ]
  in
  Route.Axes (Array.concat rows)

let move_rule m x =
  let s = shape x in
  Route.Axes [| Array.init (Array.length s) (Route.moved m s) |]

(* A bitcast to a wider dtype reads its operand's last axis whole, which it
   removes; one to a narrower dtype keeps each axis and appends one that no
   axis of its operand becomes. *)
let bitcast_rule dt x =
  let bits = D.bits dt and from = D.bits (dtype x) in
  if bits > from then Route.Axes [| reading [| rank x - 1 |] (rank x) |]
  else if bits < from then Route.Axes [| reading [||] (rank x) |]
  else Route.Elementwise

(* Attention's results [[lead…; hq; tq; dv]] and [[lead…; hq; tq]] take
   [q]'s lead axes, its heads and its rows; [k] and [v], dense or pools,
   give their key heads, which become [q]'s where they are as many and
   regroup into them in tiles otherwise, a query head reading the key head
   of its group, and dense ones their lead axes; a page table and the
   lengths their lead axes; a load its lead axes, heads and rows. The keys,
   a page table's pages and a dense call's loads' keys are the one reduced
   factor, which {!Partial} closes. Every other axis, the dimensions, a
   pool's pages and slots and a paged call's loads' keys, is read whole. *)
let attention_rule (type v s d) (q : (v, s, d) t) (kv : (v, s, d) kv)
    (lengths : d index option) (loads : d any iarray) =
  let r = rank q in
  let lead = r - 3 in
  let hq = dim q lead in
  let row n f = Array.init n (fun a -> if a < lead then Route.Becomes a else f a)
  in
  let whole _ = Route.Whole in
  let head hk =
    if hk = hq then Route.Becomes lead
    else if hk > 1 && hq > 1 then Regroups { axis = lead; extent = hq }
    else Whole
  in
  let heads x =
    row r (fun a ->
        if a = r - 2 then Route.Reduced 0
        else if a <> lead then Whole
        else head (dim x lead))
  in
  let paged = match kv with Dense _ -> false | Paged _ -> true in
  let pairs _ =
    row r (fun a ->
        if a < r - 1 then Route.Becomes a
        else if paged then Whole
        else Reduced 0)
  in
  let kv =
    match kv with
    | Dense { k; v } -> [| heads k; heads v |]
    | Paged { k; v; _ } ->
        (* A pool [[pages; hk; page; d]] holds its key heads on axis 1. *)
        let pool x =
          Array.init 4 (fun a -> if a = 1 then head (dim x 1) else Route.Whole)
        in
        [| pool k; pool v; row (lead + 1) (fun _ -> Route.Reduced 0) |]
  in
  let lengths =
    match lengths with Some _ -> [| row lead whole |] | None -> [||]
  in
  Route.Axes
    (Array.concat
       [
         [| row r (fun a -> if a < r - 1 then Route.Becomes a else Whole) |];
         kv;
         lengths;
         Array.map pairs (Iarray.to_array loads);
       ])

let rule : type r. r prim -> Route.rule =
 fun op ->
  match op with
  | Map { loads; _ } when not (Iarray.exists is_padded loads) -> Elementwise
  | Map { layout; loads; _ } -> loop_rule op (load_ranks layout loads)
  | Reduce { layout; loads; _ } -> loop_rule op (load_ranks layout loads)
  | Scan { layout; loads; carry = None; _ } ->
      loop_rule op (load_ranks layout loads)
  | Scan { layout; axis; loads; carry = Some _; _ } -> (
      (* The carry, of the loop's rank, reads the scan's axis whole and the
         others as a plain load does. *)
      let rank = L.rank layout in
      let ranks = Array.append (load_ranks layout loads) [| rank |] in
      match loop_rule op ranks with
      | Axes t ->
          t.(Iarray.length loads).(axis) <- Route.Whole;
          Axes t
      | Elementwise -> invalid_arg "Prim.rule: a loop's rule")
  | Gather { axis; idx; x } ->
      (* [x]'s axis is gone from the results, which read it at positions:
         cut, each device gathers from its tile ({!Partial}), whose results
         stack along a new leading axis, a [complex128]'s bits along a new
         last one too. Where those pass the greatest rank, it is whole. *)
      let pairs = match dtype x with D.Complex128 -> 1 | _ -> 0 in
      let stacks = rank x + 1 + pairs <= L.max_rank in
      let source a =
        if a <> axis then Route.Becomes a
        else if stacks then Reduced axis
        else Whole
      in
      Axes [| reading [||] (rank idx); Array.init (rank x) source |]
  | Gather_ranges { starts; x; _ } ->
      (* The runs are read whole along their axis [a], and [x] as a
         gather's: cut along [a], each device gathers the runs' elements in
         its tile ({!Partial}), else whole past the greatest rank. Lead and
         cell axes become the results'; the results' axis [a] is whole. *)
      let a = rank starts - 1 in
      let pairs = match dtype x with D.Complex128 -> 1 | _ -> 0 in
      let stacks = rank x + 1 + pairs <= L.max_rank in
      let runs =
        Array.init (a + 1) (fun i ->
            if i < a then Route.Becomes i else Route.Whole)
      in
      let source i =
        if i <> a then Route.Becomes i else if stacks then Reduced a else Whole
      in
      Axes [| runs; Array.copy runs; Array.init (rank x) source |]
  | Scatter { axis; idx; updates; into; _ } ->
      (* [into]'s axis is the results', which take the updates at
         positions: cut, each device writes those in its window
         ({!Partial}). Positions and updates cut along it close each
         device's partial results by the combine ({!Partial}), whose
         stack, of one more axis than [into], passes no greatest rank;
         at that rank they are read whole along it. *)
      let summed = rank into < L.max_rank in
      let along = if summed then Route.Reduced axis else Whole in
      let read r =
        Array.init r (fun a -> if a = axis then along else Route.Becomes a)
      in
      let target a = if a = axis then Route.Indexed axis else Becomes a in
      Axes
        [|
          read (rank idx); read (rank updates); Array.init (rank into) target;
        |]
  | Sort { axis; x; _ } -> Axes [| reading [| axis |] (rank x) |]
  | Unique { keys; _ } -> Axes [| reading [| 0; 1 |] (rank keys) |]
  | Fold { pad; x; _ } -> fold_rule pad x
  | Fft f ->
      let (Any x) = fft_operand f in
      Axes [| reading (fft_axes f) (rank x) |]
  | Linalg l -> linalg_rule l
  | Assemble { shape; pieces; _ } ->
      let axis a = if filled shape pieces a then Route.Becomes a else Whole in
      Axes
        (Array.of_list
           (List.map (fun (_, x) -> Array.init (rank x) axis) pieces))
  | Contract
      { batch; contracting; a; b; init; a_scale; b_scale; segments; _ } ->
      contract_rule ~batch ~contracting ~segments (shape a) (shape b)
        (Option.map rank init)
        (Option.map scale_shape a_scale)
        (Option.map scale_shape b_scale)
  | Copy _ -> Elementwise
  | Move (m, x) -> move_rule m x
  | Bitcast (dt, x) -> bitcast_rule dt x
  | Place _ | Check _ -> Elementwise
  | Attention { q; kv; lengths; loads; _ } -> attention_rule q kv lengths loads

(* Whether [p] holds the whole value on each of its devices. *)
let uncut p =
  let g = Devices.grid p in
  not (Grid.is_cut g || Grid.is_partial g)

(* The host's placement, where an operation of no operands lies unless it is
   a map that names its own. *)
let host = Devices.on Devices.host

(* Where operands lie together: none seen yet, all at physically one
   placement, or otherwise. *)
type together = Unseen | One of unit Devices.placement | Apart

(* [acc] for the operands before one at [q], then that one. It allocates
   nothing but the first placement found. *)
let join acc q =
  match acc with
  | Apart -> acc
  | Unseen -> One q
  | One p -> if p == q then acc else Apart

(* Where [op]'s operands lie together. *)
let together op = fold (fun () acc x -> join acc (placement x)) () Unseen op

(* Sums *)

(* Whether a cast from [a] to [b] keeps a sum's terms each its own value,
   rounded: between floats, or between complex numbers. *)
let float_cast (D.Any a) (D.Any b) =
  match (D.kind a, D.kind b) with
  | D.Float, D.Float | D.Complex, D.Complex -> true
  | (D.Float | D.Complex | D.Signed | D.Unsigned | D.Boolean), _ -> false

(* Whether each node of [prog] reads, linearly, an operand [summed] names: a
   node of [In] such an operand; [Add], [Sub], [Neg], [Copy] and a [Cast]
   between floats or between complex numbers of such nodes,
   [Mul] of one by a node that reads none, [Fdiv] of one by such a node, and
   [Where] of such nodes by a condition that reads none. [None] where a node
   reads one otherwise. *)
let linear (prog : P.t) (summed : int -> bool) : bool array option =
  let nodes = prog.nodes in
  let n = Iarray.length nodes in
  let d = Array.make n false in
  let rec from i =
    if i = n then Some d
    else
      let ok =
        match Iarray.get nodes i with
        | P.In k ->
            d.(i) <- summed k;
            true
        | Coord _ | Const _ -> true
        | Op1 ((Copy | Unary Neg), _, j) ->
            d.(i) <- d.(j);
            true
        | Op1 (Cast, D.Any c, j) when float_cast (P.dtype prog j) (D.Any c) ->
            d.(i) <- d.(j);
            true
        | Op1 (_, _, j) -> not d.(j)
        | Op2 (Binary (Add | Sub), j, k) ->
            d.(i) <- d.(j) || d.(k);
            true
        | Op2 (Binary Mul, j, k) ->
            d.(i) <- d.(j) || d.(k);
            not (d.(j) && d.(k))
        | Op2 (Binary Fdiv, j, k) ->
            d.(i) <- d.(j);
            not d.(k)
        | Op2 (_, j, k) -> not (d.(j) || d.(k))
        | Op3 (Where, c, j, k) ->
            d.(i) <- d.(j) || d.(k);
            not d.(c)
        | Op3 (_, j, k, l) -> not (d.(j) || d.(k) || d.(l))
      in
      if ok then from (i + 1) else None
  in
  from 0

(* [prog] as a device that holds no group's first term runs it: each
   operand of an [Add] or a [Sub] that reads no sum beside one that does is
   the additive identity, [-0] for an addend and [+0] for a subtrahend, and
   so is each case of a [Where] and each output that reads none, as an
   addend. *)
let joined (prog : P.t) summed =
  match linear prog summed with
  | None -> invalid_arg "Prim.joined: a program not linear in its sums"
  | Some d ->
      let nodes = ref ([] : P.node list) and count = ref 0 in
      let push node =
        nodes := List.cons node !nodes;
        incr count;
        !count - 1
      in
      let identity ~negative i =
        let (D.Any dt) = P.dtype prog i in
        let zero = push (P.Const (D.Any dt, P.bits dt (D.zero dt))) in
        match D.kind dt with
        | (D.Float | D.Complex) when negative ->
            push (P.Op1 (Unary Neg, D.Any dt, zero))
        | D.Float | D.Complex | D.Signed | D.Unsigned | D.Boolean -> zero
      in
      let m = Array.make (Iarray.length prog.nodes) 0 in
      Iarray.iteri
        (fun i (node : P.node) ->
          m.(i) <-
            (match node with
            | In _ | Coord _ | Const _ -> push node
            | Op1 (k, dt, j) -> push (Op1 (k, dt, m.(j)))
            | Op2 ((Binary ((Add | Sub) as b) as k), j, l) when d.(j) <> d.(l)
              ->
                let j' = if d.(j) then m.(j) else identity ~negative:true j in
                let l' =
                  if d.(l) then m.(l)
                  else identity ~negative:(b = P.Add) l
                in
                push (Op2 (k, j', l'))
            | Op2 (k, j, l) -> push (Op2 (k, m.(j), m.(l)))
            | Op3 (Where, c, j, l) when d.(j) <> d.(l) ->
                let case j = if d.(j) then m.(j) else identity ~negative:true j in
                push (Op3 (Where, m.(c), case j, case l))
            | Op3 (k, j, l, o) -> push (Op3 (k, m.(j), m.(l), m.(o)))))
        prog.nodes;
      let outs =
        Iarray.map
          (fun o -> if d.(o) then m.(o) else identity ~negative:true o)
          prog.outs
      in
      P.v ~ins:prog.ins (Iarray.of_list (List.rev !nodes)) ~outs

(* Whether a load of [loads] is padded and reads a sum. *)
let padded_sum (loads : 'd load iarray) summed =
  let found = ref false in
  Iarray.iteri
    (fun i (l : 'd load) ->
      match l with Padded _ when summed i -> found := true | _ -> ())
    loads;
  !found

let keeps prog loads summed : Route.terms =
  if padded_sum loads summed || Option.is_none (linear prog summed) then
    Closes
  else Keeps

let terms (type r) (op : r prim) summed : Route.terms =
  match op with
  | Map { prog; loads; _ } -> keeps prog loads summed
  | Reduce { prog; loads; reductions = [ Monoid (Sum, _, _) ]; _ } ->
      keeps prog loads summed
  | Gather _ -> if summed 0 then Closes else Keeps
  | Gather_ranges _ -> if summed 0 || summed 1 then Closes else Keeps
  | Copy _ | Move _ -> Keeps
  | Contract c ->
      (* Linear in [a] or in [b], which come first; [init], the scales and
         the ids follow. A sum operand keeps its sum; cut contracting
         pairs make one only where [sum] asks for it. *)
      let n = fold (fun () n _ -> n + 1) () 0 op in
      let rec rest i = i >= n || ((not (summed i)) && rest (i + 1)) in
      if (summed 0 && summed 1) || not (rest 2) then Closes
      else if c.sum || summed 0 || summed 1 then Sums
      else Closes
  | Reduce _ | Scan _ | Scatter _ | Sort _ | Unique _ | Fold _ | Fft _
  | Linalg _ | Assemble _ | Bitcast _ | Place _ | Check _ | Attention _ ->
      Closes

let summable (type v s) ~by (dt : (v, s) D.t) =
  let (D.Any c) = Nx_kernel.carrier (D.Any dt) in
  match D.kind dt with
  | D.Float | D.Complex -> ()
  | (D.Signed | D.Unsigned) when D.equal c dt -> ()
  | D.Signed | D.Unsigned | D.Boolean ->
      invalid_argf
        "%s: a %s value is no sum: a sum is of floats, complex numbers or \
         integers of 8 bits or more"
        by (D.name dt)

let rec summable_outs : type d r. by:string -> (d, r) outs -> unit =
 fun ~by -> function
  | [] -> ()
  | dt :: rest ->
      summable ~by dt;
      summable_outs ~by rest

(* Raises unless each result [op] makes at the sum [route]'s result may be
   one: a contraction's, a map's, a creation's included. Every route asks,
   so a result that is no sum allocates nothing. *)
let summed_results (type r) ~by (op : r prim) (route : unit Route.t) =
  if Devices.partial route.result then
    match op with
    | Contract { out; _ } -> summable ~by out
    | Map m -> summable_outs ~by m.outs
    | _ -> ()

let off_first (type r) (op : r prim) summed : r prim =
  match op with
  | Map m -> Map { m with prog = joined m.prog summed }
  | Reduce r -> Reduce { r with prog = joined r.prog summed }
  | _ -> op

(* [op]'s route over its operands ({!Route.route}). Where every operand lies
   at one placement that cuts no axis, all are read there and the results lie
   there whatever the rule, so neither [op]'s rule nor its operands' shapes
   are computed. An operation of no operands lies where its map names, else on
   the host, and each device of that placement computes its window. *)
let route_of (type r) ~by (op : r prim) : unit Route.t =
  match together op with
  | Unseen ->
      let result = match op with Map { at = Some p; _ } -> p | _ -> host in
      let r = { Route.operands = [||]; result; compute = Windows } in
      summed_results ~by op r;
      r
  | One p when uncut p ->
      let n = fold (fun () n _ -> n + 1) () 0 op in
      { Route.operands = Array.make n p; result = p; compute = Windows }
  | One _ | Apart ->
      let n = fold (fun () n _ -> n + 1) () 0 op in
      let ps = Array.make n host and shapes = Array.make n [||] in
      iteri
        (fun i x ->
          ps.(i) <- placement x;
          shapes.(i) <- shape x)
        op;
      (* A gather's and a scatter's positions are operand 0. *)
      let positions : int list =
        match op with
        | Gather _ | Scatter _ -> [ 0 ]
        | Gather_ranges _ -> [ 0; 1 ]
        | _ -> []
      in
      let summed i = Devices.partial ps.(i) in
      let terms =
        match op with
        | Contract _ -> terms op summed
        | _ ->
            if Array.exists Devices.partial ps then terms op summed else Closes
      in
      (* Inside a composite, a refused operand that derives from one of
         its arguments names the argument ({!Composite}). *)
      let named =
        if Composite.idle () then None
        else
          let named i q =
            let (Operands xs) = operands op in
            let (Any x) = List.nth xs i in
            Composite.named (Value x) q
          in
          Some named
      in
      let r = Route.route ~by ?named ~positions ~terms (rule op) ps shapes in
      summed_results ~by op r;
      r

let rec make_outs : type d r.
    ('a 'b 'c. int -> ('a, 'b, 'c) form -> ('a, 'b, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement ->
    (d, r) outs ->
    r =
 fun m k layout placement -> function
  | [] -> ()
  | dtype :: rest ->
      let v = m k (of_layout dtype layout placement) in
      (v, make_outs m (k + 1) layout placement rest)

(* A reduction's results, from position [k] of the operation's results. *)
let make_reduction : type d a.
    ('v 's 'c. int -> ('v, 's, 'c) form -> ('v, 's, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement ->
    (d, a) reduction ->
    a =
 fun m k layout placement -> function
  | Monoid (_, _, dtype) -> m k (of_layout dtype layout placement)
  | Logsumexp (_, dtype) -> m k (of_layout dtype layout placement)
  | Moments (_, dtype) ->
      let mean = m k (of_layout dtype layout placement) in
      (mean, m (k + 1) (of_layout dtype layout placement))
  | Arg (_, _, dtype) ->
      let extreme = m k (of_layout dtype layout placement) in
      (extreme, m (k + 1) (of_layout D.Int64 layout placement))

let rec make_reductions : type d r.
    ('v 's 'c. int -> ('v, 's, 'c) form -> ('v, 's, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement ->
    (d, r) reductions ->
    r =
 fun m k layout placement -> function
  | [] -> ()
  | r :: rest ->
      let v = make_reduction m k layout placement r in
      (v, make_reductions m (k + width r) layout placement rest)

(* [x]'s shape moved by [mv]. Raises naming [by] where [mv] does not apply: as
   {!check_shape} does for a target shape no layout has, and otherwise naming
   the movement and [x]'s shape. *)
let moved_shape ~by mv x =
  match Nx_array.Move.shape mv (layout_shape x) with
  | s' -> s'
  | exception Invalid_argument _ ->
      let name =
        match mv with
        | Reshape s' ->
            check_shape ~by s';
            "Reshape"
        | Broadcast s' ->
            check_shape ~by s';
            "Broadcast"
        | Permute _ -> "Permute"
        | Slice _ -> "Slice"
        | Window _ -> "Window"
      in
      invalid_argf "%s: %s does not apply to shape %a" by name pp_shape
        (shape x)

(* Whether [nd] is a [Coord] of an axis whose window starts at [first > 0]. *)
let shifted first (nd : P.node) =
  match nd with Coord c -> first.(c) <> 0 | _ -> false

(* [prog] with each [Coord] of an axis whose window starts at [s > 0] replaced
   by [Coord + s], so that a device computes its window. *)
let with_offsets (prog : P.t) (first : int array) =
  if
    Array.for_all (fun s -> s = 0) first
    || not (Iarray.exists (shifted first) prog.nodes)
  then prog
  else
    let offset : P.node -> P.E.any option = function
      | Coord c when first.(c) <> 0 ->
          let start = P.E.const D.Int64 (Int64.of_int first.(c)) in
          Some (P.E.any P.E.(coord c + start))
      | _ -> None
    in
    P.subst ~ins:prog.ins offset prog

let starts w = Array.map (fun (r : Nx_array.Move.range) -> r.start) w

type origin = { start : int array; extents : int array }

let no_origin = { start = [||]; extents = [||] }

let offset : type r. r prim -> origin -> r prim =
 fun op { start; _ } ->
  if Array.length start = 0 then op
  else
    match op with
    | Map m -> Map { m with prog = with_offsets m.prog start }
    | Reduce m -> Reduce { m with prog = with_offsets m.prog start }
    | Scan m -> Scan { m with prog = with_offsets m.prog start }
    | Attention a -> Attention { a with prog = with_offsets a.prog start }
    | Gather _ | Gather_ranges _ | Scatter _ | Sort _ | Unique _ | Fold _
    | Fft _ | Linalg _ | Assemble _ | Contract _ | Copy _ | Move _
    | Bitcast _ | Place _ | Check _ ->
        op

let within : type r. r prim -> Nx_array.Move.range array -> r prim * origin =
 fun op l ->
  let layout = L.contiguous (Route.extents l) in
  let origin whole =
    { start = starts l; extents = Iarray.to_array (L.shape whole) }
  in
  match op with
  | Map m -> (Map { m with layout }, origin m.layout)
  | Reduce m -> (Reduce { m with layout }, origin m.layout)
  | Scan m -> (Scan { m with layout }, origin m.layout)
  | Gather _ | Gather_ranges _ | Scatter _ | Sort _ | Unique _ | Fold _
  | Fft _ | Linalg _ | Assemble _ | Contract _ | Copy _ | Move _ | Bitcast _
  | Place _ | Check _ | Attention _ ->
      invalid_arg "Prim.within: not a loop"

(* A movement's window: a reshape's and a broadcast's operand window holds
   [w]'s elements in order, a cut keeping whole rows ({!Route.moved}); a
   slice takes all of its operand's window along an axis it keeps from 0 by
   step 1, and keeps its range along any other, which its result holds
   whole. *)
let local_move (mv : Nx_array.Move.t) w : Nx_array.Move.t =
  match mv with
  | Reshape _ -> Reshape (Route.extents w)
  | Broadcast _ -> Broadcast (Route.extents w)
  | Slice rs ->
      Slice
        (Iarray.mapi
           (fun a (r : Nx_array.Move.range) ->
             if r.start = 0 && r.step = 1 then { r with count = w.(a).count }
             else r)
           rs)
  | Permute _ | Window _ -> mv

let local : type r. r prim -> Nx_array.Move.range array -> r prim * origin =
 fun op w ->
  (* A loop's window: the window of a load of its shape. *)
  let loop layout =
    match op with
    | Map { loads; _ } when not (Iarray.exists is_padded loads) -> w
    | _ ->
        let r = loop_rule op [| L.rank layout |] in
        Route.window r ~operand:0 (Iarray.to_array (L.shape layout)) w
  in
  let looped layout = within op (loop layout) in
  match op with
  | Map m -> looped m.layout
  | Reduce m -> looped m.layout
  | Scan m -> looped m.layout
  | Fold f -> (Fold { f with shape = Route.extents w }, no_origin)
  | Move (mv, x) -> (Move (local_move mv w, x), no_origin)
  | Assemble a ->
      (* Each piece's region along an axis the window cuts, which every
         piece fills, fills the window's; the window holds any other
         whole. *)
      let range i (r : Nx_array.Move.range) =
        let d = Iarray.get a.shape i and c = w.(i).count in
        if c = d then r
        else if filled a.shape a.pieces i then
          { start = 0; count = c; step = 1 }
        else
          invalid_arg
            "Prim.local: an assembly cut along an axis a piece covers part of"
      in
      let piece (region, x) = (Iarray.mapi range region, x) in
      ( Assemble
          { a with shape = Route.extents w; pieces = List.map piece a.pieces },
        no_origin )
  (* Its program's coordinates are offset to the window ({!offset}): [o]'s
     window starts its loop's but along the keys, read whole, where [o]'s
     last axis is its values' dimension. *)
  | Attention _ ->
      let start = starts w in
      start.(Array.length start - 1) <- 0;
      (op, { no_origin with start })
  | Gather _ | Gather_ranges _ | Scatter _ | Sort _ | Unique _ | Fft _
  | Linalg _ | Contract _ | Copy _ | Bitcast _ | Place _ | Check _ ->
      (op, no_origin)

(* A bitcast's layout and shape: one width keeps them; a narrower dtype appends
   an axis of the ratio, a wider one removes it where strides allow, and a
   copy's contiguous layout otherwise. *)
let bitcast_layout ~by l (from : D.any) (into : D.any) =
  let (D.Any f) = from in
  let (D.Any t) = into in
  let bf = D.bits f and bt = D.bits t in
  let s = L.shape l and st = L.strides l and o = L.offset l in
  let r = Iarray.length s in
  if bf = bt then l
  else if bf > bt then begin
    let k = bf / bt in
    if r >= L.max_rank then
      invalid_argf "%s: a narrowing bitcast of rank %d" by r;
    L.v ~offset:(o * k)
      ~strides:(Iarray.append (Iarray.map (fun x -> x * k) st) [| 1 |])
      (Iarray.append s [| k |])
  end
  else begin
    let k = bt / bf in
    if r = 0 || Iarray.get s (r - 1) <> k then
      invalid_argf "%s: a widening bitcast needs a trailing axis of %d" by k;
    let s' = Iarray.sub s ~pos:0 ~len:(r - 1) in
    let st' = Iarray.sub st ~pos:0 ~len:(r - 1) in
    let viewable =
      L.numel l = 0
      || Iarray.get st (r - 1) = 1
         && o mod k = 0
         && Iarray.for_all (fun x -> x mod k = 0) st'
    in
    if viewable && L.numel l > 0 then
      L.v ~offset:(o / k) ~strides:(Iarray.map (fun x -> x / k) st') s'
    else L.contiguous s'
  end

(* Gathers, scatters and assemblies *)

(* A gather's result's shape: its rule. *)
let gather_shape ~by axis idx x =
  shaped ~by (Shape.gather ~axis (shape idx) (shape x))

let gather_ranges_shape ~by size starts ends x =
  shaped ~by (Shape.gather_ranges ~size (shape starts) (shape ends) (shape x))

(* A scatter's result's shape: its rule, and an [Add] of a dtype it takes. *)
let scatter_shape ~by (combine : Nx_kernel.combine) axis idx updates into =
  if combine = Add && not (P.accepts2 (Binary Add) (dtype into)) then
    invalid_argf "%s: Add does not take %s" by (D.name (dtype into));
  shaped ~by (Shape.scatter ~axis (shape into) (shape idx) (shape updates))

(* A sort's values' shape, and its positions': its rule. *)
let sort_shape ~by axis k x = shaped ~by (Shape.sort ~axis ~k (shape x))

let unique_shapes ~by size keys =
  shaped ~by (Shape.unique ~size (shape keys))

let check_fft (type d r) ~by (f : (d, r) fft) =
  let into x y fits =
    if not fits then
      invalid_argf "%s: a transform of %s into %s" by (D.name x) (D.name y)
  in
  match f with
  | C2c _ -> ()
  | R2c { dtype = c; x; _ } ->
      let r = dtype x in
      into r c (Nx_kernel.same_precision r c)
  | C2r { dtype = r; x; _ } ->
      let c = dtype x in
      into c r (Nx_kernel.same_precision r c)

(* A transform's result's shape: its rule. *)
let fft_shape (type d r) ~by (f : (d, r) fft) =
  check_fft ~by f;
  shaped ~by
    (match f with
    | C2c { axes; x; _ } -> Shape.c2c ~axes (shape x)
    | R2c { axes; x; _ } -> Shape.r2c ~axes (shape x)
    | C2r { axes; n; x; _ } -> Shape.c2r ~axes ~n (shape x))

(* A routine's results' shapes, in their order: its rule. *)
let linalg_shapes (type d r) ~by (l : (d, r) linalg) =
  List.iter
    (fun (Any x) ->
      if not (Nx_kernel.full_precision (dtype x)) then
        invalid_argf "%s: %s does not take %s" by (routine_name l)
          (D.name (dtype x)))
    (linalg_operands l);
  let into r a =
    if not (Nx_kernel.same_precision (dtype a) r) then
      invalid_argf "%s: %s of %s into %s" by (routine_name l)
        (D.name (dtype a)) (D.name r)
  in
  (match l with
  | Svd_values { dtype = r; a } -> into r a
  | Svd { dtype = r; a; _ } -> into r a
  | Eigh_values { dtype = r; a } -> into r a
  | Eigh { dtype = r; a } -> into r a
  | Eig_values { dtype = c; a } -> into c a
  | Eig { dtype = c; a } -> into c a
  | Cholesky _ | Lu _ | Qr _ | Solve_triangular _ -> ());
  let one r = Result.map (fun s -> [| s |]) r in
  let w r = Result.map (fun (w, _) -> [| w |]) r in
  let two r = Result.map (fun (s, s') -> [| s; s' |]) r in
  let three r = Result.map (fun (s, s', s'') -> [| s; s'; s'' |]) r in
  shaped ~by
    (match l with
    | Cholesky { a; _ } -> one (Shape.cholesky (shape a))
    | Lu a -> three (Shape.lu (shape a))
    | Qr { factors; a } -> two (Shape.qr factors (shape a))
    | Svd_values { a; _ } ->
        Result.map (fun (s, _, _) -> [| s |]) (Shape.svd Reduced (shape a))
    | Svd { factors; a; _ } -> three (Shape.svd factors (shape a))
    | Eigh_values { a; _ } -> w (Shape.eigh (shape a))
    | Eigh { a; _ } -> two (Shape.eigh (shape a))
    | Eig_values { a; _ } -> w (Shape.eigh (shape a))
    | Eig { a; _ } -> two (Shape.eigh (shape a))
    | Solve_triangular { a; b; _ } ->
        one (Shape.solve_triangular (shape a) (shape b)))

(* A fold's rule: [whole] is a shape a layout admits, and [x] has the shape a
   padded load of it gives, and a dtype {!Nx_kernel.Prog.Add} takes. *)
let check_fold ~by whole pad x =
  if not (P.accepts2 (Binary Add) (dtype x)) then
    invalid_argf "%s: Fold does not take %s" by (D.name (dtype x));
  check_shape ~by whole;
  let s = shaped ~by (Shape.padded pad (Iarray.to_array whole)) in
  if not (has_shape x s) then
    invalid_argf "%s: an operand of shape %a, not %a" by pp_shape (shape x)
      pp_shape s

let check_assemble (type v s d) ~by (dt : (v, s) dtype) whole (fill : v)
    (pieces : (Nx_array.Move.range iarray * (v, s, d) t) list) =
  check_shape ~by whole;
  check_value ~by dt fill;
  let regions = Array.of_list (List.map fst pieces) in
  let shapes = shaped ~by (Shape.assemble (Iarray.to_array whole) regions) in
  List.iteri
    (fun i (_, x) ->
      if not (has_shape x shapes.(i)) then
        invalid_argf "%s: piece %d has shape %a, its region %a" by i pp_shape
          (shape x) pp_shape shapes.(i))
    pieces

let contract_shape ~by ~batch ~contracting ~segments ~acc a b init a_scale
    b_scale =
  (let (D.Any dt) = acc in
   if not (Nx_kernel.accumulates dt) then
     invalid_argf "%s: an accumulator of %s" by (D.name dt));
  let segments =
    Option.map
      (fun (g : _ Nx_kernel.segments) : Shape.t Nx_kernel.segments ->
        match g with
        | Rows r -> Rows { r with ids = shape r.ids }
        | Contracting r -> Contracting { r with ids = shape r.ids })
      segments
  in
  let s =
    shaped ~by
      (Shape.contract ~batch ~contracting ~segments
         (shape a, Option.map scale_shape a_scale)
         (shape b, Option.map scale_shape b_scale))
  in
  (match init with
  | Some i when not (has_shape i s) ->
      invalid_argf "%s: init of shape %a, not %a" by pp_shape (shape i) pp_shape
        s
  | Some _ | None -> ());
  s

let linalg_results : type d r.
    ('v 's 'c. int -> ('v, 's, 'c) form -> ('v, 's, 'c) t) ->
    L.t array ->
    d Devices.placement ->
    (d, r) linalg ->
    r =
 fun m layouts placement l ->
  let one : type v s. int -> (v, s) dtype -> (v, s, d) t =
   fun k dtype -> m k (of_layout dtype layouts.(k) placement)
  in
  (* Results are made in position order. *)
  let two a b =
    let x = one 0 a in
    (x, one 1 b)
  in
  let three a b c =
    let x = one 0 a in
    let y = one 1 b in
    (x, y, one 2 c)
  in
  match l with
  | Cholesky { a; _ } -> one 0 (dtype a)
  | Lu a -> three (dtype a) D.Int64 D.Int64
  | Qr { a; _ } -> two (dtype a) (dtype a)
  | Svd_values { dtype; _ } -> one 0 dtype
  | Svd { dtype = r; a; _ } -> three r (dtype a) (dtype a)
  | Eigh_values { dtype; _ } -> one 0 dtype
  | Eigh { dtype = r; a } -> two r (dtype a)
  | Eig_values { dtype; _ } -> one 0 dtype
  | Eig { dtype; _ } -> two dtype dtype
  | Solve_triangular { a; _ } -> one 0 (dtype a)

(* [op]'s results' layouts, by its rule: one per result of a factorisation
   or a numbering, one that every result of another operation shares, none
   for a check.
   Raises naming [by] where [op]'s operands break the rule. *)
(* Values as Nx_kernel's rules read them. *)
let values =
  {
    Nx_kernel.arrays = Iarray.length;
    dtype =
      (fun a i ->
        let (Any x) = Iarray.get a i in
        D.Any (dtype x));
  }

(* Attention's rule, {!Nx_kernel.attention_refusal}'s over the dtypes, the
   program and [q]'s rank, then its results' shapes. *)
let attention_shapes (type v s w r a d) ~by (prog : P.t)
    (acc : (float, a) D.t) (out : (w, r) D.t) (q : (v, s, d) t)
    (kv : (v, s, d) kv) (lengths : d index option) (loads : d any iarray) =
  (match
     Nx_kernel.attention_refusal prog ~q:(dtype q) ~rank:(rank q) ~o:out ~acc
       ~loads:values loads
   with
  | Some (_, why) -> invalid_argf "%s: %s" by why
  | None -> ());
  let kv : (Shape.t, Shape.t) Nx_kernel.kv =
    match kv with
    | Dense { k; v } -> Dense { k = shape k; v = shape v }
    | Paged { k; v; table } ->
        Paged { k = shape k; v = shape v; table = shape table }
  in
  let loads = Array.map (fun (Any x) -> shape x) (Iarray.to_array loads) in
  shaped ~by
    (Shape.attention (shape q) kv ~lengths:(Option.map shape lengths) loads)

let layouts : type r. by:string -> r prim -> L.t array =
 fun ~by op ->
  match op with
  | Map { layout; prog; outs; loads; at } ->
      check_map ~by layout prog outs loads at;
      [| layout |]
  | Reduce { layout; axes; prog; reductions; loads } ->
      [| contiguous (check_reduce ~by layout axes prog reductions loads) |]
  | Scan { layout; axis; prog; reduction; loads; carry } ->
      check_scan ~by layout axis prog reduction loads carry;
      [| layout |]
  | Gather { axis; idx; x } -> [| contiguous (gather_shape ~by axis idx x) |]
  | Gather_ranges { size; starts; ends; x } ->
      [| contiguous (gather_ranges_shape ~by size starts ends x) |]
  | Scatter { combine; axis; idx; updates; into; _ } ->
      [| contiguous (scatter_shape ~by combine axis idx updates into) |]
  | Sort { axis; k; x; _ } -> [| contiguous (sort_shape ~by axis k x) |]
  | Unique { size; keys } ->
      let ids, first, count = unique_shapes ~by size keys in
      [| contiguous ids; contiguous first; contiguous first; contiguous count |]
  | Fft f -> [| contiguous (fft_shape ~by f) |]
  | Linalg l -> Array.map contiguous (linalg_shapes ~by l)
  | Fold { shape; pad; x } ->
      check_fold ~by shape pad x;
      [| L.contiguous shape |]
  | Assemble { dtype; shape; fill; pieces } ->
      check_assemble ~by dtype shape fill pieces;
      [| L.contiguous shape |]
  | Contract
      { batch; contracting; acc; a; b; init; a_scale; b_scale; segments; _ } ->
      [|
        contiguous
          (contract_shape ~by ~batch ~contracting ~segments ~acc a b init
             a_scale b_scale);
      |]
  | Copy x -> [| L.contiguous (layout_shape x) |]
  | Move (mv, x) -> (
      let s' = moved_shape ~by mv x in
      match L.move mv (form_layout x) with
      | Some l -> [| l |]
      | None -> [| L.contiguous s' |])
  | Bitcast (dt, x) ->
      [| bitcast_layout ~by (form_layout x) (D.Any (dtype x)) (D.Any dt) |]
  | Place (p, x) ->
      let s = shape x in
      ignore (Devices.window ~by p s 0);
      [|
        (match (x, Devices.device p) with
        | Array { a; _ }, Some _ -> Nx_array.layout a
        | _ -> contiguous s);
      |]
  | Check { ok; data; _ } ->
      let s = shape ok in
      List.iteri
        (fun i (Any x) ->
          if shape x <> s then
            invalid_argf "%s: datum %d has shape %a, the flag %a" by i pp_shape
              (shape x) pp_shape s)
        (data_any data);
      [||]
  | Attention { prog; acc; out; q; kv; lengths; loads; _ } ->
      let o, lse = attention_shapes ~by prog acc out q kv lengths loads in
      [| contiguous o; contiguous lse |]

type routed = { route : unit Route.t option; layouts : L.t array }

let route : type r. by:string -> r prim -> routed =
 fun ~by op ->
  let layouts = layouts ~by op in
  match op with
  | Place _ | Check _ -> { route = None; layouts }
  | Map _ | Reduce _ | Scan _ | Gather _ | Gather_ranges _ | Scatter _
  | Sort _ | Unique _ | Fft _ | Linalg _ | Fold _ | Assemble _ | Contract _
  | Copy _ | Move _ | Bitcast _ | Attention _ ->
      { route = Some (route_of ~by op); layouts }

let results_at : type r.
    unit Devices.placement ->
    L.t array ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
    r prim ->
    r =
 fun p layouts m op ->
  match op with
  | Map { outs; _ } -> make_outs m 0 layouts.(0) p outs
  | Reduce { reductions; _ } ->
      make_reductions m 0 layouts.(0) p reductions
  | Scan { reduction; _ } -> make_reduction m 0 layouts.(0) p reduction
  | Gather { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) p)
  | Gather_ranges { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) p)
  | Scatter { into; _ } -> m 0 (of_layout (dtype into) layouts.(0) p)
  | Sort { x; _ } ->
      let values = m 0 (of_layout (dtype x) layouts.(0) p) in
      (values, m 1 (of_layout D.Int64 layouts.(0) p))
  | Unique _ ->
      let ids = m 0 (of_layout D.Int64 layouts.(0) p) in
      let first = m 1 (of_layout D.Int64 layouts.(1) p) in
      let counts = m 2 (of_layout D.Int64 layouts.(2) p) in
      (ids, first, counts, m 3 (of_layout D.Int64 layouts.(3) p))
  | Fft f -> (
      match f with
      | C2c { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) p)
      | R2c { dtype; _ } -> m 0 (of_layout dtype layouts.(0) p)
      | C2r { dtype; _ } -> m 0 (of_layout dtype layouts.(0) p))
  | Linalg l -> linalg_results m layouts p l
  | Fold { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) p)
  | Assemble { dtype; _ } -> m 0 (of_layout dtype layouts.(0) p)
  | Contract { out; _ } -> m 0 (of_layout out layouts.(0) p)
  | Copy x -> m 0 (of_layout (dtype x) layouts.(0) p)
  | Move (_, x) -> m 0 (of_layout (dtype x) layouts.(0) p)
  | Bitcast (dt, _) -> m 0 (of_layout dt layouts.(0) p)
  | Place (p, x) -> m 0 (of_layout (dtype x) layouts.(0) p)
  | Check _ -> ()
  | Attention { acc; out; _ } ->
      let o = m 0 (of_layout out layouts.(0) p) in
      (o, m 1 (of_layout acc layouts.(1) p))

(* Where [op]'s results lie, by its route: the operands' placement where they
   lie at one that cuts no axis, as {!route_of} decides there, without
   building the route a tracer does not run. *)
let result_placement ~by op =
  match together op with
  | One p when uncut p -> p
  | Unseen | One _ | Apart -> (route_of ~by op).result

let results : type r.
    by:string ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
    r prim ->
    r =
 fun ~by m op ->
  let layouts = layouts ~by op in
  (* A placement and a check have no route: the one lies at its own
     placement and the other has no result. *)
  let p =
    match op with
    | Place _ | Check _ -> host
    | Map _ | Reduce _ | Scan _ | Gather _ | Gather_ranges _ | Scatter _
    | Sort _ | Unique _ | Fft _ | Linalg _ | Fold _ | Assemble _ | Contract _
    | Copy _ | Move _ | Bitcast _ | Attention _ ->
        result_placement ~by op
  in
  results_at p layouts m op

(* One-node maps *)

(* Raises naming [by]: [node] does not take operands of [ins]. *)
let refused ~by node ins =
  invalid_argf "%s: %s does not take %s" by (kind node)
    (String.concat ", "
       (List.map (fun (D.Any dt) -> D.name dt) (Iarray.to_list ins)))

(* One-node programs *)

(* A node's code among its arity's, which places its programs in the tables
   below. *)
let code1 : P.op1 -> int = function
  | Copy -> 0
  | Cast -> 1
  | Bitcast -> 2
  | Unary u -> fst (unary_kind u)

let code2 : P.op2 -> int = function
  | Binary b -> fst (binary_kind b)
  | Compare c -> fst (compare_kind c)

let code3 : P.op3 -> int = function Where -> 0 | Fma -> 1

(* The number of codes of each arity: one past its greatest. *)
let op1_codes = 1 + code1 (Unary Trunc)
let op2_codes = 1 + code2 (Compare Less_equal)
let op3_codes = 1 + code3 Fma

let dtypes = List.length D.all

(* The one-node programs made so far, by kind and dtypes, each made once: a
   map that finds its program builds none. A slot that two domains fill at
   once keeps either's program; they are equal. *)
let nodes1 : P.t option array =
  Array.make (op1_codes * dtypes * dtypes) None

let nodes2 : P.t option array = Array.make (op2_codes * dtypes) None

let nodes3 : P.t option array =
  Array.make (op3_codes * dtypes * dtypes) None

(* [p] kept at [table.(i)]. *)
let kept table i p =
  let p = Some p in
  table.(i) <- p;
  p

let node1 k x y =
  let i = (((code1 k * dtypes) + D.code x) * dtypes) + D.code y in
  match nodes1.(i) with
  | Some _ as p -> p
  | None when not (P.accepts1 k x y) -> None
  | None -> kept nodes1 i (P.of_node ~ins:[| D.Any x |] (Op1 (k, D.Any y, 0)))

let node2 k x =
  let i = (code2 k * dtypes) + D.code x in
  match nodes2.(i) with
  | Some _ as p -> p
  | None when not (P.accepts2 k x) -> None
  | None -> kept nodes2 i (P.of_node ~ins:[| D.Any x; D.Any x |] (Op2 (k, 0, 1)))

let node3 k c x =
  let i = (((code3 k * dtypes) + D.code c) * dtypes) + D.code x in
  match nodes3.(i) with
  | Some _ as p -> p
  | None when not (P.accepts3 k c x) -> None
  | None ->
      kept nodes3 i
        (P.of_node ~ins:[| D.Any c; D.Any x; D.Any x |] (Op3 (k, 0, 1, 2)))

(* The program that gives its one operand back, by the operand's dtype. *)
let identities : P.t option array = Array.make dtypes None

let identity dt =
  let i = D.code dt in
  match identities.(i) with
  | Some p -> p
  | None ->
      let p = P.v ~ins:[| D.Any dt |] [| In 0 |] ~outs:[| 0 |] in
      identities.(i) <- Some p;
      p

(* The literal program made last for each kind, dtype and side, beside its
   literal's bits: a scalar form that repeats its literal, as a loop does,
   builds no program and allocates the bits alone. A slot that two domains
   fill at once keeps either's pair; each pair holds its own bits. *)
let literals : (string * P.t) option array =
  Array.make (op2_codes * dtypes * 2) None

(* A literal's program: [k] of its one operand and the literal, read second,
   or first where [swapped]. A program of three nodes: it holds the literal's
   bits. *)
let literal2 ~by ~swapped k dt c =
  check_value ~by dt c;
  let bits = P.bits dt c in
  let i = (((code2 k * dtypes) + D.code dt) * 2) + Bool.to_int swapped in
  match literals.(i) with
  | Some (b, p) when String.equal b bits -> p
  | Some _ | None ->
      if not (P.accepts2 k dt) then
        refused ~by (P.Op2 (k, 0, 1)) [| D.Any dt; D.Any dt |];
      let op : P.node = if swapped then Op2 (k, 1, 0) else Op2 (k, 0, 1) in
      let p =
        P.v ~ins:[| D.Any dt |] [| In 0; Const (D.Any dt, bits); op |]
          ~outs:[| 2 |]
      in
      literals.(i) <- Some (bits, p);
      p

(* The map of [prog] over [loads], into [dt]. *)
let one_node prog dt loads =
  let (Any x) = load_any (Iarray.get loads 0) in
  Map
    {
      layout = L.contiguous (layout_shape x);
      prog;
      outs = Value.[ dt ];
      loads;
      at = None;
    }

let op1 ~by k dt x =
  match node1 k (dtype x) dt with
  | Some prog -> one_node prog dt [| Plain x |]
  | None -> refused ~by (P.Op1 (k, D.Any dt, 0)) [| D.Any (dtype x) |]

let ins2 x : D.any iarray =
  let i = D.Any (dtype x) in
  [| i; i |]

let ins3 c x : D.any iarray =
  let i = D.Any (dtype x) in
  [| D.Any (dtype c); i; i |]

let op2 ~by k dt x y =
  match node2 k (dtype x) with
  | Some prog -> one_node prog dt [| Plain x; Plain y |]
  | None -> refused ~by (P.Op2 (k, 0, 1)) (ins2 x)

let op3 ~by k c x y =
  match node3 k (dtype c) (dtype x) with
  | Some prog -> one_node prog (dtype x) [| Plain c; Plain x; Plain y |]
  | None -> refused ~by (P.Op3 (k, 0, 1, 2)) (ins3 c x)

type 'd outputs =
  | Outputs : ('d, 'r) outs * ('r -> 'd any list) -> 'd outputs

let rec outputs : type d. D.any list -> d outputs = function
  | [] -> Outputs ([], fun () -> [])
  | D.Any dt :: dts ->
      let (Outputs (outs, read)) = outputs dts in
      Outputs (dt :: outs, fun (v, r) -> Any v :: read r)

let rec iter_outs : type d q.
    ('v 's. ('v, 's, d) t -> unit) -> (d, q) outs -> q -> unit =
 fun f outs r ->
  match (outs, r) with
  | [], () -> ()
  | _ :: rest, (v, r) ->
      f v;
      iter_outs f rest r

let iter_reduction : type d a.
    ('v 's. ('v, 's, d) t -> unit) -> (d, a) reduction -> a -> unit =
 fun f r v ->
  match (r, v) with
  | Monoid _, v -> f v
  | Logsumexp _, v -> f v
  | Moments _, (a, b) ->
      f a;
      f b
  | Arg _, (a, b) ->
      f a;
      f b

let rec iter_reductions : type d q.
    ('v 's. ('v, 's, d) t -> unit) -> (d, q) reductions -> q -> unit =
 fun f rs v ->
  match (rs, v) with
  | [], () -> ()
  | r :: rest, (a, v) ->
      iter_reduction f r a;
      iter_reductions f rest v

let iter_fft : type d r.
    ('v 's. ('v, 's, d) t -> unit) -> (d, r) fft -> r -> unit =
 fun f t r ->
  match (t, r) with C2c _, x -> f x | R2c _, x -> f x | C2r _, x -> f x

let iter_linalg : type d r.
    ('v 's. ('v, 's, d) t -> unit) -> (d, r) linalg -> r -> unit =
 fun f l r ->
  match (l, r) with
  | Cholesky _, x -> f x
  | Solve_triangular _, x -> f x
  | Svd_values _, s -> f s
  | Eigh_values _, w -> f w
  | Eig_values _, w -> f w
  | Lu _, (a, b, c) ->
      f a;
      f b;
      f c
  | Svd _, (s, u, vh) ->
      f s;
      f u;
      f vh
  | Qr _, (a, b) ->
      f a;
      f b
  | Eigh _, (w, v) ->
      f w;
      f v
  | Eig _, (w, v) ->
      f w;
      f v

let iter_results : type r.
    ('v 's 'd. ('v, 's, 'd) t -> unit) -> r prim -> r -> unit =
 fun f op r ->
  match op with
  | Map { outs; _ } -> iter_outs f outs r
  | Reduce { reductions; _ } -> iter_reductions f reductions r
  | Scan { reduction; _ } -> iter_reduction f reduction r
  | Sort _ ->
      let values, positions = r in
      f values;
      f positions
  | Unique _ ->
      let ids, first, counts, count = r in
      f ids;
      f first;
      f counts;
      f count
  | Fft t -> iter_fft f t r
  | Linalg l -> iter_linalg f l r
  | Gather _ -> f r
  | Gather_ranges _ -> f r
  | Scatter _ -> f r
  | Fold _ -> f r
  | Assemble _ -> f r
  | Contract _ -> f r
  | Copy _ -> f r
  | Move _ -> f r
  | Bitcast _ -> f r
  | Place _ -> f r
  | Check _ -> ()
  | Attention _ ->
      let o, lse = r in
      f o;
      f lse

(* Zipping results *)

(* [f] of the values at one position of each result, the result of none
   [None] there. *)
type zipper = {
  f : 'v 's 'd. int -> ('v, 's, 'd) t option array -> ('v, 's, 'd) t;
}

let fsts rs = Array.map (Option.map fst) rs
let snds rs = Array.map (Option.map snd) rs

let rec zip_outs : type d q. zipper -> int -> (d, q) outs -> q option array -> q
    =
 fun z k outs rs ->
  match outs with
  | [] -> ()
  | _ :: rest ->
      let v = z.f k (fsts rs) in
      (v, zip_outs z (k + 1) rest (snds rs))

let zip_reduction : type d a.
    zipper -> int -> (d, a) reduction -> a option array -> a =
 fun z k r rs ->
  match r with
  | Monoid _ -> z.f k rs
  | Logsumexp _ -> z.f k rs
  | Moments _ ->
      let mean = z.f k (fsts rs) in
      (mean, z.f (k + 1) (snds rs))
  | Arg _ ->
      let extreme = z.f k (fsts rs) in
      (extreme, z.f (k + 1) (snds rs))

let rec zip_reductions : type d q.
    zipper -> int -> (d, q) reductions -> q option array -> q =
 fun z k rs vs ->
  match rs with
  | [] -> ()
  | r :: rest ->
      let a = zip_reduction z k r (fsts vs) in
      (a, zip_reductions z (k + width r) rest (snds vs))

let zip_linalg : type d r. zipper -> (d, r) linalg -> r option array -> r =
 fun z l rs ->
  let nth f = Array.map (Option.map f) rs in
  match l with
  | Cholesky _ -> z.f 0 rs
  | Solve_triangular _ -> z.f 0 rs
  | Svd_values _ -> z.f 0 rs
  | Eigh_values _ -> z.f 0 rs
  | Eig_values _ -> z.f 0 rs
  | Lu _ ->
      let a = z.f 0 (nth (fun (a, _, _) -> a)) in
      let b = z.f 1 (nth (fun (_, b, _) -> b)) in
      (a, b, z.f 2 (nth (fun (_, _, c) -> c)))
  | Svd _ ->
      let s = z.f 0 (nth (fun (s, _, _) -> s)) in
      let u = z.f 1 (nth (fun (_, u, _) -> u)) in
      (s, u, z.f 2 (nth (fun (_, _, vh) -> vh)))
  | Qr _ ->
      let q = z.f 0 (fsts rs) in
      (q, z.f 1 (snds rs))
  | Eigh _ ->
      let w = z.f 0 (fsts rs) in
      (w, z.f 1 (snds rs))
  | Eig _ ->
      let w = z.f 0 (fsts rs) in
      (w, z.f 1 (snds rs))

let zip : type r.
    r prim ->
    r option array ->
    ('v 's 'd. int -> ('v, 's, 'd) t option array -> ('v, 's, 'd) t) ->
    r =
 fun op rs f ->
  let z = { f } in
  match op with
  | Map { outs; _ } -> zip_outs z 0 outs rs
  | Reduce { reductions; _ } -> zip_reductions z 0 reductions rs
  | Scan { reduction; _ } -> zip_reduction z 0 reduction rs
  | Sort _ ->
      let values = f 0 (fsts rs) in
      (values, f 1 (snds rs))
  | Unique _ ->
      let nth g = Array.map (Option.map g) rs in
      let ids = f 0 (nth (fun (a, _, _, _) -> a)) in
      let first = f 1 (nth (fun (_, b, _, _) -> b)) in
      let counts = f 2 (nth (fun (_, _, c, _) -> c)) in
      (ids, first, counts, f 3 (nth (fun (_, _, _, d) -> d)))
  | Fft (C2c _) -> f 0 rs
  | Fft (R2c _) -> f 0 rs
  | Fft (C2r _) -> f 0 rs
  | Linalg l -> zip_linalg z l rs
  | Gather _ -> f 0 rs
  | Gather_ranges _ -> f 0 rs
  | Scatter _ -> f 0 rs
  | Fold _ -> f 0 rs
  | Assemble _ -> f 0 rs
  | Contract _ -> f 0 rs
  | Copy _ -> f 0 rs
  | Move _ -> f 0 rs
  | Bitcast _ -> f 0 rs
  | Place _ -> f 0 rs
  | Check _ -> ()
  | Attention _ ->
      let o = f 0 (fsts rs) in
      (o, f 1 (snds rs))

let any xs =
  Array.init (Iarray.length xs) (fun j -> Nx_array.Any (Iarray.get xs j))

(* [x]'s arrays untyped; a value on one device builds no [iarray] on the way. *)
let any_arrays : type v s d. (v, s, d) t -> Nx_array.any array = function
  | Array { a; _ } -> [| Nx_array.Any a |]
  | x -> any (arrays x)

let result_arrays op r =
  let acc = ref ([] : Nx_array.any array list) in
  iter_results (fun x -> acc := List.cons (any_arrays x) !acc) op r;
  Array.of_list (List.rev !acc)
