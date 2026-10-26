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

let at : type v s d. (v, s, d) t -> d Devices.placement option = function
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } -> Some at
  | Deferred { form; _ } | Traced { form; _ } -> form.placement

let placement : type v s d. (v, s, d) t -> d Devices.placement = function
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } -> at
  | Traced { form = { placement = Some p; _ }; _ } -> p
  | Deferred _ | Traced _ -> invalid_arg "Prim.placement: a value of every set"

let dtype : type v s d. (v, s, d) t -> (v, s) dtype = function
  | Array { a; _ } -> Nx_array.dtype a
  | Shards { arrays; _ } | Donated { arrays; _ } ->
      Nx_array.dtype (Iarray.get arrays 0)
  | Deferred { form; _ } | Traced { form; _ } -> form.dtype

(* The layout of [x]'s array, of its first shard, or of its form. *)
let own_layout : type v s d. (v, s, d) t -> L.t = function
  | Array { a; _ } -> Nx_array.layout a
  | Shards { arrays; _ } | Donated { arrays; _ } ->
      Nx_array.layout (Iarray.get arrays 0)
  | Deferred { form; _ } | Traced { form; _ } -> form.layout

let rank x = L.rank (own_layout x)

let death : type v s d. (v, s, d) t -> death = function
  | Array r -> r.dead
  | Shards r -> r.dead
  | Donated r -> Atomic.Loc.get r.donor
  | Deferred _ | Traced _ -> Live

let arrays : type v s d. (v, s, d) t -> (v, s) Nx_array.t iarray = function
  | Array { a; _ } -> [| a |]
  | Shards { arrays; _ } | Donated { arrays; _ } -> arrays
  | Deferred _ -> invalid_arg "Prim.arrays: a constant has no arrays"
  | Traced _ -> invalid_arg "Prim.arrays: a traced value has no arrays"

let of_arrays (type v s d) (p : d Devices.placement) ~shape
    (arrays : (v, s) Nx_array.t iarray) : (v, s, d) t =
  if Iarray.length arrays = 1 then
    Array { at = p; a = Iarray.get arrays 0; dead = Live }
  else Shards { at = p; shape; arrays; dead = Live }

let dim (type v s d) (x : (v, s, d) t) i =
  let l = own_layout x in
  if i < 0 || i >= L.rank l then
    invalid_arg
      (Printf.sprintf "Prim.dim: axis %d of a value of rank %d" i (L.rank l));
  match x with
  | Array _ | Deferred _ | Traced _ -> L.dim l i
  | Shards { shape; _ } | Donated { shape; _ } -> shape.(i)

let shape (type v s d) (x : (v, s, d) t) =
  match x with
  | Array _ | Deferred _ | Traced _ -> L.shape (own_layout x)
  | Shards { shape; _ } | Donated { shape; _ } -> Array.copy shape

let scale_shape (type d) (sc : d scale) =
  match sc with Exponent x -> shape x | Value x -> shape x

let has_shape x s =
  let rec go i = i = Array.length s || (dim x i = s.(i) && go (i + 1)) in
  rank x = Array.length s && go 0

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
  | Live | Offered _ -> ()
  | Died why -> invalid_argf "%s: %a was donated to %s" by pp_value x why

(* Each refusal of [Layout.fits] in its own words: they are rare, so the cheap
   predicate runs first. *)
let check_shape ~by s =
  if not (L.fits s) then begin
    if Array.exists (fun d -> d < 0) s then
      invalid_argf "%s: shape %a has a negative extent" by pp_shape s;
    if Array.length s > L.max_rank then
      invalid_argf "%s: shape %a has %d axes, more than %d" by pp_shape s
        (Array.length s) L.max_rank;
    invalid_argf "%s: the extents of shape %a other than 0 multiply past %d" by
      pp_shape s L.max_numel
  end

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
    (placement : d Devices.placement option) : (v, s, d) form =
  match placement with
  | Some p
    when Grid.count (Devices.grid p) > 1
         && not (L.is_contiguous layout && L.offset layout = 0) ->
      { dtype; layout = L.contiguous (L.shape layout); placement }
  | Some _ | None -> { dtype; layout; placement }

let form (type v s d) (x : (v, s, d) t) : (v, s, d) form =
  match x with
  | Array { at; a; _ } ->
      {
        dtype = Nx_array.dtype a;
        layout = Nx_array.layout a;
        placement = Some at;
      }
  | Donated { at; arrays; _ } when Iarray.length arrays = 1 ->
      let a = Iarray.get arrays 0 in
      {
        dtype = Nx_array.dtype a;
        layout = Nx_array.layout a;
        placement = Some at;
      }
  | Shards { at; _ } | Donated { at; _ } ->
      of_layout (dtype x) (L.contiguous (shape x)) (Some at)
  | Deferred { form; _ } | Traced { form; _ } -> form

let is_constant : type v s d. (v, s, d) t -> bool = function
  | Deferred _ -> true
  | Array _ | Shards _ | Donated _ | Traced _ -> false

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
  | Svd_values a -> [ Any a ]
  | Svd { a; _ } -> [ Any a ]
  | Eigh_values a -> [ Any a ]
  | Eigh a -> [ Any a ]
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
  | Svd_values a ->
      let a' = m 0 a in
      if a' == a then l else Svd_values a'
  | Svd r ->
      let a = m 0 r.a in
      if a == r.a then l else Svd { r with a }
  | Eigh_values a ->
      let a' = m 0 a in
      if a' == a then l else Eigh_values a'
  | Eigh a ->
      let a' = m 0 a in
      if a' == a then l else Eigh a'
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
  | Scatter _ -> "Scatter"
  | Sort _ -> "Sort"
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

let load_any (type d) (l : d load) : d any =
  match l with Plain x -> Any x | Padded { x; _ } -> Any x

let load_dtype (type d) (l : d load) =
  let (Any x) = load_any l in
  D.Any (dtype x)

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

(* A scale's array, and segments' offsets. *)
let scale_any : type d. d scale -> d any = function
  | Exponent x -> Any x
  | Value x -> Any x

let offsets : 'i Nx_kernel.segments -> 'i = function
  | Rows { offsets; _ } | Contracting { offsets; _ } -> offsets

(* A contraction's operands after [a] and [b], each it has, in order. *)
let contract_rest (type v s d) (init : (v, s, d) t option) a_scale b_scale
    (segments : d index Nx_kernel.segments option) : d any list =
  let some f o : d any list =
    match o with Some x -> List.cons (f x) List.[] | None -> List.[]
  in
  some (fun i -> Any i) init
  @ some scale_any a_scale
  @ some scale_any b_scale
  @ some (fun g -> Any (offsets g)) segments

let operands : type r. r prim -> operands = function
  | Map { loads; _ } -> Operands (Iarray.to_list (Iarray.map load_any loads))
  | Reduce { loads; _ } -> Operands (Iarray.to_list (Iarray.map load_any loads))
  | Scan { loads; carry; _ } ->
      let loads = Iarray.to_list (Iarray.map load_any loads) in
      Operands (loads @ Option.to_list carry)
  | Gather { idx; x; _ } -> Operands [ Any idx; Any x ]
  | Scatter { idx; updates; into; _ } ->
      Operands [ Any idx; Any updates; Any into ]
  | Sort { x; _ } -> Operands [ Any x ]
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

(* Whether [region] covers all of [shape] in order, as a kernel's in-place
   first piece of an assembly must. *)
let whole_region (region : Nx_array.Move.range iarray) shape =
  Iarray.length region = Iarray.length shape
  && Iarray.for_all2
       (fun (r : Nx_array.Move.range) d ->
         r.start = 0 && r.count = d && (r.step = 1 || r.count <= 1))
       region shape

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
      Some ([ 2 ], L.contiguous (shape into), D.Any (dtype into))
  | Contract { out; init = Some i; _ } ->
      Some ([ 2 ], L.contiguous (shape i), D.Any out)
  | Assemble { dtype; shape; pieces = (region, _) :: _; _ }
    when whole_region region shape ->
      Some ([ 0 ], L.contiguous (Iarray.to_array shape), D.Any dtype)
  | Map _ | Reduce _ | Scan _ | Gather _ | Contract _ | Sort _ | Fft _
  | Linalg _ | Fold _ | Assemble _ | Copy _ | Move _ | Bitcast _ | Place _
  | Check _ ->
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
    let (Any x) = load_any (Iarray.get loads i) in
    fold_loads f e (f e acc x) loads (i + 1)

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

(* [f e] folded over [op]'s operands in {!operands}' order. [e] carries what
   [f] needs, so that [f] can be closed: where it is, the fold allocates
   nothing for an operation other than a factorisation, a contraction or a
   check. *)
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
  | Scatter { idx; updates; into; _ } -> f e (f e (f e acc idx) updates) into
  | Sort { x; _ } -> f e acc x
  | Fold { x; _ } -> f e acc x
  | Fft fft ->
      let (Any x) = fft_operand fft in
      f e acc x
  | Linalg l ->
      List.fold_left (fun acc (Any x) -> f e acc x) acc (linalg_operands l)
  | Assemble { pieces; _ } -> fold_pieces f e acc pieces
  | Contract { a; b; init; a_scale; b_scale; segments; _ } -> (
      let acc = f e (f e acc a) b in
      let acc = match init with Some i -> f e acc i | None -> acc in
      let scaled acc = function
        | None -> acc
        | Some (Exponent x) -> f e acc x
        | Some (Value x) -> f e acc x
      in
      let acc = scaled (scaled acc a_scale) b_scale in
      match segments with Some g -> f e acc (offsets g) | None -> acc)
  | Copy x -> f e acc x
  | Move (_, x) -> f e acc x
  | Bitcast (_, x) -> f e acc x
  | Place (_, x) -> f e acc x
  | Check { ok; data; _ } ->
      List.fold_left
        (fun acc (Any x) -> f e acc x)
        (f e acc ok) (data_any data)

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
  | Scatter s ->
      let idx = m 0 s.idx in
      let updates = m 1 s.updates in
      let into = m 2 s.into in
      if idx == s.idx && updates == s.updates && into == s.into then op
      else Scatter { s with idx; updates; into }
  | Sort s ->
      let x = m 0 s.x in
      if x == s.x then op else Sort { s with x }
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
      let a = m 0 c.a in
      let b = m 1 c.b in
      let k = ref 2 in
      let next f = function
        | None -> None
        | Some x as o ->
            let y = f !k x in
            incr k;
            if y == x then o else Some y
      in
      let init = next m c.init in
      let scale i = function
        | Exponent x as e ->
            let y = m i x in
            if y == x then e else Exponent y
        | Value x as v ->
            let y = m i x in
            if y == x then v else Value y
      in
      let a_scale = next scale c.a_scale in
      let b_scale = next scale c.b_scale in
      let segments =
        next
          (fun i (g : _ Nx_kernel.segments) ->
            let o = offsets g in
            let o' = m i o in
            if o' == o then g
            else
              match g with
              | Rows r -> Rows { r with offsets = o' }
              | Contracting r -> Contracting { r with offsets = o' })
          c.segments
      in
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

let map : type r.
    ('v 's 'd. ('v, 's, 'd) t -> ('v, 's, 'd) t) -> r prim -> r prim =
 fun m op -> mapi (fun _ x -> m x) op

(* Printing *)

let unary_name : P.unary -> string = function
  | Neg -> "neg"
  | Recip -> "recip"
  | Abs -> "abs"
  | Sign -> "sign"
  | Sqrt -> "sqrt"
  | Exp -> "exp"
  | Exp2 -> "exp2"
  | Log -> "log"
  | Log2 -> "log2"
  | Log1p -> "log1p"
  | Expm1 -> "expm1"
  | Sin -> "sin"
  | Cos -> "cos"
  | Tan -> "tan"
  | Asin -> "asin"
  | Acos -> "acos"
  | Atan -> "atan"
  | Sinh -> "sinh"
  | Cosh -> "cosh"
  | Tanh -> "tanh"
  | Asinh -> "asinh"
  | Acosh -> "acosh"
  | Atanh -> "atanh"
  | Erf -> "erf"
  | Floor -> "floor"
  | Ceil -> "ceil"
  | Round -> "round"
  | Trunc -> "trunc"

let binary_name : P.binary -> string = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Fdiv -> "fdiv"
  | Idiv -> "idiv"
  | Mod -> "mod"
  | Pow -> "pow"
  | Atan2 -> "atan2"
  | Maximum -> "maximum"
  | Minimum -> "minimum"
  | And -> "and"
  | Or -> "or"
  | Xor -> "xor"
  | Threefry -> "threefry"

let compare_name : P.compare -> string = function
  | Equal -> "equal"
  | Not_equal -> "not_equal"
  | Less -> "less"
  | Less_equal -> "less_equal"

let any_name (D.Any dt) = D.name dt

let kind : P.node -> string = function
  | In _ -> "In"
  | Coord _ -> "Iota"
  | Const _ -> "Fill"
  | Op1 (Copy, _, _) -> "Copy"
  | Op1 (Unary k, _, _) -> String.capitalize_ascii (unary_name k)
  | Op1 (Cast, _, _) -> "Cast"
  | Op1 (Bitcast, _, _) -> "Bitcast"
  | Op2 (Binary k, _, _) -> String.capitalize_ascii (binary_name k)
  | Op2 (Compare k, _, _) -> String.capitalize_ascii (compare_name k)
  | Op3 (Where, _, _, _) -> "Where"
  | Op3 (Fma, _, _, _) -> "Fma"

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
    | Op1 (Unary k, _, j) -> Format.fprintf ppf "%s(%a)" (unary_name k) expr j
    | Op1 (Cast, dt, j) -> Format.fprintf ppf "cast_%s(%a)" (any_name dt) expr j
    | Op1 (Bitcast, dt, j) ->
        Format.fprintf ppf "bitcast_%s(%a)" (any_name dt) expr j
    | Op2 (Binary k, j, l) ->
        Format.fprintf ppf "%s(%a, %a)" (binary_name k) expr j expr l
    | Op2 (Compare k, j, l) ->
        Format.fprintf ppf "%s(%a, %a)" (compare_name k) expr j expr l
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
  let pp_at ppf = function
    | Some p -> Format.fprintf ppf "at %a" Devices.pp_placement p
    | None -> Format.pp_print_string ppf "of every set"
  in
  Format.fprintf ppf "%s %a %a"
    (D.name (dtype x))
    pp_shape (shape x) pp_at (at x)

let reduction : type d a. (d, a) reduction -> Nx_kernel.reduction * int =
  function
  | Monoid (m, k, _) -> (Monoid m, k)
  | Logsumexp (k, _) -> (Logsumexp, k)
  | Moments (k, _) -> (Moments, k)
  | Arg (e, k, _) -> (Arg e, k)

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
    (d, r) reductions -> (Nx_kernel.reduction * int) list = function
  | [] -> []
  | r :: rest -> reduction r :: reductions_list rest

let reductions rs = Array.of_list (reductions_list rs)

let pp_reduced (prog : P.t) (pr : printer) ppf (r, k) =
  Format.fprintf ppf "%s %a" (reduction_name r) pr.expr
    (Iarray.get prog.outs k)

let reduced_roots (prog : P.t) rs =
  List.map (fun (_, k) -> Iarray.get prog.outs k) rs

let pp : type r. Format.formatter -> r prim -> unit =
 fun ppf op ->
  let (Operands xs) = operands op in
  Format.fprintf ppf "%s" (name op);
  let sep ppf () = Format.pp_print_string ppf "; " in
  (match op with
  | Map { prog; _ } ->
      let outs = Iarray.to_list prog.outs in
      let pr = printer prog outs in
      Format.fprintf ppf "%a [%a]" pr.bindings ()
        (Format.pp_print_list ~pp_sep:sep pr.expr)
        outs
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
  | Contract _ | Copy _ | Move _ | Bitcast _ | Place _ | Check _ -> ());
  List.iteri (fun i x -> Format.fprintf ppf " (x%d: %a)" i pp_operand x) xs

(* Rules *)

let same_dtype a b = D.code a = D.code b

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

(* A loop's rule on its loads and coordinates: a load per operand of [prog],
   of its dtype and of [layout]'s shape, which is C-contiguous, and a
   coordinate along an axis of that shape. [what] names the loop. *)
let check_loads (type d) ~by ~what layout (prog : P.t) (loads : d load iarray)
    =
  let ins = prog.ins in
  if Iarray.length loads <> Iarray.length ins then
    invalid_argf "%s: a program of %d operands over %d loads" by
      (Iarray.length ins) (Iarray.length loads);
  Iarray.iteri
    (fun i l ->
      let (D.Any want) = Iarray.get ins i in
      let (D.Any have) = load_dtype l in
      if not (same_dtype want have) then
        invalid_argf "%s: load %d is %s where the program reads %s" by i
          (D.name have) (D.name want))
    loads;
  if not (L.is_contiguous layout && L.offset layout = 0) then
    invalid_argf "%s: a %s's layout %a is not C-contiguous" by what L.pp layout;
  Iarray.iteri
    (fun i l ->
      let fits =
        match l with
        | Plain x -> has_layout_shape x layout
        | Padded _ -> load_shape ~by l = L.shape layout
      in
      if not fits then
        invalid_argf "%s: load %d has shape %a, the %s %a" by i pp_shape
          (load_shape ~by l) what pp_shape (L.shape layout))
    loads;
  for i = 0 to Iarray.length prog.nodes - 1 do
    match Iarray.get prog.nodes i with
    | P.Coord a when a >= L.rank layout ->
        invalid_argf "%s: a coordinate along axis %d of a %s of rank %d" by a
          what (L.rank layout)
    | _ -> ()
  done

(* A map's rule; its result's layout, C-contiguous of [shape]. *)
let check_map (type d r) ~by layout (prog : P.t) (outs : (d, r) outs)
    (loads : d load iarray) =
  check_loads ~by ~what:"map" layout prog loads;
  let produced = prog.outs in
  let rec check : type q. int -> (d, q) outs -> unit =
   fun k -> function
     | [] ->
         if k <> Iarray.length produced then
           invalid_argf "%s: %d results for a program of %d outputs" by k
             (Iarray.length produced)
     | dt :: rest ->
         if k >= Iarray.length produced then
           invalid_argf "%s: more results than the program's %d outputs" by
             (Iarray.length produced);
         let (D.Any want) = P.dtype prog (Iarray.get produced k) in
         if not (same_dtype want dt) then
           invalid_argf "%s: result %d is %s where the program gives %s" by k
             (D.name dt) (D.name want);
         check (k + 1) rest
  in
  check 0 outs

(* A reduction's rule over [prog]'s outputs: its output exists and is of a
   dtype it takes. *)
let check_reduction ~by (prog : P.t) (r, k) =
  let produced = prog.outs in
  if k < 0 || k >= Iarray.length produced then
    invalid_argf "%s: a reduction of output %d of a program of %d" by k
      (Iarray.length produced);
  let (D.Any have) = P.dtype prog (Iarray.get produced k) in
  if not (Nx_kernel.accepts r have) then
    invalid_argf "%s: %s does not take %s" by (reduction_name r) (D.name have)

let reduction_width : type d a. (d, a) reduction -> int = function
  | Monoid _ | Logsumexp _ -> 1
  | Moments _ | Arg _ -> 2

let rec results_width : type d r. (d, r) reductions -> int = function
  | [] -> 0
  | r :: rest -> reduction_width r + results_width rest

(* A loop's loads and results number at most [Prog.max_operands], as a
   program's operands and outputs do. *)
let check_width ~by loads results =
  if Iarray.length loads + results > P.max_operands then
    invalid_argf "%s: %d loads and %d results, past %d" by (Iarray.length loads)
      results P.max_operands

(* The shape of the results of [rs] along [axes] of a loop of [layout]: its
   rule. *)
let reduced_shape ~by layout axes rs =
  let kinds = Array.map fst (reductions rs) in
  shaped ~by (Shape.reduce ~axes kinds (L.shape layout))

(* A reduction's rule; its results' shape. *)
let check_reduce (type d r) ~by layout axes prog (rs : (d, r) reductions)
    (loads : d load iarray) =
  check_loads ~by ~what:"reduction" layout prog loads;
  (match reductions_list rs with
  | [] -> invalid_argf "%s: a reduction of no output" by
  | l -> List.iter (check_reduction ~by prog) l);
  check_width ~by loads (results_width rs);
  reduced_shape ~by layout axes rs

let any_shape (Any x) = shape x

let check_scan (type d r) ~by layout axis prog (r : (d, r) reduction)
    (loads : d load iarray) (carry : d any option) =
  check_loads ~by ~what:"scan" layout prog loads;
  let shape = shaped ~by (Shape.scan ~axis (L.shape layout)) in
  check_reduction ~by prog (reduction r);
  check_width ~by loads (reduction_width r);
  match (carry, r) with
  | None, _ -> ()
  | Some (Any c), Monoid ((Sum | Prod | Max | Min), k, _) ->
      let want = Array.copy shape in
      want.(axis) <- 1;
      let have = any_shape (Any c) in
      if have <> want then
        invalid_argf "%s: a carry of shape %a for a scan of shape %a" by
          pp_shape have pp_shape shape;
      let (D.Any want) =
        Nx_kernel.carrier (P.dtype prog (Iarray.get prog.outs k))
      in
      if not (same_dtype want (dtype c)) then
        invalid_argf "%s: a carry of %s where the terms carry in %s" by
          (D.name (dtype c)) (D.name want)
  | Some _, (Logsumexp _ | Moments _ | Arg _) ->
      invalid_argf "%s: a carry for %s" by (reduction_name (fst (reduction r)))

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
  let whole = Array.make (Array.length (L.shape layout)) false in
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
  | Gather _ | Scatter _ | Sort _ | Fold _ | Fft _ | Linalg _ | Assemble _
  | Contract _ | Copy _ | Move _ | Bitcast _ | Place _ | Check _ ->
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
   their factor. Segments are computed uncut: [Rows]' two axes and the result's
   axis of rows are read whole, as is the result's leading axis of
   [Contracting]'s groups. A segmented contraction is read whole along its
   contracting pairs. *)
let contract_rule ~batch ~contracting ~segments sa sb init sca scb =
  let nb = Iarray.length batch and nc = Iarray.length contracting in
  let lead, whole_a, whole_b =
    match (segments : _ Nx_kernel.segments option) with
    | Some (Rows { a_axis; b_axis; _ }) -> (0, a_axis, b_axis)
    | Some (Contracting _) -> (1, -1, -1)
    | None -> (0, -1, -1)
  in
  let reduced = Option.is_none segments in
  let kept = ref (if lead = 1 then 0 else -1) in
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
  let free_a next x =
    incr next;
    if x <> whole_a then Route.Becomes (!next - 1)
    else begin
      kept := !next - 1;
      Whole
    end
  in
  let free_b next x =
    if x = whole_b then Route.Whole
    else begin
      incr next;
      Becomes (!next - 1)
    end
  in
  let ra = Array.length sa in
  let a = operand fst ra (lead + nb) free_a in
  let b = operand snd (Array.length sb) (lead + ra - nc) free_b in
  (* A scale's axis that becomes the result's reads as the operand's where
     its extent is the operand's, and regroups into it where a block holds
     more elements, so that a cut keeps whole blocks; one of one block, and
     every other, is read whole: partials take the blocks each part of a
     tile spans ({!Partial}). *)
  let scaled row s ss =
    Array.mapi
      (fun d e ->
        match (row.(d) : Route.axis) with
        | Becomes j ->
            if e = s.(d) then row.(d)
            else if e <= 1 then Whole
            else Regroups { axis = j; extent = s.(d) }
        | Whole | Reduced _ | Scanned _ | Regroups _ -> Whole)
      ss
  in
  let some f o = match o with Some x -> [| f x |] | None -> [||] in
  let reads_init r =
    Array.init r (fun i -> if i = !kept then Route.Whole else Becomes i)
  in
  let offsets _ =
    Array.init (nb + 1) (fun i ->
        if i < nb then Route.Becomes (lead + i) else Whole)
  in
  let rows : Route.axis array array list =
    [
      [| a; b |];
      some reads_init init;
      some (scaled a sa) sca;
      some (scaled b sb) scb;
      some offsets segments;
    ]
  in
  Route.Axes (Array.concat rows)

let move_rule m x =
  let s = shape x in
  Route.Axes [| Array.init (Array.length s) (Route.moved m s) |]

let bitcast_rule dt x =
  if D.bits dt > D.bits (dtype x) then
    Route.Axes [| reading [| rank x - 1 |] (rank x) |]
  else Route.Elementwise

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
      | Elementwise | Into _ -> invalid_arg "Prim.rule: a loop's rule")
  | Gather { axis; idx; x } ->
      Axes [| reading [||] (rank idx); reading [| axis |] (rank x) |]
  | Scatter { axis; idx; updates; into; _ } ->
      Into
        (Array.map (reading [| axis |]) [| rank idx; rank updates; rank into |])
  | Sort { axis; x; _ } -> Axes [| reading [| axis |] (rank x) |]
  | Fold { pad; x; _ } -> fold_rule pad x
  | Fft f ->
      let (Any x) = fft_operand f in
      Axes [| reading (fft_axes f) (rank x) |]
  | Linalg l -> linalg_rule l
  | Assemble { pieces; _ } ->
      Axes
        (Array.of_list
           (List.map (fun (_, x) -> Array.make (rank x) Route.Whole) pieces))
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

let uncut p = not (Grid.is_cut (Devices.grid p))

(* [x]'s placement at the brand routes carry. *)
let placed (type v s d) (x : (v, s, d) t) : unit Devices.placement option =
  match x with
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } ->
      Some (Devices.rebrand at)
  | Deferred { form; _ } | Traced { form; _ } -> (
      match form.placement with
      | None -> None
      | Some p -> Some (Devices.rebrand p))

(* Where operands lie together ({!Route.common}): [acc] for those before one
   at [q], then that one. *)
let join (acc : unit Route.common) q =
  match acc with
  | Route.Other -> acc
  | Route.Every_set -> Route.One q
  | Route.One p -> if p == q then acc else Route.Other

(* [join] of [x], whose placement may be none. It allocates nothing but the
   first placement found. *)
let joined (type v s d) acc (x : (v, s, d) t) =
  match x with
  | Array { at; _ } | Shards { at; _ } | Donated { at; _ } ->
      join acc (Devices.rebrand at)
  | Deferred { form; _ } | Traced { form; _ } -> (
      match form.placement with
      | None -> acc
      | Some p -> join acc (Devices.rebrand p))

(* [op]'s route over its operands ({!Route.route}). Where every operand not of
   every set lies at one placement that cuts no axis, all are read there and
   the results lie there whatever the rule, so neither [op]'s rule nor its
   operands' shapes are computed. *)
let route_of (type r) ~by (op : r prim) : unit Route.t option =
  match fold (fun () acc x -> joined acc x) () Route.Every_set op with
  | Route.Every_set -> None
  | Route.One p when uncut p ->
      let n = fold (fun () n _ -> n + 1) () 0 op in
      Some { Route.operands = Array.make n p; result = p; compute = Windows }
  | Route.One _ | Route.Other ->
      let n = fold (fun () n _ -> n + 1) () 0 op in
      let ps = Array.make n None and shapes = Array.make n [||] in
      iteri
        (fun i x ->
          ps.(i) <- placed x;
          shapes.(i) <- shape x)
        op;
      Route.route ~by (rule op) ps shapes

let rec make_outs : type d r.
    ('a 'b 'c. int -> ('a, 'b, 'c) form -> ('a, 'b, 'c) t) ->
    int ->
    L.t ->
    d Devices.placement option ->
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
    d Devices.placement option ->
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
    d Devices.placement option ->
    (d, r) reductions ->
    r =
 fun m k layout placement -> function
  | [] -> ()
  | r :: rest ->
      let v = make_reduction m k layout placement r in
      (v, make_reductions m (k + reduction_width r) layout placement rest)

(* [x]'s layout, as its form has it, allocating nothing for a value on one
   device. *)
let form_layout (type v s d) (x : (v, s, d) t) =
  match x with
  | Array { a; _ } -> Nx_array.layout a
  | Donated { arrays; _ } when Iarray.length arrays = 1 ->
      Nx_array.layout (Iarray.get arrays 0)
  | Shards _ | Donated _ -> L.contiguous (shape x)
  | Deferred { form; _ } | Traced { form; _ } -> form.layout

(* [x]'s shape moved by [mv]. Raises naming [by] where [mv] does not apply: as
   {!check_shape} does for a target shape no layout has, and otherwise naming
   the movement and [x]'s shape. *)
let moved_shape ~by mv x =
  match Nx_array.Move.shape mv (shape x) with
  | s' -> s'
  | exception Invalid_argument _ ->
      let name =
        match mv with
        | Reshape s' ->
            check_shape ~by (Iarray.to_array s');
            "Reshape"
        | Broadcast s' ->
            check_shape ~by (Iarray.to_array s');
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
   by [Coord + s], so that a device computes its window. Each node keeps its
   place after the two it gains, [at] mapping old indices to new. *)
let with_offsets (prog : P.t) (first : int array) =
  if
    Array.for_all (fun s -> s = 0) first
    || not (Iarray.exists (shifted first) prog.nodes)
  then prog
  else begin
    let nodes = ref ([] : P.node list) and count = ref 0 in
    let push nd =
      nodes := List.cons nd !nodes;
      incr count;
      !count - 1
    in
    let index = Array.make (Iarray.length prog.nodes) 0 in
    let at j = index.(j) in
    Iarray.iteri
      (fun i nd ->
        index.(i) <-
          (match (nd : P.node) with
          | Coord c when first.(c) <> 0 ->
              let coord = push nd in
              let start = P.bits D.Int64 (Int64.of_int first.(c)) in
              let start = push (Const (D.Any D.Int64, start)) in
              push (Op2 (Binary Add, coord, start))
          | Op1 (k, dt, j) -> push (Op1 (k, dt, at j))
          | Op2 (k, j, l) -> push (Op2 (k, at j, at l))
          | Op3 (k, j, l, m) -> push (Op3 (k, at j, at l, at m))
          | In _ | Coord _ | Const _ -> push nd))
      prog.nodes;
    let nodes = Iarray.of_list (List.rev !nodes) in
    P.v ~ins:prog.ins nodes ~outs:(Iarray.map at prog.outs)
  end

let starts w = Array.map (fun (r : Nx_array.Move.range) -> r.start) w
let extents w = Array.map (fun (r : Nx_array.Move.range) -> r.count) w

let within : type r. r prim -> Nx_array.Move.range array -> r prim =
 fun op l ->
  let layout = L.contiguous (extents l) in
  match op with
  | Map m -> Map { m with layout; prog = with_offsets m.prog (starts l) }
  | Reduce m -> Reduce { m with layout; prog = with_offsets m.prog (starts l) }
  | Scan m -> Scan { m with layout; prog = with_offsets m.prog (starts l) }
  | Gather _ | Scatter _ | Sort _ | Fold _ | Fft _ | Linalg _ | Assemble _
  | Contract _ | Copy _ | Move _ | Bitcast _ | Place _ | Check _ ->
      invalid_arg "Prim.within: not a loop"

let local : type r. r prim -> Nx_array.Move.range array -> r prim =
 fun op w ->
  (* A loop's window: the window of a load of its shape. *)
  let loop layout =
    match op with
    | Map { loads; _ } when not (Iarray.exists is_padded loads) -> w
    | _ ->
        let r = loop_rule op [| L.rank layout |] in
        Route.window r ~operand:0 (L.shape layout) w
  in
  match op with
  | Map m -> within op (loop m.layout)
  | Reduce m -> within op (loop m.layout)
  | Scan m -> within op (loop m.layout)
  | Fold f -> Fold { f with shape = Iarray.of_array (extents w) }
  | Assemble a ->
      if extents w <> Iarray.to_array a.shape then
        invalid_arg "Prim.local: an assembly is computed whole";
      op
  | Gather _ | Scatter _ | Sort _ | Fft _ | Linalg _ | Contract _ | Copy _
  | Move _ | Bitcast _ | Place _ | Check _ ->
      op

(* A bitcast's layout and shape: one width keeps them; a narrower dtype appends
   an axis of the ratio, a wider one removes it where strides allow, and a
   copy's contiguous layout otherwise. *)
let bitcast_layout ~by l (from : D.any) (into : D.any) =
  let (D.Any f) = from in
  let (D.Any t) = into in
  let bf = D.bits f and bt = D.bits t in
  let s = L.shape l and st = L.strides l and o = L.offset l in
  let r = Array.length s in
  if bf = bt then l
  else if bf > bt then begin
    let k = bf / bt in
    if r >= L.max_rank then
      invalid_argf "%s: a narrowing bitcast of rank %d" by r;
    L.v ~offset:(o * k)
      ~strides:(Array.append (Array.map (fun x -> x * k) st) [| 1 |])
      (Array.append s [| k |])
  end
  else begin
    let k = bt / bf in
    if r = 0 || s.(r - 1) <> k then
      invalid_argf "%s: a widening bitcast needs a trailing axis of %d" by k;
    let s' = Array.sub s 0 (r - 1) in
    let viewable =
      L.numel l = 0
      || st.(r - 1) = 1
         && o mod k = 0
         && Array.for_all (fun x -> x mod k = 0) (Array.sub st 0 (r - 1))
    in
    if viewable && L.numel l > 0 then
      L.v ~offset:(o / k)
        ~strides:(Array.map (fun x -> x / k) (Array.sub st 0 (r - 1)))
        s'
    else L.contiguous s'
  end

(* Gathers, scatters and assemblies *)

(* A gather's result's shape: its rule. *)
let gather_shape ~by axis idx x =
  shaped ~by (Shape.gather ~axis (shape idx) (shape x))

(* A scatter's result's shape: its rule, and an [Add] of a dtype it takes. *)
let scatter_shape ~by (combine : Nx_kernel.combine) axis idx updates into =
  if combine = Add && not (P.accepts2 (Binary Add) (dtype into)) then
    invalid_argf "%s: Add does not take %s" by (D.name (dtype into));
  shaped ~by (Shape.scatter ~axis (shape into) (shape idx) (shape updates))

(* A sort's values' shape, and its positions': its rule. *)
let sort_shape ~by axis k x = shaped ~by (Shape.sort ~axis ~k (shape x))

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
  let eig_into c a =
    if not (Nx_kernel.same_precision (dtype a) c) then
      invalid_argf "%s: Eig of %s into %s" by (D.name (dtype a)) (D.name c)
  in
  (match l with
  | Eig_values { dtype = c; a } -> eig_into c a
  | Eig { dtype = c; a } -> eig_into c a
  | _ -> ());
  let one r = Result.map (fun s -> [| s |]) r in
  let two r = Result.map (fun (s, s') -> [| s; s' |]) r in
  let three r = Result.map (fun (s, s', s'') -> [| s; s'; s'' |]) r in
  shaped ~by
    (match l with
    | Cholesky { a; _ } -> one (Shape.cholesky (shape a))
    | Lu a -> three (Shape.lu (shape a))
    | Qr { factors; a } -> two (Shape.qr factors (shape a))
    | Svd_values a -> one (Shape.svd_values (shape a))
    | Svd { factors; a } -> three (Shape.svd factors (shape a))
    | Eigh_values a -> one (Shape.eigh_values (shape a))
    | Eigh a -> two (Shape.eigh (shape a))
    | Eig_values { a; _ } -> one (Shape.eigh_values (shape a))
    | Eig { a; _ } -> two (Shape.eigh (shape a))
    | Solve_triangular { a; b; _ } ->
        one (Shape.solve_triangular (shape a) (shape b)))

(* A fold's rule: [whole] is a shape a layout admits, and [x] has the shape a
   padded load of it gives, and a dtype {!Nx_kernel.Prog.Add} takes. *)
let check_fold ~by whole pad x =
  if not (P.accepts2 (Binary Add) (dtype x)) then
    invalid_argf "%s: Fold does not take %s" by (D.name (dtype x));
  check_shape ~by whole;
  let s = shaped ~by (Shape.padded pad whole) in
  if not (has_shape x s) then
    invalid_argf "%s: an operand of shape %a, not %a" by pp_shape (shape x)
      pp_shape s

let check_assemble (type v s d) ~by (dt : (v, s) dtype) whole (fill : v)
    (pieces : (Nx_array.Move.range iarray * (v, s, d) t) list) =
  check_shape ~by whole;
  check_value ~by dt fill;
  let regions = Array.of_list (List.map fst pieces) in
  let shapes = shaped ~by (Shape.assemble whole regions) in
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
        | Rows r -> Rows { r with offsets = shape r.offsets }
        | Contracting r -> Contracting { r with offsets = shape r.offsets })
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
    d Devices.placement option ->
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
  | Svd_values a -> one 0 (dtype a)
  | Svd { a; _ } -> three (dtype a) (dtype a) (dtype a)
  | Eigh_values a -> one 0 (dtype a)
  | Eigh a -> two (dtype a) (dtype a)
  | Eig_values { dtype; _ } -> one 0 dtype
  | Eig { dtype; _ } -> two dtype dtype
  | Solve_triangular { a; _ } -> one 0 (dtype a)

(* [op]'s results' layouts, by its rule: one per result of a factorisation,
   one that every result of another operation shares, none for a check.
   Raises naming [by] where [op]'s operands break the rule. *)
let layouts : type r. by:string -> r prim -> L.t array =
 fun ~by op ->
  match op with
  | Map { layout; prog; outs; loads } ->
      check_map ~by layout prog outs loads;
      [| layout |]
  | Reduce { layout; axes; prog; reductions; loads } ->
      [| L.contiguous (check_reduce ~by layout axes prog reductions loads) |]
  | Scan { layout; axis; prog; reduction; loads; carry } ->
      check_scan ~by layout axis prog reduction loads carry;
      [| layout |]
  | Gather { axis; idx; x } -> [| L.contiguous (gather_shape ~by axis idx x) |]
  | Scatter { combine; axis; idx; updates; into; _ } ->
      [| L.contiguous (scatter_shape ~by combine axis idx updates into) |]
  | Sort { axis; k; x; _ } -> [| L.contiguous (sort_shape ~by axis k x) |]
  | Fft f -> [| L.contiguous (fft_shape ~by f) |]
  | Linalg l -> Array.map L.contiguous (linalg_shapes ~by l)
  | Fold { shape; pad; x } ->
      let shape = Iarray.to_array shape in
      check_fold ~by shape pad x;
      [| L.contiguous shape |]
  | Assemble { dtype; shape; fill; pieces } ->
      let shape = Iarray.to_array shape in
      check_assemble ~by dtype shape fill pieces;
      [| L.contiguous shape |]
  | Contract
      { batch; contracting; acc; a; b; init; a_scale; b_scale; segments; _ } ->
      [|
        L.contiguous
          (contract_shape ~by ~batch ~contracting ~segments ~acc a b init
             a_scale b_scale);
      |]
  | Copy x -> [| L.contiguous (shape x) |]
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
        | _ -> L.contiguous s);
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

type routed = { route : unit Route.t option; layouts : L.t array }

let route : type r. by:string -> r prim -> routed =
 fun ~by op ->
  let layouts = layouts ~by op in
  match op with
  | Place _ | Check _ -> { route = None; layouts }
  | Map _ | Reduce _ | Scan _ | Gather _ | Scatter _ | Sort _ | Fft _
  | Linalg _ | Fold _ | Assemble _ | Contract _ | Copy _ | Move _ | Bitcast _
    ->
      { route = route_of ~by op; layouts }

let results_at : type r.
    unit Devices.placement option ->
    L.t array ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
    r prim ->
    r =
 fun p layouts m op ->
  let here () = Devices.rebrand_option p in
  match op with
  | Map { outs; _ } -> make_outs m 0 layouts.(0) (here ()) outs
  | Reduce { reductions; _ } ->
      make_reductions m 0 layouts.(0) (here ()) reductions
  | Scan { reduction; _ } -> make_reduction m 0 layouts.(0) (here ()) reduction
  | Gather { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) (here ()))
  | Scatter { into; _ } -> m 0 (of_layout (dtype into) layouts.(0) (here ()))
  | Sort { x; _ } ->
      let values = m 0 (of_layout (dtype x) layouts.(0) (here ())) in
      (values, m 1 (of_layout D.Int64 layouts.(0) (here ())))
  | Fft f -> (
      match f with
      | C2c { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) (here ()))
      | R2c { dtype; _ } -> m 0 (of_layout dtype layouts.(0) (here ()))
      | C2r { dtype; _ } -> m 0 (of_layout dtype layouts.(0) (here ())))
  | Linalg l -> linalg_results m layouts (here ()) l
  | Fold { x; _ } -> m 0 (of_layout (dtype x) layouts.(0) (here ()))
  | Assemble { dtype; _ } -> m 0 (of_layout dtype layouts.(0) (here ()))
  | Contract { out; _ } -> m 0 (of_layout out layouts.(0) (here ()))
  | Copy x -> m 0 (of_layout (dtype x) layouts.(0) (here ()))
  | Move (_, x) -> m 0 (of_layout (dtype x) layouts.(0) (here ()))
  | Bitcast (dt, _) -> m 0 (of_layout dt layouts.(0) (here ()))
  | Place (p, x) -> m 0 (of_layout (dtype x) layouts.(0) (Some p))
  | Check _ -> ()

let results : type r.
    by:string ->
    ('v 's 'd. int -> ('v, 's, 'd) form -> ('v, 's, 'd) t) ->
    r prim ->
    r =
 fun ~by m op ->
  let { route; layouts } = route ~by op in
  let p = Option.map (fun (r : _ Route.t) -> r.result) route in
  results_at p layouts m op

(* One-node maps *)

(* Raises naming [by]: [node] does not take operands of [ins]. *)
let refused ~by node ins =
  invalid_argf "%s: %s does not take %s" by (kind node)
    (String.concat ", "
       (List.map (fun (D.Any dt) -> D.name dt) (Iarray.to_list ins)))

(* The map of [prog] over [loads], into [dt]. *)
let one_node prog dt loads =
  let (Any x) = load_any (Iarray.get loads 0) in
  Map { layout = L.contiguous (shape x); prog; outs = Value.[ dt ]; loads }

let op1 ~by k dt x =
  match P.node1 k (dtype x) dt with
  | Some prog -> one_node prog dt [| Plain x |]
  | None -> refused ~by (P.Op1 (k, D.Any dt, 0)) [| D.Any (dtype x) |]

let ins2 x : D.any iarray =
  let i = D.Any (dtype x) in
  [| i; i |]

let ins3 c x : D.any iarray =
  let i = D.Any (dtype x) in
  [| D.Any (dtype c); i; i |]

let op2 ~by k dt x y =
  match P.node2 k (dtype x) with
  | Some prog -> one_node prog dt [| Plain x; Plain y |]
  | None -> refused ~by (P.Op2 (k, 0, 1)) (ins2 x)

let op3 ~by k c x y =
  match P.node3 k (dtype c) (dtype x) with
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
  | Fft t -> iter_fft f t r
  | Linalg l -> iter_linalg f l r
  | Gather _ -> f r
  | Scatter _ -> f r
  | Fold _ -> f r
  | Assemble _ -> f r
  | Contract _ -> f r
  | Copy _ -> f r
  | Move _ -> f r
  | Bitcast _ -> f r
  | Place _ -> f r
  | Check _ -> ()

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

(* Constants *)

let evaluate ~same ~reads f root p =
  let computed = Hashtbl.create 16 in
  let find (Node n) q =
    List.find_map
      (fun (q', r) -> if same q' q then Some r else None)
      (Hashtbl.find_all computed n.id)
  in
  let missing (node, q) = Option.is_none (find node q) in
  let result node q = Option.get (find node q) in
  let stack = Stack.create () in
  Stack.push (root, p) stack;
  while not (Stack.is_empty stack) do
    let ((Node n as node), q) as top = Stack.top stack in
    if not (missing top) then ignore (Stack.pop stack)
    else
      match List.filter missing (reads node q) with
      | [] ->
          ignore (Stack.pop stack);
          Hashtbl.add computed n.id (q, f result node q)
      | deps -> List.iter (fun d -> Stack.push d stack) deps
  done;
  result root p
