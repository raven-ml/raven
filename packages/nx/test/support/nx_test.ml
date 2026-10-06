(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let tensor w =
  Testable.contramap
    (fun t -> (Nx.shape t, Nx.to_array t))
    (pair (array int) (array w))

(* Floats equal within [rel] of the larger magnitude or within [abs], every NaN
   equal to every NaN: the witness of a computed float that may be NaN. *)
let close ?(abs = 0.) ~rel () =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%.17g" x)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b)
      || a = b
      || Float.abs (a -. b)
         <= Float.max abs (rel *. Float.max (Float.abs a) (Float.abs b)))

let raises_invalid_arg f =
  raises_match Exn.invalid_arg (fun () -> ignore (f ()))

(* [described op r] checks that [r], [op]'s result, has the shape and dtype
   [Nx.Op.shape] and [Nx.Op.dtype] give when it is one value. *)
let described : type r. r Nx.Op.t -> r -> unit =
 fun op r ->
  let value (type a b) (op : (a, b) Nx.t Nx.Op.t) (r : (a, b) Nx.t) =
    let msg = Format.asprintf "%a" Nx.Op.pp op in
    equal ~msg (array int) (Nx.shape r) (Nx.Op.shape op);
    equal ~msg string
      (Nx_dtype.to_string (Nx.dtype r))
      (Nx_dtype.to_string (Nx.Op.dtype op))
  in
  match op with
  | Unary _ -> value op r
  | Binary _ -> value op r
  | Compare _ -> value op r
  | Where _ -> value op r
  | Fma _ -> value op r
  | Reduce _ -> value op r
  | Scan _ -> value op r
  | Arg_reduce _ -> value op r
  | Sort _ -> value op r
  | Argsort _ -> value op r
  | Group _ -> value op r
  | Pad _ -> value op r
  | Cat _ -> value op r
  | Convert _ -> value op r
  | Threefry _ -> value op r
  | Gather _ -> value op r
  | Scatter _ -> value op r
  | Update _ -> value op r
  | Unfold _ -> value op r
  | Fold _ -> value op r
  | Matmul _ -> value op r
  | Fft _ -> value op r
  | Rfft _ -> value op r
  | Irfft _ -> value op r
  | Contiguous _ -> value op r
  | Cholesky _ -> value op r
  | Solve_triangular _ -> value op r
  | Move _ -> value op r
  | Place _ -> value op r
  | Qr _ | Lu _ | Svd _ | Eig _ | Eigh _ | Read _ | Check _ -> ()

let pp_shape ppf s =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_seq s)

let pp_index ppf (i : Nx.index) =
  let ints =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
      Format.pp_print_int
  in
  match i with
  | I i -> Format.fprintf ppf "I %d" i
  | L l -> Format.fprintf ppf "L [%a]" ints l
  | R (a, b) -> Format.fprintf ppf "R (%d, %d)" a b
  | Rs (a, b, s) -> Format.fprintf ppf "Rs (%d, %d, %d)" a b s
  | A -> Format.pp_print_string ppf "A"
  | N -> Format.pp_print_string ppf "N"
  | M m ->
      Format.fprintf ppf "M [%a]"
        (Format.pp_print_seq
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           Format.pp_print_bool)
        (Array.to_seq (Nx.to_array m))
  | D (s, len) -> Format.fprintf ppf "D (%Ld, %d)" (Nx.item [] s) len

let pp_specs ppf specs =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       pp_index)
    specs

let unravel shape i =
  let idx = Array.make (Array.length shape) 0 and r = ref i in
  for d = Array.length shape - 1 downto 0 do
    idx.(d) <- !r mod shape.(d);
    r := !r / shape.(d)
  done;
  idx

let ravel shape idx =
  let i = ref 0 in
  Array.iteri (fun d k -> i := (!i * shape.(d)) + k) idx;
  !i

(* Views at the edges of the ints *)

(* [sat_add a b] is [a + b] and [sat_scale s k], for [k > 0], is [s * k], each
   clamped to [min_int] and [max_int]. *)
let sat_add a b =
  if b > 0 && a > max_int - b then max_int
  else if b < 0 && a < min_int - b then min_int
  else a + b

let sat_scale s k =
  if s > max_int / k then max_int
  else if s < min_int / k then min_int
  else s * k

(* Whether [shape] has no [0] and more than [max_int] elements. *)
let count_overflows shape =
  let count = ref 1 and over = ref false in
  Array.iter
    (fun d ->
      if d > 0 && !count > max_int / d then over := true
      else count := !count * d)
    shape;
  !over && not (Array.exists (( = ) 0) shape)

(* Whether [view] has no negative dimension, at most [max_int] elements, and
   reaches only positions [0] to [n - 1], for an [n] far below [max_int]. Its
   lowest and highest positions are summed in saturating arithmetic: a sum past
   the ints clamps to an extreme, which no position below [n] is. *)
let view_inside view n =
  let shape = Nx_array.View.shape view
  and strides = Nx_array.View.strides view
  and offset = Nx_array.View.offset view in
  if Array.exists (fun d -> d < 0) shape then false
  else if Array.exists (( = ) 0) shape then true
  else if count_overflows shape then false
  else
    let lo = ref offset and hi = ref offset in
    Array.iteri
      (fun a d ->
        if d > 1 then
          let t = sat_scale strides.(a) (d - 1) in
          if t < 0 then lo := sat_add !lo t else hi := sat_add !hi t)
      shape;
    !lo >= 0 && !hi < n

(* Offsets and strides near [lo] to [hi] and at the edges of the ints: their
   extremes, their neighbours, and powers of two whose products wrap. *)
let edge_ints lo hi =
  Gen.frequency
    [
      (6, Gen.int_range lo hi);
      (1, Gen.int);
      ( 2,
        Gen.of_list ~pp:Format.pp_print_int
          [
            min_int;
            min_int + 1;
            -(1 lsl 32);
            1 lsl 31;
            1 lsl 32;
            1 lsl 61;
            max_int - 1;
            max_int;
          ] );
    ]

(* Dimensions of up to 3 elements, negative ones, and ones whose products
   wrap. *)
let edge_dims =
  Gen.frequency
    [
      (6, Gen.int_range 0 3);
      ( 2,
        Gen.of_list ~pp:Format.pp_print_int
          [ -2; -1; 1 lsl 31; 1 lsl 32; 1 lsl 61; max_int - 1; max_int ] );
    ]

let pp_bounded_view ppf (n, view) =
  Format.fprintf ppf "%d elements, offset %d, strides %a, shape %a" n
    (Nx_array.View.offset view)
    pp_shape
    (Nx_array.View.strides view)
    pp_shape (Nx_array.View.shape view)

(* A storage of up to 8 elements and a view of rank up to 3 over it. *)
let edge_views =
  Gen.with_pp pp_bounded_view
    Gen.(
      let* rank = int_range 0 3 in
      let+ n = int_range 0 8
      and+ shape = array ~size:(constant rank) edge_dims
      and+ strides = array ~size:(constant rank) (edge_ints (-2) 3)
      and+ offset = edge_ints (-3) 8 in
      (n, Nx_array.View.create ~offset ~strides shape))

(* Layouts: ways to hold the same kind of values in a view, composed to reach
   strides, offsets and broadcasts that no single movement gives. *)

type layout = { name : string; apply : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let layout_steps =
  let rows f t =
    if Nx.ndim t = 0 || Nx.dim 0 t = 0 then t else f (Nx.dim 0 t) t
  in
  (* A stepped range gathers a copy; a window of one taken every other position
     is a view. *)
  let every_other axis t =
    if Nx.ndim t = 0 || Nx.dim axis t = 0 then t
    else Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~axis ~window:1 ~step:2 t)
  in
  [
    { name = "transposed"; apply = (fun t -> Nx.transpose t) };
    { name = "flipped"; apply = (fun t -> Nx.flip t) };
    { name = "every other row"; apply = (fun t -> every_other 0 t) };
    {
      name = "without its first row";
      apply = (fun t -> rows (fun n -> Nx.slice [ R (1, n) ]) t);
    };
    {
      name = "without its last row";
      apply = (fun t -> rows (fun n -> Nx.slice [ R (0, n - 1) ]) t);
    };
    { name = "every other column"; apply = (fun t -> every_other (-1) t) };
    {
      name = "broadcast over a new axis";
      apply = (fun t -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape t)) t);
    };
    {
      name = "with a unit axis moved last";
      apply = (fun t -> Nx.moveaxis 0 (Nx.ndim t) (Nx.unsqueeze ~axes:[ 0 ] t));
    };
    {
      name = "in windows of two along its last axis";
      apply =
        (fun t ->
          if Nx.ndim t > 0 && Nx.dim (-1) t >= 2 then
            Nx.sliding_window ~window:2 t
          else t);
    };
  ]

let pp_layout ppf = function
  | [] -> Format.pp_print_string ppf "contiguous"
  | steps ->
      Format.pp_print_list
        ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", then ")
        (fun ppf l -> Format.pp_print_string ppf l.name)
        ppf steps

let layout =
  Gen.with_pp pp_layout
    (Gen.list ~size:(Gen.int_range 0 3) (Gen.of_list layout_steps))

(* Layouts that keep the last axis of a tensor of two axes or more whole:
   reversed, or rows dropped, skipped or repeated. *)
let row_layout =
  let keeps l =
    List.mem l.name
      [
        "flipped";
        "every other row";
        "without its first row";
        "without its last row";
        "broadcast over a new axis";
      ]
  in
  Gen.with_pp pp_layout
    (Gen.list ~size:(Gen.int_range 0 2)
       (Gen.of_list (List.filter keeps layout_steps)))

let lay_out steps t = List.fold_left (fun t l -> l.apply t) t steps

(* The view of [t]'s storage: a host value's, or each device's of a placed
   one. *)
let view t = snd (Nx.shards t)

(* The storage of [t], the first device's for a placed one: views share it,
   copies do not. *)
let storage t = List.hd (fst (Nx.shards t))

(* The elements of [t] in C order, in a host buffer. *)
let elements t = Nx.Op.eval (Read { by = "Nx_test.elements"; x = t })

(* Whether buffers [a] and [b] have a byte of memory in common. *)
let share_memory = Nx_device.Buffer.overlaps

(* Where each element of [t] is in its storage, in row-major order: element
   [idx] is at [offset + sum idx.(d) * strides.(d)] of its view. *)
let positions t =
  let v = view t in
  let st = Nx_array.View.strides v in
  Array.init (Nx.numel t) (fun k ->
      let p = ref (Nx_array.View.offset v) in
      Array.iteri (fun d i -> p := !p + (i * st.(d))) (unravel (Nx.shape t) k);
      !p)

(* Whether the elements of [t] follow each other in its storage. *)
let consecutive t =
  let pos = positions t in
  Array.for_all Fun.id (Array.mapi (fun k p -> p = pos.(0) + k) pos)

(* Whether [shape] can view the elements of [t] where they are: the strides that
   the unit steps of [shape] give must place every element. *)
let viewable t shape =
  let pos = positions t in
  let unit d =
    ravel shape (Array.mapi (fun e _ -> if e = d then 1 else 0) shape)
  in
  Array.length pos = 0
  ||
  let strides =
    Array.mapi (fun d s -> if s > 1 then pos.(unit d) - pos.(0) else 0) shape
  in
  List.for_all
    (fun k ->
      let p = ref pos.(0) in
      Array.iteri (fun d i -> p := !p + (i * strides.(d))) (unravel shape k);
      !p = pos.(k))
    (List.init (Array.length pos) Fun.id)

(* A tensor as nx.mli describes it: a shape and its elements in row-major order.
   Every operation is written from the documented semantics as an index map,
   never from strides, so it is the trusted side of the suites. *)
module Ref = struct
  type 'a t = { shape : int array; data : 'a array }

  let numel shape = Array.fold_left ( * ) 1 shape
  let ndim t = Array.length t.shape

  let create shape data =
    if Array.exists (fun d -> d < 0) shape || Array.length data <> numel shape
    then invalid_arg "create";
    { shape; data }

  let unravel = unravel
  let ravel = ravel

  let init shape f =
    { shape; data = Array.init (numel shape) (fun i -> f (unravel shape i)) }

  let get t idx = t.data.(ravel t.shape idx)
  let of_nx s = { shape = Nx.shape s; data = Nx.to_array s }

  (* The elements of [s] read from its storage where [positions] says. *)
  let of_layout s =
    {
      shape = Nx.shape s;
      data =
        Array.map (Nx_array.Elements.get (Nx.dtype s) (storage s)) (positions s);
    }

  let witness w =
    Testable.contramap (fun t -> (t.shape, t.data)) (pair (array int) (array w))

  let pp pp_elt ppf t =
    Format.fprintf ppf "@[<h>%a %a@]" pp_shape t.shape
      (Format.pp_print_seq
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
         pp_elt)
      (Array.to_seq t.data)

  let axis t a =
    let a' = if a < 0 then a + ndim t else a in
    if a' < 0 || a' >= ndim t then invalid_arg "axis";
    a'

  (* Movement *)

  (* [shape] with its [-1], if it has one, resolved so that it holds [n]
     elements. *)
  let resolve n shape =
    if Array.exists (fun d -> d < -1) shape then invalid_arg "reshape";
    let holes =
      Array.fold_left (fun c d -> if d = -1 then c + 1 else c) 0 shape
    in
    let known =
      Array.fold_left (fun p d -> if d = -1 then p else p * d) 1 shape
    in
    if holes > 1 || (holes = 1 && (known = 0 || n mod known <> 0)) then
      invalid_arg "reshape";
    let shape = Array.map (fun d -> if d = -1 then n / known else d) shape in
    if numel shape <> n then invalid_arg "reshape";
    shape

  let reshape shape t = { t with shape = resolve (numel t.shape) shape }

  let transpose ?axes t =
    let n = ndim t in
    let axes =
      match axes with
      | None -> Array.init n (fun i -> n - 1 - i)
      | Some l -> Array.of_list (List.map (axis t) l)
    in
    if List.sort compare (Array.to_list axes) <> List.init n Fun.id then
      invalid_arg "transpose";
    init
      (Array.map (fun a -> t.shape.(a)) axes)
      (fun idx ->
        let src = Array.make n 0 in
        Array.iteri (fun j a -> src.(a) <- idx.(j)) axes;
        get t src)

  let flip ?axes t =
    let flipped = Array.make (ndim t) (axes = None) in
    Option.iter (List.iter (fun a -> flipped.(axis t a) <- true)) axes;
    init t.shape (fun idx ->
        get t
          (Array.mapi
             (fun d k -> if flipped.(d) then t.shape.(d) - 1 - k else k)
             idx))

  let broadcast_to shape t =
    let n = Array.length shape and m = ndim t in
    if m > n || Array.exists (fun d -> d < 0) shape then
      invalid_arg "broadcast_to";
    Array.iteri
      (fun i d ->
        if d <> 1 && d <> shape.(n - m + i) then invalid_arg "broadcast_to")
      t.shape;
    init shape (fun idx ->
        get t
          (Array.init m (fun i ->
               if t.shape.(i) = 1 then 0 else idx.(n - m + i))))

  let pad widths v t =
    if
      Array.length widths <> ndim t
      || Array.exists (fun (a, b) -> a < 0 || b < 0) widths
    then invalid_arg "pad";
    init
      (Array.mapi (fun d (a, b) -> t.shape.(d) + a + b) widths)
      (fun idx ->
        let src = Array.mapi (fun d k -> k - fst widths.(d)) idx in
        let inside = ref true in
        Array.iteri
          (fun d k -> if k < 0 || k >= t.shape.(d) then inside := false)
          src;
        if !inside then get t src else v)

  let shrink ranges t =
    init
      (Array.map (fun (a, b) -> b - a) ranges)
      (fun idx -> get t (Array.mapi (fun d k -> k + fst ranges.(d)) idx))

  let squeeze ?axes t =
    let drop =
      match axes with
      | None -> Array.map (fun d -> d = 1) t.shape
      | Some l ->
          let drop = Array.make (ndim t) false in
          List.iter
            (fun a ->
              let a = axis t a in
              if t.shape.(a) <> 1 then invalid_arg "squeeze";
              drop.(a) <- true)
            l;
          drop
    in
    let shape =
      List.filteri (fun d _ -> not drop.(d)) (Array.to_list t.shape)
    in
    { t with shape = Array.of_list shape }

  let unsqueeze ?axes t =
    let axes =
      match axes with None -> invalid_arg "unsqueeze" | Some l -> l
    in
    let n = ndim t + List.length axes in
    let axes = List.map (fun a -> if a < 0 then a + n else a) axes in
    if
      List.exists (fun a -> a < 0 || a >= n) axes
      || List.length (List.sort_uniq compare axes) <> List.length axes
    then invalid_arg "unsqueeze";
    let dims = ref (Array.to_list t.shape) in
    let shape =
      Array.init n (fun a ->
          if List.mem a axes then 1
          else
            match !dims with
            | d :: rest ->
                dims := rest;
                d
            | [] -> assert false)
    in
    { t with shape }

  let sliding_window ?axis:(a = -1) ~window ?(step = 1) t =
    let a = axis t a in
    let n = t.shape.(a) and m = ndim t in
    if window < 1 || step < 1 || window > n then invalid_arg "sliding_window";
    let shape =
      Array.mapi
        (fun d s -> if d = a then ((n - window) / step) + 1 else s)
        t.shape
    in
    init (Array.append shape [| window |]) (fun idx ->
        let src = Array.sub idx 0 m in
        src.(a) <- (idx.(a) * step) + idx.(m);
        get t src)

  let concatenate ~axis:a ts =
    match ts with
    | [] -> invalid_arg "concatenate"
    | t0 :: _ ->
        let a = axis t0 a in
        List.iter
          (fun t ->
            if ndim t <> ndim t0 then invalid_arg "concatenate";
            Array.iteri
              (fun d s ->
                if d <> a && s <> t0.shape.(d) then invalid_arg "concatenate")
              t.shape)
          ts;
        let shape = Array.copy t0.shape in
        shape.(a) <- List.fold_left (fun n t -> n + t.shape.(a)) 0 ts;
        init shape (fun idx ->
            let rec find k = function
              | t :: rest ->
                  if k < t.shape.(a) then (t, k)
                  else find (k - t.shape.(a)) rest
              | [] -> assert false
            in
            let t, k = find idx.(a) ts in
            let src = Array.copy idx in
            src.(a) <- k;
            get t src)

  (* [-1] keeps the size of the axis it aligns with, from the right as
     [broadcast_to] aligns. *)
  let expand shape t =
    let n = Array.length shape and m = ndim t in
    if n < m || Array.exists (fun d -> d < -1) shape then invalid_arg "expand";
    let resolve i d =
      if d <> -1 then d
      else if i < n - m then invalid_arg "expand"
      else t.shape.(i - (n - m))
    in
    broadcast_to (Array.mapi resolve shape) t

  let moveaxis src dst t =
    let src = axis t src and dst = axis t dst in
    let rest = List.filter (( <> ) src) (List.init (ndim t) Fun.id) in
    let axes =
      List.filteri (fun i _ -> i < dst) rest
      @ (src :: List.filteri (fun i _ -> i >= dst) rest)
    in
    transpose ~axes t

  let swapaxes a b t =
    let a = axis t a and b = axis t b in
    transpose
      ~axes:
        (List.init (ndim t) (fun i ->
             if i = a then b else if i = b then a else i))
      t

  let flat t = { t with shape = [| numel t.shape |] }

  (* The tensor to work on and its axis: [t] flattened when there is none. *)
  let flat_or_axis t = function None -> (flat t, 0) | Some a -> (t, axis t a)

  let roll ?axis:a shift t =
    let t', a = flat_or_axis t a in
    let n = t'.shape.(a) in
    let rolled =
      init t'.shape (fun idx ->
          let src = Array.copy idx in
          src.(a) <- (((idx.(a) - shift) mod n) + n) mod n;
          get t' src)
    in
    { rolled with shape = t.shape }

  (* nx.mli states no error for fewer repetitions than axes: they are
     refused. *)
  let tile reps t =
    let n = Array.length reps and m = ndim t in
    if n < m || Array.exists (fun r -> r < 0) reps then invalid_arg "tile";
    let s = Array.append (Array.make (n - m) 1) t.shape in
    let t = { t with shape = s } in
    init (Array.map2 ( * ) s reps) (fun idx ->
        get t (Array.mapi (fun d k -> k mod s.(d)) idx))

  let repeat ?axis:a k t =
    if k < 0 then invalid_arg "repeat";
    let t, a = flat_or_axis t a in
    let shape = Array.copy t.shape in
    shape.(a) <- shape.(a) * k;
    init shape (fun idx ->
        let src = Array.copy idx in
        src.(a) <- idx.(a) / k;
        get t src)

  let stack ~axis:a ts =
    match ts with
    | [] -> invalid_arg "stack"
    | t0 :: _ ->
        let n = ndim t0 + 1 in
        let a = if a < 0 then a + n else a in
        if a < 0 || a >= n then invalid_arg "stack";
        concatenate ~axis:a (List.map (unsqueeze ~axes:[ a ]) ts)

  let flatten ~start_dim ~end_dim t =
    let s = axis t start_dim and e = axis t end_dim in
    if s > e then invalid_arg "flatten";
    let sub a b = Array.sub t.shape a (b - a) in
    {
      t with
      shape =
        Array.concat
          [ sub 0 s; [| numel (sub s (e + 1)) |]; sub (e + 1) (ndim t) ];
    }

  let unflatten dim sizes t =
    let d = axis t dim in
    let sizes = resolve t.shape.(d) sizes in
    {
      t with
      shape =
        Array.concat
          [
            Array.sub t.shape 0 d;
            sizes;
            Array.sub t.shape (d + 1) (ndim t - d - 1);
          ];
    }

  (* A part holds the positions of its interval that lie in the axis. *)
  let array_split ~axis:a spec t =
    let a = axis t a in
    let size = t.shape.(a) in
    let bounds =
      match spec with
      | `Count n ->
          if n < 1 then invalid_arg "array_split";
          List.init (n + 1) (fun i -> (i * (size / n)) + Int.min i (size mod n))
      | `Indices l -> (0 :: l) @ [ size ]
    in
    let rec parts = function
      | lo :: (hi :: _ as rest) ->
          let lo = Int.min lo size in
          let hi = Int.max lo (Int.min hi size) in
          shrink
            (Array.mapi (fun d n -> if d = a then (lo, hi) else (0, n)) t.shape)
            t
          :: parts rest
      | _ -> []
    in
    parts bounds

  let split ~axis:a n t =
    if n < 1 || t.shape.(axis t a) mod n <> 0 then invalid_arg "split";
    array_split ~axis:a (`Count n) t

  let fill v t = { t with data = Array.make (Array.length t.data) v }

  (* Indexing. [R] and [Rs] clamp their bounds into the axis as Python slices
     do, which nx.mli does not state. *)

  type sel = Keep of int array | Drop of int | New

  let index dim i =
    let i' = if i < 0 then i + dim else i in
    if i' < 0 || i' >= dim then invalid_arg "index";
    i'

  let range dim start stop step =
    if step = 0 then invalid_arg "slice";
    let lo, hi = if step > 0 then (0, dim) else (-1, dim - 1) in
    let clamp i = Int.max lo (Int.min hi (if i < 0 then i + dim else i)) in
    let stop = clamp stop in
    let rec go i =
      if (step > 0 && i < stop) || (step < 0 && i > stop) then i :: go (i + step)
      else []
    in
    Array.of_list (go (clamp start))

  let select dim : Nx.index -> sel = function
    | I i -> Drop (index dim i)
    | A -> Keep (Array.init dim Fun.id)
    | R (a, b) -> Keep (range dim a b 1)
    | Rs (a, b, s) -> Keep (range dim a b s)
    | L l -> Keep (Array.of_list (List.map (index dim) l))
    | M m ->
        if Nx.ndim m <> 1 || Nx.numel m <> dim then invalid_arg "slice";
        let bits = Nx.to_array m in
        Keep
          (Array.of_list
             (List.filter (fun i -> bits.(i)) (List.init dim Fun.id)))
    | N -> New
    | D (s, len) ->
        if Nx.ndim s <> 0 || len < 0 || len > dim then invalid_arg "slice";
        let first =
          Int.max 0 (Int.min (dim - len) (Int64.to_int (Nx.item [] s)))
        in
        Keep (Array.init len (fun i -> first + i))

  let selection specs t =
    let rec go d = function
      | [] ->
          List.init
            (ndim t - d)
            (fun k -> Keep (Array.init t.shape.(d + k) Fun.id))
      | (Nx.N : Nx.index) :: rest -> New :: go d rest
      | spec :: rest ->
          if d >= ndim t then invalid_arg "slice";
          let s = select t.shape.(d) spec in
          s :: go (d + 1) rest
    in
    let sels = go 0 specs in
    let shape =
      List.filter_map
        (function
          | Keep p -> Some (Array.length p) | Drop _ -> None | New -> Some 1)
        sels
    in
    let source idx =
      let k = ref 0 in
      let src =
        List.filter_map
          (function
            | Keep p ->
                let i = p.(idx.(!k)) in
                incr k;
                Some i
            | Drop i -> Some i
            | New ->
                incr k;
                None)
          sels
      in
      Array.of_list src
    in
    (sels, Array.of_list shape, source)

  let slice specs t =
    let _, shape, source = selection specs t in
    init shape (fun idx -> get t (source idx))

  let set specs v t =
    let sels, shape, source = selection specs t in
    if numel shape > 0 then
      List.iter
        (function
          | Keep p
            when List.length (List.sort_uniq compare (Array.to_list p))
                 <> Array.length p ->
              invalid_arg "set"
          | _ -> ())
        sels;
    let v = broadcast_to shape v in
    let data = Array.copy t.data in
    for i = 0 to numel shape - 1 do
      let idx = unravel shape i in
      data.(ravel t.shape (source idx)) <- get v idx
    done;
    { t with data }

  let item indices t =
    if List.length indices <> ndim t then invalid_arg "item";
    get t (Array.of_list (List.mapi (fun d i -> index t.shape.(d) i) indices))

  let take ?axis:a ~zero indices t =
    let t, a = flat_or_axis t a in
    let n = t.shape.(a) in
    let shape = Array.copy t.shape in
    shape.(a) <- Array.length indices;
    init shape (fun idx ->
        let k = indices.(idx.(a)) in
        if k < 0 || k >= n then zero
        else
          let src = Array.copy idx in
          src.(a) <- k;
          get t src)

  let compress ?axis:a condition t =
    let t, a = flat_or_axis t a in
    if Array.length condition <> t.shape.(a) then invalid_arg "compress";
    let kept =
      List.filter (fun i -> condition.(i)) (List.init t.shape.(a) Fun.id)
    in
    slice (List.init a (fun _ -> Nx.A) @ [ Nx.L kept ]) t

  (* Elementwise *)

  (* [along ~axis ~length f t] replaces each lane of [t] along [axis] by the
     [length] elements [f] gives for it. *)
  let along ~axis:a ~length f t =
    let a = axis t a in
    let rest = Array.copy t.shape in
    rest.(a) <- 1;
    let lanes =
      Array.init (numel rest) (fun l ->
          let idx = unravel rest l in
          f
            (Array.init t.shape.(a) (fun k ->
                 let i = Array.copy idx in
                 i.(a) <- k;
                 get t i)))
    in
    let shape = Array.copy t.shape in
    shape.(a) <- length;
    init shape (fun idx ->
        let i = Array.copy idx in
        i.(a) <- 0;
        lanes.(ravel rest i).(idx.(a)))

  (* [reduce ?axes ?keepdims f init t] folds [f] over [axes], one axis after the
     other, from [init]. *)
  let reduce ?axes ?(keepdims = false) f init t =
    let axes =
      match axes with
      | None -> List.init (ndim t) Fun.id
      | Some l -> List.sort_uniq compare (List.map (axis t) l)
    in
    let kept =
      List.fold_left
        (fun t a ->
          along ~axis:a ~length:1
            (fun lane -> [| Array.fold_left f init lane |])
            t)
        t axes
    in
    if keepdims then kept
    else
      let shape =
        List.filteri
          (fun d _ -> not (List.mem d axes))
          (Array.to_list kept.shape)
      in
      { kept with shape = Array.of_list shape }

  let map f t = { shape = t.shape; data = Array.map f t.data }

  let broadcast_shapes a b =
    let n = Int.max (Array.length a) (Array.length b) in
    let dim s i =
      let k = i - (n - Array.length s) in
      if k < 0 then 1 else s.(k)
    in
    Array.init n (fun i ->
        match (dim a i, dim b i) with
        | x, y when x = y -> x
        | 1, y -> y
        | x, 1 -> x
        | _ -> invalid_arg "broadcast")

  let map2 f a b =
    let shape = broadcast_shapes a.shape b.shape in
    let a = broadcast_to shape a and b = broadcast_to shape b in
    { shape; data = Array.map2 f a.data b.data }
end

(* A tensor of [dtype] under some layout, its elements drawn from [value]. It
   prints as its layout and its elements. *)
let viewed ?(shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4))
    ?(layout = layout) ~pp dtype value =
  let drawn =
    let open Gen in
    let* s = shape in
    let* steps = layout in
    let+ xs = array ~size:(constant (Ref.numel s)) value in
    (steps, lay_out steps (Nx.create dtype s xs))
  in
  Gen.map snd
    (Gen.with_pp
       (fun ppf (steps, t) ->
         Format.fprintf ppf "%a: %a" pp_layout steps (Ref.pp pp) (Ref.of_nx t))
       drawn)

(* The integer dtypes, their values seen as int64: [bits] and [signed] say where
   they wrap and how they order. *)
type int_dtype =
  | Int_dtype : {
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      bits : int;
      signed : bool;
      to_i64 : 'a -> int64;
      of_i64 : int64 -> 'a;
      exact : 'a testable;
    }
      -> int_dtype

let wrap ~bits ~signed v =
  if bits = 64 then v
  else
    let m = Int64.shift_left 1L bits in
    let x = Int64.logand v (Int64.pred m) in
    if signed && Int64.compare x (Int64.shift_right m 1) >= 0 then Int64.sub x m
    else x

let int_dtypes =
  let small name dtype bits signed =
    Int_dtype
      {
        name;
        dtype;
        bits;
        signed;
        to_i64 = Int64.of_int;
        of_i64 = Int64.to_int;
        exact = int;
      }
  in
  [
    small "int8" Nx.int8 8 true;
    small "uint8" Nx.uint8 8 false;
    small "int16" Nx.int16 16 true;
    small "uint16" Nx.uint16 16 false;
    Int_dtype
      {
        name = "int32";
        dtype = Nx.int32;
        bits = 32;
        signed = true;
        to_i64 = Int64.of_int32;
        of_i64 = Int64.to_int32;
        exact = int32;
      };
    Int_dtype
      {
        name = "uint32";
        dtype = Nx.uint32;
        bits = 32;
        signed = false;
        to_i64 = (fun v -> Int64.logand (Int64.of_int32 v) 0xFFFF_FFFFL);
        of_i64 = Int64.to_int32;
        exact = int32;
      };
    Int_dtype
      {
        name = "int64";
        dtype = Nx.int64;
        bits = 64;
        signed = true;
        to_i64 = Fun.id;
        of_i64 = Fun.id;
        exact = int64;
      };
    Int_dtype
      {
        name = "uint64";
        dtype = Nx.uint64;
        bits = 64;
        signed = false;
        to_i64 = Fun.id;
        of_i64 = Fun.id;
        exact = int64;
      };
  ]

(* The 4-bit integer dtypes, which compute as the others. *)
let int4_dtypes =
  let nibble name dtype signed =
    Int_dtype
      {
        name;
        dtype;
        bits = 4;
        signed;
        to_i64 = Int64.of_int;
        of_i64 = Int64.to_int;
        exact = int;
      }
  in
  [ nibble "int4" Nx.int4 true; nibble "uint4" Nx.uint4 false ]

(* The least and greatest values of a width, as int64. *)
let int_range ~bits ~signed =
  if signed then
    ( Int64.neg (Int64.shift_left 1L (bits - 1)),
      Int64.pred (Int64.shift_left 1L (bits - 1)) )
  else (0L, if bits = 64 then -1L else Int64.pred (Int64.shift_left 1L bits))

(* Values of a width that break arithmetic: its ends, their neighbours, zero,
   ones, and values in between. *)
let int_value ~bits ~signed =
  let lo, hi = int_range ~bits ~signed in
  let wrapped = Gen.map (fun v -> wrap ~bits ~signed (Int64.of_int v)) in
  Gen.frequency
    [
      (4, wrapped (Gen.int_range (-9) 9));
      (2, Gen.map (wrap ~bits ~signed) Gen.int64);
      ( 1,
        Gen.of_list
          ~pp:(fun ppf v -> Format.fprintf ppf "%Ld" v)
          [ lo; hi; Int64.succ lo; Int64.pred hi; 0L; 1L ] );
    ]

let int_compare ~signed a b =
  if signed then Int64.compare a b else Int64.unsigned_compare a b

(* nx.cpu's kernels under another name: a backend of its own, which a test pairs
   with a device whose default backend is nx.cpu, and which owns no memory. *)
module Renamed = struct
  include (val Nx_backend.kernels Nx_cpu.backend)

  let name = "nx.cpu renamed"
  let owns _ = false
end

let renamed = Nx_backend.v (module Renamed)

(* Test devices: nx.cpu over memories of the host's, whose statistics count the
   bytes they receive and send. *)
module Devices = struct
  let memory name =
    Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
      (Host_visible { memory = Nx_device.Driver.host_memory; mapping = None })

  let memories = List.map memory [ "TEST:1"; "TEST:2"; "TEST:3"; "TEST:4" ]
  let runtimes = List.map Nx.Device.make memories

  let d1, d2, d3, d4 =
    match runtimes with
    | [ d1; d2; d3; d4 ] -> (d1, d2, d3, d4)
    | _ -> assert false

  (* A device beside the four. *)
  let other = Nx.Device.make (memory "OTHER")

  let total count =
    List.fold_left (fun n r -> n + count (Nx_device.stats r)) 0 memories

  (* The bytes the four devices have received, and sent. *)
  let bytes_in () = total Nx_device.Stats.bytes_in
  let bytes_out () = total Nx_device.Stats.bytes_out
  let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal
  let device = Testable.make ~pp:Nx.Device.pp ~equal:Nx.Device.equal

  (* The storage of the placed value [x]. *)
  let storage_of x =
    match Nx.Repr.v x with
    | Placed p -> Nx.Repr.Placed.storage p
    | Host _ | Traced _ -> fail "expected a placed value"

  (* [s] consumed at [path], as a compiled call consumes it. *)
  let consume s ~path =
    Nx.Repr.Storage.borrow s;
    Nx.Repr.Storage.upgrade s;
    Nx.Repr.Storage.consume s ~path;
    ignore (Nx.Repr.Storage.finish s);
    Nx.Repr.Storage.release s
end

(* Tensors of every dtype as they are stored: drawn as bit patterns under every
   layout, and compared bit for bit. *)
module Stored = struct
  (* Tensors compare bit for bit: dtype, shape and the bytes of the elements in
     row-major order. *)

  let storage (Nx.P t) =
    let bytes = Nx_device.Buffer.bigarray Bigarray.char (elements t) in
    ( Nx_dtype.to_string (Nx.dtype t),
      Nx.shape t,
      String.init (Bigarray.Array1.dim bytes) (Bigarray.Array1.get bytes) )

  let pp_packed ppf (Nx.P t as p) =
    let _, _, bytes = storage p in
    Format.fprintf ppf "%a (bytes %S)" Nx.pp t bytes

  let packed =
    Testable.make ~pp:pp_packed ~equal:(fun a b -> storage a = storage b)

  (* A dtype, tensors of it under every layout, and the witness of their values
     that a text file keeps: every NaN equal to every NaN. *)
  type case =
    | Case : {
        name : string;
        dtype : ('a, 'b) Nx.dtype;
        tensors : ('a, 'b) Nx.t Gen.t;
        values : ('a, 'b) Nx.t testable;
      }
        -> case

  let case name dtype tensors values = Case { name; dtype; tensors; values }
  let shape = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 4)
  let pp_bits ppf v = Format.fprintf ppf "0x%Lx" v
  let ones n = Int64.pred (Int64.shift_left 1L n)

  (* The bit patterns of a float format with [e] exponent and [m] significand
     bits: signed zeros, the least and greatest subnormals, the least normal,
     the greatest finite, infinities, quiet and signalling NaNs with payloads,
     and any pattern. *)
  let float_bits ~e ~m =
    let inf = Int64.shift_left (ones e) m in
    let nans =
      [ Int64.logor inf (Int64.shift_left 1L (m - 1)); Int64.succ inf ]
    in
    let corners =
      [ 0L; 1L; ones m; Int64.shift_left 1L m; Int64.pred inf; inf ] @ nans
    in
    let sign = Int64.logor (Int64.shift_left 1L (e + m)) in
    let mask = if e + m = 63 then -1L else ones (1 + e + m) in
    Gen.frequency
      [
        (1, Gen.of_list ~pp:pp_bits (corners @ List.map sign corners));
        (2, Gen.with_pp pp_bits (Gen.map (Int64.logand mask) Gen.int64));
      ]

  (* Tensors of [dtype] whose elements are the bits [bits] draws, read through
     the integer dtype [name] of the same width. *)
  let from_bits name dtype bits =
    match List.find (fun (Int_dtype d) -> d.name = name) int_dtypes with
    | Int_dtype u ->
        let drawn =
          let open Gen in
          let* s = shape in
          let* steps = layout in
          let+ xs = array ~size:(constant (Ref.numel s)) bits in
          (steps, Ref.create s xs)
        in
        let pp ppf (steps, r) =
          Format.fprintf ppf "%a: %a" pp_layout steps (Ref.pp pp_bits) r
        in
        Gen.map
          (fun (steps, (r : int64 Ref.t)) ->
            let words = Nx.create u.dtype r.shape (Array.map u.of_i64 r.data) in
            lay_out steps (Nx.bitcast dtype words))
          (Gen.with_pp pp drawn)

  let float_tensors dtype ~e ~m =
    from_bits ("uint" ^ string_of_int (1 + e + m)) dtype (float_bits ~e ~m)

  let float_case name dtype ~e ~m =
    case name dtype (float_tensors dtype ~e ~m) (tensor float_exact)

  let ints =
    List.map
      (fun (Int_dtype d) ->
        let value = int_value ~bits:d.bits ~signed:d.signed in
        case d.name d.dtype (from_bits d.name d.dtype value) (tensor d.exact))
      int_dtypes

  let bool =
    let pp = Format.pp_print_bool in
    case "bool" Nx.bool (viewed ~pp Nx.bool Gen.bool) (tensor bool)

  let float16 = float_case "float16" Nx.float16 ~e:5 ~m:10
  let bfloat16 = float_case "bfloat16" Nx.bfloat16 ~e:8 ~m:7
  let float32 = float_case "float32" Nx.float32 ~e:8 ~m:23
  let float64s = float_tensors Nx.float64 ~e:11 ~m:52
  let float64 = case "float64" Nx.float64 float64s (tensor float_exact)
  let float8_e4m3 = float_case "float8_e4m3" Nx.float8_e4m3 ~e:4 ~m:3
  let float8_e5m2 = float_case "float8_e5m2" Nx.float8_e5m2 ~e:5 ~m:2

  let complex_exact =
    Testable.contramap
      (fun (c : Complex.t) -> (c.re, c.im))
      (pair float_exact float_exact)

  (* A complex64 element is two float32 bit patterns, real part first. *)
  let complex64s =
    let f32 = float_bits ~e:8 ~m:23 in
    let word (re, im) = Int64.logor (Int64.shift_left im 32) re in
    from_bits "int64" Nx.complex64 (Gen.map word (Gen.pair f32 f32))

  let complex64 =
    case "complex64" Nx.complex64 complex64s (tensor complex_exact)

  let complex128s =
    let pp ppf (c : Complex.t) = Format.fprintf ppf "%h%+hi" c.re c.im in
    let value (re, im) = { Complex.re; im } in
    let values = Gen.map value (Gen.pair Gen.any_float Gen.any_float) in
    viewed ~pp Nx.complex128 values

  let complex128 =
    case "complex128" Nx.complex128 complex128s (tensor complex_exact)

  let every =
    (bool :: ints)
    @ [
        float8_e4m3;
        float8_e5m2;
        float16;
        bfloat16;
        float32;
        float64;
        complex64;
        complex128;
      ]
end

(* The contract of runtime devices, checked on the runtimes of a suite: test
   runtimes over host memory, Metal, CUDA. A value of every dtype placed on one
   runtime, copied on all or split along an axis reads back bit for bit, each
   runtime receiving the bytes of its window; an operation gives the host's
   result on the elements placed, placed where its operand is; and an allocation
   a runtime cannot make raises Out_of_memory with its device and bytes. *)
module Runtimes = struct
  open Stored

  let host x = Nx.place Nx.Placement.host x

  (* Values of the dtypes narrower than a byte, 4-bit integers two to a byte and
     bits eight: they compare by value, since the bits of a last byte past the
     last element belong to no element. *)
  let narrow =
    let pp = Format.pp_print_int in
    let nibbles name dtype lo hi =
      case name dtype (viewed ~pp dtype (Gen.int_range lo hi)) (tensor int)
    in
    [
      nibbles "int4" Nx.int4 (-8) 7;
      nibbles "uint4" Nx.uint4 0 15;
      case "bit" Nx.bit
        (viewed ~pp:Format.pp_print_bool Nx.bit Gen.bool)
        (tensor Windtrap.bool);
    ]

  (* Each device alone, a copy on all, and a split along each axis that divides
     among them. *)
  let placements ds shape =
    let n = List.length ds in
    let splits =
      List.filter
        (fun a -> n > 1 && shape.(a) mod n = 0)
        (List.init (Array.length shape) Fun.id)
    in
    Gen.of_list ~pp:Nx.Placement.pp
      (List.map Nx.Placement.on ds
      @ (if n > 1 then [ Nx.Placement.replicated ds ] else [])
      @ List.map (fun axis -> Nx.Placement.sharded ~axis ds) splits)

  let placed ds tensors =
    Gen.with_pp
      (fun ppf (t, p) ->
        Format.fprintf ppf "%a at %a" Nx.pp t Nx.Placement.pp p)
      (Gen.bind tensors (fun t ->
           Gen.map (fun p -> (t, p)) (placements ds (Nx.shape t))))

  (* The bytes of the elements of [x] that [d] holds at [p], as stored. *)
  let window_bytes p x d =
    if List.exists (Nx.Device.equal d) (Nx.Placement.devices p) then
      let window = Nx.Placement.window p (Nx.shape x) d in
      let n = Array.fold_left (fun n (lo, hi) -> n * (hi - lo)) 1 window in
      let bits = Nx_dtype.Scalar.(bitsize (of_dtype (Nx.dtype x))) in
      ((n * bits) + 7) / 8
    else 0

  let operations =
    [
      ("neg", Nx.neg);
      ("x + x", fun x -> Nx.add x x);
      ("exp", Nx.exp);
      ("sum", fun x -> Nx.sum x);
      ("transpose", fun x -> Nx.transpose x);
      ("flip", fun x -> Nx.flip x);
      ("flatten", fun x -> Nx.flatten x);
    ]

  (* [x] written to a fresh file, as a value on the disk over it. *)
  let on_disk x =
    let module B = Nx_device.Buffer in
    let src = elements x and path = temp_file () in
    let pp = Format.pp_print_string in
    B.copy ~src ~dst:(require_ok ~pp (B.create_file path (B.nbytes src)));
    Nx.of_buffer (Nx.dtype x) (Nx.shape x)
      (B.view
         (require_ok ~pp (B.of_file path))
         ~offset:0 (B.dtype src) (B.length src))

  (* [f ()] with no room in [r]'s budget. *)
  let full r f =
    let budget = Nx_device.budget r in
    Fun.protect ~finally:(fun () -> Nx_device.set_budget r budget) @@ fun () ->
    Nx_device.set_budget r 0;
    f ()

  (* [laws ms] checks the runtimes [ms], each a memory: operations on its device
     compute where nx.cpu computes on it, and are refused otherwise. *)
  let laws = function
    | [] -> [ test "on no runtime" (fun () -> skip ~reason:"no device" ()) ]
    | ms ->
        let ds = List.map Nx.Device.make ms in
        let received f =
          let before = List.map Nx_device.stats ms in
          let y = f () in
          let bytes_in r s =
            Nx_device.Stats.(bytes_in (diff s (Nx_device.stats r)))
          in
          (y, List.map2 bytes_in ms before)
        in
        (* A value on the disk is borrowed from its file's pages by devices
           whose memory the host addresses, which receive no byte, unless a
           window of elements narrower than a byte starts inside a byte. *)
        let borrows = List.for_all Nx_device.shares_host_memory ms in
        let round_trip ~disk (Case c) =
          prop
            (c.name ^ " values"
            ^ (if disk then " on the disk" else "")
            ^ " read back bit for bit, each device receiving its window")
            (placed ds c.tensors)
            (fun (x, p) ->
              let source = if disk then on_disk x else x in
              let y, bytes = received (fun () -> Nx.place p source) in
              equal Devices.placement p (Nx.placement y);
              let windows = List.map (window_bytes p x) ds in
              (match c.dtype with
              | (Int4 | UInt4 | Bit) when disk && borrows ->
                  is_true ~msg:"bytes received: none, or the windows"
                    (List.for_all (( = ) 0) bytes || bytes = windows)
              | _ when disk && borrows ->
                  equal ~msg:"bytes received" (list int)
                    (List.map (fun _ -> 0) ds)
                    bytes
              | _ -> equal ~msg:"bytes received" (list int) windows bytes);
              match c.dtype with
              | Int4 | UInt4 | Bit -> equal c.values x (host y)
              | _ -> equal packed (Nx.P x) (Nx.P (host y)))
        in
        let ops =
          List.map Nx.Placement.on ds
          @ if List.length ds > 1 then [ Nx.Placement.replicated ds ] else []
        in
        let m = List.hd ms in
        let on_d = Nx.Placement.on (List.hd ds) in
        let out_of_memory n = function
          | Nx_device.Out_of_memory (m', k) -> Nx_device.equal m m' && k = n
          | _ -> false
        in
        let computing =
          match List.for_all Nx_device.runs_on_host ms with
          | true ->
              [
                prop
                  "an operation gives the host's result on the elements \
                   placed, placed where its operand is"
                  (Gen.triple
                     (Gen.of_list
                        ~pp:(fun ppf (name, _) ->
                          Format.pp_print_string ppf name)
                        operations)
                     (float_tensors Nx.float32 ~e:8 ~m:23)
                     (Gen.of_list ~pp:Nx.Placement.pp ops))
                  (fun ((_, f), x, p) ->
                    let y = f (Nx.place p x) in
                    equal Devices.placement p (Nx.placement y);
                    equal packed (Nx.P (f (Nx.copy x))) (Nx.P (host y)));
                test
                  "an operation's result the runtime cannot allocate raises \
                   Out_of_memory with the device and its bytes" (fun () ->
                    let x = Nx.place on_d (Nx.zeros Nx.float32 [| 4 |]) in
                    full m @@ fun () ->
                    raises_match (out_of_memory 16) (fun () -> Nx.add x x);
                    (* Held through the addition: the refused allocation
                       collects garbage and tries again, and could take a dead
                       [x]'s memory. *)
                    ignore (Sys.opaque_identity x));
              ]
          | false ->
              [
                test
                  "an operation on the runtime raises before any work, naming \
                   the remedies" (fun () ->
                    let x = Nx.place on_d (Nx.ones Nx.float32 [| 4 |]) in
                    raises_match
                      (function
                        | Invalid_argument why ->
                            String.starts_with
                              ~prefix:
                                ("Nx.add: " ^ Nx_device.name m
                               ^ " has no eager kernels")
                              why
                            && String.ends_with
                                 ~suffix:
                                   "or place the operands on Nx.Placement.host."
                                 why
                        | _ -> false)
                      (fun () -> Nx.add x x));
              ]
        in
        [
          group "placing" (List.map (round_trip ~disk:false) (every @ narrow));
          group "placing from the disk"
            (List.map (round_trip ~disk:true) (every @ narrow));
          test
            "a placement the runtime cannot allocate raises Out_of_memory with \
             the device and its bytes" (fun () ->
              full m @@ fun () ->
              raises_match (out_of_memory 400) (fun () ->
                  Nx.place on_d (Nx.zeros Nx.float32 [| 100 |])));
        ]
        @ computing
end

module Profiles = struct
  module P = Nx_device.Profile
  module B = Nx_device.Buffer

  let profiled f =
    let p = P.start () in
    match f () with
    | () -> P.stop p
    | exception e ->
        ignore (P.stop p);
        raise e

  (* Copies from the host to [d] and back, staged and through a borrow: each is
     a span of the host named after its devices, and each that [d]'s copy queue
     ran is also a span of its copy lane within the host's, [slack] nanoseconds
     either side allowed for the calibration of [d]'s clock. *)
  let copies ?(slack = 0) = function
    | [] -> [ test "on no device" (fun () -> skip ~reason:"no device" ()) ]
    | ds ->
        List.map
          (fun d ->
            let name = Nx_device.name d in
            test
              (name
             ^ "'s copies are spans of the host, and of its copy lane within \
                them") (fun () ->
                let host = Nx_device.host and u8 = Nx_dtype.Scalar.UInt8 in
                let small = B.create host u8 4096
                and big = B.create host u8 (1 lsl 20) in
                let small_d = B.create d u8 4096
                and big_d = B.create d u8 (1 lsl 20) in
                let borrowed =
                  require_ok ~pp:Format.pp_print_string (B.borrow d big)
                in
                let events =
                  profiled (fun () ->
                      B.copy ~src:small ~dst:small_d;
                      B.copy ~src:small_d ~dst:small;
                      B.copy ~src:big ~dst:big_d;
                      B.copy ~src:big_d ~dst:big)
                in
                ignore (Sys.opaque_identity borrowed);
                let spans on lane =
                  List.filter_map
                    (function
                      | P.Span s when Nx_device.equal s.device on && lane s.lane
                        ->
                          Some (s.name, s.start, s.stop)
                      | _ -> None)
                    events
                in
                let hosted = spans host (String.starts_with ~prefix:"domain ")
                and queued = spans d (String.equal "copy") in
                let names = [ "CPU -> " ^ name; name ^ " -> CPU" ] in
                let names = names @ names in
                let addressed =
                  Nx_device.Driver.Region.(host_address (of_buffer small_d))
                  <> None
                in
                let name (n, _, _) = n in
                equal ~msg:"host spans" (list string) names
                  (List.map name hosted);
                equal ~msg:"copy lane spans" (list string)
                  (if addressed then [] else names)
                  (List.map name queued);
                if not addressed then
                  List.iter2
                    (fun (n, t0, t1) (_, s0, s1) ->
                      is_true
                        ~msg:
                          (Printf.sprintf "%s: %d to %d within %d to %d" n s0 s1
                             t0 t1)
                        (s0 <= s1 && t0 - slack <= s0 && s1 <= t1 + slack))
                    hosted queued))
          ds
end

(* Test memories that a test loses: ["FAULTY:k"] holds its values in the host's
   memory, which nx.cpu computes on, as [Nx.Device.cpu k] does, and its driver
   reports the fault a test names at its next synchronization, as a GPU's driver
   reports one at a wait. [device k] is one memory until it is lost, and a fresh
   one after, as a vendor's library reopens a lost GPU. *)
module Faulty = struct
  type memory = { memory : Nx_device.t; fault : string option Atomic.t }

  let lock = Mutex.create ()
  let current : (int, memory) Hashtbl.t = Hashtbl.create 4
  let made = ref []

  let open_memory k =
    let fault = Atomic.make None in
    let memory =
      Nx_device.Driver.device
        ~name:(Printf.sprintf "FAULTY:%d" k)
        ~arch:"test" ~budget:max_int
        ~synchronized:(fun () -> Option.iter failwith (Atomic.get fault))
        (Host_visible
           { memory = Nx_device.Driver.host_memory; mapping = Some Identity })
    in
    let m = { memory; fault } in
    Hashtbl.replace current k m;
    made := m :: !made;
    m

  let device k =
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt current k with
    | Some m when Nx_device.lost m.memory = None -> Nx.Device.make m.memory
    | Some _ | None -> Nx.Device.make (open_memory k).memory

  (* [lose d why] loses [d]'s memory with the fault [why], which its driver
     reports at the synchronization this runs. *)
  let lose d why =
    let memory = Nx.Device.memory d in
    match
      Mutex.protect lock (fun () ->
          List.find_opt (fun m -> m.memory == memory) !made)
    with
    | None -> invalid_arg "Nx_test.Faulty.lose: not a faulty memory"
    | Some m -> (
        Atomic.set m.fault (Some why);
        try Nx_device.synchronize memory with Nx_device.Lost _ -> ())
end

module Accuracy = Accuracy

(* Special-function goldens

   A golden of [golden/special/] lists, per dtype, the arguments of a function
   and its value correctly rounded to the dtype, as hexadecimal floats, and for
   some functions a last column the function's bound reads: an inverse's
   condition number, or the scale of an absolute bound. *)

module Special = struct
  (* [overflow] when [value] is an infinity a finite exact value rounds to past
     the dtype's range. *)
  type row = {
    args : float array;
    value : float;
    overflow : bool;
    scale : float;
  }

  (* The rows of [file] at the dtype named [short] ("f32", "f64"). *)
  let rows file short =
    let path = Filename.concat (Filename.dirname Sys.executable_name) file in
    let lines = In_channel.with_open_bin path In_channel.input_lines in
    match
      List.filter (fun l -> not (String.starts_with ~prefix:"#" l)) lines
    with
    | [] -> failwith (file ^ ": no header")
    | header :: lines ->
        (* The columns are the dtype, the arguments, [f] and maybe a scale. *)
        let columns = String.split_on_char '\t' header in
        let scaled = List.nth columns (List.length columns - 1) <> "f" in
        let n = List.length columns - if scaled then 3 else 2 in
        List.filter_map
          (fun line ->
            match String.split_on_char '\t' line with
            | dtype :: cells when dtype = short ->
                let overflow =
                  match List.nth cells n with
                  | "overflow" | "-overflow" -> true
                  | _ -> false
                in
                let float = function
                  | "overflow" -> Float.infinity
                  | "-overflow" -> Float.neg_infinity
                  | cell -> float_of_string cell
                in
                let cells = Array.of_list (List.map float cells) in
                Some
                  {
                    args = Array.sub cells 0 n;
                    value = cells.(n);
                    overflow;
                    scale = (if scaled then cells.(n + 1) else Float.nan);
                  }
            | _ -> None)
          lines

  (* The forms of a bound: [k] ulps; [2k (1 + |log f|) + 4] ulps, the
     exponential of a logarithm within [k] ulps, absolute near zeros; [k] ulps
     or [|f̂ - f| <= a eps] where [|f| < 1]; [k] ulps or [|f̂ - f| <= k eps
     scale]; an inverse's [k] ulps plus its condition number times its forward
     function's bound; and a relative bound at float64 and at float32, alone or
     times the scale. *)
  type bound =
    | Ulps of int
    | Log_ulps of int
    | Near_zeros of int * int
    | Scaled of int
    | Inverse of int * int
    | Relative of float * float
    | Relative_scaled of float * float

  let pp_bound ppf = function
    | Ulps k -> Format.fprintf ppf "%d ulps" k
    | Near_zeros (k, a) ->
        Format.fprintf ppf "%d ulps, or %d eps where below 1" k a
    | Scaled k -> Format.fprintf ppf "%d ulps, or %d eps times the scale" k k
    | Inverse (k, f) -> Format.fprintf ppf "%d + kappa * %d ulps" k f
    | Log_ulps k -> Format.fprintf ppf "%d (1 + |log f|) + 4 ulps" (2 * k)
    | Relative (r64, r32) ->
        Format.fprintf ppf "%g relative (%g at float32)" r64 r32
    | Relative_scaled (r64, r32) ->
        Format.fprintf ppf "%g relative (%g at float32) times the scale" r64 r32

  (* The position of a float of [width] bits [b] among its format's values in
     order, [-0.] just below [+0.]. *)
  let rank ~width b = Accuracy.position width b

  let rank_of ~f32 x =
    if f32 then
      rank ~width:32
        (Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xffff_ffffL)
    else rank ~width:64 (Int64.bits_of_float x)

  (* [ulps ~f32 a b] is the distance between [a] and [b] in floats, as a float
     so that it reaches past [max_int]: zero between two NaNs, infinite between
     a NaN and a number. *)
  let ulps ~f32 a b =
    if Float.is_nan a || Float.is_nan b then
      if Float.is_nan a && Float.is_nan b then 0. else Float.infinity
    else
      Int64.to_float (Int64.abs (Int64.sub (rank_of ~f32 a) (rank_of ~f32 b)))

  (* [log_ulps k f] is [2k (1 + |log f|) + 4], [Log_ulps k]'s bound at [f]. *)
  let log_ulps k f =
    (2. *. Float.of_int k *. (1. +. Float.abs (Stdlib.log (Float.abs f)))) +. 4.

  (* Whether [got] meets [bound] against the correctly rounded [row]: exactly
     where the value is an exact infinity or NaN, and with its sign where it is
     zero. An overflow, read as the infinity one rank past the largest float, is
     held by rank to the bound in ulps, since its absolute error is infinite. *)
  let within ?(zeros = `Signed) ~f32 bound row got =
    let eps = if f32 then 0x1p-23 else 0x1p-52 in
    let r = row.value in
    let d = ulps ~f32 r got in
    let error = Float.abs (got -. r) in
    if (not (Float.is_finite r)) && not row.overflow then d = 0.
    else if zeros = `Signed && r = 0. && Float.sign_bit got <> Float.sign_bit r
    then false
    else if row.overflow then
      d
      <=
      match bound with
      | Ulps k | Near_zeros (k, _) | Scaled k -> Float.of_int k
      | Log_ulps k -> log_ulps k r
      | Inverse (k, forward) ->
          Float.of_int k +. (row.scale *. Float.of_int forward)
      | Relative (r64, r32) -> (if f32 then r32 else r64) /. eps
      | Relative_scaled (r64, r32) ->
          (if f32 then r32 else r64) *. row.scale /. eps
    else
      match bound with
      | Ulps k -> d <= Float.of_int k
      | Near_zeros (k, a) ->
          d <= Float.of_int k
          || (Float.abs r < 1. && error <= Float.of_int a *. eps)
      | Scaled k ->
          d <= Float.of_int k || error <= Float.of_int k *. eps *. row.scale
      | Inverse (k, forward) ->
          d <= Float.of_int k +. (row.scale *. Float.of_int forward)
      | Log_ulps k -> d <= log_ulps k r
      | Relative (r64, r32) ->
          d = 0. || error <= (if f32 then r32 else r64) *. Float.abs r
      | Relative_scaled (r64, r32) ->
          d = 0.
          || error <= (if f32 then r32 else r64) *. row.scale *. Float.abs r

  type f = { f : 'b. (float, 'b) Nx.t array -> (float, 'b) Nx.t }

  let eval (type b) (dt : (float, b) Nx.dtype) { f } rows =
    let n = List.length rows in
    let arity = match rows with [] -> 0 | r :: _ -> Array.length r.args in
    let args =
      Array.init arity (fun i ->
          Nx.create dt [| n |]
            (Array.of_list (List.map (fun r -> r.args.(i)) rows)))
    in
    Nx.to_array (Nx.cast Nx.float64 (f args))

  (* A row with a subnormal argument or value, which a device that flushes
     subnormals reads or writes as zero. *)
  let subnormal row =
    let sub x = x <> 0. && Float.abs x < 0x1p-126 in
    sub row.value || Array.exists sub row.args

  (* [check ~bound file f] holds [f] to [bound args] at every row of [file] that
     [keep] keeps (all by default), at each of [dtypes] (both by default), and
     fails listing the rows that miss it, worst first. [zeros] says whether a
     zero is held to its sign ([`Signed], by default) or not ([`Unsigned], for a
     derivative, whose zeros nx does not sign). *)
  let check ?(keep = fun _ -> true) ?(dtypes = [ `F32; `F64 ]) ?zeros ~bound
      file f =
    List.map
      (fun dtype ->
        let short, f32 =
          match dtype with `F32 -> ("f32", true) | `F64 -> ("f64", false)
        in
        Windtrap.test short (fun () ->
            let rows = List.filter keep (rows file short) in
            let got =
              if f32 then eval Nx.float32 f rows else eval Nx.float64 f rows
            in
            let misses =
              List.concat
                (List.mapi
                   (fun i row ->
                     let b = bound row.args in
                     if within ?zeros ~f32 b row got.(i) then []
                     else [ (ulps ~f32 row.value got.(i), row, got.(i), b) ])
                   rows)
            in
            let misses =
              List.sort (fun (a, _, _, _) (b, _, _, _) -> compare b a) misses
            in
            match misses with
            | [] -> ()
            | _ ->
                let pp_miss ppf (d, row, got, b) =
                  Format.fprintf ppf "at (%a): %h, not %h (%g ulps), bound %a"
                    (Format.pp_print_list
                       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
                       (fun ppf x -> Format.fprintf ppf "%h" x))
                    (Array.to_list row.args) got row.value d pp_bound b;
                  if Float.is_finite row.scale then
                    Format.fprintf ppf " (scale %g)" row.scale
                in
                Windtrap.failf "%d of %d rows miss their bound:@\n%a"
                  (List.length misses) (List.length rows)
                  (Format.pp_print_list pp_miss)
                  (List.filteri (fun i _ -> i < 8) misses)))
      dtypes

  (* Narrow floats

     A narrow float computes at float32 and rounds once. Its reference is the
     float64 result, whose own bound the goldens hold: rounded to the narrow
     dtype it is the correctly rounded value, or one of its neighbours when the
     float64 result lies within its bound of a narrow midpoint. *)

  type narrow =
    | Narrow : string * (float, 'b) Nx.dtype * int * int * int -> narrow
        (** A name, a dtype, its width, its precision and its least exponent. *)

  let narrows =
    [
      Narrow ("float16", Nx.float16, 16, 11, -14);
      Narrow ("bfloat16", Nx.bfloat16, 16, 8, -126);
      Narrow ("float8_e4m3", Nx.float8_e4m3, 8, 4, -6);
      Narrow ("float8_e5m2", Nx.float8_e5m2, 8, 3, -14);
    ]

  (* The unit in the last place of [x] in a format of precision [p] and least
     exponent [emin]: the least subnormal at zero. *)
  let ulp ~p ~emin x =
    let _, e = Float.frexp x in
    let e = if x = 0. then emin else Int.max (e - 1) emin in
    Float.ldexp 1. (e - (p - 1))

  (* Every float of a narrow dtype of [width] bits, and the ranks of a tensor of
     it. *)
  let every (type b) (dt : (float, b) Nx.dtype) width : (float, b) Nx.t =
    let n = 1 lsl width in
    match width with
    | 16 ->
        Nx.bitcast dt
          (Nx.create Nx.int16 [| n |] (Array.init n (fun i -> i - (n / 2))))
    | _ -> Nx.bitcast dt (Nx.create Nx.uint8 [| n |] (Array.init n Fun.id))

  let ranks (type b) width (x : (float, b) Nx.t) =
    let bits =
      match width with
      | 16 -> Nx.to_array (Nx.bitcast Nx.int16 x)
      | _ -> Nx.to_array (Nx.bitcast Nx.uint8 x)
    in
    let mask = (1 lsl width) - 1 in
    Array.map
      (fun b -> Int64.to_int (rank ~width (Int64.of_int (b land mask))))
      bits

  (* [hold dt ~bound ?scale f args n] holds [f] at [args], [n] floats of the
     narrow dtype [dt] each, as [narrow] states. *)
  let hold (type b) (dt : (float, b) Nx.dtype) width p emin ~bound ?scale { f }
      (args : (float, b) Nx.t array) n =
    let got = f args in
    let args64 = Array.map (Nx.cast Nx.float64) args in
    let v = f args64 in
    let xs = Array.map Nx.to_array args64 in
    let at i = Array.map (fun a -> a.(i)) xs in
    let scales =
      match scale with
      | Some { f } -> Nx.to_array (f args64)
      | None -> Array.make n Float.nan
    in
    let vs = Nx.to_array v in
    (* The float64 bound, in ulps, at each input. *)
    let bounds =
      Array.init n (fun i ->
          match bound (at i) with
          | Ulps k | Near_zeros (k, _) | Scaled k -> Float.of_int k
          | Log_ulps k -> log_ulps k vs.(i)
          | Inverse (k, fw) -> Float.of_int k +. (scales.(i) *. Float.of_int fw)
          | Relative (r, _) -> r *. 0x1p52
          | Relative_scaled (r, _) -> r *. scales.(i) *. 0x1p52)
    in
    let slack =
      Nx.create Nx.float64 [| n |]
        (Array.map
           (fun b -> if Float.is_finite b then b *. 0x1p-52 else 0.)
           bounds)
    in
    let rounded k = Nx.cast dt (Nx.mul v (Nx.add_s (Nx.mul_s slack k) 1.)) in
    let candidates =
      List.map (fun k -> ranks width (rounded k)) [ 0.; -1.; 1. ]
    in
    let got_ranks = ranks width got in
    let gs = Nx.to_array (Nx.cast Nx.float64 got) in
    (* What the float64 result is in the narrow dtype: a dtype with no
       infinities has no other value for one. *)
    let cs = Nx.to_array (Nx.cast Nx.float64 (Nx.cast dt v)) in
    let misses = ref [] in
    for i = 0 to n - 1 do
      let g = gs.(i) and v = vs.(i) in
      let d =
        if Float.is_nan v || Float.is_nan g then
          if Float.is_nan v && Float.is_nan g then 0 else max_int
        else
          List.fold_left
            (fun d c -> Int.min d (abs (got_ranks.(i) - c.(i))))
            max_int candidates
      in
      let h = 0.5 *. ulp ~p ~emin v in
      (* Where the bound is absolute, its float32 bound and half an ulp of the
         narrow dtype. *)
      let absolute =
        match bound (at i) with
        | Ulps _ | Log_ulps _ | Inverse _ -> false
        | Near_zeros (_, a) ->
            Float.abs v < 1.
            && Float.abs (g -. v) <= (Float.of_int a *. 0x1p-23) +. h
        | Scaled k ->
            Float.abs (g -. v) <= (Float.of_int k *. 0x1p-23 *. scales.(i)) +. h
        | Relative (_, r) -> Float.abs (g -. v) <= (r *. Float.abs v) +. h
        | Relative_scaled (_, r) ->
            Float.abs (g -. v) <= (r *. scales.(i) *. Float.abs v) +. h
      in
      let ok =
        if not (Float.is_finite v) then
          (Float.is_nan g && Float.is_nan cs.(i)) || g = cs.(i)
        else
          d <= 1
          && (g <> 0.
             || cs.(i) <> 0.
             || Float.sign_bit g = Float.sign_bit cs.(i))
          || absolute
      in
      if not ok then misses := (at i, g, v) :: !misses
    done;
    match !misses with
    | [] -> ()
    | misses ->
        Windtrap.failf "%d of %d inputs miss:@\n%a" (List.length misses) n
          (Format.pp_print_list (fun ppf (x, g, v) ->
               Format.fprintf ppf "at (%a): %h, float64 %h"
                 (Format.pp_print_list
                    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
                    (fun ppf x -> Format.fprintf ppf "%h" x))
                 (Array.to_list x) g v))
          (List.filteri (fun i _ -> i < 8) (List.rev misses))

  (* [narrow ~bound ?scale f] holds [f] at every float of each narrow dtype
     within one ulp of its correctly rounded value, and a zero to its sign,
     [scale] being an inverse's condition number at the float64 argument. The
     reference is the float64 result, within [bound] of its own correctly
     rounded value: rounded to the narrow dtype it is the correctly rounded
     value, or a neighbour where that bound reaches a narrow midpoint. A
     non-finite value is held exactly. *)
  let narrow ~bound ?scale f =
    List.map
      (fun (Narrow (name, dt, width, p, emin)) ->
        Windtrap.test name (fun () ->
            hold dt width p emin ~bound ?scale f
              [| every dt width |]
              (1 lsl width)))
      narrows

  (* [narrow2 ?keep ~bound ?scale ~firsts ~seconds f] holds [f] of two arguments
     as [narrow] holds one: at every pair of floats of an 8-bit dtype, and at a
     16-bit one at every float as either argument against each of [firsts] or
     [seconds] as the other, rounded to the dtype; at the pairs [keep] keeps,
     all by default. *)
  let narrow2 ?(keep = fun _ -> true) ~bound ?scale ~firsts ~seconds f =
    List.map
      (fun (Narrow (name, dt, width, p, emin)) ->
        Windtrap.test name (fun () ->
            let all =
              Array.to_list (Nx.to_array (Nx.cast Nx.float64 (every dt width)))
            in
            let pairs =
              if width = 8 then
                List.concat_map (fun a -> List.map (fun x -> (a, x)) all) all
              else
                List.concat_map (fun a -> List.map (fun x -> (a, x)) all) firsts
                @ List.concat_map
                    (fun x -> List.map (fun a -> (a, x)) all)
                    seconds
            in
            let pairs = List.filter (fun (a, x) -> keep [| a; x |]) pairs in
            let n = List.length pairs in
            let column sel =
              Nx.cast dt
                (Nx.create Nx.float64 [| n |]
                   (Array.of_list (List.map sel pairs)))
            in
            hold dt width p emin ~bound ?scale f [| column fst; column snd |] n))
      narrows
end
