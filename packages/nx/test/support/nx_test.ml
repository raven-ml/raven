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
  | D (s, len) -> Format.fprintf ppf "D (%ld, %d)" (Nx.item [] s) len

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

let lay_out steps t = List.fold_left (fun t l -> l.apply t) t steps

(* The storage of [t]: views share it, copies do not. *)
let storage t = Nx_effect.to_host t

(* Where each element of [t] is in its storage, in row-major order: element
   [idx] is at [offset + sum idx.(d) * strides.(d)] of its view. *)
let positions t =
  let v = Nx_effect.view t in
  let st = Nx_core.View.strides v in
  Array.init (Nx.numel t) (fun k ->
      let p = ref (Nx_core.View.offset v) in
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
        Array.map (Nx_core.Elements.get (Nx.dtype s) (storage s)) (positions s);
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
          Int.max 0 (Int.min (dim - len) (Int32.to_int (Nx.item [] s)))
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

(* Test devices: an engine whose devices hold their storage in host memory of
   their own, one buffer per device of a value's placement in its order (a split
   value's shards, a replicated value's copies), and count what moves. *)
module Devices = struct
  type Nx_effect.storage += Mem of Nx_device.Buffer.t list

  let elements_read = ref 0
  let uploads = ref 0

  (* The elements view [v] reaches in [mem], in C order. *)
  let gather mem v =
    elements_read := !elements_read + Nx_core.View.numel v;
    Nx_core.Elements.gather mem v

  let rec engine =
    {
      Nx_effect.read =
        (fun r ->
          match r.r_cell.state with
          | Live (Mem shards) ->
              let shape =
                Nx_effect.global r.r_placement (Nx_core.View.shape r.r_view)
              in
              Nx_effect.assemble r
                (Array.map (fun n -> (0, n)) shape)
                (fun d v ->
                  let devices = Nx.Placement.devices r.r_cell.placement in
                  let k = Option.get (List.find_index (( == ) d) devices) in
                  gather (List.nth shards k) v)
          | _ -> assert false);
      place = (fun p x -> place p x);
    }

  and place : type a b.
      Nx_effect.placement -> (a, b) Nx_effect.t -> (a, b) Nx_effect.t =
   fun p x ->
    if Nx.numel x > 0 then incr uploads;
    let h = Nx.place Nx.Placement.host x in
    let devices = Nx.Placement.devices p in
    let windows = List.map (Nx.Placement.window p (Nx.shape h)) devices in
    let shards =
      List.map (fun w -> Nx_effect.to_host (Nx.copy (Nx.shrink w h))) windows
    in
    let shape = Array.map (fun (lo, hi) -> hi - lo) (List.hd windows) in
    Nx_effect.placed p (Nx.dtype x)
      (Nx_core.View.create shape)
      (Nx_effect.cell ~placement:p
         ~length:(Array.fold_left ( * ) 1 shape)
         (Mem shards))

  let d1 = Nx_effect.Device.make "TEST:1" engine
  let d2 = Nx_effect.Device.make "TEST:2" engine
  let d3 = Nx_effect.Device.make "TEST:3" engine
  let d4 = Nx_effect.Device.make "TEST:4" engine

  (* A device of another engine. *)
  let other = Nx_effect.Device.make "OTHER" { engine with place = engine.place }
  let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

  let cell_of (type a b) (x : (a, b) Nx.t) =
    match x with
    | Nx_effect.Placed r -> r.r_cell
    | _ -> fail "expected a placed value"
end

(* Tensors of every dtype as they are stored: drawn as bit patterns under every
   layout, and compared bit for bit. *)
module Stored = struct
  (* Tensors compare bit for bit: dtype, shape and the bytes of the elements in
     row-major order. *)

  let storage (Nx.P t) =
    let t = Nx.contiguous t in
    let b = Nx_effect.to_host t in
    let b = Nx_device.Buffer.view b ~offset:0 (Nx_device.Buffer.dtype b) (Nx.numel t) in
    let bytes = Nx_device.Buffer.bigarray Bigarray.char b in
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
  let complex64 =
    let f32 = float_bits ~e:8 ~m:23 in
    let word (re, im) = Int64.logor (Int64.shift_left im 32) re in
    let tensors =
      from_bits "int64" Nx.complex64 (Gen.map word (Gen.pair f32 f32))
    in
    case "complex64" Nx.complex64 tensors (tensor complex_exact)

  let complex128 =
    let pp ppf (c : Complex.t) = Format.fprintf ppf "%h%+hi" c.re c.im in
    let value (re, im) = { Complex.re; im } in
    let values = Gen.map value (Gen.pair Gen.any_float Gen.any_float) in
    case "complex128" Nx.complex128
      (viewed ~pp Nx.complex128 values)
      (tensor complex_exact)

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
