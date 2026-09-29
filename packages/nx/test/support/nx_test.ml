(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let tensor w =
  Testable.contramap
    (fun t -> (Nx.shape t, Nx.to_array t))
    (pair (array int) (array w))

(* Floats equal within [rel] of the larger magnitude, every NaN equal to every
   NaN: the witness of a computed float that may be NaN. *)
let close ~rel =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%.17g" x)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b)
      || a = b
      || Float.abs (a -. b) <= rel *. Float.max (Float.abs a) (Float.abs b))

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
  [
    { name = "transposed"; apply = (fun t -> Nx.transpose t) };
    { name = "flipped"; apply = (fun t -> Nx.flip t) };
    {
      name = "every other row";
      apply = (fun t -> rows (fun n -> Nx.slice [ Rs (0, n, 2) ]) t);
    };
    {
      name = "without its first row";
      apply = (fun t -> rows (fun n -> Nx.slice [ R (1, n) ]) t);
    };
    {
      name = "broadcast over a new axis";
      apply = (fun t -> Nx.broadcast_to (Array.append [| 2 |] (Nx.shape t)) t);
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

(* Whether [shape] can view the elements of [t] where they are: the strides that
   the unit steps of [shape] give must place every element. *)
let viewable t shape =
  let st = Array.map (fun b -> b / Nx.itemsize t) (Nx.strides t) in
  let pos k =
    let p = ref 0 in
    Array.iteri (fun d i -> p := !p + (i * st.(d))) (unravel (Nx.shape t) k);
    !p
  in
  let n = Nx.numel t in
  let unit d =
    ravel shape (Array.mapi (fun e _ -> if e = d then 1 else 0) shape)
  in
  let strides =
    Array.mapi (fun d s -> if s > 1 then pos (unit d) - pos 0 else 0) shape
  in
  List.for_all
    (fun k ->
      let p = ref (pos 0) in
      Array.iteri (fun d i -> p := !p + (i * strides.(d))) (unravel shape k);
      !p = pos k)
    (List.init n Fun.id)

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

  (* The elements of [s] read as [Nx.data] documents: element [idx] is at
     [offset + sum idx.(d) * strides.(d) / itemsize] of the buffer. *)
  let of_layout s =
    let buf = Nx.data s and off = Nx.offset s in
    let strides = Array.map (fun b -> b / Nx.itemsize s) (Nx.strides s) in
    init (Nx.shape s) (fun idx ->
        let i = ref off in
        Array.iteri (fun d k -> i := !i + (k * strides.(d))) idx;
        Nx_buffer.get buf !i)

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

  let reshape shape t =
    if Array.exists (fun d -> d < -1) shape then invalid_arg "reshape";
    let holes =
      Array.fold_left (fun n d -> if d = -1 then n + 1 else n) 0 shape
    in
    let known =
      Array.fold_left (fun p d -> if d = -1 then p else p * d) 1 shape
    in
    let n = numel t.shape in
    if holes > 1 || (holes = 1 && (known = 0 || n mod known <> 0)) then
      invalid_arg "reshape";
    let shape = Array.map (fun d -> if d = -1 then n / known else d) shape in
    if numel shape <> n then invalid_arg "reshape";
    { t with shape }

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
    | D _ -> invalid_arg "Ref: D windows are not modelled"

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

  (* Elementwise *)

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
