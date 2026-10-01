(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Elements *)

(* A compiled result may lie on a device that computes nothing eagerly, such
   as Metal: the cast runs on the host. *)
let complexes x =
  let x = Nx.place Nx.Placement.host x in
  let dt = Nx.dtype x in
  if Nx_dtype.is_complex dt then Nx.to_array (Nx.cast Nx.complex128 x)
  else if Nx_dtype.is_float dt then
    Array.map
      (fun re -> { Complex.re; im = 0. })
      (Nx.to_array (Nx.cast Nx.float64 x))
  else invalid_arg ("Reference.complexes: " ^ Nx_dtype.to_string dt)

let norm a = Array.fold_left (fun m z -> Float.max m (Complex.norm z)) 0. a

(* Witnesses *)

let bits f = if Float.is_nan f then Int64.minus_one else Int64.bits_of_float f

let same_bits (type a b) (x : (a, b) Nx.t) (y : (a, b) Nx.t) =
  let dt = Nx.dtype x in
  if Nx_dtype.is_float dt || Nx_dtype.is_complex dt then
    let same a b =
      Int64.equal (bits a.Complex.re) (bits b.Complex.re)
      && Int64.equal (bits a.im) (bits b.im)
    in
    Array.for_all2 same (complexes x) (complexes y)
  else Nx.to_array x = Nx.to_array y

let exact () =
  Windtrap.Testable.make
    ~pp:(fun ppf x ->
      Format.fprintf ppf "%a %a@ %a" Nx.pp_dtype (Nx.dtype x) Nx.pp_shape
        (Nx.shape x) Nx.pp x)
    ~equal:(fun x y ->
      Nx_dtype.equal (Nx.dtype x) (Nx.dtype y)
      && Nx.shape x = Nx.shape y
      && same_bits x y)

(* [agree x y] is [Some d], the distance between two finite components, [Some
   0.] for two NaNs or two infinities of one sign, and [None] otherwise. *)
let agree x y =
  if Float.is_finite x && Float.is_finite y then Some (Float.abs (x -. y))
  else if (Float.is_nan x && Float.is_nan y) || x = y then Some 0.
  else None

let finite z = Float.is_finite z.Complex.re && Float.is_finite z.im

let close_arrays ~rel ~floor a b =
  Array.length a = Array.length b
  &&
  let diff = ref 0. and ok = ref true in
  Array.iteri
    (fun i x ->
      let y = b.(i) in
      match (agree x.Complex.re y.Complex.re, agree x.im y.im) with
      | Some d, Some e -> diff := Float.max !diff (Float.max d e)
      | None, _ | _, None -> ok := false)
    a;
  let scale a = norm (Array.of_list (List.filter finite (Array.to_list a))) in
  !ok && !diff <= (rel *. Float.max (scale a) (scale b)) +. floor

let pp_complex ppf { Complex.re; im } =
  if im = 0. then Format.fprintf ppf "%.17g" re
  else Format.fprintf ppf "%.17g%+.17gi" re im

let pp_arrays ppf l =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
    (fun ppf a ->
      Format.fprintf ppf "[@[%a@]]"
        (Format.pp_print_array
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
           pp_complex)
        a)
    ppf l

let close ~rel ?(floor = 0.) () =
  Windtrap.Testable.make ~pp:pp_arrays ~equal:(fun a b ->
      List.length a = List.length b
      && List.for_all2 (close_arrays ~rel ~floor) a b)

(* Structures *)

let leaves s x =
  List.rev (Nx.Ptree.fold s (fun _ t acc -> complexes t :: acc) x [])

let direction r s x =
  let element () = Random.State.float r 4. -. 2. in
  let component () =
    let m = 0.1 +. Random.State.float r 1.9 in
    if Random.State.bool r then m else -.m
  in
  Nx.Ptree.map s
    (fun _ t ->
      let shape = Nx.shape t and n = Nx.numel t in
      if Nx_dtype.is_complex (Nx.dtype t) then
        Nx.cast (Nx.dtype t)
          (Nx.create Nx.complex128 shape
             (Array.init n (fun _ ->
                  { Complex.re = component (); im = component () })))
      else
        Nx.cast (Nx.dtype t)
          (Nx.create Nx.float64 shape (Array.init n (fun _ -> element ()))))
    x

let step s x h v =
  Nx.Ptree.map2 s
    (fun _ x v ->
      Nx.add x
        (Nx.mul v
           (Nx.full (Nx.dtype v) [||] (Nx_dtype.of_float (Nx.dtype v) h))))
    x v

let fold2 s f u v acc =
  List.fold_left2
    (fun acc a b ->
      let acc = ref acc in
      Array.iteri (fun i x -> acc := f x b.(i) !acc) a;
      !acc)
    acc (leaves s u) (leaves s v)

let dot s u v =
  fold2 s (fun a b acc -> acc +. (Complex.mul (Complex.conj a) b).re) u v 0.

let magnitude s u v =
  fold2 s (fun a b acc -> acc +. (Complex.norm a *. Complex.norm b)) u v 0.

let central s r ~eps f x v =
  let hi = leaves r (f (step s x eps v))
  and lo = leaves r (f (step s x (-.eps) v)) in
  List.map2
    (Array.map2 (fun a b ->
         Complex.div (Complex.sub a b) { re = 2. *. eps; im = 0. }))
    hi lo

(* The cumulative operations' definitions *)

let cumprod_derivative xs s =
  let distinct = List.sort_uniq Int.compare s in
  if List.length distinct <> List.length s then 0.
  else
    let low = List.fold_left Int.max 0 s in
    let total = ref 0. in
    for k = low to Array.length xs - 1 do
      let p = ref 1. in
      for j = 0 to k do
        if not (List.mem j s) then p := !p *. xs.(j)
      done;
      total := !total +. !p
    done;
    !total

let running_arg better xs =
  let arg = ref 0 in
  Array.mapi
    (fun k x ->
      if better x xs.(!arg) then arg := k;
      !arg)
    xs
