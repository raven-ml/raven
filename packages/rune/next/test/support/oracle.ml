(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Witnesses *)

(* A tensor's elements, real ones as float64 and complex ones as complex128:
   every dtype the suites compare widens exactly to one of them. *)
type elements = Real of float array | Cplx of Complex.t array

let elements (type a b) (t : (a, b) Nx.t) =
  if Nx_dtype.is_complex (Nx.dtype t) then
    Cplx (Nx.to_array (Nx.cast Nx.complex128 t))
  else Real (Nx.to_array (Nx.cast Nx.float64 t))

(* A tolerance: [rel] of the larger magnitude, or [abs]. *)
type tolerance = { rel : float; abs : float }

let tolerance ?rel ?abs () =
  match (rel, abs) with
  | None, None -> None
  | _ ->
      Some
        {
          rel = Option.value rel ~default:0.;
          abs = Option.value abs ~default:0.;
        }

let same_float ?tol a b =
  match tol with
  | None -> Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
  | Some { rel; abs } ->
      (Float.is_nan a && Float.is_nan b)
      || a = b
      || Float.abs (a -. b)
         <= Float.max abs (rel *. Float.max (Float.abs a) (Float.abs b))

let same_elements ?tol a b =
  match (a, b) with
  | Real a, Real b ->
      Array.length a = Array.length b && Array.for_all2 (same_float ?tol) a b
  | Cplx a, Cplx b ->
      Array.length a = Array.length b
      && Array.for_all2
           (fun (x : Complex.t) (y : Complex.t) ->
             same_float ?tol x.re y.re && same_float ?tol x.im y.im)
           a b
  | Real _, Cplx _ | Cplx _, Real _ -> false

let pp_elements ppf = function
  | Real a ->
      Format.fprintf ppf "@[<hov 1>[%a]@]"
        (Format.pp_print_array ~pp_sep:Format.pp_print_space (fun ppf x ->
             Format.fprintf ppf "%h" x))
        a
  | Cplx a ->
      Format.fprintf ppf "@[<hov 1>[%a]@]"
        (Format.pp_print_array ~pp_sep:Format.pp_print_space
           (fun ppf (z : Complex.t) -> Format.fprintf ppf "%h%+hi" z.re z.im))
        a

(* What a witness compares of one tensor. *)
type leaf = { dtype : string; shape : int array; elements : elements }

let leaf t =
  {
    dtype = Nx_dtype.to_string (Nx.dtype t);
    shape = Nx.shape t;
    elements = elements t;
  }

let same_leaf ?tol a b =
  String.equal a.dtype b.dtype
  && a.shape = b.shape
  && same_elements ?tol a.elements b.elements

let pp_leaf ppf l =
  Format.fprintf ppf "@[<hov 2>%s%a@ %a@]" l.dtype Nx.pp_shape l.shape
    pp_elements l.elements

let tensor ?rel ?abs () =
  let tol = tolerance ?rel ?abs () in
  Testable.contramap leaf (Testable.make ~pp:pp_leaf ~equal:(same_leaf ?tol))

let structure ?rel ?abs s =
  let tol = tolerance ?rel ?abs () in
  let leaves x =
    ( Nx.Ptree.visits s x,
      Nx.Ptree.fold s (fun path t acc -> (path, leaf t) :: acc) x [] |> List.rev
    )
  in
  let pp ppf (visits, ls) =
    Format.fprintf ppf "@[<v>%a@,%a@]"
      (Format.pp_print_list Nx.Ptree.pp_visit)
      visits
      (Format.pp_print_list (fun ppf (p, l) ->
           Format.fprintf ppf "%a: %a" Nx.Ptree.Path.pp p pp_leaf l))
      ls
  in
  let equal (v, a) (w, b) =
    v = w
    && List.length a = List.length b
    && List.for_all2 (fun (_, x) (_, y) -> same_leaf ?tol x y) a b
  in
  Testable.contramap leaves (Testable.make ~pp ~equal)

(* Inner products *)

let dot u v =
  if Nx.shape u <> Nx.shape v then invalid_arg "Oracle.dot: two shapes";
  match (elements u, elements v) with
  | Real a, Real b ->
      let s = ref 0. in
      Array.iteri (fun i x -> s := !s +. (x *. b.(i))) a;
      !s
  | Cplx a, Cplx b ->
      let s = ref 0. in
      Array.iteri
        (fun i (x : Complex.t) ->
          s := !s +. (Complex.mul (Complex.conj x) b.(i)).re)
        a;
      !s
  | Real _, Cplx _ | Cplx _, Real _ -> invalid_arg "Oracle.dot: two kinds"

(* Finite differences *)

let central ~eps f x v =
  let step s = f (Nx.add x (Nx.mul_s v s)) in
  Nx.div_s (Nx.sub (step eps) (step (-.eps))) (2. *. eps)

(* The composition function *)

let term x = Float.tanh (x *. x) *. Float.exp x

let term' x =
  let t = Float.tanh (x *. x) in
  let s = 1. -. (t *. t) in
  Float.exp x *. (t +. (2. *. x *. s))

let term'' x =
  let t = Float.tanh (x *. x) in
  let s = 1. -. (t *. t) in
  Float.exp x *. (t +. (4. *. x *. s) +. (2. *. s) -. (8. *. x *. x *. t *. s))

let map f x = Nx.create (Nx.dtype x) (Nx.shape x) (Array.map f (Nx.to_array x))

(* Errors *)

let message f =
  match f () with
  | _ -> failwith "Oracle.message: the call returned"
  | exception Invalid_argument m -> m
  | exception e ->
      failwith ("Oracle.message: the call raised " ^ Printexc.to_string e)
