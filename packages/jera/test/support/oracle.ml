(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Witnesses *)

let floats t = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Placement.host t))

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

type leaf = { dtype : string; shape : int array; elements : float array }

let leaf t =
  {
    dtype = Nx_dtype.to_string (Nx.dtype t);
    shape = Nx.shape t;
    elements = floats t;
  }

let same_leaf ?tol a b =
  String.equal a.dtype b.dtype
  && a.shape = b.shape
  && Array.for_all2 (same_float ?tol) a.elements b.elements

let pp_leaf ppf l =
  Format.fprintf ppf "@[<hov 2>%s%a@ @[<hov 1>[%a]@]@]" l.dtype Nx.pp_shape
    l.shape
    (Format.pp_print_array ~pp_sep:Format.pp_print_space (fun ppf x ->
         Format.fprintf ppf "%.17g" x))
    l.elements

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

(* Host values *)

let dot u v =
  if Nx.shape u <> Nx.shape v then invalid_arg "Oracle.dot: two shapes";
  let a = floats u and b = floats v in
  let s = ref 0. in
  Array.iteri (fun i x -> s := !s +. (x *. b.(i))) a;
  !s

(* Finite differences *)

let central ~eps f x v =
  let step s = f (Nx.add x (Nx.mul_s v s)) in
  Nx.div_s (Nx.sub (step eps) (step (-.eps))) (2. *. eps)

let slope e1 e2 = Float.log2 (e1 /. e2)

(* Errors *)

let message f =
  match f () with
  | _ -> failwith "Oracle.message: the call returned"
  | exception Invalid_argument m -> m
  | exception e ->
      failwith ("Oracle.message: the call raised " ^ Printexc.to_string e)
