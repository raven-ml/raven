(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Outcomes *)

type 'a outcome = Value of 'a | Raised of string

let outcome f =
  match f () with v -> Value v | exception Invalid_argument m -> Raised m

let outcome_t t =
  Testable.make
    ~pp:(fun ppf -> function
      | Value v -> Testable.pp t ppf v
      | Raised m -> Format.fprintf ppf "raised %S" m)
    ~equal:(fun a b ->
      match (a, b) with
      | Value a, Value b -> Testable.equal t a b
      | Raised a, Raised b -> String.equal a b
      | _ -> false)

(* Bits *)

let code f = if Float.is_nan f then Int64.minus_one else Int64.bits_of_float f

type bits = int array * int64 array

let bits x = (Nx.shape x, Array.map code (Nx.to_array x))

let pp_elements pp ppf xs =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
    pp ppf (Array.to_list xs)

let bits_t =
  Testable.make
    ~pp:(fun ppf (s, b) ->
      Format.fprintf ppf "%a %a" Operand.pp_shape s
        (pp_elements (fun ppf b ->
             Format.fprintf ppf "%h" (Int64.float_of_bits b)))
        b)
    ~equal:( = )

type codes = int array * string array

let codes (type v s d) (x : (v, s, d) Nx.t) =
  let dt = Nx.dtype x in
  let code : v -> string =
    match Nx.Dtype.kind dt with
    | Float -> fun f -> Printf.sprintf "%Lx" (code f)
    | Complex ->
        fun c -> Printf.sprintf "%Lx,%Lx" (code c.Complex.re) (code c.im)
    | _ -> Format.asprintf "%a" (Nx.Dtype.pp_value dt)
  in
  (Nx.shape x, Array.map code (Nx.to_array x))

let codes_t =
  Testable.make
    ~pp:(fun ppf (s, c) ->
      Format.fprintf ppf "%a [%a]" Operand.pp_shape s
        (pp_elements Format.pp_print_string)
        c)
    ~equal:( = )

(* Distances *)

(* A float's magnitude bits at [dt]'s width: its rank among the floats of its
   sign. *)
let magnitude (type s) (dt : (float, s) Nx.dtype) x =
  match dt with
  | Nx.Dtype.Float64 -> Int64.logand (Int64.bits_of_float x) Int64.max_int
  | Nx.Dtype.Float32 ->
      Int64.of_int32 (Int32.logand (Int32.bits_of_float x) Int32.max_int)
  | _ -> invalid_arg "Witness.ulps: a dtype other than float32 or float64"

let ulps dt a b =
  let ma = magnitude dt a and mb = magnitude dt b in
  let infinite x = Float.abs x = Float.infinity in
  if Float.is_nan a || Float.is_nan b then
    if Float.is_nan a && Float.is_nan b then 0. else Float.infinity
  else if infinite a || infinite b then if a = b then 0. else Float.infinity
  else if Float.sign_bit a = Float.sign_bit b then
    Int64.to_float (Int64.abs (Int64.sub ma mb))
  else Int64.to_float (Int64.add (Int64.add ma mb) 1L)
