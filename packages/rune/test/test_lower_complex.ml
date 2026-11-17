(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The complex kinds at complex64 and complex128, compiled by Rune.jit, which
   must compile them whole: no call is evaluated. Each is compiled to nx's own
   sequence and gives eager's bits (Kinds), over drawn operands and of a
   constant inside the compiled function. tolk's folding gives a NaN it
   generates a clear sign bit, where the host's arithmetic sets it, so a folded
   pow may give a zero part the other sign where it copies one (Kinds.law's
   nan_signs). *)

open Windtrap
module Kinds = Rune_support.Kinds

let ported k =
  let name = Kinds.name k in
  [
    prop ~count:12
      ~examples:[ Kinds.example k ]
      (name ^ " gives eager's bits")
      (Kinds.gen k)
      (Kinds.law (module Nx.Host) ~f64:true k);
    test (name ^ " of a constant gives eager's bits") (fun () ->
        Kinds.folded ~nan_signs:true (module Nx.Host) ~f64:true k);
  ]

let kinds =
  Kinds.
    [
      cmul;
      cdiv;
      crecip;
      csqrt;
      csign;
      clog;
      clog2;
      cexp;
      cexp2;
      cpow;
      csinh;
      ccosh;
      ctanh;
      csin;
      ccos;
      ctan;
      casin;
      cacos;
      catan;
      casinh;
      cacosh;
      catanh;
    ]

(* NaN parts

   rune.mli's rule, from the spec: an operand with a NaN part and none infinite
   gives a result with a NaN part, but z^w is 1 at w = 0, and 1 ± 0i at z = 1 ±
   0i with w's imaginary part zero, where the float pow is 1 at x = 1. Each
   function compiled on the host at complex64 and complex128, over numbers of
   the parts NaN, -NaN, ±0, ±1 and 2 with a NaN part among the operands. *)

type fn =
  | F1 of { f1 : 's 'd. (Complex.t, 's, 'd) Nx.t -> (Complex.t, 's, 'd) Nx.t }
  | F2 of {
      f2 :
        's 'd.
        (Complex.t, 's, 'd) Nx.t ->
        (Complex.t, 's, 'd) Nx.t ->
        (Complex.t, 's, 'd) Nx.t;
    }

let fns =
  [
    ("mul", F2 { f2 = Nx.mul });
    ("div", F2 { f2 = Nx.div });
    ("pow", F2 { f2 = Nx.pow });
    ("recip", F1 { f1 = Nx.recip });
    ("sqrt", F1 { f1 = Nx.sqrt });
    ("sign", F1 { f1 = Nx.sign });
    ("log", F1 { f1 = Nx.log });
    ("log2", F1 { f1 = Nx.log2 });
    ("exp", F1 { f1 = Nx.exp });
    ("exp2", F1 { f1 = Nx.exp2 });
    ("sinh", F1 { f1 = Nx.sinh });
    ("cosh", F1 { f1 = Nx.cosh });
    ("tanh", F1 { f1 = Nx.tanh });
    ("sin", F1 { f1 = Nx.sin });
    ("cos", F1 { f1 = Nx.cos });
    ("tan", F1 { f1 = Nx.tan });
    ("asin", F1 { f1 = Nx.asin });
    ("acos", F1 { f1 = Nx.acos });
    ("atan", F1 { f1 = Nx.atan });
    ("asinh", F1 { f1 = Nx.asinh });
    ("acosh", F1 { f1 = Nx.acosh });
    ("atanh", F1 { f1 = Nx.atanh });
  ]

let numbers =
  let ps = [ Float.nan; -.Float.nan; 0.; -0.; 1.; -1.; 2. ] in
  List.concat_map (fun re -> List.map (fun im -> { Complex.re; im }) ps) ps

let has_nan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im

(* The rule's value at operands [zs]: [`One] where it is 1 + ±0i, else a NaN
   part. *)
let expected name zs =
  match (name, zs) with
  | "pow", [ _; (w : Complex.t) ] when w.re = 0. && w.im = 0. -> `One
  | "pow", [ (z : Complex.t); (w : Complex.t) ]
    when z.re = 1. && z.im = 0. && w.im = 0. ->
      `One
  | _ -> `Nan

let nan_parts (name, f) =
  let tuples =
    match f with
    | F1 _ -> List.map (fun z -> [ z ]) numbers
    | F2 _ ->
        List.concat_map (fun z -> List.map (fun w -> [ z; w ]) numbers) numbers
  in
  let tuples = List.filter (List.exists has_nan) tuples in
  let at (type s) (dt : (Complex.t, s) Nx.dtype) =
    let col j =
      Nx.create dt
        [| List.length tuples |]
        (Array.of_list (List.map (fun t -> List.nth t j) tuples))
    in
    let before = Rune.evaluations () in
    let y =
      match f with
      | F1 { f1 } -> Rune.jit Nx.Ptree.(tensor @-> returns tensor) f1 (col 0)
      | F2 { f2 } ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            f2 (col 0) (col 1)
    in
    equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
    let show (z : Complex.t) = Printf.sprintf "(%h, %h)" z.re z.im in
    List.concat
      (List.mapi
         (fun i (r : Complex.t) ->
           let zs = List.nth tuples i in
           let ok =
             match expected name zs with
             | `One -> r.re = 1. && r.im = 0.
             | `Nan -> has_nan r
           in
           if ok then []
           else
             [
               Printf.sprintf "%s(%s) at %s = %s" name
                 (String.concat ", " (List.map show zs))
                 (Nx.Dtype.name dt) (show r);
             ])
         (Array.to_list (Nx.to_array y)))
  in
  equal (list string) [] (at Nx.complex64 @ at Nx.complex128)

let () =
  exit
    (run "Lower_complex"
       [
         group "kinds" (List.concat_map ported kinds);
         Windtrap.cases ~id:fst
           "a NaN part and no infinite one give a NaN part, but pow's \
            exceptions"
           fns nan_parts;
       ])
