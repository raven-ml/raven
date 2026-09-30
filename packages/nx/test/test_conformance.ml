(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's operations at an nx.cpu placement: each computing operation, on operands
   placed on a test device that holds its own memory, gives the host's result
   bit for bit, placed on that device. *)

open Windtrap
open Nx_test

(* How a case receives its operands: as they are, or placed. *)
type at = { at : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let p = Nx.Placement.device Devices.d1
let host = { at = Fun.id }
let placed = { at = (fun x -> Nx.place p x) }
let f64 shape xs = Nx.create Nx.float64 shape xs
let unit_interval = f64 [| 2; 3 |] [| 0.1; 0.25; 0.4; 0.55; 0.7; 0.85 |]
let signed = f64 [| 2; 3 |] [| -1.5; 0.5; 2.5; -0.25; 3.; -2. |]
let ints = Nx.create Nx.int32 [| 2; 3 |] [| 5l; -3l; 12l; 7l; 0l; -9l |]
let ints' = Nx.create Nx.int32 [| 2; 3 |] [| 2l; 5l; 3l; -4l; 1l; 6l |]
let square = f64 [| 3; 3 |] [| 4.; 1.; 2.; 1.; 5.; 3.; 2.; 3.; 6. |]
let wide = f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 7. |]
let rows = Nx.create Nx.int32 [| 2; 2 |] [| 2l; 0l; 1l; 2l |]
let complex = Nx.cast Nx.complex128 signed
let images = f64 [| 1; 4; 4 |] (Array.init 16 float_of_int)
let patches = f64 [| 1; 4; 9 |] (Array.init 36 float_of_int)
let unary name f = (name, fun { at } -> [ Nx.P (f (at unit_interval)) ])

let binary name f =
  (name, fun { at } -> [ Nx.P (f (at signed) (at unit_interval)) ])

let integer name f = (name, fun { at } -> [ Nx.P (f (at ints) (at ints')) ])

let cases : (string * (at -> Nx.packed list)) list =
  [
    unary "neg" Nx.neg;
    unary "recip" Nx.recip;
    unary "abs" Nx.abs;
    unary "sqrt" Nx.sqrt;
    unary "sign" Nx.sign;
    unary "exp" Nx.exp;
    unary "log" Nx.log;
    unary "sin" Nx.sin;
    unary "cos" Nx.cos;
    unary "tan" Nx.tan;
    unary "asin" Nx.asin;
    unary "acos" Nx.acos;
    unary "atan" Nx.atan;
    unary "sinh" Nx.sinh;
    unary "cosh" Nx.cosh;
    unary "tanh" Nx.tanh;
    unary "trunc" Nx.trunc;
    unary "ceil" Nx.ceil;
    unary "floor" Nx.floor;
    unary "round" Nx.round;
    unary "erf" Nx.erf;
    binary "add" Nx.add;
    binary "sub" Nx.sub;
    binary "mul" Nx.mul;
    binary "div" Nx.div;
    binary "pow" (fun a b -> Nx.pow (Nx.abs a) b);
    binary "atan2" Nx.atan2;
    binary "maximum" Nx.maximum;
    binary "minimum" Nx.minimum;
    binary "mod" Nx.mod_;
    integer "integer div" Nx.div;
    integer "bitwise_and" Nx.bitwise_and;
    integer "bitwise_or" Nx.bitwise_or;
    integer "bitwise_xor" Nx.bitwise_xor;
    binary "equal" Nx.equal;
    binary "not_equal" Nx.not_equal;
    binary "less" Nx.less;
    binary "less_equal" Nx.less_equal;
    ( "where",
      fun { at } ->
        [
          Nx.P
            (Nx.where (at (Nx.less signed unit_interval)) (at signed) (at wide));
        ] );
    unary "sum" (fun x -> Nx.sum ~axes:[ 1 ] x);
    unary "prod" (fun x -> Nx.prod ~axes:[ 0 ] x);
    unary "max" (fun x -> Nx.max ~axes:[ 1 ] x);
    unary "min" (fun x -> Nx.min x);
    unary "cumsum" (fun x -> Nx.cumsum ~axis:1 x);
    unary "cumprod" (fun x -> Nx.cumprod ~axis:0 x);
    unary "cummax" (fun x -> Nx.cummax ~axis:1 x);
    unary "cummin" (fun x -> Nx.cummin ~axis:1 x);
    ("argmax", fun { at } -> [ Nx.P (Nx.argmax ~axis:1 (at signed)) ]);
    ("argmin", fun { at } -> [ Nx.P (Nx.argmin ~axis:0 (at signed)) ]);
    ( "sort",
      fun { at } ->
        let s, i = Nx.sort ~axis:1 (at signed) in
        [ Nx.P s; Nx.P i ] );
    ("argsort", fun { at } -> [ Nx.P (Nx.argsort ~descending:true (at signed)) ]);
    unary "pad" (Nx.pad [| (1, 0); (0, 2) |] 9.);
    ( "concatenate",
      fun { at } -> [ Nx.P (Nx.concatenate ~axis:0 [ at signed; at wide ]) ] );
    ("cast", fun { at } -> [ Nx.P (Nx.cast Nx.float32 (at signed)) ]);
    ("bitcast", fun { at } -> [ Nx.P (Nx.bitcast Nx.int64 (at signed)) ]);
    ( "threefry",
      fun { at } ->
        let key = Nx.Rng.of_tensor (at (Nx.Rng.key 7 :> Nx.int32_t)) in
        [ Nx.P (Nx.Rng.uniform key Nx.float32 [| 5 |]) ] );
    ( "take",
      fun { at } -> [ Nx.P (Nx.take ~axis:1 ~indices:(at rows) (at signed)) ] );
    ( "scatter",
      fun { at } ->
        [
          Nx.P
            (Nx.scatter ~axis:1 ~indices:(at rows)
               ~values:(at (Nx.slice [ R (0, 2); R (0, 2) ] wide))
               (at signed));
        ] );
    ( "scatter add",
      fun { at } ->
        [
          Nx.P
            (Nx.scatter ~mode:`Add ~axis:1 ~indices:(at rows)
               ~values:(at (Nx.slice [ R (0, 2); R (0, 2) ] wide))
               (at signed));
        ] );
    ( "set",
      fun { at } ->
        [
          Nx.P
            (Nx.set
               [ I 1; R (1, 3) ]
               (at (Nx.slice [ I 0; R (0, 2) ] wide))
               (at signed));
        ] );
    ( "extract_patches",
      fun { at } ->
        [
          Nx.P
            (Nx.extract_patches ~kernel_size:[| 2; 2 |] ~stride:[| 1; 1 |]
               ~dilation:[| 1; 1 |]
               ~padding:[| (0, 0); (0, 0) |]
               (at images));
        ] );
    ( "combine_patches",
      fun { at } ->
        [
          Nx.P
            (Nx.combine_patches ~output_size:[| 4; 4 |] ~kernel_size:[| 2; 2 |]
               ~stride:[| 1; 1 |] ~dilation:[| 1; 1 |]
               ~padding:[| (0, 0); (0, 0) |]
               (at patches));
        ] );
    ("matmul", fun { at } -> [ Nx.P (Nx.matmul (at wide) (at square)) ]);
    ("fftn", fun { at } -> [ Nx.P (Nx.fftn (at complex)) ]);
    ("ifftn", fun { at } -> [ Nx.P (Nx.ifftn ~axes:[ 1 ] (at complex)) ]);
    ("rfftn", fun { at } -> [ Nx.P (Nx.rfftn Nx.complex128 (at signed)) ]);
    ( "irfftn",
      fun { at } -> [ Nx.P (Nx.irfftn Nx.float64 ~axes:[ 1 ] (at complex)) ] );
    ( "irfftn to a given size",
      fun { at } ->
        [ Nx.P (Nx.irfftn Nx.float64 ~axes:[ 1 ] ~s:[ 6 ] (at complex)) ] );
    ("copy", fun { at } -> [ Nx.P (Nx.copy (at (Nx.transpose signed))) ]);
    ("cholesky", fun { at } -> [ Nx.P (Nx.cholesky (at square)) ]);
    ( "qr",
      fun { at } ->
        let q, r = Nx.qr (at wide) in
        [ Nx.P q; Nx.P r ] );
    ( "qr complete",
      fun { at } ->
        let q, r = Nx.qr ~mode:`Complete (at wide) in
        [ Nx.P q; Nx.P r ] );
    ( "lu",
      fun { at } ->
        let perm, l, u = Nx.lu (at square) in
        [ Nx.P perm; Nx.P l; Nx.P u ] );
    ( "svd",
      fun { at } ->
        let u, s, vt = Nx.svd (at wide) in
        [ Nx.P u; Nx.P s; Nx.P vt ] );
    ( "svd full",
      fun { at } ->
        let u, s, vt = Nx.svd ~full_matrices:true (at wide) in
        [ Nx.P u; Nx.P s; Nx.P vt ] );
    ("eigvals", fun { at } -> [ Nx.P (Nx.eigvals (at square)) ]);
    ( "eig",
      fun { at } ->
        let w, v = Nx.eig (at square) in
        [ Nx.P w; Nx.P v ] );
    ( "eigh",
      fun { at } ->
        let w, v = Nx.eigh (at square) in
        [ Nx.P w; Nx.P v ] );
    ("eigvalsh", fun { at } -> [ Nx.P (Nx.eigvalsh (at square)) ]);
    ( "solve_triangular",
      fun { at } ->
        [
          Nx.P
            (Nx.solve_triangular (at (Nx.triu square)) (at (Nx.transpose wide)));
        ] );
  ]

let placed_result (Nx.P y) =
  is_true ~msg:"placed on the device" (Nx.Placement.equal (Nx.placement y) p);
  Nx.P (Nx.place Nx.Placement.host y)

let conformance =
  group "at a placement of nx.cpu on a device of its own memory"
    (List.map
       (fun (name, f) ->
         test name (fun () ->
             List.iter2
               (fun expected got ->
                 equal Stored.packed expected (placed_result got))
               (f host) (f placed)))
       cases)

let () = exit (run "nx conformance" [ conformance ])
