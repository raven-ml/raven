(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled calls. A call computes eager's values over every layout of its
   arguments; traces once per key, and once more for each part of the key that
   changes; gives each result storage of its own; consumes and lends storage as
   its signature says, refusing before any work what it cannot consume; binds
   its captures once; raises before consuming anything; runs from several
   domains; runs where its arguments and captures lie, on the host and on
   devices over the host's memory; and folds a scan inside its trace. *)

open Windtrap
open Nx_test
module Rune = Rune_internals.Rune

let floats = tensor float_exact
let close = Oracle.tensor ~rel:1e-5 ~abs:1e-30 ()
let x () = Nx.create Nx.float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |]
let y () = Nx.create Nx.float32 [| 4 |] [| 2.; 0.; -1.; 4. |]
let poly x = Nx.add (Nx.mul x x) x
let host t = Nx.place Nx.Placement.host t
let address t = List.hd (Witness.addresses t)
let consumes = Nx.Ptree.(consumes tensor @@ returns tensor)
let two = Nx.Ptree.(tensor @-> tensor @-> returns tensor)

(* Observing a call *)

(* [profiled f] is [f ()] and the host spans the compiled call recorded
   meanwhile, in order. *)
let profiled f =
  let p = Nx_device.Profile.start () in
  match f () with
  | y ->
      let spans =
        List.filter_map
          (function
            | Nx_device.Profile.Span s
              when String.starts_with ~prefix:"rune.jit: " s.name ->
                Some s.name
            | _ -> None)
          (Nx_device.Profile.stop p)
      in
      (y, spans)
  | exception e ->
      ignore (Nx_device.Profile.stop p);
      raise e

(* [traces f] is the number of traces while [f ()] runs. *)
let traces f =
  let (), spans = profiled f in
  List.length (List.filter (String.equal "rune.jit: trace") spans)

(* [loaded_on d f] is [f ()] and the number of programs loaded on [d]
   meanwhile. *)
let loaded_on d f =
  let p = Nx_device.Profile.start () in
  match f () with
  | y ->
      let loads =
        List.filter
          (function
            | Nx_device.Profile.Load l ->
                Nx_device.equal d (Nx_device.Program.device l.program)
            | _ -> false)
          (Nx_device.Profile.stop p)
      in
      (y, List.length loads)
  | exception e ->
      ignore (Nx_device.Profile.stop p);
      raise e

(* [counted f] is [f] and the number of times it ran. *)
let counted f =
  let n = ref 0 in
  ( (fun x ->
      incr n;
      f x),
    n )

let raises_jit_error f =
  raises_match
    (function Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

let message f = Oracle.message (fun () -> ignore (f ()))

(* Devices over the host's memory, whose programs are the host's *)

let driver ?(mapping = Some Nx_device.Driver.Identity) name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping })

let d1, d2, d3, d4 =
  match List.map driver [ "J1"; "J2"; "J3"; "J4" ] with
  | [ a; b; c; d ] -> (a, b, c, d)
  | _ -> assert false

let on d = Nx.Placement.device ~backend:Rune.compiled d
let placed d t = Nx.place (on d) t
let stats = Nx_device.stats
let bytes_in d = Nx_device.Stats.bytes_in (stats d)
let allocated d = Nx_device.Stats.allocated (stats d)

(* [allocated d] once the memory of what was dropped has returned to [d]. A
   device buffer returns one major cycle after the last value holding it dies,
   and a value that a finaliser closure keeps, as a compiled call keeps the
   storages it binds, dies only once that finaliser has run, a cycle after the
   call: a chain of such holders takes a cycle per link. A fixed number of
   rounds covers the chains these tests build; [allocated] alone cannot tell
   when they are done, since a round may return nothing yet free a holder. *)
let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    Nx_device.synchronize d
  done;
  allocated d

(* [warmed measure] is [measure ()] after a first, uncounted run of it. A device
   keeps some memory for its life from the first work that needs it, such as an
   NV device's local memory, which a measure of what one call holds leaves
   out. *)
let warmed measure =
  ignore (measure ());
  measure ()

(* Values *)

(* How a compiled value agrees with eager's: bit for bit; bit for bit but for
   the sign of a zero, which a compiled extreme leaves to its target; or as a
   value computed in another order. *)
type agreement = Exact | Exact_up_to_zero | Rounded

(* An operation of one family, by name, over two float32 arguments of one shape,
   and how its compiled value agrees with eager's. *)
type family = {
  name : string;
  agreement : agreement;
  light : bool;  (** Whether the default run takes it. *)
  apply : Nx.float32_t -> Nx.float32_t -> Nx.float32_t;
}

let families =
  let f ?(agreement = Exact) ?(light = false) name apply =
    { name; agreement; light; apply }
  in
  [
    f ~agreement:Exact_up_to_zero ~light:true "neg, abs, max" (fun a b ->
        Nx.maximum (Nx.neg a) (Nx.abs b));
    f "where a less than b" (fun a b -> Nx.where (Nx.less a b) a b);
    f ~agreement:Rounded "exp and sin" (fun a b -> Nx.add (Nx.exp a) (Nx.sin b));
    f ~agreement:Rounded ~light:true "a sum over the last axis" (fun a b ->
        Nx.add a (Nx.sum ~axes:[ -1 ] ~keepdims:true b));
    f ~agreement:Exact_up_to_zero "a maximum over every axis" (fun a b ->
        Nx.mul a (Nx.max b));
    f ~agreement:Rounded "a running sum" (fun a b ->
        Nx.add a (Nx.cumsum ~axis:0 b));
    f "a transpose made contiguous" (fun a b ->
        Nx.add a (Nx.transpose (Nx.contiguous (Nx.transpose b))));
    f ~light:true "a flip and a pad" (fun a b ->
        Nx.add a
          (Nx.shrink
             (Array.map (fun n -> (1, n + 1)) (Nx.shape b))
             (Nx.pad (Array.map (fun _ -> (1, 1)) (Nx.shape b)) 0. (Nx.flip b))));
    f "a concatenation sliced back" (fun a b ->
        Nx.add a
          (Nx.shrink
             (Array.mapi
                (fun i n -> if i = 0 then (n, 2 * n) else (0, n))
                (Nx.shape b))
             (Nx.concatenate ~axis:0 [ a; b ])));
    f "a cast to int32 and back" (fun a b ->
        Nx.add a (Nx.cast Nx.float32 (Nx.cast Nx.int32 b)));
    f ~agreement:Rounded "a product with the transpose" (fun a b ->
        Nx.add a (Nx.matmul (Nx.matmul a (Nx.matrix_transpose b)) b));
  ]

(* Two arguments of one shape under one drawn layout: transposed, flipped, every
   other row, offset, broadcast, in windows. *)
let operands =
  let open Gen in
  let* rows = int_range 1 4 in
  let* cols = int_range 1 4 in
  let value = float_range (-2.) 2. in
  let* a = array ~size:(constant (rows * cols)) value in
  let* b = array ~size:(constant (rows * cols)) value in
  let+ steps = layout in
  let make xs = lay_out steps (Nx.create Nx.float32 [| rows; cols |] xs) in
  (make a, make b)

let laid =
  Gen.with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "%a@ %a" Nx.pp a Nx.pp b)
    operands

(* A triangular solve of 80 right-hand sides, wide enough to be solved in
   blocks, with each flag: the residual of the system each flag states. *)
let wide_solve =
  cases ~name:fst "a triangular solve of 80 right-hand sides solves its system"
    [
      ("lower", (false, false, false));
      ("upper, transposed, unit diagonal", (true, true, true));
    ]
    (fun (_, (upper, transpose, unit_diag)) ->
      let n = 80 in
      let a =
        Nx.init Nx.float64 [| n; n |] (fun i ->
            if i.(0) = i.(1) then 2.
            else
              float_of_int ((((i.(0) * 37) + (i.(1) * 11)) mod 13) - 6) /. 64.)
      in
      let b =
        Nx.init Nx.float64 [| n; n |] (fun i ->
            float_of_int ((((i.(0) * 5) + i.(1)) mod 7) - 3))
      in
      let system m =
        let t = if upper then Nx.triu ~k:1 m else Nx.tril ~k:(-1) m in
        let d =
          if unit_diag then Nx.eye Nx.float64 n else Nx.diag (Nx.diagonal m)
        in
        let m = Nx.add t d in
        if transpose then Nx.matrix_transpose m else m
      in
      let residual m =
        let x = Nx.solve_triangular ~upper ~transpose ~unit_diag m b in
        Nx.max (Nx.abs (Nx.sub (Nx.matmul (system m) x) b))
      in
      at_most float_exact ~than:1e-9 (Nx.item [] (Rune.jit' residual a)))

(* Values computed in another order than eager's: within a relative 1e-5, or
   within 2^-20 of the largest magnitude their terms reach, so that a sum that
   cancels to a value far below its terms is compared at its terms' scale. *)
let rounded a b =
  let largest t =
    Array.fold_left
      (fun m v -> if Float.is_finite v then Float.max m (Float.abs v) else m)
      0. (Nx.to_array t)
  in
  let n = Array.fold_left max 1 (Nx.shape b) in
  Oracle.tensor ~rel:1e-5
    ~abs:(Float.ldexp (largest a +. (float_of_int n *. largest b)) (-20))
    ()

(* Indices along an axis of 4, some 2^32 from one of its positions, which a
   truncation to 32 bits would bring back to it. *)
let far = 1 lsl 32

let far_index =
  Gen.frequency
    [
      (2, Gen.int_range (-2) 5);
      ( 1,
        let open Gen in
        let+ i = int_range 0 3
        and+ k = of_list ~pp:Format.pp_print_int [ -2; -1; 1; 2 ] in
        i + (k * far) );
    ]

(* The laws of values, [count] cases each; [heavy] adds the families the default
   run leaves out, and the tests whose programs take longest to compile. *)
let values ~count ~heavy =
  let law { name; agreement; apply; _ } =
    prop ~count name laid (fun (a, b) ->
        match apply a b with
        | expected -> (
            let actual = Rune.jit two apply a b in
            match agreement with
            | Exact -> equal floats expected actual
            | Exact_up_to_zero -> Traces.exact_up_to_zero expected actual
            | Rounded -> equal (rounded a b) expected actual)
        | exception Invalid_argument m ->
            (* An empty extreme raises eagerly; compiled, it raises too. *)
            raises_match ~msg:m Exn.invalid_arg (fun () ->
                Rune.jit two apply a b))
  in
  group "values"
    ([
       group "one operation per family equals eager"
         (List.map law (List.filter (fun f -> heavy || f.light) families));
       test "a replay reads its new arguments, and an earlier call's again"
         (fun () ->
           let g = Rune.jit' poly in
           equal floats (poly (x ())) (g (x ()));
           equal floats (poly (y ())) (g (y ()));
           equal floats (poly (x ())) (g (x ())));
       test "a structured result equals eager's leaf by leaf" (fun () ->
           let s = Nx.Ptree.(pair tensor (list (option tensor))) in
           let f a = (Nx.neg a, [ Some (poly a); None; Some a ]) in
           equal (Oracle.structure s)
             (f (x ()))
             (Rune.jit Nx.Ptree.(tensor @-> returns s) f (x ())));
       test "64-bit integer constants keep every bit" (fun () ->
           let a = Nx.create Nx.int64 [| 2 |] [| 1L; -1L |] in
           let f a =
             Nx.add a
               (Nx.create Nx.int64 [| 2 |] [| Int64.max_int; Int64.min_int |])
           in
           equal (tensor int64) (f a) (Rune.jit' f a));
       test "integer constants wrap at the operand's width" (fun () ->
           let a = Nx.create Nx.int8 [| 3 |] [| 127; -128; 100 |] in
           let f a = Nx.add (Nx.mul_s a 3) (Nx.full Nx.int8 [| 3 |] 100) in
           equal (tensor int) (f a) (Rune.jit' f a));
       test "float identities hold only where IEEE keeps them" (fun () ->
           let a =
             Nx.create Nx.float32 [| 4 |]
               [| -0.; 0.; Float.infinity; Float.nan |]
           in
           let f a =
             Nx.stack ~axis:0
               [
                 Nx.add a (Nx.zeros_like a);
                 Nx.div a a;
                 Nx.mul a (Nx.zeros_like a);
               ]
           in
           equal floats (f a) (Rune.jit' f a));
       test "a bitcast's result has its dtype and its argument's bits"
         (fun () ->
           let a = Nx.create Nx.float32 [| 3 |] [| 1.; -0.; Float.nan |] in
           let r = Rune.jit' (Nx.bitcast Nx.int32) a in
           equal (tensor int32)
             (Nx.create Nx.int32 [| 3 |]
                [| 0x3f800000l; Int32.min_int; Int32.bits_of_float Float.nan |])
             r);
       test "a bitcast between widths reads the bytes eager reads" (fun () ->
           let bytes =
             Nx.init Nx.uint8 [| 2; 8 |] (fun i ->
                 ((i.(0) * 8) + i.(1)) * 29 mod 256)
           in
           let words = Nx.bitcast Nx.uint64 bytes in
           equal (tensor int64)
             (Nx.bitcast Nx.int64 words)
             (Nx.bitcast Nx.int64 (Rune.jit' (Nx.bitcast Nx.uint64) bytes));
           equal (tensor int) bytes (Rune.jit' (Nx.bitcast Nx.uint8) words));
       cases ~name:fst "an arange inside a compiled call equals eager's"
         [
           ("1 element", (0, 1, 1));
           ("257 elements", (0, 257, 1));
           ("513 elements", (0, 513, 1));
           ("2^20 elements", (0, 1 lsl 20, 1));
           ("down by 2 from 1000", (1000, -26, -2));
           ( "from 2^40 by more than 2^34",
             ( 1 lsl 40,
               (1 lsl 40) + (8 * ((1 lsl 34) + 12345)),
               (1 lsl 34) + 12345 ) );
         ]
         (fun (_, (start, stop, step)) ->
           let arange () = Nx.arange Nx.int64 start stop step in
           let eager = arange () in
           equal (tensor int64) eager
             (Rune.jit' (fun z -> Nx.add z (arange ())) (Nx.zeros_like eager)));
       test "an int32 and a float32 arange inside a compiled call equal eager's"
         (fun () ->
           let i () = Nx.arange Nx.int32 0 513 1 in
           equal (tensor int32) (i ())
             (Rune.jit' (fun z -> Nx.add z (i ())) (Nx.zeros_like (i ())));
           let f () = Nx.arange Nx.float32 0 513 1 in
           equal floats (f ())
             (Rune.jit' (fun z -> Nx.add z (f ())) (Nx.zeros_like (f ()))));
       test "a bfloat16 arange from 2^40 inside a compiled call equals eager's"
         (fun () ->
           let a () =
             Nx.arange Nx.bfloat16 (1 lsl 40)
               ((1 lsl 40) + (8 * ((1 lsl 31) + 12345)))
               ((1 lsl 31) + 12345)
           in
           equal floats
             (Nx.cast Nx.float32 (a ()))
             (Nx.cast Nx.float32
                (Rune.jit' (fun z -> Nx.add z (a ())) (Nx.zeros_like (a ())))));
       test "top_k puts NaN first, as eager does" (fun () ->
           let scores =
             Nx.init Nx.float32 [| 2; 24 |] (fun i ->
                 let i = (i.(0) * 24) + i.(1) in
                 if i mod 5 = 3 then Float.nan else float_of_int (i * 7 mod 11))
           in
           List.iter
             (fun k ->
               let indices x = snd (Nx.top_k ~k x) in
               let eager = indices scores in
               equal ~msg:"the first NaN first" (tensor int64)
                 (Nx.scalar Nx.int64 3L)
                 (Nx.slice [ I 0; I 0 ] eager);
               equal
                 ~msg:(Printf.sprintf "top %d" k)
                 (tensor int64) eager (Rune.jit' indices scores))
             [ 2; 17 ]);
       test "top_k of a short row ranked by counting is eager's" (fun () ->
           let x =
             Nx.create Nx.float32 [| 2; 6 |]
               [|
                 1.; -0.; Float.nan; 0.; 1.; -1.; 2.; 2.; -0.; Float.nan; 0.; 2.;
               |]
           in
           List.iter
             (fun k ->
               let top x = Nx.top_k ~k ~axis:1 x in
               let v, i = top x in
               let v', i' =
                 Rune.jit
                   Nx.Ptree.(tensor @-> returns (pair tensor tensor))
                   top x
               in
               equal ~msg:(Printf.sprintf "values, top %d" k) floats v v';
               equal
                 ~msg:(Printf.sprintf "indices, top %d" k)
                 (tensor int64) i i')
             [ 1; 3; 6 ]);
       prop "gather and scatter at indices 2^32 from a position equal eager's"
         ~examples:[ [| far + 1; 1 - far; 2; -1; 4; far |] ]
         (Gen.array ~size:(Gen.constant 6) far_index)
         (fun idx ->
           let indices =
             Nx.create Nx.int64 [| 6 |] (Array.map Int64.of_int idx)
           in
           let t = Nx.reshape [| 4; 2 |] (Nx.arange_f Nx.float32 1. 9. 1.) in
           let scatter mode indices t =
             Nx.scatter ~mode ~axis:0
               ~indices:
                 (Nx.broadcast_to [| 6; 2 |] (Nx.reshape [| 6; 1 |] indices))
               ~values:
                 (Nx.reshape [| 6; 2 |] (Nx.arange_f Nx.float32 10. 22. 1.))
               t
           in
           List.iter
             (fun (msg, f) ->
               equal ~msg floats (f indices t)
                 (Rune.jit
                    Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                    f indices t))
             [
               ("take", fun indices t -> Nx.take ~axis:0 ~indices t);
               ("scatter set", scatter `Set);
               ("scatter add", scatter `Add);
             ]);
       test "quantiles inside a compiled call equal eager's" (fun () ->
           let a =
             Nx.create Nx.float32 [| 2; 5 |]
               [| 3.; Float.nan; -0.; 1.; 2.; 4.; 4.; Float.infinity; 0.; -1. |]
           in
           let f = Nx.quantile ~axis:1 [| 0.; 0.3; 0.5; 0.9; 1. |] in
           equal floats (f a) (Rune.jit' f a);
           let g = Nx.quantile [| 0.25; 0.75 |] in
           let b = Nx.cast Nx.float64 a in
           equal (tensor float_exact) (g b) (Rune.jit' g b));
       test "a bitmap packed and read inside a compiled call is eager's"
         (fun () ->
           let m =
             Nx.init Nx.bool [| 21 |] (fun i -> i.(0) mod 3 = 0 || i.(0) = 7)
           in
           let packed m = fst (Nx_bits.bytes (Nx_bits.of_bool m)) in
           equal (tensor int) (packed m) (Rune.jit' packed m);
           let b = Nx_bits.sub (Nx_bits.of_bool m) ~offset:3 ~length:15 in
           let bytes, offset = Nx_bits.bytes b in
           let read f bytes = f (Nx_bits.v ~offset ~length:15 bytes) in
           equal (tensor bool) (Nx_bits.to_bool b)
             (Rune.jit' (read Nx_bits.to_bool) bytes);
           equal (tensor int64) (Nx_bits.count b)
             (Rune.jit' (read Nx_bits.count) bytes));
       test "a ragged array grouped by ids inside a compiled call is eager's"
         (fun () ->
           let ids = Nx.create Nx.int64 [| 6 |] [| 2L; 0L; -1L; 2L; 3L; 0L |] in
           let x = Nx.reshape [| 6; 2 |] (Nx.arange_f Nx.float32 0. 12. 1.) in
           let grouped ids x =
             let r = Nx_ragged.of_ids ~segments:3 ids x in
             (Nx_ragged.offsets r, Nx_ragged.values (Nx_ragged.map Nx.neg r))
           in
           let offsets, values = grouped ids x in
           let offsets', values' =
             Rune.jit
               Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
               grouped ids x
           in
           equal (tensor int64) offsets offsets';
           equal floats values values';
           let medians ids x =
             Nx_ragged.quantile [| 0.; 0.5; 1. |]
               (Nx_ragged.of_ids ~segments:3 ids (Nx.flatten x))
           in
           let ids = Nx.concatenate ~axis:0 [ ids; ids ] in
           equal floats (medians ids x)
             (Rune.jit
                Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                medians ids x));
       test "a zero-size result is an empty tensor" (fun () ->
           let a = Nx.zeros Nx.float32 [| 0; 3 |] in
           let r = Rune.jit' poly a in
           equal (array int) [| 0; 3 |] (Nx.shape r));
     ]
    @ if heavy then [ wide_solve ] else [])

(* Keys *)

(* A structure that reports an integer. *)
module Windowed = struct
  type 'a t = { n : int; x : 'a }

  let walk c { n; x } =
    let open Nx.Ptree.Walk in
    let n = field c "n" int n in
    let x = field c "x" leaf x in
    { n; x }
end

let windowed : (float, Nx.float32_elt) Nx.t Windowed.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Windowed)

(* A structure with two cases. *)
module Choice = struct
  type 'a t = Left of 'a | Right of 'a

  let walk c =
    let open Nx.Ptree.Walk in
    function
    | Left x ->
        case c "left";
        Left (leaf c x)
    | Right x ->
        case c "right";
        Right (leaf c x)
end

let choice : (float, Nx.float32_elt) Nx.t Choice.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Choice)

(* [retraces first other] asserts that [first ()] traces once, [other ()] once
   more, and [other ()] again not at all. *)
let retraces first other =
  equal ~msg:"the first call" int 1 (traces first);
  equal ~msg:"the other key" int 1 (traces other);
  equal ~msg:"the other key again" int 0 (traces other)

(* [shares first other] asserts that [other ()] replays the program of [first
   ()]. *)
let shares first other =
  first ();
  equal ~msg:"traces" int 0 (traces other)

(* [checked g f a] is [g a], checked against eager's [f a]. *)
let checked g f a () = equal close (f a) (g a)
let arange n = Nx.arange_f Nx.float32 0. (float_of_int n) 1.
let grid r c = Nx.reshape [| r; c |] (arange (r * c))

(* Every other element of [t] along its first axis, as a view. *)
let every_other t =
  Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~axis:0 ~window:1 ~step:2 t)

let keys =
  let g () = Rune.jit' poly in
  group "keys"
    [
      test "a call with the key of an earlier one replays its program"
        (fun () ->
          let f, ran = counted poly in
          let g = Rune.jit' f in
          ignore (g (x ()));
          equal int 0 (traces (fun () -> ignore (g (y ()))));
          equal int 1 !ran);
      test "another extent retraces once" (fun () ->
          let g = g () in
          retraces (checked g poly (x ())) (checked g poly (arange 5)));
      test "another rank retraces once" (fun () ->
          let g = g () in
          retraces (checked g poly (arange 4)) (checked g poly (grid 2 2)));
      test "strides out of C order retrace once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 2 3))
            (checked g poly (Nx.transpose (grid 3 2))));
      test
        "an argument starting 4 bytes further within 16 bytes of memory \
         retraces once" (fun () ->
          let g = g () in
          let a = arange 12 in
          retraces
            (checked g poly (Nx.slice [ R (0, 4) ] a))
            (checked g poly (Nx.slice [ R (1, 5) ] a)));
      test "an argument starting 16 bytes further shares the program" (fun () ->
          let g = g () in
          let a = arange 12 in
          shares
            (checked g poly (Nx.slice [ R (0, 4) ] a))
            (checked g poly (Nx.slice [ R (4, 8) ] a)));
      test
        "strides out of C order only on axes of one element share the program"
        (fun () ->
          let g = g () in
          shares
            (checked g poly (grid 4 1))
            (checked g poly (Nx.transpose (grid 1 4))));
      test "another reported integer retraces once" (fun () ->
          let f { Windowed.n; x } = Nx.mul_s x (float_of_int n) in
          let g = Rune.jit Nx.Ptree.(windowed @-> returns tensor) f in
          let w n () = { Windowed.n; x = x () } in
          retraces
            (fun () -> equal close (f (w 1 ())) (g (w 1 ())))
            (fun () -> equal close (f (w 2 ())) (g (w 2 ()))));
      slow "another case retraces once" (fun () ->
          let f = function Choice.Left x -> Nx.neg x | Right x -> poly x in
          let g = Rune.jit Nx.Ptree.(choice @-> returns tensor) f in
          retraces
            (fun () -> equal close (f (Left (x ()))) (g (Left (x ()))))
            (fun () -> equal close (f (Right (x ()))) (g (Right (x ())))));
      slow "another list length retraces once" (fun () ->
          let f l = List.fold_left Nx.add (x ()) l in
          let g = Rune.jit Nx.Ptree.(list tensor @-> returns tensor) f in
          retraces
            (fun () -> equal close (f [ y () ]) (g [ y () ]))
            (fun () -> equal close (f [ y (); y () ]) (g [ y (); y () ])));
      slow "an option's presence retraces once" (fun () ->
          let f = function None -> x () | Some a -> poly a in
          let g = Rune.jit Nx.Ptree.(option tensor @-> returns tensor) f in
          retraces
            (fun () -> equal close (f None) (g None))
            (fun () -> equal close (f (Some (y ()))) (g (Some (y ())))));
      slow "a key met again after another replays its first program" (fun () ->
          let g = g () in
          ignore (g (x ()));
          ignore (g (arange 5));
          equal int 0 (traces (fun () -> ignore (g (y ())))));
      slow "two compiled functions of one function keep their own programs"
        (fun () ->
          let g1 = Rune.jit' poly and g2 = Rune.jit' poly in
          equal int 1 (traces (fun () -> ignore (g1 (x ()))));
          equal int 1 (traces (fun () -> ignore (g2 (x ())))));
      test "a change of NOOPT around a call retraces once" (fun () ->
          let g = g () in
          let noopt f =
            Tolk.Helpers.context [ B (Tolk.Helpers.noopt, true) ] f
          in
          retraces
            (checked g poly (x ()))
            (fun () -> noopt (checked g poly (x ()))));
      (* A program of its own, whose kernel no earlier search chose: BEAM asks
         for a search, which times candidates on the host. *)
      test "a call under BEAM=1 searches its kernel and computes eager's values"
        (fun () ->
          let f a = Nx.add_s (poly a) 0.375 in
          let r =
            Tolk.Helpers.context
              [ B (Tolk.Helpers.beam, 1) ]
              (fun () -> Rune.jit' f (x ()))
          in
          equal close (f (x ())) r);
      test "a call compiled with ~beam:1 computes eager's values" (fun () ->
          let f a = Nx.add_s (poly a) 0.6875 in
          equal close (f (x ())) (Rune.jit' ~beam:1 ~parallel:2 f (x ())));
      slow "a flipped view retraces once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 2 3))
            (checked g poly (Nx.flip (grid 2 3))));
      slow "a broadcast view retraces once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 2 3))
            (checked g poly (Nx.broadcast_to [| 2; 3 |] (grid 1 3))));
      slow "a view skipping elements retraces once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (arange 3))
            (checked g poly (every_other (arange 6))));
      slow "overlapping windows retrace once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 4 2))
            (checked g poly (Nx.sliding_window ~window:2 (arange 5))));
      slow "another device retraces once" (fun () ->
          let g = g () in
          retraces
            (fun () -> equal close (poly (x ())) (host (g (placed d1 (x ())))))
            (fun () -> equal close (poly (x ())) (host (g (placed d2 (x ()))))));
      slow "another backend on one device retraces once" (fun () ->
          let g = g () in
          retraces
            (fun () -> equal close (poly (x ())) (host (g (placed d1 (x ())))))
            (fun () ->
              equal close
                (poly (x ()))
                (host (g (Nx.place (Nx.Placement.device d1) (x ()))))));
      slow "a split value after a replicated one retraces once" (fun () ->
          let g = g () in
          let on p () =
            equal close (poly (x ())) (host (g (Nx.place p (x ()))))
          in
          retraces
            (on (Nx.Placement.replicated ~backend:Rune.compiled [ d1; d2 ]))
            (on
               (Nx.Placement.sharded ~backend:Rune.compiled ~axis:0 [ d1; d2 ])));
    ]

(* Results *)

let distinct a b =
  is_false ~msg:"distinct storage"
    (List.exists2 Nativeint.equal (Witness.addresses a) (Witness.addresses b))

let results =
  group "results"
    [
      test "every result leaf has storage of its own" (fun () ->
          let a, b =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              (fun a -> (Nx.neg a, poly a))
              (x ())
          in
          distinct a b);
      test "a result that returns a read argument is a copy" (fun () ->
          let a = x () in
          let r = Rune.jit' Fun.id a in
          equal floats a r;
          distinct a r);
      test "a result that returns a capture is a copy" (fun () ->
          let w = y () in
          let r = Rune.jit' (fun _ -> w) (x ()) in
          equal floats w r;
          distinct w r);
      test "a value at two result leaves comes back as two values" (fun () ->
          let a, b =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              (fun a ->
                let r = poly a in
                (r, r))
              (x ())
          in
          equal floats a b;
          distinct a b);
      test "one value passed at two read leaves is read at both" (fun () ->
          let a = x () in
          equal floats (Nx.add a a) (Rune.jit two Nx.add a a));
      test "a result can be a structure that checks its leaves' shapes"
        (fun () ->
          let next =
            Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
          in
          let step k = (Nx.Rng.split k).(0) in
          let k = Nx.Rng.key 42 in
          equal (tensor int32)
            (step k :> (int32, Nx.int32_elt) Nx.t)
            (next step k :> (int32, Nx.int32_elt) Nx.t));
    ]

(* Consumption *)

let consumed_message a = message (fun () -> Nx.to_array a)

(* The host value of the [n] floats of [b] from its [first]. *)
let window b first n =
  Nx.Repr.host
    {
      Nx_array.dtype = Nx.float32;
      view = Nx_array.View.create [| n |];
      buffer = Nx_device.Buffer.view b ~offset:(4 * first) Float32 n;
    }

let floats_buffer v =
  let b = Nx_device.Buffer.create Nx_device.host Float32 (Array.length v) in
  let ba = Nx_device.Buffer.bigarray Bigarray.float32 b in
  Array.iteri (Bigarray.Array1.set ba) v;
  b

let test_consumed_window () =
  let b = floats_buffer [| 1.; -2.; 3.; 0.5; 2.; 0.; -1.; 4. |] in
  let first = window b 0 4 and second = window b 4 4 in
  let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) second in
  equal floats (Nx.add_s (y ()) 1.) r;
  equal ~msg:"the window" floats (y ()) second;
  equal ~msg:"its sibling" floats (x ()) first

(* A call consuming a slice of [n] elements at [offset] of a longer value at
   [at], whose storage it consumes. *)
let consumed_slice ~at offset n =
  let parent =
    Nx.place at
      (Nx.init Nx.float32 [| offset + n + 3 |] (fun i -> Float.of_int i.(0)))
  in
  let a = Nx.slice [ R (offset, offset + n) ] parent in
  let f a = Nx.add_s (Nx.mul_s a 2.) 1. in
  let expected =
    f (Nx.init Nx.float32 [| n |] (fun i -> Float.of_int (offset + i.(0))))
  in
  let r = Rune.jit consumes f a in
  equal ~msg:"the result" floats expected (Nx.place Nx.Placement.host r);
  raises_match ~msg:"the parent" (Exn.invalid_arg ~substring:"consumed at 0")
    (fun () -> Nx.to_array parent)

(* The memory of host value [a]. *)
let memory a =
  match Nx.Repr.v a with
  | Host h -> h.buffer
  | Placed _ | Traced _ -> fail "expected a host value"

(* Another domain's read of [a], in flight while a call consumes it, is a read
   claim on its memory: the call cannot have it exclusive, so it computes from a
   copy, and the read sees the elements it started with. *)
let test_consumed_while_read () =
  let a = x () in
  let m = memory a in
  let address = Nx_device.Buffer.address m in
  let elements = Nx_device.Buffer.bigarray Bigarray.float32 m in
  Nx_device.Buffer.Claim.read m;
  let r =
    Fun.protect ~finally:(fun () -> Nx_device.Buffer.Claim.release m)
    @@ fun () ->
    let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
    equal ~msg:"the read's elements" (array float_exact) [| 1.; -2.; 3.; 0.5 |]
      (Array.init 4 (Bigarray.Array1.get elements));
    r
  in
  is_false ~msg:"lent" (Witness.addresses r = [ address ]);
  equal floats (Nx.add_s (x ()) 1.) r;
  raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
      Nx.to_array a)

(* A call that lends [a] holds its memory exclusive while it runs; another
   domain's read of [a] then raises at once and never sees the write. *)
let test_read_while_lent () =
  let a = x () in
  let m = memory a in
  Nx_device.Buffer.Claim.read m;
  is_true ~msg:"exclusive" (Nx_device.Buffer.Claim.try_exclusive m);
  Fun.protect
    ~finally:(fun () ->
      Nx_device.Buffer.Claim.finish m;
      Nx_device.Buffer.Claim.release m)
    (fun () ->
      raises_match (Exn.invalid_arg ~substring:"in use") (fun () -> Nx.neg a);
      raises_match (Exn.invalid_arg ~substring:"in use") (fun () ->
          Nx.to_array a));
  equal floats (x ()) a

let consumption =
  group "consumption"
    [
      test "a consumed argument raises on read, naming its path" (fun () ->
          let a = x () in
          ignore (Rune.jit consumes (fun a -> Nx.add_s a 1.) a);
          let m = consumed_message a in
          is_true ~msg:m (String.length m > 0);
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              ignore (Nx.to_array a)));
      test "a consumed argument keeps its shape and dtype" (fun () ->
          let a = Nx.zeros Nx.float32 [| 2; 3 |] in
          ignore (Rune.jit consumes Nx.neg a);
          equal (array int) [| 2; 3 |] (Nx.shape a);
          is_true (Nx_dtype.equal Nx.float32 (Nx.dtype a)));
      test "a consumed argument raises as an operand and as an argument"
        (fun () ->
          let a = x () in
          ignore (Rune.jit consumes Nx.neg a);
          raises_invalid_arg (fun () -> Nx.add a a);
          raises_invalid_arg (fun () -> Rune.jit' Nx.neg a);
          raises_invalid_arg (fun () -> Rune.jit consumes Nx.neg a));
      test "a view of consumed storage taken before the call raises on read"
        (fun () ->
          let a = x () in
          let v = Nx.slice [ R (0, 2) ] a in
          ignore (Rune.jit consumes Nx.neg a);
          raises_invalid_arg (fun () -> Nx.to_array v));
      test
        "a consumed slice is computed from a copy, and its storage dies with it"
        (fun () ->
          let whole = x () in
          let a = Nx.slice [ R (0, 2) ] whole in
          let r = Rune.jit consumes Nx.neg a in
          equal floats (Nx.neg (Nx.slice [ R (0, 2) ] (x ()))) r;
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              Nx.to_array a);
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              Nx.to_array whole));
      test
        "a consumed broadcast of one element of its storage is computed from a \
         copy, and dies" (fun () ->
          let a = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
          let v = Nx.broadcast_to [| 2 |] (Nx.slice [ R (0, 1) ] a) in
          equal floats
            (Nx.create Nx.float32 [| 2 |] [| -1.; -1. |])
            (Rune.jit consumes Nx.neg v);
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              Nx.to_array v));
      cases "a consumed slice at an offset is computed into storage of its own"
        ~name:(fun (o, n) -> Printf.sprintf "%d elements at %d" n o)
        [ (1, 1); (1, 7); (5, 7); (5, 1027) ]
        (fun (offset, n) -> consumed_slice ~at:Nx.Placement.host offset n);
      test "a call that consumes zeros as its state runs, and the state dies"
        (fun () ->
          let state = Nx.zeros Nx.float32 [| 4 |] in
          let r = Rune.jit consumes (fun s -> Nx.add s (x ())) state in
          equal floats (x ()) r;
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              Nx.to_array state));
      test
        "a consumed window of a larger memory is computed from a copy, and it \
         and its siblings stay live"
        test_consumed_window;
      test "a consumed value another domain reads is copied, not lent"
        test_consumed_while_read;
      test "a value read while a call holds its memory exclusive raises busy"
        test_read_while_lent;
      test
        "a consumed leaf that another leaf reaches raises before any work, \
         naming both paths" (fun () ->
          let a = x () in
          let g =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> returns tensor)
              Nx.add
          in
          raises_match
            (Exn.invalid_arg
               ~substring:"0 is consumed and 1 reaches its storage") (fun () ->
              g a a);
          equal floats (x ()) a);
      test "two consumed leaves over one storage raise before any work"
        (fun () ->
          let a = x () in
          let g =
            Rune.jit
              Nx.Ptree.(consumes (pair tensor tensor) @@ returns tensor)
              (fun (a, b) -> Nx.add a b)
          in
          raises_invalid_arg (fun () -> g (a, a));
          equal floats (x ()) a);
      test "a consumed leaf whose storage the function captures raises"
        (fun () ->
          let w = y () in
          let g = Rune.jit consumes (fun a -> Nx.add a w) in
          raises_match
            (Exn.invalid_arg
               ~substring:"0 is consumed and the function captures its storage")
            (fun () -> g w);
          equal floats (y ()) w);
      test "a call that raises while tracing consumes nothing" (fun () ->
          let a = x () in
          raises_jit_error (fun () ->
              Rune.jit consumes
                (fun a -> if Nx.item [ 0 ] a > 0. then a else Nx.neg a)
                a);
          equal floats (x ()) a);
      test "read arguments stay readable after any number of calls" (fun () ->
          let a = x () in
          let g = Rune.jit' poly in
          for _ = 1 to 3 do
            ignore (g a)
          done;
          equal floats (x ()) a);
    ]

(* Lending *)

let state =
  Nx.Ptree.(consumes (pair tensor tensor) @@ returns (pair tensor tensor))

(* A write of rows into a pool of [n] rows of [2; 3] (64 by default), at one
   index per row: a projection of [x], one row of [x] per index, of integers, so
   that every sum is exact. *)
let pool ?(n = 64) () =
  Nx.reshape [| n; 2; 3 |] (Nx.arange_f Nx.float32 0. (Float.of_int (6 * n)) 1.)

let write_rows ?(rows_as = Fun.id) pool x indices =
  let k = Nx.dim 0 indices in
  let rows =
    Nx.reshape [| k; 2; 3 |]
      (Nx.matmul x (Nx.reshape [| 6; 6 |] (Nx.arange_f Nx.float32 0. 36. 1.)))
  in
  Nx.scatter ~unique_indices:true ~axis:0
    ~indices:(Nx.broadcast_to [| k; 2; 3 |] (Nx.reshape [| k; 1; 1 |] indices))
    ~values:(rows_as rows) pool

let projections k =
  Nx.reshape [| k; 6 |]
    (Nx.arange_f Nx.float32 1. (Float.of_int ((6 * k) + 1)) 1.)

let indices l = Nx.create Nx.int64 [| List.length l |] (Array.of_list l)

(* [kernels f] is [f ()] and the names of the kernels it ran, in order. *)
let kernels f =
  let p = Nx_device.Profile.start () in
  let y = f () in
  let kernel name =
    String.length name >= 1
    && (name.[0] = 'E' || name.[0] = 'r')
    && (String.length name = 1 || name.[1] = '_')
  in
  ( y,
    List.filter_map
      (function
        | Nx_device.Profile.Span { name; _ } when kernel name -> Some name
        | _ -> None)
      (Nx_device.Profile.stop p) )

(* The elements a kernel ranges over, read from its name: [E_2_4], 8. *)
let ranged name =
  List.fold_left
    (fun n part ->
      match int_of_string_opt part with Some d -> n * d | None -> n)
    1
    (List.tl (String.split_on_char '_' name))

(* A decode step as gpt-oss's attention takes it: the token's slot looked up in
   a table at its position, its key and value rows written there into two pools,
   and an attention of its query over the pools' rows read back through the
   table. *)
let decode_step (keys, values) x pos table =
  let slot = Nx.take_along_axis ~axis:1 ~indices:pos table in
  let indices = Nx.broadcast_to [| 1; 2; 3 |] (Nx.reshape [| 1; 1; 1 |] slot) in
  let weight k =
    Nx.reshape [| 6; 6 |] (Nx.arange_f Nx.float32 k (k +. 36.) 1.)
  in
  let row k = Nx.reshape [| 1; 2; 3 |] (Nx.matmul x (weight k)) in
  let write pool k =
    Nx.scatter ~unique_indices:true ~axis:0 ~indices ~values:(row k) pool
  in
  let keys = write keys 0. and values = write values 36. in
  let read pool = Nx.take ~axis:0 ~indices:(Nx.reshape [| 16 |] table) pool in
  let q = Nx.contiguous (Nx.reshape [| 2; 3 |] (row 72.)) in
  let scores = Nx.sum ~axes:[ 2 ] (Nx.mul (read keys) q) in
  let weights = Nx.softmax ~axes:[ 0 ] (Nx.div_s scores 1e4) in
  let out =
    Nx.sum ~axes:[ 0 ] (Nx.mul (Nx.unsqueeze ~axes:[ 2 ] weights) (read values))
  in
  (out, (keys, values))

(* [rows_written ?at name] is the tests of a lent write of rows whose values are
   placed at [at], on the host by default. *)
let rows_written ?at name =
  let write =
    Nx.Ptree.(consumes tensor @@ tensor @-> tensor @-> returns tensor)
  in
  let on t = match at with None -> t | Some p -> Nx.place p t in
  let agrees ?n x i =
    equal floats
      (write_rows (pool ?n ()) x i)
      (host (Rune.jit write write_rows (on (pool ?n ())) (on x) (on i)))
  in
  let everywhere =
    [
      cases
        ~name:(fun (n, l) ->
          Printf.sprintf "%d rows: %s" n
            (String.concat ", " (List.map Int64.to_string l)))
        "a dropped row changes no other row"
        [
          (8, [ 2L; -1L ]);
          (8, [ -1L; 2L; 3L ]);
          (8, [ 5L; -1L; 6L ]);
          (8, [ 6L; 0L; -1L; 8L ]);
          (64, [ -1L; 0L ]);
          (64, [ 0L; -1L ]);
          (64, [ -1L; -5L ]);
        ]
        (fun (n, l) -> agrees ~n (projections (List.length l)) (indices l));
      test
        "decode steps write their cache rows and attend over them, as eager \
         does" (fun () ->
          let pools () = (pool ~n:16 (), pool ~n:16 ()) in
          let eager = ref (pools ()) in
          let compiled =
            let k, v = pools () in
            ref (on k, on v)
          in
          let table =
            Nx.create Nx.int64 [| 1; 16 |]
              (Array.init 16 (fun j -> Int64.of_int (15 - j)))
          in
          let both = Nx.Ptree.(pair tensor tensor) in
          let step =
            Rune.jit
              Nx.Ptree.(
                consumes both @@ tensor @-> tensor @-> tensor
                @-> returns (pair tensor both))
              decode_step
          in
          List.iteri
            (fun i p ->
              let x = Nx.mul_s (projections 1) (Float.of_int (i + 1))
              and pos = Nx.create Nx.int64 [| 1; 1 |] [| p |] in
              let out, (keys, values) = decode_step !eager x pos table in
              let out', (keys', values') =
                step !compiled (on x) (on pos) (on table)
              in
              eager := (keys, values);
              compiled := (keys', values');
              equal close out (host out');
              equal floats keys (host keys');
              equal floats values (host values'))
            [ 0L; 7L; 15L; 7L ]);
      test "an unlent write of rows leaves the pool it writes into" (fun () ->
          let a = on (pool ())
          and x = projections 2
          and i = indices [ 3L; 5L ] in
          let r =
            Rune.jit
              Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
              write_rows a (on x) (on i)
          in
          equal floats (write_rows (pool ()) x i) (host r);
          equal floats (pool ()) (host a));
    ]
  in
  let on_the_host =
    [
      cases ~name:Int64.to_string
        "a row is stored by the kernel that computes it, beside its offset"
        [ 0L; 63L; -1L; 64L ] (fun at ->
          let x = projections 1 and i = indices [ at ] in
          let r, names =
            kernels (fun () -> Rune.jit write write_rows (pool ()) x i)
          in
          equal floats (write_rows (pool ()) x i) r;
          equal ~msg:"kernels" int 2 (List.length names);
          equal ~msg:"reductions" int 1
            (List.length (List.filter (fun n -> n.[0] = 'r') names)));
      test
        "a row made contiguous is stored by the kernel that computes it, as a \
         key-value cache writes its rows" (fun () ->
          let x = projections 1 and i = indices [ 5L ] in
          let rows_as = Nx.contiguous in
          let r, names =
            kernels (fun () ->
                Rune.jit write (write_rows ~rows_as) (pool ()) x i)
          in
          equal floats (write_rows ~rows_as (pool ()) x i) r;
          equal ~msg:"kernels" int 2 (List.length names));
      test
        "stores only its rows, at the pool's first and last rows, and drops \
         the rows outside it" (fun () ->
          let x = projections 4 and i = indices [ 0L; 63L; -1L; 64L ] in
          let r, names =
            kernels (fun () -> Rune.jit write write_rows (pool ()) x i)
          in
          equal floats (write_rows (pool ()) x i) r;
          greater ~msg:"kernels recorded" int ~than:0 (List.length names);
          List.iter
            (fun name ->
              less ~msg:"elements a kernel ranges over" int ~than:384
                (ranged name))
            names);
      cases
        ~name:(fun l -> String.concat ", " (List.map Int64.to_string l))
        "a pool split along the written axis is written whole"
        [ [ 0L; 63L ]; [ 31L; 32L ]; [ 63L; 0L; -1L; 64L ]; [ -1L; -2L ] ]
        (fun l ->
          let both =
            Nx.Placement.replicated ~backend:Rune.compiled [ d1; d2 ]
          in
          let x = projections (List.length l) and i = indices l in
          equal floats
            (write_rows (pool ()) x i)
            (host
               (Rune.jit write write_rows
                  (Nx.place
                     (Nx.Placement.sharded ~backend:Rune.compiled ~axis:0
                        [ d1; d2 ])
                     (pool ()))
                  (Nx.place both x) (Nx.place both i))));
    ]
  in
  group name (everywhere @ if at = None then on_the_host else [])

let lending =
  group "lending"
    [
      test "a consumed host argument lends its storage to the result" (fun () ->
          let a = x () in
          let before = address a in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal floats (Nx.add_s (x ()) 1.) r;
          equal nativeint before (address r));
      test "a result takes the consumed leaf it derives from at its own index"
        (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let r1, r2 =
            Rune.jit state (fun (a, b) -> (Nx.add_s b 1., Nx.mul_s a 2.)) (a, b)
          in
          equal (pair nativeint nativeint) ab (address r2, address r1));
      test
        "a result read through a flip of a leaf takes a leaf it does not read, \
         and the leaf goes to a result derived at its own index" (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let flipped, own =
            Rune.jit state
              (fun (a, _) ->
                let flipped = Nx.add_s (Nx.flip a) 1. in
                let own = Nx.mul_s a 2. in
                (flipped, own))
              (a, b)
          in
          equal floats (Nx.add_s (Nx.flip (x ())) 1.) flipped;
          equal floats (Nx.mul_s (x ()) 2.) own;
          equal (pair nativeint nativeint) ab (address own, address flipped));
      test "an indexed write takes the leaf it writes before any other result"
        (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let r1, r2 =
            Rune.jit state
              (fun (a, _) ->
                ( Nx.full Nx.float32 [| 4 |] 7.,
                  Nx.set [ I 0 ] (Nx.scalar Nx.float32 9.) a ))
              (a, b)
          in
          equal floats (Nx.set [ I 0 ] (Nx.scalar Nx.float32 9.) (x ())) r2;
          equal (pair nativeint nativeint) ab (address r2, address r1));
      test "the other results take the free leaves in walk order" (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let r1, r2 =
            Rune.jit state
              (fun _ ->
                let r1 = Nx.full Nx.float32 [| 4 |] 1. in
                let r2 = Nx.full Nx.float32 [| 4 |] 2. in
                (r1, r2))
              (a, b)
          in
          equal (pair nativeint nativeint) ab (address r1, address r2));
      test "a storage lends to one result" (fun () ->
          let a = x () in
          let before = address a in
          let r1, r2 =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ returns (pair tensor tensor))
              (fun a -> (Nx.add_s a 1., Nx.mul_s a 2.))
              a
          in
          equal nativeint before (address r1);
          distinct r1 r2;
          equal floats (Nx.mul_s (x ()) 2.) r2);
      cases ~name:fst
        "a result that reads its consumed leaf at other indices takes fresh \
         storage"
        [
          ( "left rotation",
            ( (fun a ->
                Nx.concatenate ~axis:0
                  [ Nx.slice [ R (1, 8) ] a; Nx.slice [ R (0, 1) ] a ]),
              [| 2.; 3.; 4.; 5.; 6.; 7.; 8.; 1. |] ) );
          ( "right rotation",
            ( (fun a ->
                Nx.concatenate ~axis:0
                  [ Nx.slice [ R (7, 8) ] a; Nx.slice [ R (0, 7) ] a ]),
              [| 8.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] ) );
          ( "flip",
            ( (fun a -> Nx.add_s (Nx.flip a) 1.),
              [| 9.; 8.; 7.; 6.; 5.; 4.; 3.; 2. |] ) );
        ]
        (fun (_, (f, expected)) ->
          let a = Nx.arange_f Nx.float32 1. 9. 1. in
          let before = address a in
          let r = Rune.jit consumes f a in
          equal floats (Nx.create Nx.float32 [| 8 |] expected) r;
          is_false (Nativeint.equal before (address r));
          raises_invalid_arg (fun () -> Nx.to_array a));
      test "a result derived through an equal-width bitcast takes the leaf"
        (fun () ->
          let a = x () in
          let before = address a in
          let f a =
            Nx.bitcast Nx.float32 (Nx.add_s (Nx.bitcast Nx.int32 a) 1l)
          in
          let r = Rune.jit consumes f a in
          equal floats (f (x ())) r;
          equal nativeint before (address r));
      test "a result of another dtype does not take the leaf" (fun () ->
          let a = x () in
          let before = address a in
          let r =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ returns tensor)
              (fun a -> Nx.cast Nx.float64 a)
              a
          in
          equal (tensor float_exact) (Nx.cast Nx.float64 (x ())) r;
          is_false (Nativeint.equal before (address r)));
      test "a consumed leaf returned unchanged is lent with no store" (fun () ->
          let a = x () in
          let before = address a in
          let r = Rune.jit consumes Fun.id a in
          equal floats (x ()) r;
          equal nativeint before (address r));
      test "a borrowed consumed argument is copied, and still consumed"
        (fun () ->
          let ba =
            Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
              [| 1.; 2.; 3.; 4. |]
          in
          let a = Nx.of_bigarray (Bigarray.genarray_of_array1 ba) in
          let before = address a in
          let r = Rune.jit consumes (fun a -> Nx.mul_s a 2.) a in
          equal floats (Nx.create Nx.float32 [| 4 |] [| 2.; 4.; 6.; 8. |]) r;
          is_false (Nativeint.equal before (address r));
          raises_invalid_arg (fun () -> Nx.to_array a));
      test
        "a window written at a position read when the call runs reuses the \
         cache" (fun () ->
          let cache = Nx.zeros Nx.float32 [| 4; 3 |] in
          let before = address cache in
          let step =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> tensor @-> returns tensor)
              (fun cache pos row -> Nx.set [ D (pos, 1) ] row cache)
          in
          let row = Nx.ones Nx.float32 [| 1; 3 |] in
          let r = step cache (Nx.scalar Nx.int64 2L) row in
          let expected =
            Nx.set
              [ R (2, 3) ]
              (Nx.ones Nx.float32 [| 1; 3 |])
              (Nx.zeros Nx.float32 [| 4; 3 |])
          in
          equal floats expected r;
          equal nativeint before (address r));
      test "two programs alternating on one consumed state keep its storage"
        (fun () ->
          let a = x () in
          let before = address a in
          let inc = Rune.jit consumes (fun a -> Nx.add_s a 1.)
          and dbl = Rune.jit consumes (fun a -> Nx.mul_s a 2.) in
          let r = ref a in
          for _ = 1 to 3 do
            r := dbl (inc !r)
          done;
          let expected =
            List.fold_left
              (fun a _ -> Nx.mul_s (Nx.add_s a 1.) 2.)
              (x ()) [ 1; 2; 3 ]
          in
          equal floats expected !r;
          equal nativeint before (address !r));
      test "every leaf of a consumed state derived at its own index is lent"
        (fun () ->
          let s = Nx.Ptree.(list tensor) in
          let leaves () =
            List.init 4 (fun i -> Nx.full Nx.float32 [| 3 |] (float_of_int i))
          in
          let ls = leaves () in
          let before = List.map address ls in
          let r =
            Rune.jit
              Nx.Ptree.(consumes s @@ returns s)
              (List.map (fun l -> Nx.add_s l 1.))
              ls
          in
          equal (list nativeint) before (List.map address r));
      (* A momentum step: the parameters read the new velocity, which reads the
         parameters. Each result is written over its own state, the velocity
         first, and the parameters read it from there. *)
      test "a step whose results read each other's state lends both" (fun () ->
          let step (w, v) =
            let v = Nx.add (Nx.mul_s v 0.9) (Nx.mul_s (Nx.sub w (x ())) 2.) in
            (Nx.sub w (Nx.mul_s v 0.1), v)
          in
          let g = Rune.jit state step in
          let zeros () =
            (Nx.zeros Nx.float32 [| 4 |], Nx.zeros Nx.float32 [| 4 |])
          in
          let compiled = ref (zeros ()) and eager = ref (zeros ()) in
          for _ = 1 to 3 do
            let before = (address (fst !compiled), address (snd !compiled)) in
            compiled := g !compiled;
            eager := step !eager;
            equal (pair nativeint nativeint) before
              (address (fst !compiled), address (snd !compiled))
          done;
          equal floats (fst !eager) (fst !compiled);
          equal floats (snd !eager) (snd !compiled));
      (* Each result reads the other's state before its store: no order exists,
         and the latest result takes storage of its own. *)
      test "results that read each other's state in a cycle lend the earliest"
        (fun () ->
          let step (a, b) = (Nx.add a b, Nx.sub b a) in
          let g = Rune.jit state step in
          let compiled = ref (x (), y ()) and eager = ref (x (), y ()) in
          for _ = 1 to 3 do
            let a = address (fst !compiled) and b = address (snd !compiled) in
            compiled := g !compiled;
            eager := step !eager;
            equal nativeint a (address (fst !compiled));
            is_false (Nativeint.equal b (address (snd !compiled)))
          done;
          equal floats (fst !eager) (fst !compiled);
          equal floats (snd !eager) (snd !compiled));
      test "a consumed leaf returned as it is keeps its value" (fun () ->
          let both = Nx.Ptree.(pair tensor tensor) in
          let step (a, b) = (Nx.add_s a 1., b) in
          let g = Rune.jit Nx.Ptree.(consumes both @@ returns both) step in
          let a, b = g (g (x (), Nx.ones Nx.float32 [| 4 |])) in
          equal floats (Nx.add_s (x ()) 2.) a;
          equal floats (Nx.ones Nx.float32 [| 4 |]) b);
    ]

(* [together fs] runs each of [fs] on a domain of its own, all released at once,
   and is their results. *)
let together fs =
  let go = Atomic.make false in
  let ds =
    List.map
      (fun f ->
        Domain.spawn (fun () ->
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done;
            f ()))
      fs
  in
  Atomic.set go true;
  List.map Domain.join ds

(* Captures *)

(* Whether a call that consumes [w] lends its storage to its result, which it
   does unless a program pins it. It consumes [w]. *)
let lends w =
  let before = Witness.addresses w in
  let r = Rune.jit consumes (fun a -> Nx.add_s a 0.) w in
  Witness.addresses r = before

let captures =
  group "captures"
    [
      test "a captured tensor is a constant of the program" (fun () ->
          let w = y () in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          equal floats (Nx.mul (x ()) w) (g (x ()));
          equal floats (Nx.mul (y ()) w) (g (y ())));
      test "a capture of one element on a device is bound as a constant"
        (fun () ->
          let w = placed d1 (Nx.scalar Nx.float32 3.) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          equal close (Nx.mul_s (x ()) 3.) (host (g (placed d1 (x ()))));
          is_true ~msg:"not pinned" (lends w));
      test "a host capture of a call on a device is placed there once"
        (fun () ->
          let w = y () in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = placed d2 (x ()) in
          ignore (g a);
          let before = bytes_in d2 in
          for _ = 1 to 3 do
            ignore (g a)
          done;
          equal ~msg:"bytes received by later calls" int before (bytes_in d2));
      test "a capture decides the device of a call of host arguments" (fun () ->
          let w = placed d1 (y ()) in
          let r = Rune.jit' (fun a -> Nx.mul a w) (x ()) in
          is_true (Nx.Placement.equal (on d1) (Nx.placement r));
          equal close (Nx.mul (x ()) (y ())) (host r));
      test
        "a capture another call consumes is computed from a copy and stays \
         live for the program" (fun () ->
          let w = y () in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          ignore (g (x ()));
          equal ~msg:"the consuming call" floats
            (Nx.neg (y ()))
            (Rune.jit consumes Nx.neg w);
          equal ~msg:"the program" close (Nx.mul (x ()) (y ())) (g (x ()));
          equal ~msg:"the capture" floats (y ()) w);
      test "two compiled functions share one captured buffer" (fun () ->
          let w = placed d1 (y ()) in
          let g1 = Rune.jit' (fun a -> Nx.mul a w)
          and g2 = Rune.jit' (fun a -> Nx.add a w) in
          let a = placed d1 (x ()) in
          equal close (Nx.mul (x ()) (y ())) (host (g1 a));
          equal close (Nx.add (x ()) (y ())) (host (g2 a));
          is_false ~msg:"pinned" (lends w);
          equal ~msg:"the program after" close
            (Nx.mul (x ()) (y ()))
            (host (g1 a)));
      test "a bound capture read between calls stays bound" (fun () ->
          let w = placed d1 (y ()) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = placed d1 (x ()) in
          ignore (g a);
          equal floats (y ()) (host w);
          equal close (Nx.mul (x ()) (y ())) (host (g a));
          is_false ~msg:"pinned" (lends w);
          equal ~msg:"the program after" close
            (Nx.mul (x ()) (y ()))
            (host (g a)));
      test "a capture is released with the compiled function that binds it"
        (fun () ->
          let w = placed d1 (y ()) in
          let run () =
            let g = Rune.jit' (fun a -> Nx.mul a w) in
            ignore (g (placed d1 (x ())))
          in
          run ();
          Gc.full_major ();
          is_true ~msg:"unpinned" (lends w));
      test
        "a dropped compiled function's captures are back in allocated after \
         one collection and one operation of their device" (fun () ->
          Gc.full_major ();
          let before = allocated d4 in
          let run () =
            let w = placed d4 (y ()) in
            let g = Rune.jit' (fun a -> Nx.mul a w) in
            ignore (host (g (placed d4 (x ()))))
          in
          run ();
          Gc.full_major ();
          Nx_device.synchronize d4;
          equal int before (allocated d4));
      test "two compiled functions binding one capture run from two domains"
        (fun () ->
          let w = placed d1 (y ()) in
          let g1 = Rune.jit' (fun a -> Nx.mul a w)
          and g2 = Rune.jit' (fun a -> Nx.add a w) in
          let a = placed d1 (x ()) in
          match
            together [ (fun () -> host (g1 a)); (fun () -> host (g2 a)) ]
          with
          | [ r1; r2 ] ->
              equal close (Nx.mul (x ()) (y ())) r1;
              equal close (Nx.add (x ()) (y ())) r2
          | _ -> fail "two results");
      test "a draw from a key the function captures raises Jit_error" (fun () ->
          raises_jit_error (fun () ->
              Rune.jit'
                (fun a ->
                  Nx.add a
                    (Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
                         Nx.rand Nx.float32 [| 4 |])))
                (x ())));
      test "a draw from a captured key at counters of the arguments computes"
        (fun () ->
          let key =
            Nx.create Nx.int32 [| 4; 2 |]
              (Array.init 8 (fun i -> Int32.of_int (7 + (i mod 2))))
          in
          let draw c = Nx.Op.eval (Nx.Op.Threefry (key, c)) in
          let c = Nx.create Nx.int32 [| 4; 2 |] (Array.init 8 Int32.of_int) in
          equal (tensor int32) (draw c) (Rune.jit' draw c));
      test "an empty draw from a captured key computes" (fun () ->
          let key = Nx.zeros Nx.int32 [| 0; 2 |] in
          let draw x = Nx.Op.eval (Nx.Op.Threefry (key, Nx.cast Nx.int32 x)) in
          let x = Nx.zeros Nx.float64 [| 0; 2 |] in
          equal (tensor int32) (draw x) (Rune.jit' draw x));
      slow "a draw from a key the function takes draws again at each call"
        (fun () ->
          let g =
            Rune.jit
              Nx.Ptree.(tensor @-> returns tensor)
              (fun k ->
                Nx.Rng.with_key (Nx.Rng.of_tensor k) (fun () ->
                    Nx.rand Nx.float32 [| 4 |]))
          in
          let draw k =
            Nx.Rng.with_key k (fun () -> Nx.rand Nx.float32 [| 4 |])
          in
          let k1 = Nx.Rng.key 1 and k2 = Nx.Rng.key 2 in
          equal floats (draw k1) (g (k1 :> Nx.int32_t));
          equal floats (draw k2) (g (k2 :> Nx.int32_t)));
    ]

(* Errors *)

let twin1 = driver "TWIN"
let twin2 = driver "TWIN"

let errors =
  let leaked = ref None in
  let messages =
    [
      ( "operands on two devices",
        fun () ->
          Rune.jit two Nx.add (placed d1 (x ())) (placed d2 (y ())) |> ignore );
      ( "a name met with two devices",
        fun () ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
            (fun a b -> (Nx.neg a, Nx.neg b))
            (placed twin1 (x ()))
            (placed twin2 (y ()))
          |> ignore );
      ( "a consumed leaf another leaf reaches",
        fun () ->
          let a = x () in
          Rune.jit
            Nx.Ptree.(consumes tensor @@ tensor @-> returns tensor)
            Nx.add a a
          |> ignore );
    ]
  in
  group "errors"
    [
      test "reading a traced value raises Jit_error" (fun () ->
          raises_jit_error (fun () ->
              Rune.jit'
                (fun a -> if Nx.item [ 0 ] a > 0. then a else Nx.neg a)
                (x ())));
      test "a ragged take inside a compiled call names Nx_ragged.take"
        (fun () ->
          let r =
            Nx_ragged.of_lengths
              (Nx.create Nx.int64 [| 2 |] [| 1L; 3L |])
              (x ())
          in
          let take indices = Nx_ragged.values (Nx_ragged.take ~indices r) in
          raises_match
            (function
              | Rune.Jit_error m ->
                  String.starts_with ~prefix:"Nx_ragged.take: " m
              | _ -> false)
            (fun () ->
              ignore (Rune.jit' take (Nx.create Nx.int64 [| 2 |] [| 1L; 0L |]))));
      test "an operation no target computes raises Jit_error" (fun () ->
          raises_jit_error (fun () ->
              Rune.jit'
                (fun a -> Nx.real Nx.float32 (Nx.fft (Nx.cast Nx.complex64 a)))
                (x ())));
      test "operands on two devices raise nx's message" (fun () ->
          raises_invalid_arg (List.assoc "operands on two devices" messages));
      test "a name met with two devices raises" (fun () ->
          raises_invalid_arg (List.assoc "a name met with two devices" messages));
      test "a traced value kept after the call raises on read" (fun () ->
          let r =
            Rune.jit'
              (fun a ->
                let t = poly a in
                leaked := Some t;
                t)
              (x ())
          in
          equal close (poly (x ())) r;
          match !leaked with
          | Some t -> raises_invalid_arg (fun () -> Nx.to_array t)
          | None -> fail "the function did not run");
      test
        "a traced value kept after its call raises as another call's argument"
        (fun () ->
          let kept = ref None in
          ignore
            (Rune.jit'
               (fun a ->
                 kept := Some (poly a);
                 a)
               (x ()));
          match !kept with
          | Some t ->
              raises_match
                (Exn.invalid_arg ~substring:"a traced tensor has no bytes")
                (fun () -> Rune.jit' Nx.neg t)
          | None -> fail "the function did not run");
      test "a call that raised traces again at the next call" (fun () ->
          let g = Rune.jit' (fun a -> if Nx.item [ 0 ] a > 0. then a else a) in
          equal int 1 (traces (fun () -> raises_jit_error (fun () -> g (x ()))));
          equal int 1 (traces (fun () -> raises_jit_error (fun () -> g (x ())))));
      test "no message names the compiled call's internals" (fun () ->
          List.iter
            (fun (name, f) ->
              let m = message f in
              List.iter
                (fun word -> not_contains ~msg:name ~sub:word m)
                [ "PARAM"; "UOp"; "slot"; "latch"; "Lower"; "Staged" ])
            messages);
    ]

(* Reports *)

let reports =
  group "reports"
    [
      test "a first call records its phases in order, and a replay none"
        (fun () ->
          let g = Rune.jit' poly in
          let _, first = profiled (fun () -> g (x ())) in
          equal (list string)
            [
              "rune.jit: trace";
              "rune.jit: schedule";
              "rune.jit: compile";
              "rune.jit: link";
            ]
            first;
          let _, again = profiled (fun () -> g (y ())) in
          equal (list string) [] again);
    ]

(* Domains *)

let domains =
  group "domains"
    [
      test
        "a call that reads a storage and one that consumes it, from two \
         domains, each read it whole or refuse" (fun () ->
          let a = x () in
          let reader = Rune.jit' poly in
          let consumer = Rune.jit consumes (fun v -> Nx.add_s v 1.) in
          let outcome f () =
            match f () with
            | r -> Some (Nx.to_array r)
            | exception Invalid_argument _ -> None
          in
          match
            together
              [ outcome (fun () -> reader a); outcome (fun () -> consumer a) ]
          with
          | [ read; consumed ] ->
              is_true ~msg:"one of them ran"
                (Option.is_some read || Option.is_some consumed);
              Option.iter
                (equal ~msg:"the reader" (array float_exact)
                   (Nx.to_array (poly (x ()))))
                read;
              Option.iter
                (equal ~msg:"the consumer" (array float_exact)
                   (Nx.to_array (Nx.add_s (x ()) 1.)))
                consumed
          | _ -> fail "two outcomes");
      test "two domains meeting one new key trace it once" (fun () ->
          let g = Rune.jit' poly in
          let rs, spans =
            profiled (fun () ->
                together [ (fun () -> g (x ())); (fun () -> g (x ())) ])
          in
          List.iter (equal close (poly (x ()))) rs;
          equal int 1
            (List.length (List.filter (String.equal "rune.jit: trace") spans)));
      test "two domains replay one program, each reading its own arguments"
        (fun () ->
          let g = Rune.jit' poly in
          ignore (g (x ()));
          let run k () =
            List.init 20 (fun i ->
                let v =
                  Nx.full Nx.float32 [| 4 |] (float_of_int ((k * 100) + i))
                in
                (poly v, g v))
          in
          List.iter
            (List.iter (fun (e, a) -> equal close e a))
            (together [ run 1; run 2 ]));
    ]

(* Transformations *)

let transformations =
  group "transformations"
    [
      test "under grad a compiled function consumes nothing" (fun () ->
          let a = x () in
          let f a = Nx.sum (Rune.jit consumes poly a) in
          ignore (Rune.grad' f a);
          equal floats (x ()) a);
      test "a detached value inside a compiled call is its value" (fun () ->
          let f a = Nx.add a (Rune.detach (poly a)) in
          equal close (f (x ())) (Rune.jit' f (x ())));
      test "under a transformation a compiled function runs its function"
        (fun () ->
          let f a = Nx.sum (poly a) in
          equal floats (Rune.grad' f (x ())) (Rune.grad' (Rune.jit' f) (x ())));
      test
        "a compiled function called inside another one's trace traces through"
        (fun () ->
          let inner = Rune.jit' Nx.neg in
          let g = Rune.jit' (fun a -> inner (inner a)) in
          equal int 1 (traces (fun () -> equal floats (x ()) (g (x ())))));
      test "a remat's gradient under jit is its function's" (fun () ->
          let g =
            Rune.remat
              Nx.Ptree.(tensor @-> returns tensor)
              (fun x -> Nx.tanh (Nx.mul x x))
          in
          let f x = Nx.sum (g (Nx.mul_s x 2.)) in
          let x = Nx.create Nx.float64 [| 4 |] [| 0.5; -1.; 0.25; 2. |] in
          equal
            (Oracle.tensor ~abs:1e-12 ~rel:1e-9 ())
            (Rune.grad'
               (fun x ->
                 Nx.sum (Nx.tanh (Nx.mul (Nx.mul_s x 2.) (Nx.mul_s x 2.))))
               x)
            (Rune.jit' (Rune.grad' f) x));
    ]

(* Placement and views *)

let placement =
  group "placement"
    [
      test "a host argument of a call on a device is uploaded at each call"
        (fun () ->
          let g = Rune.jit two Nx.mul in
          let a = placed d2 (x ()) in
          ignore (g a (y ()));
          let before = bytes_in d2 in
          ignore (g a (y ()));
          equal ~msg:"bytes received" int (before + 16) (bytes_in d2));
      slow
        "a state starting on the host retraces once on a device, then replays"
        (fun () ->
          let step = Rune.jit consumes (fun a -> Nx.add_s a 1.) in
          let s = ref (x ()) in
          equal int 1 (traces (fun () -> s := step !s));
          equal int 1 (traces (fun () -> s := step (placed d2 (host !s))));
          equal int 0 (traces (fun () -> s := step !s));
          equal floats (Nx.add_s (x ()) 3.) (host !s));
      test
        "a value placed on another device inside the function meets its source \
         and raises" (fun () ->
          raises_invalid_arg (fun () ->
              Rune.jit'
                (fun a -> Nx.add a (Nx.place (on d2) a))
                (placed d1 (x ()))));
      test "a split argument computes on each device, and stays split"
        (fun () ->
          let p =
            Nx.Placement.sharded ~backend:Rune.compiled ~axis:0 [ d1; d2 ]
          in
          let r = Rune.jit' poly (Nx.place p (x ())) in
          is_true (Nx.Placement.equal p (Nx.placement r));
          equal close (poly (x ())) (host r));
      test "a consumed split state is lent on every device" (fun () ->
          let p =
            Nx.Placement.sharded ~backend:Rune.compiled ~axis:0 [ d3; d4 ]
          in
          let a = Nx.place p (x ()) in
          let before = Witness.addresses a in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal (list nativeint) before (Witness.addresses r);
          is_true (Nx.Placement.equal p (Nx.placement r)));
      slow "a product and a sum over four devices equal one device" (fun () ->
          let p =
            Nx.Placement.sharded ~backend:Rune.compiled ~axis:0
              [ d1; d2; d3; d4 ]
          in
          let w =
            Nx.place
              (Nx.Placement.replicated ~backend:Rune.compiled [ d1; d2; d3; d4 ])
              (grid 3 3)
          in
          let f a = Nx.sum ~axes:[ 1 ] (Nx.matmul a w) in
          let a = grid 4 3 in
          let eager = host (f (Nx.place p a)) in
          equal close (Nx.sum ~axes:[ 1 ] (Nx.matmul a (grid 3 3))) eager;
          equal close eager (host (Rune.jit' f (Nx.place p a))));
    ]

(* Staged scans *)

(* [scanned ~init f] is the scan of [f] from [init], and how many times its step
   ran. *)
let scanned ~init f =
  let steps = ref 0 in
  let step c x =
    incr steps;
    f c x
  in
  ((fun xs -> Rune.scan' ~f:step ~init xs), steps)

let rows n k = Nx.mul_s (grid n k) 0.01
let zeros k = Nx.zeros Nx.float32 [| k |]
let ones k = Nx.ones Nx.float32 [| k |]
let near = Oracle.tensor ~rel:1e-4 ~abs:1e-5 ()

let decay c x =
  let c = Nx.add (Nx.mul_s c 0.5) x in
  (c, Nx.sin c)

let sum c x =
  let c = Nx.add c x in
  (c, c)

(* A step that scans the four rows of four its row holds. *)
let nested c x =
  let c, _ = Rune.scan' ~f:sum ~init:c (Nx.reshape [| 4; 4 |] x) in
  (c, Nx.reshape [| 1 |] (Nx.sum c))

(* A scan of rows [m * 4] wide whose step scans its row as [m] rows of four,
   from its carry [c], the inner step reading [c] beside its own carry and row:
   its final carry and the sums of its inner outputs. *)
let reading_outer m xs =
  let c, ys =
    Rune.scan'
      ~f:(fun c x ->
        let ic, iys =
          Rune.scan'
            ~f:(fun d y ->
              let d = Nx.add (Nx.mul_s d 0.9) (Nx.mul y c) in
              (d, Nx.sin d))
            ~init:c
            (Nx.reshape [| m; 4 |] x)
        in
        (Nx.add ic (Nx.sum ~axes:[ 0 ] iys), Nx.reshape [| 1 |] (Nx.sum iys)))
      ~init:(Nx.ones Nx.float32 [| 4 |])
      xs
  in
  Nx.concatenate ~axis:0 [ c; Nx.flatten ys ]

(* [reads_outer at] checks a scan staged in a staged scan's step that reads the
   enclosing step's carry, placed at [at], against eager: its values, its
   gradient and its batch. *)
let reads_outer at =
  let xs = Nx.mul_s (Nx.sin (Nx.reshape [| 5; 12 |] (arange 60))) 0.5 in
  let loss xs = Nx.sum (Nx.mul (reading_outer 3 xs) (reading_outer 3 xs)) in
  let batch = Nx.stack ~axis:0 [ xs; Nx.mul_s xs 2. ] in
  group "a scan in a staged scan's step that reads the step's carry"
    [
      test "computes eager's values" (fun () ->
          equal near (reading_outer 3 xs)
            (host (Rune.jit' (reading_outer 3) (Nx.place at xs))));
      test "differentiates as eager" (fun () ->
          equal near (Rune.grad' loss xs)
            (host (Rune.jit' (Rune.grad' loss) (Nx.place at xs))));
      test "batches as eager" (fun () ->
          let f = Rune.vmap' (reading_outer 3) in
          equal near (f batch) (host (Rune.jit' f (Nx.place at batch))));
    ]

let product c x =
  let w = Nx.mul_s (grid 3 3) 0.1 in
  let c =
    Nx.add (Nx.reshape [| 3 |] (Nx.matmul w (Nx.reshape [| 3; 1 |] c))) x
  in
  (c, c)

let rotated c x =
  let c =
    Nx.add
      (Nx.concatenate ~axis:0
         [ Nx.slice [ R (1, 3) ] c; Nx.slice [ R (0, 1) ] c ])
      x
  in
  (c, c)

let flipped c x =
  let c = Nx.add (Nx.flip c) x in
  (c, Nx.mul_s c 2.)

(* [staged at name ~steps ~init f xs] checks that the scan of [f] over [xs],
   compiled with [xs] at [at], computes the eager scan's values, its step
   running [steps n] times for [n] rows. *)
let staged at name ~steps ~init f xs =
  test name (fun () ->
      let scan, ran = scanned ~init f in
      let c, ys = scan xs in
      ran := 0;
      let g =
        Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) scan
      in
      let c', ys' = g (Nx.place at xs) in
      equal near c (host c');
      equal near ys (host ys');
      equal int (steps (Nx.shape xs).(0)) !ran)

(* [carried at name ~steps ~init f xs] is [staged] for a list of carries. *)
let carried at name ~steps ~init f xs =
  test name (fun () ->
      let ran = ref 0 in
      let scan xs =
        let cs, ys =
          Rune.scan
            Nx.Ptree.(list tensor)
            Nx.Ptree.tensor Nx.Ptree.tensor
            ~f:(fun c x ->
              incr ran;
              f c x)
            ~init xs
        in
        ys :: cs
      in
      let eager = scan xs in
      ran := 0;
      let g = Rune.jit Nx.Ptree.(tensor @-> returns (list tensor)) scan in
      List.iter2 (fun e c -> equal near e (host c)) eager (g (Nx.place at xs));
      equal int steps !ran)

(* [summed ran xs] is the sum of the outputs of a scan over [xs], counting its
   steps in [ran]: the transpose's rows of output cotangents are a broadcast
   constant. *)
let summed ran xs =
  let step c x =
    incr ran;
    decay c x
  in
  Nx.sum (snd (Rune.scan' ~f:step ~init:(zeros 4) xs))

(* [transformed at name ~steps f] checks that [f ran] over nine rows at [at]
   computes under a compiled call what it computes eagerly, its scans' steps
   running [steps] times. *)
let transformed at name ~steps f =
  test name (fun () ->
      let ran = ref 0 in
      let r = f ran (rows 9 4) in
      ran := 0;
      equal near r (host (Rune.jit' (f ran) (Nx.place at (rows 9 4))));
      equal int steps !ran)

(* [weighted ran w xs] is a loss over a scan of [xs] whose step reads [w] and
   whose carry becomes tracked after its first step, counting its steps in
   [ran]; [weights] and [row_weights] weigh the final carry and the outputs, so
   no cotangent row is a constant. *)
let weights = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |]

let row_weights =
  Nx.create Nx.float32 [| 9; 4 |]
    (Array.init 36 (fun i -> 1. +. (Float.of_int i /. 36.)))

let weighted ran w xs =
  let step c x =
    incr ran;
    decay (Nx.add c w) (Nx.mul x c)
  in
  let c, ys = Rune.scan' ~f:step ~init:(ones 4) xs in
  Nx.add (Nx.sum (Nx.mul c weights)) (Nx.sum (Nx.mul ys row_weights))

let w0 = Nx.full Nx.float32 [| 4 |] 0.3
let ws = Nx.stack ~axis:0 [ w0; Nx.mul_s w0 2.; Nx.mul_s w0 0.5 ]

(* [stages at name f] checks that [f ran xs], over 9 and over 17 rows of three
   at [at], computes under a compiled call what it computes eagerly, its scans'
   steps running as many times for either length: the scans stage. *)
let stages at name f =
  test name (fun () ->
      let runs n =
        let ran = ref 0 in
        let xs = Nx.mul_s (grid n 3) 0.01 in
        let expected = f ran xs in
        ran := 0;
        let r = Rune.jit' (f ran) (Nx.place at xs) in
        equal ~msg:(Printf.sprintf "%d rows" n) near expected (host r);
        !ran
      in
      let nine = runs 9 in
      equal ~msg:"steps for 17 rows against 9" int nine (runs 17))

(* A recurrent cell of weight [wr], and a scan of it over rows of three from
   [h]. *)
let wr = Nx.mul_s (Nx.sub_s (grid 3 3) 4.) 0.1
let hr = Nx.create Nx.float32 [| 3 |] [| 0.5; -0.25; 0.1 |]

let cell w h x =
  Nx.tanh
    (Nx.add (Nx.reshape [| 3 |] (Nx.matmul w (Nx.reshape [| 3; 1 |] h))) x)

let rollout ran w h xs =
  snd
    (Rune.scan'
       ~f:(fun h x ->
         incr ran;
         let h = cell w h x in
         (h, h))
       ~init:h xs)

(* [constant_rows at] checks scans over rows computed from constants alone, of 4
   and 20 bytes, with a carry at [at]: the rows have no device until the program
   places them, padded to the loop's row stride. *)
(* [both_gradients at] checks compiled gradients of a scan in its initial carry
   and its rows, at [at], by grad and by a pullback, for rows of 16 bytes, which
   the transpose's loop writes in the result's storage, and of 8, which it
   copies: the two results share the loop, which runs once. *)
let both_gradients at =
  let scan (c, xs) =
    snd
      (Rune.scan'
         ~f:(fun d y ->
           let d = Nx.add (Nx.mul_s d 0.9) y in
           (d, Nx.sin d))
         ~init:c xs)
  in
  let both = Nx.Ptree.(pair tensor tensor) in
  let check name g k =
    test
      (Printf.sprintf "%s in a scan's carry and rows of %d bytes" name (4 * k))
      (fun () ->
        let c = Nx.full Nx.float32 [| k |] 0.7 and xs = rows 6 k in
        let ec, ex = g (c, xs) in
        let jc, jx =
          Rune.jit
            Nx.Ptree.(both @-> returns both)
            g
            (Nx.place at c, Nx.place at xs)
        in
        equal near ec (host jc);
        equal near ex (host jx))
  in
  let grad = Rune.grad both (fun a -> Nx.sum (scan a)) in
  let pullback a =
    let ys, back = Rune.vjp both Nx.Ptree.tensor scan a in
    back (Nx.cos ys)
  in
  List.concat_map
    (fun k -> [ check "a gradient" grad k; check "a pullback" pullback k ])
    [ 4; 2 ]

(* [lent_beside_taken at] checks a compiled call at [at] that consumes its
   state, writes it updated over it, and starts a scan from the update: the
   scan's carry and rows are written in the results' storage, and read the
   update, not what its store overwrote. *)
let lent_beside_taken at =
  test "a scan from a consumed state's update takes its carry and rows"
    (fun () ->
      let f (a, xs) =
        let a = Nx.add_s a 1. in
        let c, ys =
          Rune.scan'
            ~f:(fun d y ->
              let d = Nx.add (Nx.mul_s d 0.9) y in
              (d, d))
            ~init:a xs
        in
        (a, (c, ys))
      in
      let g =
        Rune.jit
          Nx.Ptree.(
            consumes (pair tensor tensor)
            @@ returns (pair tensor (pair tensor tensor)))
          f
      in
      let a = Nx.full Nx.float32 [| 4 |] 0.5 and xs = rows 6 4 in
      let ea, (ec, ey) = f (a, xs) in
      let ja, (jc, jy) = g (Nx.place at a, Nx.place at xs) in
      equal near ea (host ja);
      equal near ec (host jc);
      equal near ey (host jy))

let constant_rows at =
  let check name rows ~carry ~ys =
    test name (fun () ->
        let f c =
          Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor
            ~f:(fun c l -> (Nx.add c (Nx.sum l), Nx.mul_s l 10l))
            ~init:c (rows ())
        in
        let c, y =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (pair tensor tensor))
            f
            (Nx.place at (Nx.zeros Nx.int32 [||]))
        in
        equal (array int32) [| carry |]
          (Nx.to_array (Nx.reshape [| 1 |] (host c)));
        equal (array int32) ys (Nx.to_array (host y)))
  in
  [
    check "a scan over rows of 4 bytes computed from constants sums as eager"
      (fun () -> Nx.cumsum (Nx.ones Nx.int32 [| 8 |]))
      ~carry:36l
      ~ys:(Array.init 8 (fun i -> Int32.of_int (10 * (i + 1))));
    check "a scan over rows of 20 bytes computed from constants sums as eager"
      (fun () -> Nx.cumsum ~axis:0 (Nx.ones Nx.int32 [| 8; 5 |]))
      ~carry:180l
      ~ys:(Array.init 40 (fun k -> Int32.of_int (10 * ((k / 5) + 1))));
  ]

(* [held_by_steps at ~than a b] checks that a staged scan of [b * chunk_calls]
   steps at [at] holds less than [than] times the memory one of [a *
   chunk_calls] steps holds, beyond the one it holds at [a]: the memory a
   program holds for its loop does not grow with its steps. A step makes at
   least one call, so either count runs several of the batches the engine
   reruns; [b] above 16 is slow. The scan's outputs are rows of 16 bytes, which
   the program writes in its result: the measure leaves the result out. *)
let held_by_steps at ~than a b =
  let d = match Nx.Placement.devices at with [ d ] -> d | _ -> assert false in
  (if b > 16 then slow else test)
    (Printf.sprintf
       "a staged scan of %d chunks of calls holds less than %d times the loop \
        memory of one of %d"
       b than a) (fun () ->
      let held k =
        let n = k * Tolk.Hcq2.chunk_calls in
        let xs = Nx.place at (rows n 4) in
        let g =
          Rune.jit' (fun xs -> Rune.scan' ~f:sum ~init:(zeros 4) xs |> snd)
        in
        let before = settled d in
        let r = g xs in
        let held = settled d - before - Nx.nbytes r in
        ignore (Sys.opaque_identity (g, xs));
        ignore (host r);
        held
      in
      let ha = warmed (fun () -> held a) in
      let hb = held b in
      less
        ~msg:(Printf.sprintf "%d bytes, against %d for %d chunks" hb ha a)
        int ~than:(than * ha) (hb - ha))

(* Scans on a device whose work runs from command queues. *)
let staged_scans d =
  let at = on d and once _ = 1 in
  group "staged scans"
    [
      group "constant rows" (constant_rows at);
      group "gradients" (both_gradients at);
      lent_beside_taken at;
      staged at "stage, their step once, over rows 16 bytes apart" ~steps:once
        ~init:(zeros 4) decay (rows 7 4);
      staged at "stage over rows that are not, through a padded copy"
        ~steps:once ~init:(zeros 3) decay (rows 5 3);
      staged at "stage a thousand steps" ~steps:once ~init:(zeros 4) decay
        (rows 1000 4);
      (* More steps than one batch holds: they run as several chunks of steps
         and the steps left. *)
      staged at "stage more steps than a batch holds, in chunks and the rest"
        ~steps:once ~init:(zeros 4) decay (rows 3001 4);
      held_by_steps at ~than:1 4 16;
      held_by_steps at ~than:2 4 64;
      (* A loop's calls run on devices with queues or all on the host: a step
         that computes on the host between steps on the device is written
         out. *)
      staged at
        "write out a step that computes on the host between device steps"
        ~steps:(fun n -> n + 1)
        ~init:(ones 4)
        (fun c x ->
          let h = Nx.sqrt (Nx.place Nx.Placement.host c) in
          let c = Nx.add (Nx.place at h) x in
          (c, c))
        (rows 5 4);
      staged at "update a carry its next value reads through a product"
        ~steps:once ~init:(ones 3) product (rows 6 3);
      staged at "stage a scan whose step stages a scan of its own" ~steps:once
        ~init:(zeros 4) nested (rows 6 16);
      reads_outer at;
      staged at "update a carry its next value reads rotated" ~steps:once
        ~init:(ones 3) rotated (rows 5 3);
      staged at "update a carry its next value reads reversed" ~steps:once
        ~init:(ones 3) flipped (rows 5 3);
      carried at "swap two carries" ~steps:1
        ~init:[ ones 4; Nx.full Nx.float32 [| 4 |] 3. ]
        (fun cs x ->
          match cs with
          | [ a; b ] -> ([ b; a ], Nx.add (Nx.mul_s a 2.) x)
          | _ -> assert false)
        (rows 5 4);
      carried at "update two carries that read each other" ~steps:1
        ~init:[ ones 4; Nx.full Nx.float32 [| 4 |] 3. ]
        (fun cs x ->
          match cs with
          | [ a; b ] -> ([ Nx.add b x; a ], Nx.add a b)
          | _ -> assert false)
        (rows 6 4);
      carried at "update carries as a Fibonacci sequence" ~steps:1
        ~init:[ ones 4; ones 4 ]
        (fun cs x ->
          match cs with
          | [ a; b ] -> ([ b; Nx.add (Nx.add a b) x ], a)
          | _ -> assert false)
        (rows 6 4);
      carried at "rotate three carries" ~steps:1
        ~init:[ ones 4; Nx.full Nx.float32 [| 4 |] 2.; zeros 4 ]
        (fun cs x ->
          match cs with
          | [ a; b; c ] -> ([ b; Nx.add c x; a ], Nx.add a c)
          | _ -> assert false)
        (rows 6 4);
      (* Each carry's next value reads every earlier carry: checking each for a
         cycle walks the carries once, where following every path would take
         minutes. *)
      carried at "update forty carries that each read the earlier ones" ~steps:1
        ~init:(List.init 40 (fun _ -> ones 4))
        (fun cs x ->
          let next, _ =
            List.fold_left
              (fun (next, sum) c ->
                (Nx.add (Nx.add c x) (Nx.mul_s sum 0.001) :: next, Nx.add sum c))
              ([], zeros 4)
              cs
          in
          (List.rev next, Nx.sum (List.hd next)))
        (rows 3 4);
      transformed at "stage grad of a sum over a scan's outputs" ~steps:3
        (fun ran -> Rune.grad' (summed ran));
      transformed at "stage grad of grad of a sum over a scan's outputs"
        ~steps:7 (fun ran ->
          Rune.grad' (fun xs -> Nx.sum (Rune.grad' (summed ran) xs)));
      transformed at "stage jvp of a loss reading a host capture after the scan"
        ~steps:2 (fun ran xs ->
          let k = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
          let loss xs =
            let step c x =
              incr ran;
              decay c x
            in
            let c, ys = Rune.scan' ~f:step ~init:(zeros 4) xs in
            Nx.add (Nx.sum ys) (Nx.sum (Nx.mul c k))
          in
          snd (Rune.jvp' loss xs (Nx.ones_like xs)));
      transformed at
        "stage grad in a weight the step reads, the forward step twice as the \
         carry becomes tracked"
        ~steps:3 (fun ran xs -> Rune.grad' (fun w -> weighted ran w xs) w0);
      transformed at "stage jvp in the rows, the scan of their tangents"
        ~steps:2 (fun ran xs ->
          snd (Rune.jvp' (weighted ran w0) xs (Nx.mul_s xs 2.)));
      transformed at "stage vmap over weights, the scan of their lanes" ~steps:2
        (fun ran xs -> Rune.vmap' (fun w -> weighted ran w xs) ws);
      transformed at "stage vmap of grad, both scans batched" ~steps:5
        (fun ran xs -> Rune.vmap' (Rune.grad' (fun w -> weighted ran w xs)) ws);
      transformed at "stage jvp of grad, the scans of the tangents" ~steps:5
        (fun ran xs ->
          snd (Rune.jvp' (Rune.grad' (fun w -> weighted ran w xs)) w0 weights));
      test
        "raise at the transpose for a step whose rerun reads a tracked value \
         its first run did not" (fun () ->
          let again = ref false in
          let loss w =
            let step c x =
              let c = if !again then Nx.mul (Nx.add c x) w else Nx.add c x in
              (c, c)
            in
            let xs = Nx.mul (Nx.place at (rows 9 4)) w in
            let _, ys = Rune.scan' ~f:step ~init:(zeros 4) xs in
            again := true;
            Nx.sum (Nx.mul ys row_weights)
          in
          raises
            (Invalid_argument
               "Rune.grad': a function run again for its transpose reads a \
                value the differentiation tracks that its first run did not")
            (fun () -> Rune.jit' (Rune.grad' loss) (Nx.place at w0)));
      stages at "stage jvp of a scan" (fun ran xs ->
          snd (Rune.jvp' (fun h -> rollout ran wr h xs) hr (Nx.ones_like hr)));
      test "stage jvp of a scan whose counter carry takes no tangent" (fun () ->
          let f ran xs h =
            let (h, v), ys =
              Rune.scan
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.tensor Nx.Ptree.tensor
                ~f:(fun (h, v) x ->
                  incr ran;
                  let h = cell wr h x in
                  ((h, Nx.add_s v 1.), Nx.mul_s h 2.))
                ~init:(h, Nx.scalar Nx.float32 0.)
                xs
            in
            Nx.concatenate ~axis:0 [ h; Nx.reshape [| 1 |] v; Nx.flatten ys ]
          in
          let ran = ref 0 in
          let xs = Nx.mul_s (grid 9 3) 0.01 in
          let dh = Nx.create Nx.float32 [| 3 |] [| 1.; -0.5; 0.25 |] in
          let expected = Rune.jvp' (f ran xs) hr dh in
          ran := 0;
          let y, dy =
            Rune.jit
              Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
              (fun xs dh -> Rune.jvp' (f ran xs) hr dh)
              (Nx.place at xs) dh
          in
          equal ~msg:"primal" near (fst expected) (host y);
          equal ~msg:"tangent" near (snd expected) (host dy);
          equal ~msg:"the counter's tangent" float_exact 0.
            (Nx.item [ 3 ] (host dy));
          less ~msg:"steps" int ~than:9 !ran);
      stages at "stage jvp of jvp of a scan" (fun ran xs ->
          let f h = rollout ran wr h xs in
          snd
            (Rune.jvp'
               (fun h -> snd (Rune.jvp' f h (Nx.ones_like h)))
               hr (Nx.ones_like hr)));
      stages at "stage vmap of a scan" (fun ran xs ->
          Rune.vmap' (fun h -> rollout ran wr h xs) (Nx.stack [ hr; Nx.neg hr ]));
      stages at "stage vmap over jvp of a scan" (fun ran xs ->
          Rune.vmap'
            (fun dh -> snd (Rune.jvp' (fun h -> rollout ran wr h xs) hr dh))
            (Nx.stack [ Nx.ones_like hr; hr ]));
      stages at "stage vmap over jvp of a scan from an active initial carry"
        (fun ran xs ->
          let dws = Nx.stack [ wr; Nx.ones_like wr ] in
          let dhs = Nx.stack [ hr; Nx.ones_like hr ] in
          Rune.vmap
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            (fun dw dh ->
              snd
                (Rune.jvp
                   Nx.Ptree.(pair tensor tensor)
                   Nx.Ptree.tensor
                   (fun (w, h) -> rollout ran w h xs)
                   (wr, hr) (dw, dh)))
            dws dhs);
      stages at "stage grad of jvp of a scan" (fun ran xs ->
          Rune.grad'
            (fun w ->
              Nx.sum
                (snd
                   (Rune.jvp'
                      (fun h -> rollout ran w h xs)
                      hr (Nx.ones_like hr))))
            wr);
      stages at "stage grad of vmap of a scan" (fun ran xs ->
          Rune.grad'
            (fun w ->
              Nx.sum
                (Rune.vmap'
                   (fun h -> rollout ran w h xs)
                   (Nx.stack [ hr; Nx.neg hr ])))
            wr);
      stages at "stage jvp and vmap of a gradient through a scan" (fun ran xs ->
          let g w = Rune.grad' (fun w -> Nx.sum (rollout ran w hr xs)) w in
          Nx.concatenate ~axis:0
            [
              Nx.flatten (snd (Rune.jvp' g wr (Nx.ones_like wr)));
              Nx.flatten (Rune.vmap' g (Nx.stack [ wr; Nx.neg wr ]));
            ]);
      stages at "stage grad of a scan reading an undifferentiated capture"
        (fun ran xs ->
          let d = Nx.mul_s (grid 3 3) 0.05 in
          Rune.grad'
            (fun w ->
              Nx.sum
                (snd
                   (Rune.scan'
                      ~f:(fun h x ->
                        incr ran;
                        let h =
                          cell w
                            (Nx.reshape [| 3 |]
                               (Nx.matmul d (Nx.reshape [| 3; 1 |] h)))
                            x
                        in
                        (h, h))
                      ~init:hr xs)))
            wr);
      stages at "stage a carry the step returns unchanged, and its gradient"
        (fun ran xs ->
          let c0 = Nx.create Nx.float32 [| 3 |] [| 0.1; 0.2; -0.3 |] in
          let f (w, c) =
            let (h, c), ys =
              Rune.scan
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.tensor Nx.Ptree.tensor
                ~f:(fun (h, c) x ->
                  incr ran;
                  let h = cell w h (Nx.add x c) in
                  ((h, c), h))
                ~init:(hr, c) xs
            in
            Nx.add (Nx.sum ys) (Nx.add (Nx.sum h) (Nx.sum c))
          in
          let p = Nx.Ptree.(pair tensor tensor) in
          let gw, gc = Rune.grad p f (wr, c0) in
          Nx.concatenate ~axis:0
            [ Nx.reshape [| 1 |] (f (wr, c0)); Nx.flatten gw; gc ]);
      stages at "stage a scan over the rows of three another scan wrote"
        (fun ran xs ->
          let ys = rollout ran wr hr xs in
          rollout ran (Nx.neg wr) (Nx.neg hr) ys);
      test
        "stage a scan whose step reads a draw and another scan's result made \
         before it" (fun () ->
          let settle d =
            fst
              (Rune.scan'
                 ~f:(fun d _ -> (Nx.tanh (Nx.matmul d d), d))
                 ~init:d
                 (zeros 4 |> Nx.reshape [| 4; 1 |]))
          in
          let f ran (k, xs) =
            let d =
              settle
                (Nx.mul_s
                   (Nx.Rng.with_key k (fun () -> Nx.randn Nx.float32 [| 3; 3 |]))
                   0.3)
            in
            fst
              (Rune.scan'
                 ~f:(fun h x ->
                   incr ran;
                   (cell d h x, h))
                 ~init:hr xs)
          in
          let k = Nx.Rng.key 7 in
          let runs n =
            let ran = ref 0 in
            let xs = Nx.mul_s (grid n 3) 0.01 in
            let expected = f ran (k, xs) in
            ran := 0;
            let g =
              Rune.jit
                Nx.Ptree.(pair Nx.Rng.ptree tensor @-> returns tensor)
                (f ran)
            in
            equal near expected (host (g (k, Nx.place at xs)));
            !ran
          in
          equal ~msg:"steps for 17 rows against 9" int (runs 9) (runs 17));
      test "a consumed leaf beside a staged scan is lent" (fun () ->
          let both = Nx.Ptree.(pair tensor tensor) in
          let step (u, v) =
            (Nx.add_s u 1., fst (Rune.scan' ~f:sum ~init:(zeros 4) v))
          in
          let g = Rune.jit Nx.Ptree.(consumes both @@ returns both) step in
          let u = Nx.place at (x ()) and v = Nx.place at (rows 6 4) in
          let before = address u in
          let u', v' = g (u, v) in
          equal ~msg:"lent" nativeint before (address u');
          equal near (Nx.add_s (x ()) 1.) (host u');
          equal near
            (fst (Rune.scan' ~f:sum ~init:(zeros 4) (rows 6 4)))
            (host v'));
      test "a staged carry is one buffer, whatever the number of steps"
        (fun () ->
          let held n k =
            let xs = Nx.place at (rows n 4) in
            let g =
              Rune.jit' (fun xs ->
                  fst
                    (Rune.scan'
                       ~f:(fun c x ->
                         let c = Nx.add (Nx.mul_s c 0.5) (Nx.sum x) in
                         (c, Nx.sum c))
                       ~init:(zeros k) xs))
            in
            let before = settled d in
            let r = g xs in
            (* What the call holds once its temporaries are collected, with its
               program and result alive: a temporary may or may not be collected
               by the end of the call. *)
            let held = settled d - before in
            ignore (Sys.opaque_identity (g, xs));
            ignore (host r);
            held
          in
          let larger n =
            let wide = held n 4096 in
            wide - held n 4
          in
          let few = warmed (fun () -> larger 64) in
          let many = larger 512 in
          equal ~msg:"for 64 and 512 steps" int few many);
      test "a staged scan reads its rows in place" (fun () ->
          let held w =
            let xs =
              Nx.place at (Nx.mul_s (Nx.ones Nx.float32 [| 64; w |]) 0.01)
            in
            let g =
              Rune.jit' (fun xs ->
                  fst
                    (Rune.scan'
                       ~f:(fun c x ->
                         (Nx.add (Nx.mul_s c 0.5) (Nx.sum x), Nx.sum c))
                       ~init:(zeros 4) xs))
            in
            let before = settled d in
            let r = g xs in
            let held = allocated d - before in
            ignore (host r);
            held
          in
          let wide = warmed (fun () -> held 1024) in
          let narrow = held 4 in
          less ~msg:"bytes held for rows 1,024 values wide against 4" int
            ~than:(64 * 1020 * 4)
            (wide - narrow));
      staged at "stage a step whose output is a constant" ~steps:once
        ~init:(zeros 4)
        (fun c x -> (Nx.add (Nx.mul_s c 0.5) x, Nx.zeros Nx.float32 [||]))
        (rows 300 4);
      staged at "stage a step whose output is empty" ~steps:once ~init:(zeros 4)
        (fun c x ->
          let c = Nx.add (Nx.mul_s c 0.5) x in
          (c, Nx.slice [ R (0, 0) ] c))
        (rows 300 4);
      (* Written out, each step's carry is stored before the next reads it: four
         hundred steps, past the 256 levels Metal nests, compile as kernels of
         one step each. *)
      test "write out four hundred steps, each carry stored" (fun () ->
          let ran = ref 0 in
          let f (k, xs) =
            Nx.Rng.with_key k (fun () ->
                Rune.scan'
                  ~f:(fun c x ->
                    incr ran;
                    let c = Nx.add (Nx.mul_s c 0.5) x in
                    (c, Nx.add c (Nx.rand Nx.float32 [| 4 |])))
                  ~init:(zeros 4) xs)
          in
          let k = Nx.Rng.key 7 in
          let c, ys = f (k, rows 400 4) in
          ran := 0;
          let c', ys' =
            Rune.jit
              Nx.Ptree.(
                pair Nx.Rng.ptree tensor @-> returns (pair tensor tensor))
              f
              (k, Nx.place at (rows 400 4))
          in
          equal near c (host c');
          equal near ys (host ys');
          equal ~msg:"a probe, then a step per row" int 401 !ran);
      test
        "write out a step that draws under a key scope, drawing as eager does \
         before, inside and after the scan" (fun () ->
          let ran = ref 0 in
          let f (k, xs) =
            Nx.Rng.with_key k (fun () ->
                let before = Nx.rand Nx.float32 [| 2 |] in
                let c, ys =
                  Rune.scan'
                    ~f:(fun c x ->
                      incr ran;
                      (Nx.add c x, Nx.add x (Nx.rand Nx.float32 [| 4 |])))
                    ~init:(zeros 4) xs
                in
                [ before; c; ys; Nx.rand Nx.float32 [| 3 |] ])
          in
          let k = Nx.Rng.key 42 in
          let eager = f (k, rows 5 4) in
          ran := 0;
          let g =
            Rune.jit
              Nx.Ptree.(pair Nx.Rng.ptree tensor @-> returns (list tensor))
              f
          in
          List.iter2
            (fun e c -> equal near e (host c))
            eager
            (g (k, Nx.place at (rows 5 4)));
          equal ~msg:"a probe, then a step per row" int 6 !ran);
    ]

(* Scans and remats *)

let scans =
  let cumulative xs =
    Rune.scan'
      ~f:(fun c x -> (Nx.add c x, Nx.mul c x))
      ~init:(Nx.zeros Nx.float32 [| 2 |])
      xs
  in
  group "scans"
    [
      group "constant rows" (constant_rows Nx.Placement.host);
      group "gradients" (both_gradients Nx.Placement.host);
      lent_beside_taken Nx.Placement.host;
      test "a scan folds inside the trace and equals eager" (fun () ->
          let f xs = snd (cumulative xs) in
          equal close (f (grid 3 2)) (Rune.jit' f (grid 3 2)));
      staged Nx.Placement.host
        "a scan on the host stages, its step running once"
        ~steps:(fun _ -> 1)
        ~init:(zeros 3) sum (rows 5 3);
      staged Nx.Placement.host "a scan of four hundred steps on the host stages"
        ~steps:(fun _ -> 1)
        ~init:(zeros 4) decay (rows 400 4);
      test "a scan over rows computed from constants alone equals eager"
        (fun () ->
          List.iter
            (fun rows ->
              let f c =
                Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor
                  ~f:(fun c l -> (Nx.add c (Nx.sum l), Nx.mul_s l 10l))
                  ~init:c (rows ())
              in
              let g =
                Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) f
              in
              equal
                (pair (tensor int32) (tensor int32))
                (f (Nx.zeros Nx.int32 [||]))
                (g (Nx.zeros Nx.int32 [||])))
            [
              (fun () -> Nx.cumsum (Nx.ones Nx.int32 [| 8 |]));
              (fun () -> Nx.cumsum ~axis:0 (Nx.ones Nx.int32 [| 8; 5 |]));
            ]);
      test "a gradient through a scan equals eager's" (fun () ->
          let f xs = Nx.sum (snd (cumulative xs)) in
          equal close
            (Rune.grad' f (grid 3 2))
            (Rune.jit' (Rune.grad' f) (grid 3 2)));
      test "a scan over float16 rows short of 16 bytes equals eager" (fun () ->
          let f xs =
            snd
              (Rune.scan'
                 ~f:(fun c x -> (Nx.add c x, Nx.mul c x))
                 ~init:(Nx.zeros Nx.float16 [| 3 |])
                 xs)
          in
          let xs = Nx.cast Nx.float16 (Nx.mul_s (grid 4 3) 0.25) in
          equal floats
            (Nx.cast Nx.float32 (f xs))
            (Nx.cast Nx.float32 (Rune.jit' f xs)));
      test "nested scans inside a compiled call equal eager" (fun () ->
          let inner c row =
            fst (Rune.scan' ~f:(fun c x -> (Nx.add c x, x)) ~init:c row)
          in
          let f xs =
            snd
              (Rune.scan'
                 ~f:(fun c x ->
                   let c = inner c (Nx.reshape [| 3; 1 |] x) in
                   (c, c))
                 ~init:(Nx.zeros Nx.float32 [| 1 |])
                 xs)
          in
          equal close (f (grid 2 3)) (Rune.jit' f (grid 2 3)));
      test
        "a scan in a staged scan's step is a loop of its own, its step traced \
         as often whatever its rows" (fun () ->
          let ran = ref 0 in
          let f m xs =
            snd
              (Rune.scan'
                 ~f:(fun c x ->
                   let c, _ =
                     Rune.scan'
                       ~f:(fun c x ->
                         incr ran;
                         (Nx.add c x, x))
                       ~init:c
                       (Nx.reshape [| m; 1 |] x)
                   in
                   (c, c))
                 ~init:(Nx.zeros Nx.float32 [| 1 |])
                 xs)
          in
          let traced m =
            let xs = grid 2 m in
            let expected = f m xs in
            ran := 0;
            let r = Rune.jit' (f m) xs in
            equal close expected r;
            !ran
          in
          equal int (traced 3) (traced 7));
      reads_outer Nx.Placement.host;
      test "a carry that changes its shape across steps is written out"
        (fun () ->
          let f xs =
            fst
              (Rune.scan'
                 ~f:(fun c x -> (Nx.concatenate ~axis:0 [ c; x ], x))
                 ~init:(Nx.zeros Nx.float32 [| 1 |])
                 xs)
          in
          let xs = Nx.reshape [| 3; 1 |] (arange 3) in
          equal close (f xs) (Rune.jit' f xs));
      test "an empty scan axis raises Rune.scan's message" (fun () ->
          let f xs = snd (cumulative xs) in
          let xs = Nx.zeros Nx.float32 [| 0; 2 |] in
          equal string
            (message (fun () -> f xs))
            (message (fun () -> Rune.jit' f xs)));
      test "a remat under a compiled gradient equals eager's" (fun () ->
          let block a = Nx.tanh (Nx.mul a a) in
          let f a =
            Nx.sum (Rune.remat Nx.Ptree.(tensor @-> returns tensor) block a)
          in
          (* Away from tanh's saturation, where 1 - tanh² is a difference of
             nearly equal numbers. *)
          let a = Nx.create Nx.float32 [| 4 |] [| 0.5; -0.3; 0.8; 0.1 |] in
          equal close (Rune.grad' f a) (Rune.jit' (Rune.grad' f) a));
    ]

(* Device lists *)

let split ?(axis = 0) ds = Nx.Placement.sharded ~backend:Rune.compiled ~axis ds
let copies ds = Nx.Placement.replicated ~backend:Rune.compiled ds

let device_lists =
  let pair = [ d1; d2 ] in
  group "device lists"
    [
      test "elementwise operations and a sum over two devices equal one device"
        (fun () ->
          let f a = Nx.sum ~axes:[ 1 ] (Nx.mul (Nx.exp a) a) in
          let a = grid 4 3 in
          equal close (f a) (host (Rune.jit' f (Nx.place (split pair) a))));
      slow "an elementwise chain over two devices has one device's bits"
        (fun () ->
          let a = grid 4 3 in
          equal floats (poly a)
            (host (Rune.jit' poly (Nx.place (split pair) a))));
      slow "a value split along its second axis computes" (fun () ->
          let a = grid 3 4 in
          let r = Rune.jit' poly (Nx.place (split ~axis:1 pair) a) in
          is_true (Nx.Placement.equal (split ~axis:1 pair) (Nx.placement r));
          equal floats (poly a) (host r));
      (* [zeros_like] of a split value is split alike, so the scatter is too and
         stores each device's rows. *)
      test "a scatter-add into zeros like a split value equals eager" (fun () ->
          let f x t =
            Nx.Op.eval
              (Nx.Op.Scatter
                 {
                   mode = `Add;
                   unique = false;
                   axis = 1;
                   indices = Nx.unsqueeze ~axes:[ -1 ] t;
                   updates = Nx.ones Nx.float32 [| 4; 1 |];
                   into = Nx.zeros_like x;
                 })
          in
          let x = grid 4 3
          and t = Nx.create Nx.int64 [| 4 |] [| 2L; 0L; 1L; 2L |] in
          let at a = Nx.place (split pair) a in
          equal close
            (host (f (at x) (at t)))
            (host
               (Rune.jit
                  Nx.Ptree.(tensor @-> tensor @-> returns tensor)
                  f (at x) (at t))));
      test "the gradient of a mean over a split batch equals one device's"
        (fun () ->
          let f a = Nx.mean (Nx.mul a a) in
          let a = grid 4 3 in
          equal close (Rune.grad' f a)
            (host (Rune.jit' (Rune.grad' f) (Nx.place (split pair) a))));
      slow "two collectively reduced results equal one device's" (fun () ->
          let a = grid 4 3 in
          let s, m =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              (fun a -> (Nx.sum a, Nx.max a))
              (Nx.place (split pair) a)
          in
          equal close (Nx.sum a) (host s);
          equal close (Nx.max a) (host m));
      test "arguments placed with another backend bind, and results keep it"
        (fun () ->
          let p = Nx.Placement.sharded ~axis:0 pair in
          let r = Rune.jit' poly (Nx.place p (grid 4 3)) in
          is_true (Nx.Placement.equal p (Nx.placement r));
          equal floats (poly (grid 4 3)) (host r));
      slow "a result fed back to the call moves no bytes" (fun () ->
          let g = Rune.jit' (fun a -> Nx.mul_s a 0.5) in
          let r = ref (g (Nx.place (split pair) (grid 4 3))) in
          let before = bytes_in d1 + bytes_in d2 in
          for _ = 1 to 3 do
            r := g !r
          done;
          equal int before (bytes_in d1 + bytes_in d2));
      test "a capture copied to every device is bound on each" (fun () ->
          let w = Nx.place (copies pair) (y ()) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = Nx.place (copies pair) (x ()) in
          ignore (g a);
          let before = bytes_in d1 + bytes_in d2 in
          equal close (Nx.mul (x ()) (y ())) (host (g a));
          equal int before (bytes_in d1 + bytes_in d2));
      slow "one shard's slice of a split value computes on its device alone"
        (fun () ->
          let a = Nx.place (split pair) (grid 4 3) in
          let row = Nx.slice [ I 3 ] a in
          let r = Rune.jit' poly row in
          is_true (Nx.Placement.equal (on d2) (Nx.placement r));
          equal floats (poly (Nx.slice [ I 3 ] (grid 4 3))) (host r));
      test "a loop consuming a split state holds two generations on each device"
        (fun () ->
          let n = 1 lsl 15 in
          let step = Rune.jit consumes (fun a -> Nx.add_s a 1.) in
          let s =
            ref (Nx.place (split [ d3; d4 ]) (Nx.zeros Nx.float32 [| 2 * n |]))
          in
          s := step !s;
          let b3 = allocated d3 and b4 = allocated d4 in
          for _ = 1 to 10 do
            s := step !s
          done;
          at_most ~msg:"on the first device" int ~than:(4 * n)
            (allocated d3 - b3);
          at_most ~msg:"on the second device" int ~than:(4 * n)
            (allocated d4 - b4);
          equal floats (Nx.full Nx.float32 [| 2 * n |] 11.) (host !s));
      slow
        "gradients through max, sum and mean keeping their axes equal one \
         device's" (fun () ->
          let a = grid 4 3 in
          List.iter
            (fun (name, f) ->
              equal ~msg:name close (Rune.grad' f a)
                (host (Rune.jit' (Rune.grad' f) (Nx.place (split pair) a))))
            [
              ( "max",
                fun a -> Nx.sum (Nx.mul a (Nx.max ~axes:[ 0 ] ~keepdims:true a))
              );
              ( "sum",
                fun a -> Nx.sum (Nx.mul a (Nx.sum ~axes:[ 0 ] ~keepdims:true a))
              );
              ( "mean",
                fun a ->
                  Nx.sum (Nx.mul a (Nx.mean ~axes:[ 0 ] ~keepdims:true a)) );
            ]);
      slow "a remat over a split batch equals one device's gradient" (fun () ->
          let block a = Nx.tanh (Nx.mul_s a 0.5) in
          let f a =
            Nx.sum (Rune.remat Nx.Ptree.(tensor @-> returns tensor) block a)
          in
          let a = Nx.mul_s (grid 4 3) 0.1 in
          equal close (Rune.grad' f a)
            (host (Rune.jit' (Rune.grad' f) (Nx.place (split pair) a))));
      slow "a gradient through a scan over split rows equals one device's"
        (fun () ->
          let f xs =
            Nx.sum
              (snd
                 (Rune.scan'
                    ~f:(fun c x -> (Nx.add c x, Nx.mul c x))
                    ~init:(Nx.zeros Nx.float32 [| 4 |])
                    xs))
          in
          let xs = Nx.mul_s (grid 3 4) 0.1 in
          equal close (Rune.grad' f xs)
            (host (Rune.jit' (Rune.grad' f) (Nx.place (split ~axis:1 pair) xs))));
      slow "a scan over split rows equals one device's" (fun () ->
          let ran = ref 0 in
          let f xs = rollout ran wr hr xs in
          let xs = Nx.mul_s (grid 6 3) 0.01 in
          equal near (f xs)
            (host (Rune.jit' f (Nx.place (split ~axis:1 [ d1; d2; d3 ]) xs))));
      slow "a scan over the split rows another scan wrote equals one device's"
        (fun () ->
          let ran = ref 0 in
          let f xs = rollout ran (Nx.neg wr) hr (rollout ran wr hr xs) in
          let xs = Nx.mul_s (grid 6 3) 0.01 in
          equal near (f xs)
            (host (Rune.jit' f (Nx.place (split ~axis:1 [ d1; d2; d3 ]) xs))));
      slow "a carry the step places on the rows' devices equals one device's"
        (fun () ->
          let ran = ref 0 in
          let f xs =
            let (a, b), ys =
              Rune.scan
                Nx.Ptree.(pair tensor tensor)
                Nx.Ptree.tensor Nx.Ptree.tensor
                ~f:(fun (a, b) x ->
                  incr ran;
                  let a' = Nx.add (Nx.mul_s a 0.5) x in
                  let b' = Nx.add (Nx.mul_s b 0.5) a in
                  ((a', b'), Nx.mul a' b'))
                ~init:
                  (Nx.zeros Nx.float32 [| 16 |], Nx.zeros Nx.float32 [| 16 |])
                xs
            in
            Nx.add (Nx.sum a) (Nx.add (Nx.sum b) (Nx.sum ys))
          in
          let xs = Nx.mul_s (grid 6 16) 0.01 in
          equal near (f xs)
            (host
               (Rune.jit' f (Nx.place (split ~axis:1 [ d1; d2; d3; d4 ]) xs))));
      slow "a column-then-row split MLP equals one device" (fun () ->
          let w1 = Nx.mul_s (grid 3 4) 0.1 and w2 = Nx.mul_s (grid 4 3) 0.1 in
          let f (a, (w1, w2)) = Nx.matmul (Nx.relu (Nx.matmul a w1)) w2 in
          let s = Nx.Ptree.(pair tensor (pair tensor tensor)) in
          let a = Nx.mul_s (grid 2 3) 0.1 in
          let r =
            Rune.jit
              Nx.Ptree.(s @-> returns tensor)
              f
              ( Nx.place (copies pair) a,
                ( Nx.place (split ~axis:1 pair) w1,
                  Nx.place (split ~axis:0 pair) w2 ) )
          in
          equal close (f (a, (w1, w2))) (host r));
      slow "host arguments beside a split one enter as copies" (fun () ->
          let a = grid 4 3 and h = Nx.mul_s (grid 4 3) 2. in
          let r = Rune.jit two Nx.add (Nx.place (split pair) a) h in
          is_true (Nx.Placement.equal (split pair) (Nx.placement r));
          equal floats (Nx.add a h) (host r));
      slow "a split capture is bound on its devices, moving no bytes" (fun () ->
          let w = Nx.place (split pair) (Nx.mul_s (grid 4 3) 2.) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = Nx.place (split pair) (grid 4 3) in
          ignore (g a);
          let before = bytes_in d1 + bytes_in d2 in
          equal floats (Nx.mul (grid 4 3) (Nx.mul_s (grid 4 3) 2.)) (host (g a));
          equal int before (bytes_in d1 + bytes_in d2));
      slow "a value placed inside the function is split as it says" (fun () ->
          let r =
            Rune.jit' (fun a -> Nx.place (split pair) (poly a)) (grid 4 3)
          in
          is_true (Nx.Placement.equal (split pair) (Nx.placement r));
          equal floats (poly (grid 4 3)) (host r));
      slow "an indexed write into a split value equals eager" (fun () ->
          let f a = Nx.set [ A; I 1 ] (Nx.zeros Nx.float32 [| 4 |]) a in
          let a = grid 4 3 in
          equal floats (f a) (host (Rune.jit' f (Nx.place (split pair) a))));
      cases ~name:fst "a window written across the split axis equals eager"
        [
          ("at a start the program holds", fun (_ : Nx.int64_t) -> Nx.R (1, 3));
          ("at a start read when the call runs", fun pos -> Nx.D (pos, 2));
        ]
        (fun (_, at) ->
          let f x pos =
            Nx.set [ at pos; A ] (Nx.full Nx.float32 [| 2; 3 |] 9.) x
          in
          let pos = Nx.scalar Nx.int64 1L in
          let g = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f in
          equal floats
            (f (grid 4 3) pos)
            (host (g (Nx.place (split pair) (grid 4 3)) pos)));
      slow "a map over a split axis computes each lane" (fun () ->
          let f = Rune.vmap' (fun a -> Nx.add_s (Nx.mul a a) 1.) in
          let a = grid 4 3 in
          let expected = Nx.add_s (Nx.mul a a) 1. in
          equal floats expected (host (f (Nx.place (split pair) a)));
          equal floats expected (host (Rune.jit' f (Nx.place (split pair) a))));
      slow
        "a mask drawn from a key folded with each lane's index is that lane's, \
         over split lanes" (fun () ->
          let key = Nx.Rng.key 7 in
          let draw k =
            Nx.cast Nx.float32
              (Nx.Rng.bernoulli k
                 (Nx.broadcast_to [| 16 |] (Nx.scalar Nx.float32 0.5)))
          in
          let masks a key =
            Rune.vmap'
              (fun a ->
                Rune.grad'
                  (fun a ->
                    let m =
                      draw (Nx.Rng.fold_in_tensor key (Rune.lane_index ()))
                    in
                    Nx.mul_s (Nx.sum (Nx.mul (Nx.mul a a) m)) 0.5)
                  a)
              a
          in
          let masks =
            Rune.jit
              Nx.Ptree.(tensor @-> Nx.Rng.ptree @-> returns tensor)
              masks
              (Nx.place (split pair) (Nx.ones Nx.float32 [| 2; 16 |]))
              key
          in
          let lane i = draw (Nx.Rng.fold_in key i) in
          equal floats (Nx.stack [ lane 0; lane 1 ]) (host masks);
          is_false (Nx.array_equal (lane 0) (lane 1) |> Nx.item []));
      slow
        "a sum over an axis split over four devices replays each call's values"
        (fun () ->
          let reduce = Rune.jit' (Nx.sum ~axes:[ 0 ]) in
          List.iter
            (fun call ->
              let a =
                Nx.init Nx.float32 [| 4; 16 |] (fun i ->
                    float_of_int ((1000 * call) + (100 * i.(0)) + i.(1)))
              in
              let expected =
                Nx.init Nx.float32 [| 16 |] (fun i ->
                    float_of_int ((4000 * call) + 600 + (4 * i.(0))))
              in
              equal
                ~msg:(Printf.sprintf "call %d" call)
                floats expected
                (host (reduce (Nx.place (split [ d1; d2; d3; d4 ]) a))))
            [ 0; 1; 2 ]);
      slow "data-parallel training follows one device" (fun () ->
          let loss w a = Nx.mean (Nx.square (Nx.matmul a w)) in
          let step =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> returns tensor)
              (fun w a ->
                Nx.sub w (Nx.mul_s (Rune.grad' (fun w -> loss w a) w) 0.1))
          in
          let eager w a =
            Nx.sub w (Nx.mul_s (Rune.grad' (fun w -> loss w a) w) 0.1)
          in
          let a = Nx.mul_s (grid 4 3) 0.1 in
          let w0 = Nx.mul_s (grid 3 2) 0.1 in
          let we = ref w0 and wc = ref (Nx.place (copies pair) w0) in
          for _ = 1 to 5 do
            we := eager !we a;
            wc := step !wc (Nx.place (split pair) a)
          done;
          equal close !we (host !wc));
    ]

(* Values on the disk *)

(* The float32 values of the file at [path], four of them, as a value on the
   disk. *)
let on_disk_at_read path =
  let module B = Nx_device.Buffer in
  let pp = Format.pp_print_string in
  let p = Nx.Placement.device Nx_device.disk in
  Nx.Repr.Placed.v p Nx.float32
    (Nx_array.View.create [| 4 |])
    (Nx.Repr.Storage.v p
       [
         B.view
           (require_ok ~pp (B.of_file path))
           ~offset:0 Nx_dtype.Scalar.Float32 4;
       ])

(* [x] written to the file at [path], as a value on the disk over it. *)
let on_disk_at path x =
  let module B = Nx_device.Buffer in
  let src = elements x in
  let pp = Format.pp_print_string in
  B.copy ~src ~dst:(require_ok ~pp (B.create_file path (B.nbytes src)));
  let p = Nx.Placement.device Nx_device.disk in
  Nx.Repr.Placed.v p (Nx.dtype x)
    (Nx_array.View.create (Nx.shape x))
    (Nx.Repr.Storage.v p
       [
         B.view
           (require_ok ~pp (B.of_file path))
           ~offset:0 (B.dtype src) (B.length src);
       ])

(* The int32 values [v] in a file, two bytes after its start, as a value on the
   disk whose elements are not aligned to their width. *)
let unaligned_on_disk v =
  let module B = Nx_device.Buffer in
  let n = Array.length v in
  let path = temp_file () in
  let bytes =
    Nx.init Nx.uint8
      [| 2 + (4 * n) |]
      (fun i ->
        let i = i.(0) - 2 in
        if i < 0 then 0
        else
          Int32.to_int (Int32.shift_right_logical v.(i / 4) (8 * (i mod 4)))
          land 255)
  in
  ignore (on_disk_at path bytes);
  let p = Nx.Placement.device Nx_device.disk in
  Nx.Repr.Placed.v p Nx.int32
    (Nx_array.View.create [| n |])
    (Nx.Repr.Storage.v p
       [
         B.view
           (require_ok ~pp:Format.pp_print_string (B.of_file path))
           ~offset:2 Nx_dtype.Scalar.Int32 n;
       ])

let disk =
  group "values on the disk"
    [
      test "a value on the disk not aligned to its elements is read" (fun () ->
          let v = [| 1l; -2l; 70000l; Int32.min_int; Int32.max_int; 0l |] in
          let a = unaligned_on_disk v in
          equal (tensor int32)
            (Nx.mul_s (Nx.create Nx.int32 [| 6 |] v) 3l)
            (Rune.jit' (fun a -> Nx.mul_s a 3l) a));
      test "a leaf and a capture on the disk are read as host values" (fun () ->
          let a = on_disk_at (temp_file ()) (x ()) in
          let w = on_disk_at (temp_file ()) (y ()) in
          equal close
            (Nx.mul (poly (x ())) (y ()))
            (Rune.jit' (fun a -> Nx.mul (poly a) w) a));
      test "a consumed value on the disk is copied, and its file unchanged"
        (fun () ->
          let path = temp_file () in
          let a = on_disk_at path (x ()) in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal floats (Nx.add_s (x ()) 1.) (host r);
          equal floats (x ()) (host (on_disk_at_read path)));
      test
        "a consumed weight that is a window of its file is computed from a \
         copy and stays readable, as its sibling does" (fun () ->
          let module B = Nx_device.Buffer in
          let path = temp_file () in
          ignore (on_disk_at path (Nx.concatenate ~axis:0 [ x (); y () ]));
          let p = Nx.Placement.device Nx_device.disk in
          let file = require_ok ~pp:Format.pp_print_string (B.of_file path) in
          let weight first =
            Nx.Repr.Placed.v p Nx.float32
              (Nx_array.View.create [| 4 |])
              (Nx.Repr.Storage.v p
                 [ B.view file ~offset:(4 * first) Float32 4 ])
          in
          let sibling = weight 0 and w = weight 4 in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) w in
          equal floats (Nx.add_s (y ()) 1.) (host r);
          equal ~msg:"the weight" floats (y ()) (host w);
          equal ~msg:"its sibling" floats (x ()) (host sibling));
      test "the file opened is read, not the one at its path now" (fun () ->
          let path = temp_file () in
          let a = on_disk_at path (x ()) in
          let replacement = temp_file () in
          ignore (on_disk_at replacement (y ()));
          Sys.rename replacement path;
          equal close (poly (x ())) (host (Rune.jit' poly a)));
    ]

(* Gathers *)

(* Rows read at indices from memory, by calls whose arguments are placed at
   [at], computing on [d]: a gather that its reader broadcasts or reads twice is
   stored by a kernel of its own and read back, and so is a sum masked by a
   bound from memory. A device loads a program once however many calls run it,
   so each test's rows have a width of their own, and no kernel of one test is
   loaded by another. *)
let gathers ~at d =
  let rows n w scale =
    Nx.init Nx.float32 [| n; w |] (fun i ->
        Float.of_int ((i.(0) * w) + i.(1) + 1) *. scale)
  in
  let scores w c =
    Nx.matmul
      (Nx.mul (rows 8 w 0.01) c)
      (Nx.transpose (Nx.mul (rows 8 w 0.02) c))
  in
  let kernels name n f x =
    test name (fun () ->
        let r, loaded = loaded_on d (fun () -> Rune.jit' f (Nx.place at x)) in
        equal ~msg:"kernels" int n loaded;
        equal close (f x) (host r))
  in
  let indices =
    Nx.create Nx.int64 [| 8 |] [| 3L; 5L; 7L; 0L; 47L; 12L; 12L; 40L |]
  in
  group "gathers"
    [
      kernels "a cache's rows read by a product are a kernel of their own" 2
        (fun slots ->
          Nx.matmul (rows 8 32 0.01)
            (Nx.transpose (Nx.take ~axis:0 (rows 20 32 0.001) ~indices:slots)))
        (Nx.create Nx.int64 [| 12 |]
           (Array.init 12 (fun i -> Int64.of_int (19 - i))));
      kernels
        "rows at positions from an offset, read twice, are a kernel of their \
         own"
        2
        (fun p ->
          scores 16
            (Nx.take ~axis:0 (rows 64 16 0.001)
               ~indices:(Nx.add (Nx.arange Nx.int64 0 8 1) p)))
        (Nx.scalar Nx.int64 4L);
      kernels "rows read by two sums are a kernel of their own" 2
        (fun indices ->
          let g = Nx.take ~axis:0 (rows 64 40 0.001) ~indices in
          Nx.add
            (Nx.sum ~axes:[ 1 ] (Nx.mul g (Nx.slice [ I 0 ] (rows 8 40 0.01))))
            (Nx.sum ~axes:[ 1 ] (Nx.mul g (Nx.slice [ I 1 ] (rows 8 40 0.02)))))
        indices;
      kernels
        "rows summed below a bound from memory, read twice, are a kernel of \
         their own"
        2
        (fun bound ->
          let below =
            Nx.less
              (Nx.reshape [| 1; 48; 1 |] (Nx.arange Nx.int64 0 48 1))
              bound
          in
          scores 24
            (Nx.sum ~axes:[ 1 ]
               (Nx.where below
                  (Nx.reshape [| 1; 48; 24 |] (rows 48 24 0.001))
                  (Nx.scalar Nx.float32 0.))))
        (Nx.reshape [| 8; 1; 1 |] indices);
    ]

(* One device *)

(* The calls whose bytes and memory a device counts, on [d]: the test devices by
   default, Metal in the slow run. *)
let on_one_device ~name d =
  let block (w1, w2) a = Nx.add a (Nx.matmul (Nx.relu (Nx.matmul a w1)) w2) in
  group name
    [
      test "a call runs where its arguments lie, and leaves its results there"
        (fun () ->
          let r = Rune.jit' poly (placed d (x ())) in
          is_true (Nx.Placement.equal (on d) (Nx.placement r));
          equal close (poly (x ())) (host r));
      test "a consumed slice at an offset is computed into storage of its own"
        (fun () ->
          consumed_slice ~at:(on d) 1 7;
          consumed_slice ~at:(on d) 5 1027);
      test "a call searched on several domains computes eager's values"
        (fun () ->
          let f a = Nx.add_s (poly a) 0.8125 in
          equal close
            (f (x ()))
            (host (Rune.jit' ~beam:1 ~parallel:2 f (placed d (x ())))));
      (* The gather broadcasts its indices to the rows it reads: the copy moves
         the indices, and the broadcast is taken on the device. *)
      test "a host index a gather reads uploads its own bytes" (fun () ->
          let table = placed d (grid 16 4) in
          let g =
            Rune.jit' (fun ids ->
                Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] ids) table)
          in
          let ids = Nx.create Nx.int64 [| 1; 4 |] [| 3L; 1L; 2L; 0L |] in
          ignore (g ids);
          let before = bytes_in d in
          let r = g ids in
          equal ~msg:"bytes received" int (Nx.nbytes ids) (bytes_in d - before);
          equal close
            (Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] ids) (grid 16 4))
            (host r));
      test "a placed argument feeds a call with no transfer" (fun () ->
          let g = Rune.jit' poly in
          let a = placed d (x ()) in
          ignore (g a);
          let before = bytes_in d in
          ignore (g a);
          equal int before (bytes_in d));
      test "a placed view is read where it lies" (fun () ->
          let g = Rune.jit' poly in
          let a = Nx.transpose (placed d (grid 2 3)) in
          ignore (g a);
          let before = bytes_in d in
          let r = g a in
          equal ~msg:"bytes received" int before (bytes_in d);
          equal close (poly (Nx.transpose (grid 2 3))) (host r));
      test
        "a float16 argument starting 2 bytes further retraces once, and is \
         read where it lies" (fun () ->
          let g = Rune.jit' poly in
          let a = placed d (Nx.cast Nx.float16 (arange 12)) in
          let read lo () =
            let v = Nx.slice [ R (lo, lo + 4) ] a in
            equal close
              (Nx.cast Nx.float32
                 (poly
                    (Nx.slice
                       [ R (lo, lo + 4) ]
                       (Nx.cast Nx.float16 (arange 12)))))
              (Nx.cast Nx.float32 (host (g v)))
          in
          retraces (read 0) (read 1));
      test "a chain of 8-bit float operations rounds after each, as eager does"
        (fun () ->
          let chain (type b) (dt : (float, b) Nx.dtype) =
            let x =
              Nx.create Nx.float32 [| 6 |]
                [| 13.7; 1.3; 0.1; 3.3; -2.7; 0.0123 |]
            in
            let f x =
              let y = Nx.cast dt x in
              Nx.cast Nx.float32 (Nx.mul (Nx.add y y) y)
            in
            equal floats (f x) (host (Rune.jit' f (placed d x)))
          in
          chain Nx.float8_e4m3;
          chain Nx.float8_e5m2);
      test "a call whose trace raises allocates nothing" (fun () ->
          let a = placed d (x ()) in
          let before = allocated d in
          raises_jit_error (fun () ->
              Rune.jit' (fun a -> if Nx.item [ 0 ] a > 0. then poly a else a) a);
          equal ~msg:"bytes allocated" int before (allocated d));
      test "a consumed placed argument lends its storage" (fun () ->
          let a = placed d (x ()) in
          let before = Witness.addresses a in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal (list nativeint) before (Witness.addresses r);
          equal close (Nx.add_s (x ()) 1.) (host r));
      test "a capture placed where the call computes is bound, not uploaded"
        (fun () ->
          let w = placed d (y ()) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = placed d (x ()) in
          ignore (g a);
          let before = bytes_in d in
          let r = g a in
          equal ~msg:"bytes received" int before (bytes_in d);
          equal close (Nx.mul (x ()) (y ())) (host r);
          is_false ~msg:"pinned" (lends w);
          equal ~msg:"the program after" close
            (Nx.mul (x ()) (y ()))
            (host (g a)));
      test "a value the call computes from host captures is uploaded once"
        (fun () ->
          let limit = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 2l; 3l |] in
          let f a =
            let mask = Nx.less_s limit 2l in
            let scale = Nx.mul_s (Nx.cast Nx.float32 limit) 0.5 in
            Nx.where
              (Nx.place (on d) mask)
              (Nx.mul a (Nx.place (on d) scale))
              (Nx.zeros_like a)
          in
          let g = Rune.jit' f in
          let a = placed d (x ()) in
          ignore (g a);
          let before = bytes_in d in
          let r = g a in
          equal ~msg:"bytes received" int before (bytes_in d);
          equal close (host (f (x ()))) (host r));
      test "a loop consuming its state holds two generations of it" (fun () ->
          let n = 1 lsl 16 in
          let step = Rune.jit consumes (fun a -> Nx.add_s a 1.) in
          let s = ref (placed d (Nx.zeros Nx.float32 [| n |])) in
          s := step !s;
          let base = allocated d in
          for _ = 1 to 20 do
            s := step !s
          done;
          at_most ~msg:"bytes allocated across 20 steps" int ~than:(4 * n)
            (allocated d - base);
          equal floats (Nx.full Nx.float32 [| n |] 21.) (host !s));
      test "a view of a weight on the disk placed on the device is captured"
        (fun () ->
          let w = on_disk_at (temp_file ()) (grid 4 4) in
          let p = Nx.place (on d) (Nx.matrix_transpose w) in
          let g = Rune.jit' (fun a -> Nx.matmul a p) in
          equal close
            (Nx.matmul (grid 2 4) (Nx.matrix_transpose (grid 4 4)))
            (host (g (placed d (grid 2 4)))));
      test
        "a consumed value placed from a file lends its storage only where the \
         file was copied, and the file keeps its elements" (fun () ->
          let path = temp_file () in
          let elements = Nx.create Nx.float32 [| 4 |] [| 5.; 6.; 1.; 2. |] in
          let pool = Nx.place (on d) (on_disk_at path elements) in
          let before = Witness.addresses pool in
          let indices = Nx.create Nx.int64 [| 2 |] [| 0L; 2L |] in
          let values = Nx.create Nx.float32 [| 2 |] [| 10.; 30. |] in
          let r =
            Rune.jit consumes (Nx.scatter ~axis:0 ~indices ~values) pool
          in
          equal floats
            (Nx.create Nx.float32 [| 4 |] [| 10.; 6.; 30.; 2. |])
            (host r);
          (* A device that shares the host's memory borrows the file's pages,
             which it must not lend; another copies them into its own. *)
          let copied = not (Nx_device.shares_host_memory d) in
          equal bool ~msg:"lent" copied
            (List.equal Nativeint.equal before (Witness.addresses r));
          raises_invalid_arg (fun () -> Nx.to_array pool);
          equal floats elements (host (on_disk_at_read path)));
      slow "a compiled gradient through remats keeps under half the activations"
        (fun () ->
          let layers = 8 and batch = 256 and dim = 32 in
          let hidden = 8 * dim in
          let weights =
            List.init layers (fun i ->
                let w r c =
                  placed d
                    (Nx.mul_s
                       (Nx.Rng.with_key (Nx.Rng.key i) (fun () ->
                            Nx.randn Nx.float32 [| r; c |]))
                       0.05)
                in
                (w dim hidden, w hidden dim))
          in
          let a = placed d (Nx.ones Nx.float32 [| batch; dim |]) in
          let loss remat a =
            Nx.sum
              (List.fold_left
                 (fun a w ->
                   if remat then
                     Rune.remat Nx.Ptree.(tensor @-> returns tensor) (block w) a
                   else block w a)
                 a weights)
          in
          let peak remat =
            let g = Rune.jit' (Rune.grad' (loss remat)) in
            let base = settled d in
            let r = g a in
            let used = allocated d - base in
            ignore (host r);
            used
          in
          let plain = warmed (fun () -> peak false) in
          let recomputed = peak true in
          less
            ~msg:
              (Printf.sprintf "%d bytes with remat, %d without" recomputed plain)
            int ~than:(plain / 2) recomputed);
    ]

(* A value computed from constants alone, used on [d], whose programs load
   there, is computed in the kernel that reads it. *)
let constants_where_used d =
  test "a value computed from no capture is computed where it is used"
    (fun () ->
      let f p = Nx.add (Nx.arange Nx.int64 0 8 1) p in
      let p = Nx.scalar Nx.int64 4L in
      let r, loaded = loaded_on d (fun () -> Rune.jit' f (placed d p)) in
      equal ~msg:"programs" int 1 loaded;
      equal (tensor int64) (f p) (host r))

(* A compiled sum adds each product into its running sum rounded once: the
   products -(1 + 2^-11) and (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24 sum to 2^-24,
   where rounding the second product first gives 0. Both are in the class of a
   rounded sum. *)
let sums_fuse_products d =
  test "a compiled sum adds each product into its running sum rounded once"
    (fun () ->
      let x = 1. +. 0x1p-12 in
      let a = Nx.create Nx.float32 [| 1; 2 |] [| -.(1. +. 0x1p-11); x |] in
      let b = placed d (Nx.create Nx.float32 [| 2; 1 |] [| 1.; x |]) in
      let f a = Nx.matmul a b in
      equal floats
        (Nx.create Nx.float32 [| 1; 1 |] [| 0x1p-24 |])
        (host (Rune.jit' f (placed d a))))

(* The calls on a GPU of [kind], if this machine has one. *)
let on_gpu kind = function
  | Some m ->
      [
        on_one_device ~name:"one device" m;
        constants_where_used m;
        sums_fuse_products m;
        staged_scans m;
        rows_written
          ~at:(Nx.Placement.device ~backend:Rune.compiled m)
          "a lent write of rows";
        gathers ~at:(on m) m;
      ]
  | None ->
      let why = "no " ^ kind ^ " device" in
      [ slow why (fun () -> skip ~reason:why ()) ]

let () =
  exit
    (run "Rune_internals.Jit"
       [
         values ~count:1 ~heavy:false;
         keys;
         results;
         consumption;
         lending;
         rows_written "a lent write of rows";
         captures;
         errors;
         reports;
         domains;
         transformations;
         placement;
         scans;
         gathers ~at:Nx.Placement.host Nx_device.host;
         device_lists;
         disk;
         on_one_device ~name:"one device" d4;
         sums_fuse_products Nx_device.host;
         group ~tags:[ "slow" ] "metal" (on_gpu "Metal" Metal.device);
         group ~tags:[ "slow" ] "cuda" (on_gpu "CUDA" Nvidia.cuda);
         group ~tags:[ "slow" ] "nv" (on_gpu "NV" Nvidia.nv);
         group ~tags:[ "slow" ] "swept" [ values ~count:25 ~heavy:true ];
       ])
