(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Padded loads and Fold through Nx.Prim: the values Nx_kernel.load and
   Nx_kernel.S.fold state for small operands, their rules before any kernel,
   a split placement, and a library that declines. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

(* nx.cpu, declining every fold and every loop with a padded load. *)
module Unpadded = struct
  include Nx_cpu

  let name = "nx.unpadded"

  let padded loads =
    Array.exists
      (function Nx_kernel.Padded _ -> true | Nx_kernel.Plain _ -> false)
      loads

  let map p ~dsts loads =
    if padded loads then A.Declined else Nx_cpu.map p ~dsts loads

  let reduce p ~shape ~axes rs ~dsts loads =
    if padded loads then A.Declined
    else Nx_cpu.reduce p ~shape ~axes rs ~dsts loads

  let fold _ ~dst:_ _ = A.Declined
end

module Dec = (val Nx.devices ~kernels:(module Unpadded) [ m 2 ])

let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))
let host a = Nx.Repr.of_array Nx.Host.v a
let identity dt = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |]

let pad ?(lo = [| 0 |]) ?(hi = [| 0 |]) ?(interior = [| 0 |]) windows =
  { Nx_kernel.lo; hi; interior; windows }

let window size step = { A.Move.axis = 0; size; step; dilation = 1 }

(* The shape [x] has once loaded through [pad]. *)
let loaded x pad =
  match Nx_kernel.Shape.padded pad (Nx.shape x) with
  | Ok s -> s
  | Error _ -> fail "no loaded shape"

(* Sum over the last axis of [x] loaded through [pad], padded with [fill]. *)
let sum_of x ~fill pad =
  let layout = L.contiguous (loaded x pad) in
  let r = L.rank layout in
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Reduce
         {
           layout;
           axes = [| r - 1 |];
           prog = identity (Nx.dtype x);
           reductions = [ Monoid (Sum, 0, Nx.dtype x) ];
           loads = [| Padded { x; fill; pad } |];
         })
  in
  y

(* [x] loaded through [pad], padded with [fill]. *)
let load x ~fill pad =
  let y, () =
    Nx.Prim.eval ~by:"t"
      (Map
         {
           layout = L.contiguous (loaded x pad);
           prog = identity (Nx.dtype x);
           outs = [ Nx.dtype x ];
           loads = [| Padded { x; fill; pad } |];
         })
  in
  y

let f32 xs = A.of_array D.Float32 [| Array.length xs |] xs

let test_sum_pool () =
  let x = Nx.place Count.on (host (f32 [| 1.; 2.; 3.; 4. |])) in
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  equal ~msg:"a sum over windows of three, padded with zeros"
    (array float_exact) [| 3.; 6.; 9.; 7. |]
    (read (sum_of x ~fill:0. p));
  equal ~msg:"padded with 10" (array float_exact) [| 13.; 6.; 9.; 17. |]
    (read (sum_of x ~fill:10. p));
  let p = pad ~lo:[| -1 |] ~interior:[| 1 |] [| window 2 2 |] in
  (* Interior padding [1; 0; 2; 0; 3; 0; 4], cropped by one: [0; 2; 0; 3; 0; 4],
     in windows of two, stepping by two. *)
  equal ~msg:"interior padding, a negative lo crops" (array float_exact)
    [| 2.; 3.; 4. |]
    (read (sum_of x ~fill:0. p))

let test_fold () =
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  let ones =
    Nx.place Count.on
      (host (A.of_array D.Float32 [| 4; 3 |] (Array.make 12 1.)))
  in
  let y = Nx.Prim.eval ~by:"t" (Fold { shape = [| 4 |]; pad = p; x = ones }) in
  equal ~msg:"each element, the number of taps that read it" (array float_exact)
    [| 2.; 3.; 3.; 2. |] (read y);
  (* A narrow float and a sub-byte dtype fold in one kernel call. *)
  let counts (type v s) (dt : (v, s) D.t) (one : v) (pp : v testable) want =
    let ones =
      Nx.place Count.on (host (A.of_array dt [| 4; 3 |] (Array.make 12 one)))
    in
    C.reset ();
    let y =
      Nx.Prim.eval ~by:"t" (Fold { shape = [| 4 |]; pad = p; x = ones })
    in
    let got = read y in
    equal ~msg:(D.name dt ^ " in one kernel call") int 1 (C.calls ());
    equal ~msg:(D.name dt) (array pp) want got
  in
  counts D.Bfloat16 1. float_exact [| 2.; 3.; 3.; 2. |];
  counts D.Float4_e2m1fn 1. float_exact [| 2.; 3.; 3.; 2. |];
  counts D.Int4 1 int [| 2; 3; 3; 2 |]

let test_rules () =
  let x = Nx.place Count.on (host (f32 [| 1.; 2.; 3. |])) in
  let raises f = raises_match (Exn.invalid_arg ~substring:"t: ") f in
  C.reset ();
  raises (fun () ->
      Nx.Prim.eval ~by:"t"
        (Map
           {
             layout = L.contiguous [| 3 |];
             prog = identity D.Float32;
             outs = [ Nx.float32 ];
             loads = [| Padded { x; fill = 0.; pad = pad ~lo:[| 1 |] [||] } |];
           }));
  raises (fun () ->
      Nx.Prim.eval ~by:"t"
        (Fold { shape = [| 3 |]; pad = pad ~lo:[| 1 |] [||]; x }));
  raises (fun () ->
      Nx.Prim.eval ~by:"t"
        (Fold
           {
             shape = [| 3 |];
             pad = pad [||];
             x =
               Nx.place Count.on
                 (host (A.of_array D.Bool [| 3 |] [| true; false; true |]));
           }));
  equal ~msg:"kernel calls" int 0 (C.calls ())

let test_split () =
  let a = f32 [| 1.; 2.; 3.; 4. |] in
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  let one = sum_of (Nx.place Count.on (host a)) ~fill:0. p in
  let two = sum_of (Nx.place (Count2.split ~axis:0) (host a)) ~fill:0. p in
  equal ~msg:"as on one device" (array float_exact) (read one) (read two)

(* A load and a fold keep the cut of an axis they neither pad nor window, as
   their rule reads only the padded and windowed axes whole. *)
let test_split_kept () =
  let a =
    A.of_array D.Float32 [| 2; 4 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let p =
    pad ~lo:[| 0; 1 |] ~hi:[| 0; 1 |] ~interior:[| 0; 0 |]
      [| { (window 3 1) with axis = 1 } |]
  in
  let split = Count2.split ~axis:0 in
  let one = sum_of (Nx.place Count.on (host a)) ~fill:0. p in
  let two = sum_of (Nx.place split (host a)) ~fill:0. p in
  equal ~msg:"the load keeps the cut" bool true
    (Nx.Placement.equal split (Option.get (Nx.placement two)));
  equal ~msg:"as on one device" (array float_exact) (read one) (read two);
  let fold x = Nx.Prim.eval ~by:"t" (Fold { shape = [| 2; 4 |]; pad = p; x }) in
  let x =
    A.of_array D.Float32 (loaded (host a) p) (Array.init 24 Float.of_int)
  in
  let one = fold (Nx.place Count.on (host x)) in
  let two = fold (Nx.place split (host x)) in
  equal ~msg:"the fold keeps the cut" bool true
    (Nx.Placement.equal split (Option.get (Nx.placement two)));
  equal ~msg:"as on one device" (array float_exact) (read one) (read two)

(* Declines *)

module Dec2 = (val Nx.devices ~kernels:(module Unpadded) [ m 2; m 3 ])

(* The elements [pad] loads from [xs], of shape [s], in C order of the loaded
   shape [shape]: an index's windows read the padded index [w·step +
   j·dilation], which holds [xs]'s element [(p - lo) / (interior + 1)] where
   that divides and lies in [s], and [fill] elsewhere. *)
let reference s xs ~fill (p : Nx_kernel.pad) shape =
  let r = Array.length s in
  let n = Array.fold_left ( * ) 1 shape in
  Array.init n (fun flat ->
      let ix = Array.make (Array.length shape) 0 and k = ref flat in
      for a = Array.length shape - 1 downto 0 do
        ix.(a) <- !k mod shape.(a);
        k := !k / shape.(a)
      done;
      let padded = Array.sub ix 0 r in
      Array.iteri
        (fun j (w : A.Move.window) ->
          padded.(w.axis) <- (ix.(w.axis) * w.step) + (ix.(r + j) * w.dilation))
        p.windows;
      let at = ref 0 and inside = ref true in
      for a = 0 to r - 1 do
        let q = padded.(a) - p.lo.(a) and gap = p.interior.(a) + 1 in
        if q < 0 || q mod gap <> 0 || q / gap >= s.(a) then inside := false
        else at := (!at * s.(a)) + (q / gap)
      done;
      if !inside then xs.(!at) else fill)

type padded = {
  s : int array;
  p : Nx_kernel.pad;
  xs : float array;
  fill : float;
}

let pp_padded ppf c =
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "[%s] lo [%s] hi [%s] interior [%s] windows [%s] fill %g"
    (ints c.s) (ints c.p.lo) (ints c.p.hi) (ints c.p.interior)
    (String.concat "; "
       (Array.to_list
          (Array.map
             (fun (w : A.Move.window) ->
               Printf.sprintf "%d:%d/%d*%d" w.axis w.size w.step w.dilation)
             c.p.windows)))
    c.fill

let padded_case =
  let open Gen in
  with_pp pp_padded
  @@ let* s = array ~size:(int_range 1 2) (int_range 0 4) in
     let r = Array.length s in
     let ints lo hi = array ~size:(constant r) (int_range lo hi) in
     let* lo = ints (-2) 2 in
     let* hi = ints (-2) 2 in
     let* interior = ints 0 2 in
     let* chosen =
       array ~size:(constant r)
         (option
            (let+ size = int_range 1 3
             and+ step = int_range 1 2
             and+ dilation = int_range 1 2 in
             (size, step, dilation)))
     in
     let windows =
       Array.of_list
         (List.filter_map Fun.id
            (List.mapi
               (fun axis ->
                 Option.map (fun (size, step, dilation) ->
                     { A.Move.axis; size; step; dilation }))
               (Array.to_list chosen)))
     in
     let* fill = of_list [ 0.; -1.5; 10. ] in
     let+ xs =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (map Float.of_int (int_range (-8) 8))
     in
     { s; p = { Nx_kernel.lo; hi; interior; windows }; xs; fill }

let law_padded c =
  match Nx_kernel.Shape.padded c.p c.s with
  | Error _ -> reject ()
  | Ok shape ->
      cover "a negative lo or hi"
        (Array.exists (fun e -> e < 0) (Array.append c.p.lo c.p.hi));
      cover "interior padding" (Array.exists (fun e -> e > 0) c.p.interior);
      cover "windows" (c.p.windows <> [||]);
      cover "an element loaded" (Array.fold_left ( * ) 1 shape > 0);
      let a = A.of_array D.Float32 c.s c.xs in
      let want = reference c.s c.xs ~fill:c.fill c.p shape in
      let on p = load (Nx.place p (host a)) ~fill:c.fill c.p in
      equal ~msg:"declined map" (array float_exact) want (read (on Dec.on));
      if c.s.(0) mod 2 = 0 then
        equal ~msg:"declined map, split" (array float_exact) want
          (read (on (Dec2.split ~axis:0)));
      let x = Nx.place Count.on (host a) in
      C.reset ();
      let y = load x ~fill:c.fill c.p in
      if Array.fold_left ( * ) 1 shape > 0 then
        equal ~msg:"nx.cpu's map in one kernel call" int 1 (C.calls ());
      equal ~msg:"nx.cpu's map" (array float_exact) want (read y);
      if Array.length shape > 0 then
        equal ~msg:"declined sum, as nx.cpu's" (array float_exact)
          (read (sum_of (Nx.place Count.on (host a)) ~fill:c.fill c.p))
          (read (sum_of (Nx.place Dec.on (host a)) ~fill:c.fill c.p))

(* A fold on a library that declines it, against nx.cpu's: values whose sums
   round, so that the order of the taps shows, and windows of up to 25 taps, so
   that a fold takes more than one map's worth. *)
let fold_case =
  let open Gen in
  with_pp (fun ppf (c, _) -> pp_padded ppf c)
  @@ let* c = padded_case in
     let window =
       let+ size = int_range 1 5
       and+ step = int_range 1 2
       and+ dilation = int_range 1 2 in
       (size, step, dilation)
     in
     let* chosen =
       array
         ~size:(constant (Array.length c.s))
         (frequency [ (3, map Option.some window); (1, constant None) ])
     in
     let windows =
       Array.of_list
         (List.filter_map Fun.id
            (List.mapi
               (fun axis ->
                 Option.map (fun (size, step, dilation) ->
                     { A.Move.axis; size; step; dilation }))
               (Array.to_list chosen)))
     in
     let+ xs =
       array ~size:(constant 64) (of_list [ 1e8; 1.; -1e8; 0.1; -0.; 3. ])
     in
     ({ c with p = { c.p with windows } }, xs)

(* The operand shape a fold [c] reads, if [c]'s padding fits its shape. *)
let fold_operand c =
  Result.to_option (Nx_kernel.Shape.padded c.p c.s)

(* [c]'s fold of an operand of [xshape] drawn from [pool], on a declining
   library and on nx.cpu, bit for bit. *)
let fold_agrees c xshape pool =
  let n = Array.fold_left ( * ) 1 xshape in
  let x =
    A.of_array D.Float32 xshape
      (Array.init n (fun i -> pool.(i mod Array.length pool)))
  in
  let fold p =
    Nx.Prim.eval ~by:"t"
      (Fold { shape = c.s; pad = c.p; x = Nx.place p (host x) })
  in
  equal ~msg:"as nx.cpu's, bit for bit" (array int32)
    (Array.map Int32.bits_of_float (read (fold Count.on)))
    (Array.map Int32.bits_of_float (read (fold Dec.on)))

let law_fold (c, pool) =
  match fold_operand c with
  | None -> reject ()
  | Some xshape ->
      cover "an axis without a window"
        (Array.length c.p.windows < Array.length c.s);
      cover "interior padding" (Array.exists (fun e -> e > 0) c.p.interior);
      cover "a negative lo or hi"
        (Array.exists (fun e -> e < 0) (Array.append c.p.lo c.p.hi));
      fold_agrees c xshape pool

let test_many_taps () =
  let w axis = { A.Move.axis; size = 5; step = 1; dilation = 1 } in
  let c =
    {
      s = [| 3; 4 |];
      p =
        {
          Nx_kernel.lo = [| 2; 2 |];
          hi = [| 2; 2 |];
          interior = [| 1; 0 |];
          windows = [| w 0; w 1 |];
        };
      xs = [||];
      fill = 0.;
    }
  in
  fold_agrees c (Option.get (fold_operand c)) [| 1e8; 1.; -1e8; 0.1; 3. |]

let () =
  exit
    (run "nx padded"
       [
         group "loads and folds"
           [
             test "a padded, windowed load" test_sum_pool;
             test "a fold counts the taps" test_fold;
             test "the rules raise before any kernel" test_rules;
             test "a split value loads as on one device" test_split;
             test "a load and a fold keep the cut of an axis they do not pad"
               test_split_kept;
             prop "a declined fold adds as nx.cpu's" fold_case law_fold;
             test "a declined fold of 25 taps, in two maps" test_many_taps;
             prop "a padded load is the padded operand's windows" padded_case
               law_padded;
           ];
       ])
