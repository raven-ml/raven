(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Padded loads and Fold through Nx.Prim: the values Nx_kernel.load and
   Nx_kernel.S.fold state for small operands, their rules before any kernel,
   and a split placement. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))
let host a = Nx.Repr.of_array Nx.Host.v a
let identity dt = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |]

let pad ?(lo : int iarray = [| 0 |]) ?(hi : int iarray = [| 0 |])
    ?(interior : int iarray = [| 0 |]) windows =
  { Nx_kernel.lo; hi; interior; windows }

let window size step = { A.Move.axis = 0; size; step; dilation = 1 }

(* The shape [x] has once loaded through [pad]. *)
let loaded x pad =
  match Nx_kernel.Shape.padded pad (Nx.shape x) with
  | Ok s -> Iarray.of_array s
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
           at = None;
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

(* A fold sums each element's taps with no +0 before them, so a single -0
   tap keeps its sign, and is +0 where no tap lands: here interior padding
   puts the operand's elements at 0, 2 and 4 of five, and a crop drops the
   result's first element from every tap. *)
let test_signed_zeros () =
  let bits a = Array.map Int64.bits_of_float a in
  let fold (type s) (dt : (float, s) D.t) shape p xs =
    let x = Nx.place Count.on (host (A.of_array dt [| Array.length xs |] xs)) in
    bits (read (Nx.Prim.eval ~by:"t" (Fold { shape; pad = p; x })))
  in
  let spread = pad ~interior:[| 1 |] [||] and cropped = pad ~lo:[| -1 |] [||] in
  List.iter
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float ->
          let name = D.name dt in
          equal ~msg:(name ^ ": one -0 tap") (array int64)
            (bits [| -0.; -0.; 3. |])
            (fold dt [| 3 |] spread [| -0.; 6.; -0.; 6.; 3. |]);
          equal ~msg:(name ^ ": no tap") (array int64)
            (bits [| 0.; -0.; 1. |])
            (fold dt [| 3 |] cropped [| -0.; 1. |])
      | _ -> ())
    [
      D.Any D.Float32;
      D.Any D.Float64;
      D.Any D.Bfloat16;
      D.Any D.Float16;
      D.Any D.Float8_e4m3fn;
      D.Any D.Float8_e5m2;
      D.Any D.Float4_e2m1fn;
    ];
  (* A complex number's parts each keep a -0. *)
  let c re im = { Complex.re; im } in
  let parts (z : Complex.t array) =
    let both (z : Complex.t) = bits [| z.re; z.im |] in
    Array.concat (Array.to_list (Array.map both z))
  in
  let fold_c (type s) (dt : (Complex.t, s) D.t) shape p zs =
    let x = Nx.place Count.on (host (A.of_array dt [| Array.length zs |] zs)) in
    parts (read (Nx.Prim.eval ~by:"t" (Fold { shape; pad = p; x })))
  in
  List.iter
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Complex ->
          let name = D.name dt in
          equal ~msg:(name ^ ": one -0 tap") (array int64)
            (parts [| c (-0.) (-0.); c (-0.) 1.; c 3. (-0.) |])
            (fold_c dt [| 3 |] spread
               [| c (-0.) (-0.); c 6. 6.; c (-0.) 1.; c 6. 6.; c 3. (-0.) |]);
          equal ~msg:(name ^ ": no tap") (array int64)
            (parts [| c 0. 0.; c (-0.) (-0.); c 1. 1. |])
            (fold_c dt [| 3 |] cropped [| c (-0.) (-0.); c 1. 1. |])
      | _ -> ())
    [ D.Any D.Complex64; D.Any D.Complex128 ]

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
             at = None;
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

(* A load reads its padded axis whole, so a value cut along it raises with
   the placement to write; placed there, it loads as on one device. *)
let test_split () =
  let a = f32 [| 1.; 2.; 3.; 4. |] in
  let p = pad ~lo:[| 1 |] ~hi:[| 1 |] [| window 3 1 |] in
  let one = sum_of (Nx.place Count.on (host a)) ~fill:0. p in
  let split = Nx.place (Count2.split ~axis:0) (host a) in
  raises
    (Invalid_argument
       "t: operand 0 lies at split ~axis:0 [m0; m1] and is read at on [m0; \
        m1]; Nx.place (Placement.on s) x")
    (fun () -> sum_of split ~fill:0. p);
  let two = sum_of (Nx.place Count2.on split) ~fill:0. p in
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
    (Nx.Placement.equal split (Nx.placement two));
  equal ~msg:"as on one device" (array float_exact) (read one) (read two);
  let fold x = Nx.Prim.eval ~by:"t" (Fold { shape = [| 2; 4 |]; pad = p; x }) in
  let x =
    A.of_array D.Float32 (loaded (host a) p) (Array.init 24 Float.of_int)
  in
  let one = fold (Nx.place Count.on (host x)) in
  let two = fold (Nx.place split (host x)) in
  equal ~msg:"the fold keeps the cut" bool true
    (Nx.Placement.equal split (Nx.placement two));
  equal ~msg:"as on one device" (array float_exact) (read one) (read two)

(* Loads *)

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
      Iarray.iteri
        (fun j (w : A.Move.window) ->
          padded.(w.axis) <- (ix.(w.axis) * w.step) + (ix.(r + j) * w.dilation))
        p.windows;
      let at = ref 0 and inside = ref true in
      for a = 0 to r - 1 do
        let q = padded.(a) - Iarray.get p.lo a
        and gap = Iarray.get p.interior a + 1 in
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
    (ints c.s)
    (ints (Iarray.to_array c.p.lo))
    (ints (Iarray.to_array c.p.hi))
    (ints (Iarray.to_array c.p.interior))
    (String.concat "; "
       (Iarray.to_list
          (Iarray.map
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
     let lo = Iarray.of_array lo and hi = Iarray.of_array hi in
     let interior = Iarray.of_array interior in
     let windows = Iarray.of_array windows in
     { s; p = { Nx_kernel.lo; hi; interior; windows }; xs; fill }

let law_padded c =
  match Nx_kernel.Shape.padded c.p c.s with
  | Error _ -> reject ()
  | Ok shape ->
      cover "a negative lo or hi"
        (Iarray.exists (fun e -> e < 0) (Iarray.append c.p.lo c.p.hi));
      cover "interior padding" (Iarray.exists (fun e -> e > 0) c.p.interior);
      cover "windows" (Iarray.length c.p.windows > 0);
      cover "an element loaded" (Array.fold_left ( * ) 1 shape > 0);
      let a = A.of_array D.Float32 (Iarray.of_array c.s) c.xs in
      let want = reference c.s c.xs ~fill:c.fill c.p shape in
      let x = Nx.place Count.on (host a) in
      C.reset ();
      let y = load x ~fill:c.fill c.p in
      if Array.fold_left ( * ) 1 shape > 0 then
        equal ~msg:"nx.cpu's map in one kernel call" int 1 (C.calls ());
      equal ~msg:"nx.cpu's map" (array float_exact) want (read y);
      (* Small integers and fills sum exactly, in any order. *)
      if Array.length shape > 0 then begin
        let r = Array.length shape in
        let last = shape.(r - 1) in
        let outer = Array.fold_left ( * ) 1 (Array.sub shape 0 (r - 1)) in
        let sums =
          Array.init outer (fun i ->
              Array.fold_left ( +. ) 0. (Array.sub want (i * last) last))
        in
        equal ~msg:"nx.cpu's sum along the last axis" (array float_exact) sums
          (read (sum_of (Nx.place Count.on (host a)) ~fill:c.fill c.p))
      end

let () =
  exit
    (run "nx padded"
       [
         group "loads and folds"
           [
             test "a padded, windowed load" test_sum_pool;
             test "a fold counts the taps" test_fold;
             test "a fold keeps one -0 tap's sign and is +0 with none"
               test_signed_zeros;
             test "the rules raise before any kernel" test_rules;
             test "a split value along its padded axis raises" test_split;
             test "a load and a fold keep the cut of an axis they do not pad"
               test_split_kept;
             prop "a padded load is the padded operand's windows" padded_case
               law_padded;
           ];
       ])
