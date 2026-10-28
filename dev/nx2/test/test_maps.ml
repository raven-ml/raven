(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What only the engine's operations reach, through its private modules, copied
   here: maps of several nodes, programs of coordinates on split placements,
   operations of no operand, and a device's view of a value's window. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module C = Nx_support.Counting

type b

let m = Nx_support.memory
let s1 : b Devices.t = Devices.mint ~by:"t" ~kernels:(module C) [ m 0 ]
let s2 : b Devices.t = Devices.mint ~by:"t" ~kernels:(module C) [ m 0; m 1 ]
let at1 = Devices.one s1 0
let split = Devices.split ~by:"t" ~axis:0 s2

let bits =
  Testable.make ~pp:Format.pp_print_float ~equal:(fun a b ->
      Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b))

let f32 data : (float, D.float32_elt, b) Value.t =
  Value.Array
    {
      at = at1;
      a = A.to_device (m 0) (A.of_array D.Float32 [| Array.length data |] data);
      dead = Value.Live;
    }

let elements (type v s) (x : (v, s, b) Value.t) : v array =
  match
    Repr.array
      (Place.value ~by:"t" (Devices.one Devices.host 0) x)
  with
  | Some a -> A.to_array a
  | None -> fail "no array"

let first (x, ()) = x

(* The map [prog] of one float32 output over [loads], of shape [n]. *)
let map1 n prog loads =
  first
    (Eval.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [| n |];
            prog;
            outs = Value.[ D.Float32 ];
            loads;
            at = None;
          }))

(* A map of [w] arrays, an output and [w - 1] loads, gives the bits of the
   same sum computed in pieces of two loads: Σ x_k · c_k in order of k, each
   product rounded, then each sum. *)
let test_wide_maps () =
  let n = 5 and f = D.Any D.Float32 in
  let c k = P.Const (f, P.bits D.Float32 (1. /. Float.of_int (k + 3))) in
  let x k = f32 (Array.init n (fun i -> Float.of_int ((7 * k) + i) /. 10.)) in
  List.iter
    (fun w ->
      let nl = w - 1 in
      let xs = Array.init nl x in
      (* Loads 0 to nl - 1, then per load its constant and product, then
         the sums. *)
      let nodes =
        List.init nl (fun k -> P.In k)
        @ List.concat
            (List.init nl (fun k ->
                 [ c k; P.Op2 (Binary Mul, k, nl + (2 * k)) ]))
        @ List.init (nl - 1) (fun i ->
              let prod k = nl + (2 * k) + 1 in
              let acc = if i = 0 then prod 0 else (3 * nl) + i - 1 in
              P.Op2 (Binary Add, acc, prod (i + 1)))
      in
      let prog =
        P.v ~ins:(Iarray.init nl (fun _ -> f)) (Iarray.of_list nodes)
          ~outs:[| List.length nodes - 1 |]
      in
      let whole =
        map1 n prog (Iarray.map (fun x -> Value.Plain x) (Iarray.of_array xs))
      in
      (* The pieces: x_0 · c_0, then acc + x_k · c_k for each next load. *)
      let first_piece =
        map1 n
          (P.v ~ins:[| f |]
             [| In 0; c 0; Op2 (Binary Mul, 0, 1) |]
             ~outs:[| 2 |])
          [| Plain xs.(0) |]
      in
      let step acc k =
        map1 n
          (P.v ~ins:[| f; f |]
             [|
               In 0; In 1; c k; Op2 (Binary Mul, 1, 2); Op2 (Binary Add, 0, 3);
             |]
             ~outs:[| 4 |])
          [| Plain acc; Plain xs.(k) |]
      in
      let pieces = ref first_piece in
      for k = 1 to nl - 1 do
        pieces := step !pieces k
      done;
      equal ~msg:(Printf.sprintf "%d arrays" w) (array bits)
        (elements !pieces) (elements whole))
    [ 17; 18; 19; 32; 100 ]

let maps =
  group "maps"
    [
      test "a map of several nodes computes each node" (fun () ->
          (* x0 * x1 + x0 *)
          let prog =
            P.v
              ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
              [| In 0; In 1; Op2 (Binary Mul, 0, 1); Op2 (Binary Add, 2, 0) |]
              ~outs:[| 3 |]
          in
          let x = f32 [| 1.; 2.; 3. |] and y = f32 [| 4.; 5.; 6. |] in
          let z =
            first
              (Eval.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 3 |];
                      prog;
                      outs = Value.[ D.Float32 ];
                      loads = [| Plain x; Plain y |];
                      at = None;
                    }))
          in
          equal (array bits) [| 5.; 12.; 21. |] (elements z));
      test "a map of no loads computes each device's window where it lies"
        (fun () ->
          let prog =
            P.v ~ins:[||]
              [| Coord 0; Op1 (Cast, D.Any D.Float32, 0) |]
              ~outs:[| 1 |]
          in
          C.reset ();
          let c =
            first
              (Eval.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 4 |];
                      prog;
                      outs = Value.[ D.Float32 ];
                      loads = [||];
                      at = Some split;
                    }))
          in
          equal ~msg:"positions" int 4 (C.positions ());
          equal bool true (Devices.equal (Prim.placement c) split);
          equal
            (array (array bits))
            [| [| 0.; 1. |]; [| 2.; 3. |] |]
            (Iarray.to_array (Iarray.map A.to_array (Repr.shards c))));
      test "a map of any number of arrays computes as its pieces do"
        test_wide_maps;
    ]

(* A fill of [1.] on [at1]: a node of its own. *)
let fresh_one () =
  first
    (Eval.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [||];
            prog =
              P.of_node ~ins:[||]
                (Const (D.Any D.Float32, P.bits D.Float32 1.));
            outs = Value.[ D.Float32 ];
            loads = [||];
            at = Some at1;
          }))

(* The coordinate along [axis], as a float32: a program of a coordinate
   alone. *)
let coords axis =
  P.v ~ins:[||] [| Coord axis; Op1 (Cast, D.Any D.Float32, 0) |] ~outs:[| 1 |]

let eager =
  group "eager"
    [
      test "an operation computes once, when it is called" (fun () ->
          C.reset ();
          let one = fresh_one () in
          let c = Eval.apply2 ~by:"t" (Binary Add) D.Float32 one one in
          let d = Eval.apply2 ~by:"t" (Binary Mul) D.Float32 c c in
          let e = Eval.apply2 ~by:"t" (Binary Add) D.Float32 c d in
          (* one, c, d, e: four calls. *)
          equal ~msg:"making e" int 4 (C.calls ());
          C.reset ();
          equal (array bits) [| 6. |] (elements e);
          equal ~msg:"reading e" int 0 (C.calls ()));
    ]

(* Operations of no operand other than a map: an assembly of no piece and
   loops of no load lie on the host, and an assembly of pieces where they
   lie, each against the elements the definition gives. *)

let d0 = Devices.one s2 0
let on_host = Devices.on Devices.host

let assembly pieces : (float, D.float32_elt, b) Value.t =
  Eval.run ~by:"t"
    (Value.Assemble { dtype = D.Float32; shape = [| 4 |]; fill = 7.; pieces })

let ones p n : (float, D.float32_elt, b) Value.t =
  first
    (Eval.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [| n |];
            prog =
              P.of_node ~ins:[||]
                (Const (D.Any D.Float32, P.bits D.Float32 1.));
            outs = Value.[ D.Float32 ];
            loads = [||];
            at = Some p;
          }))

(* The sum and the running sum, along the last axis, of [[0; 1; 2]] repeated
   over four rows. *)
let row_sums () : (float, D.float32_elt, b) Value.t =
  let s, () =
    Eval.run ~by:"t"
      (Value.Reduce
         {
           layout = Nx_array.Layout.contiguous [| 4; 3 |];
           axes = [| 1 |];
           prog = coords 1;
           reductions = Value.[ Monoid (Sum, 0, D.Float32) ];
           loads = [||];
         })
  in
  s

let row_scans () : (float, D.float32_elt, b) Value.t =
  Eval.run ~by:"t"
    (Value.Scan
       {
         layout = Nx_array.Layout.contiguous [| 4; 3 |];
         axis = 1;
         prog = coords 1;
         reduction = Monoid (Sum, 0, D.Float32);
         loads = [||];
         carry = None;
       })

let unloaded =
  group "operations of no operand"
    [
      test "an empty assembly lies on the host, filled" (fun () ->
          let y = assembly [] in
          equal bool true (Devices.equal (Prim.placement y) on_host);
          equal (array bits) [| 7.; 7.; 7.; 7. |] (elements y));
      test "pieces on one device assemble there" (fun () ->
          let r : _ iarray =
            [| { Nx_array.Move.start = 1; count = 2; step = 1 } |]
          in
          let y = assembly [ (r, ones d0 2) ] in
          equal bool true (Devices.equal (Prim.placement y) d0);
          equal (array bits) [| 7.; 1.; 1.; 7. |] (elements y));
      test "a sum of coordinates lies on the host" (fun () ->
          let y = row_sums () in
          equal bool true (Devices.equal (Prim.placement y) on_host);
          equal (array bits) [| 3.; 3.; 3.; 3. |] (elements y));
      test "a running sum of coordinates lies on the host" (fun () ->
          let y = row_scans () in
          equal bool true (Devices.equal (Prim.placement y) on_host);
          equal (array bits)
            [| 0.; 1.; 3.; 0.; 1.; 3.; 0.; 1.; 3.; 0.; 1.; 3. |]
            (elements y));
    ]

(* A Logsumexp's terms: shifts from the maximum about the least integers whose
   exp is a normal float32 (-87) and float64 (-708), infinities, signed zeros
   and NaNs with payloads. *)
type lse =
  | Lse :
      (float, 's) D.t * int array * int array * float array * int list
      -> lse

let pp_lse ppf (Lse (dt, shape, axes, xs, _)) =
  Format.fprintf ppf "%s %a along %a: %a" (D.name dt)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
    (Array.to_list shape)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
    (Array.to_list axes)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_float)
    (Array.to_list xs)

let lse_terms =
  [
    0.; -0.; 1.; 50.; -37.; -86.5; -87.; -87.25; -88.; -100.; -660.; -707.5;
    -708.; -708.25; -750.; Float.infinity; Float.neg_infinity; Float.nan;
    Int64.float_of_bits 0x7ff8_0000_2000_0000L;
    Int64.float_of_bits 0xfff8_0000_0000_0001L;
  ]

let lse =
  let open Gen in
  with_pp pp_lse
  @@ let* rank = int_range 1 2 in
     let* shape = array ~size:(constant rank) (int_range 0 9) in
     let* axes =
       such_that
         (fun a -> a <> [||])
         (map
            (fun keep ->
              Array.of_list
                (List.filteri (fun a _ -> keep.(a)) (List.init rank Fun.id)))
            (array ~size:(constant rank) bool))
     in
     let n = Array.fold_left ( * ) 1 shape in
     let* xs = array ~size:(constant n) (of_list lse_terms) in
     let* signalling =
       array ~size:(constant n)
         (frequency [ (11, constant false); (1, constant true) ])
     in
     let sn = List.filter (fun i -> signalling.(i)) (List.init n Fun.id) in
     let+ dt = int_range 0 3 in
     match dt with
     | 0 -> Lse (D.Float32, shape, axes, xs, sn)
     | 1 -> Lse (D.Float64, shape, axes, xs, sn)
     | 2 -> Lse (D.Float16, shape, axes, xs, sn)
     | _ -> Lse (D.Bfloat16, shape, axes, xs, sn)

(* [xs] of [shape] at [dt], a signalling NaN with a payload at each flat
   position of [sn]. *)
let signalling (type s) (dt : (float, s) D.t) shape xs sn :
    (float, s, b) Value.t =
  let a = A.of_array dt (Iarray.of_array shape) xs in
  let index i =
    let idx = Array.make (Array.length shape) 0 and r = ref i in
    for k = Array.length shape - 1 downto 0 do
      idx.(k) <- !r mod shape.(k);
      r := !r / shape.(k)
    done;
    Iarray.of_array idx
  in
  let put (type u r) (u : (u, r) D.t) (code : u) =
    let raw = Option.get (A.bitcast u a) in
    List.iter (fun i -> A.set raw (index i) code) sn
  in
  (match dt with
  | D.Float16 -> put D.Uint16 0x7d01
  | D.Bfloat16 -> put D.Uint16 0x7f81
  | D.Float32 -> put D.Uint32 0x7f800101l
  | D.Float64 -> put D.Uint64 0x7ff0_0000_0000_0101L
  | _ -> ());
  Value.Array { at = at1; a = A.to_device (m 0) a; dead = Value.Live }

(* [L + D + 5 + 6 ln n] of nx.cpu's Logsumexp bound for [n] terms. *)
let lse_rounds n =
  let n' = Float.of_int n in
  let l = ((min n 1024 + 15) / 16) + 3 in
  let blocks = (n + 1023) / 1024 in
  let rec depth b d = if b <= 1 then d else depth ((b + 1) / 2) (d + 1) in
  Float.of_int (l + depth blocks 0 + 5) +. (6. *. Float.log n')

(* [m + log Σ exp (t - m)] of [ts] in float64, [m] their maximum: a NaN
   where a term is, [m] where it is infinite. *)
let lse_ref ts =
  if List.exists Float.is_nan ts then Float.nan
  else
    let m = List.fold_left Float.max Float.neg_infinity ts in
    if not (Float.is_finite m) then m
    else
      let s = List.fold_left (fun s t -> s +. Float.exp (t -. m)) 0. ts in
      m +. Float.log s

(* nx.cpu's Logsumexp is [m + log Σ exp (x - m)] within the bound {!Nx_cpu}
   states, in the float32 carrier for a narrow float, which adds its own
   rounding: a NaN where a term is, [-∞] for no term or terms all [-∞], [+∞]
   for a [+∞] term. *)
let law_logsumexp (Lse (dt, shape, axes, xs, sn)) =
  let cut = match dt with D.Float64 -> -708. | _ -> -87. in
  let x = signalling dt shape xs sn in
  cover "no term" (Array.exists (fun a -> shape.(a) = 0) axes);
  cover "a NaN term" (Array.exists Float.is_nan xs);
  cover "a signalling NaN term" (sn <> []);
  cover "a signalling NaN of a narrow float" (sn <> [] && D.bits dt = 16);
  (* Each output's terms, as [x] holds them, in C order of the outputs. *)
  let kept =
    List.filter
      (fun a -> not (Array.mem a axes))
      (List.init (Array.length shape) Fun.id)
  in
  let outs = List.fold_left (fun n a -> n * shape.(a)) 1 kept in
  let rows = Array.make outs [] in
  Array.iteri
    (fun i t ->
      let idx = Array.make (Array.length shape) 0 and r = ref i in
      for a = Array.length shape - 1 downto 0 do
        idx.(a) <- !r mod shape.(a);
        r := !r / shape.(a)
      done;
      let o = List.fold_left (fun o a -> (o * shape.(a)) + idx.(a)) 0 kept in
      rows.(o) <- t :: rows.(o))
    (elements x);
  let below =
    Array.exists
      (fun row ->
        let m = List.fold_left Float.max Float.neg_infinity row in
        Float.is_finite m
        && List.exists (fun t -> t -. m < cut && t -. m > -1000.) row)
      rows
  in
  cover "a term past the cutoff below a finite maximum" below;
  let op =
    Value.Reduce
      {
        layout = A.Layout.contiguous (Iarray.of_array shape);
        axes = Iarray.of_array axes;
        prog = P.of_node ~ins:[| D.Any dt |] (In 0);
        reductions = Value.[ Logsumexp (0, dt) ];
        loads = [| Plain x |];
      }
  in
  let got = elements (first (Eval.run ~by:"t" op)) in
  let u = match dt with D.Float64 -> Float.epsilon /. 2. | _ -> 0x1p-24 in
  let narrow =
    match dt with D.Float16 -> 0x1p-11 | D.Bfloat16 -> 0x1p-8 | _ -> 0.
  in
  equal ~msg:"one result per output" int outs (Array.length got);
  Array.iteri
    (fun o row ->
      let want = lse_ref row and y = got.(o) in
      let msg = Printf.sprintf "output %d" o in
      if not (Float.is_finite want) then
        equal ~msg (option float_exact)
          (if Float.is_nan want then None else Some want)
          (if Float.is_nan y then None else Some y)
      else
        let n = List.length row in
        (* The float64 reference errs by about [n + 4] roundings. *)
        let slack =
          match dt with
          | D.Float64 -> Float.of_int (n + 4) *. u *. (1. +. Float.abs want)
          | _ -> 0.
        in
        let bound =
          (lse_rounds n *. u) +. ((u +. narrow) *. Float.abs want) +. slack
        in
        at_most ~msg float_exact ~than:bound (Float.abs (y -. want)))
    rows

let arithmetic =
  group "arithmetic"
    [
      prop ~count:500
        "a Logsumexp is within its bound, NaNs and infinities as stated" lse
        law_logsumexp;
    ]

(* A device's view of an empty window is an array of no bytes of its own on
   that device, whether or not the value has an array there; a nonempty window
   on a device without one raises. *)
let test_empty_view () =
  let x : (float, D.float32_elt, b) Value.t =
    Value.Array
      {
        at = Devices.one s2 0;
        a = A.to_device (m 0) (A.of_array D.Float32 [| 2 |] [| 1.; 2. |]);
        dead = Value.Live;
      }
  in
  let rig = Testable.make ~pp:Rig.pp ~equal:Rig.equal in
  let empty = [| { Nx_array.Move.start = 1; count = 0; step = 1 } |] in
  List.iter
    (fun k ->
      let v = Place.view ~by:"t" x k empty in
      let b = A.buffer v in
      let msg what = Printf.sprintf "device %d: %s" k what in
      equal ~msg:(msg "shape") (array int) [| 0 |]
        (Iarray.to_array (Nx_array.Layout.shape (A.layout v)));
      equal ~msg:(msg "device") rig (Devices.rig s2 k) (A.device v);
      equal ~msg:(msg "bytes") int 0 (Rig.Buffer.length b);
      equal ~msg:(msg "all of its memory") bool true (Rig.Buffer.spans b))
    [ 0; 1 ];
  raises_match
    (Exn.invalid_arg ~substring:"Place.view: ")
    (fun () ->
      Place.view ~by:"t" x 1
        [| { Nx_array.Move.start = 0; count = 1; step = 1 } |])

let views =
  group "views"
    [ test "an empty window is an array of its own" test_empty_view ]

let () =
  exit
    (run "nx maps" [ maps; eager; unloaded; arithmetic; views ])
