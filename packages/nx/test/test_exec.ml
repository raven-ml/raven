(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Computing through a set's kernels, through Nx: each function against a
   reference over nx.array's elements, refusals before any kernel, declines,
   creations and allocation. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module C = Nx_support.Counting

let m = Nx_support.memory

module Count = (val Nx.devices ~kernels:(module C) [ m 0 ])
module Count2 = (val Nx.devices ~kernels:(module C) [ m 0; m 1 ])

module Count4 =
  (val Nx.devices ~kernels:(module C) [ m 0; m 1; m 2; m 3 ])

module Dec = (val Nx.devices ~kernels:(module Nx_support.Declining) [ m 1 ])

module Dec2 =
  (val Nx.devices ~kernels:(module Nx_support.Declining) [ m 0; m 1 ])

(* Counting nx.cpu, declining the one-node maps of a copy, [Neg], [Add],
   [Less] and [Where], every [Logsumexp] and [Arg] reduction, sort and
   contraction, as a library may, and counting those declines. *)
module Refusing = struct
  include C

  let name = "nx.refusing"
  let declines = Atomic.make 0

  let decline () =
    Atomic.incr declines;
    A.Declined

  let map p ~dsts loads =
    match Nx_kernel.Prog.single p with
    | Some
        ( Op1 ((Copy | Unary Neg), _, _)
        | Op2 ((Binary Add | Compare Less), _, _)
        | Op3 (Where, _, _, _) ) ->
        decline ()
    | Some _ | None -> C.map p ~dsts loads

  let reduce p ~shape ~axes rs ~dsts loads =
    let declined ((r : Nx_kernel.reduction), _) =
      match r with Logsumexp | Arg _ -> true | _ -> false
    in
    if Iarray.exists declined rs then decline ()
    else C.reduce p ~shape ~axes rs ~dsts loads

  let sort ~axis:_ ~descending:_ ~values:_ ~positions:_ _ = decline ()

  let contract ~batch:_ ~contracting:_ ~segments:_ ~acc:_ ~dst:_ ~init:_ _ _
      =
    decline ()
end

(* Counting nx.cpu, and counting its LU and SVD factorizations apart. *)
module Factoring = struct
  include C

  let name = "nx.factoring"
  let factored = Atomic.make 0

  let lu ~lu ~pivots ~perm a =
    Atomic.incr factored;
    C.lu ~lu ~pivots ~perm a

  let svd ~s ?vectors a =
    Atomic.incr factored;
    C.svd ~s ?vectors a
end

module Factor = (val Nx.devices ~kernels:(module Factoring) [ m 3 ])
module Factor2 = (val Nx.devices ~kernels:(module Factoring) [ m 2; m 3 ])
module Refuse = (val Nx.devices ~kernels:(module Refusing) [ m 2 ])
module Refuse2 = (val Nx.devices ~kernels:(module Refusing) [ m 2; m 3 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let elements x =
  A.to_array (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

(* Operands: a shape, and arrays of it laid out C-contiguous, transposed, offset
   or broadcast, each holding [data]. *)
type layout = Plain | Transposed | Offset | Broadcast

let pp_layout ppf l =
  Format.pp_print_string ppf
    (match l with
    | Plain -> "plain"
    | Transposed -> "transposed"
    | Offset -> "offset"
    | Broadcast -> "broadcast")

(* [data] of [shape] in C order, laid out as [l], on the host. *)
let lay (type v s) l (dt : (v, s) D.t) shape (data : v array) : (v, s) A.t =
  let r = Array.length shape in
  match l with
  | Plain -> A.of_array dt (Iarray.of_array shape) data
  | Transposed when r >= 2 && Array.length data > 0 ->
      let rev = Array.init r (fun i -> r - 1 - i) in
      let t = Array.map (fun a -> shape.(a)) rev in
      let base =
        A.of_array dt (Iarray.of_array t)
          (Array.of_list (List.init (Array.length data) (fun _ -> data.(0))))
      in
      (* Fill the transposed base so that its view reads [data] in C order. *)
      let view = Option.get (A.move (M.Permute (Iarray.of_array rev)) base) in
      let idx = Array.make r 0 in
      Array.iteri
        (fun k v ->
          let k = ref k in
          for a = r - 1 downto 0 do
            idx.(a) <- !k mod shape.(a);
            k := !k / shape.(a)
          done;
          A.set view (Iarray.of_array idx) v)
        data;
      view
  | Transposed -> A.of_array dt (Iarray.of_array shape) data
  | Offset ->
      let n = Array.length data in
      let padded =
        A.of_array dt
          [| n + 1 |]
          (Array.append [| (if n = 0 then D.zero dt else data.(0)) |] data)
      in
      let tail =
        Option.get
          (A.move (M.Slice [| { start = 1; count = n; step = 1 } |]) padded)
      in
      Option.get (A.move (M.Reshape (Iarray.of_array shape)) tail)
  | Broadcast ->
      if Array.length data = 0 then A.of_array dt (Iarray.of_array shape) data
      else
        let one = A.of_array dt [||] [| data.(0) |] in
        Option.get (A.move (M.Broadcast (Iarray.of_array shape)) one)

(* What [lay] reads back: a broadcast holds its first element everywhere. *)
let read l data =
  match l with
  | Broadcast when Array.length data > 0 -> Array.map (fun _ -> data.(0)) data
  | _ -> data

let case =
  let open Gen in
  with_pp
    (fun ppf (s, la, lb, _, _) ->
      Format.fprintf ppf "%a, %a and %a" pp_ints s pp_layout la pp_layout lb)
    (let* rank = int_range 0 3 in
     let* s = array ~size:(constant rank) (int_range 0 3) in
     let n = Array.fold_left ( * ) 1 s in
     let layout =
       of_list ~pp:pp_layout [ Plain; Transposed; Offset; Broadcast ]
     in
     let* la = layout in
     let* lb = layout in
     let* xs = array ~size:(constant n) any_float in
     let+ ys = array ~size:(constant n) any_float in
     (s, la, lb, xs, ys))

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let on_host l dt s data = Nx.Repr.of_array Nx.Host.v (lay l dt s data)

let bits =
  Testable.make ~pp:Format.pp_print_float ~equal:(fun a b ->
      Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
      || (Float.is_nan a && Float.is_nan b))

(* [x]'s bytes in C order, read on the host. *)
let bytes (type v s d) (x : (v, s, d) Nx.t) =
  let a = Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)) in
  let flat = A.move (M.Reshape [| A.Layout.numel (A.layout a) |]) (A.copy a) in
  A.to_array (Option.get (A.bitcast D.Uint8 (Option.get flat)))

(* The one-node map of [node] over [loads], of [shape], into [dt], run as an
   operation. *)
let run_map (type w r) node ins (dt : (w, r) D.t) shape loads =
  let prog = Nx_kernel.Prog.of_node ~ins node in
  let op : ((w, r, Nx.host) Nx.t * unit) Nx.Prim.t =
    let layout = A.Layout.contiguous (Iarray.of_array shape) in
    Map { layout; prog; outs = [ dt ]; loads; at = None }
  in
  fst (Nx.Prim.eval ~by:"test" op)

(* Kinds of each arity over float32 operands [x], [y] and a flag [c]: the
   bytes of each through its Nx function, which takes the fast path, and
   through its operation run. *)
let one_node_maps =
  let f = D.Any D.Float32 and b = D.Any D.Bool in
  let maps name arity slim general =
    (name, arity, fun xyc -> (bytes (slim xyc), bytes (general xyc)))
  in
  [
    maps "neg" 1
      (fun (x, _, _) -> Nx.neg x)
      (fun (x, _, _) ->
        run_map (Op1 (Unary Neg, f, 0)) [| f |] D.Float32 (Nx.shape x)
          [| Plain x |]);
    maps "cast to float16" 1
      (fun (x, _, _) -> Nx.cast D.Float16 x)
      (fun (x, _, _) ->
        run_map
          (Op1 (Cast, D.Any D.Float16, 0))
          [| f |] D.Float16 (Nx.shape x) [| Plain x |]);
    (* Half a byte each, compared as int32. *)
    maps "cast to int4" 1
      (fun (x, _, _) -> Nx.cast D.Int32 (Nx.cast D.Int4 x))
      (fun (x, _, _) ->
        Nx.cast D.Int32
          (run_map
             (Op1 (Cast, D.Any D.Int4, 0))
             [| f |] D.Int4 (Nx.shape x) [| Plain x |]));
    maps "add" 2
      (fun (x, y, _) -> Nx.add x y)
      (fun (x, y, _) ->
        run_map (Op2 (Binary Add, 0, 1)) [| f; f |] D.Float32 (Nx.shape x)
          [| Plain x; Plain y |]);
    maps "less" 2
      (fun (x, y, _) -> Nx.less x y)
      (fun (x, y, _) ->
        run_map (Op2 (Compare Less, 0, 1)) [| f; f |] D.Bool (Nx.shape x)
          [| Plain x; Plain y |]);
    maps "where" 3
      (fun (x, y, c) -> Nx.where c x y)
      (fun (x, y, c) ->
        run_map (Op3 (Where, 0, 1, 2)) [| b; f; f |] D.Float32 (Nx.shape x)
          [| Plain c; Plain x; Plain y |]);
    maps "fma" 3
      (fun (x, y, _) -> Nx.fma x y x)
      (fun (x, y, _) ->
        run_map (Op3 (Fma, 0, 1, 2)) [| f; f; f |] D.Float32 (Nx.shape x)
          [| Plain x; Plain y; Plain x |]);
  ]

(* Maps of several nodes, as an interpreter fuses them: [x + y] and
   [(x + y) * y] as two results, and [(x + y) * y] alone. *)
let chain =
  let f = D.Any D.Float32 in
  Nx_kernel.Prog.v ~ins:[| f; f |]
    [| In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 2, 1) |]
    ~outs:[| 2; 3 |]

let fused =
  let f = D.Any D.Float32 in
  Nx_kernel.Prog.v ~ins:[| f; f |]
    [| In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 2, 1) |]
    ~outs:[| 3 |]

(* The bytes of [chain]'s results and of [fused]'s, each over the operands
   [operands ()] gives. *)
let fused_maps_over operands =
  let loads () : Nx.host Nx.Prim.load iarray =
    let x, y = operands () in
    [| Plain x; Plain y |]
  in
  let x, _ = operands () in
  let layout = A.Layout.contiguous (Iarray.of_array (Nx.shape x)) in
  let s, (p, ()) =
    Nx.Prim.eval ~by:"test"
      (Map
         {
           layout;
           prog = chain;
           outs = [ D.Float32; D.Float32 ];
           loads = loads ();
           at = None;
         })
  in
  let q, () =
    Nx.Prim.eval ~by:"test"
      (Map
         {
           layout;
           prog = fused;
           outs = [ D.Float32 ];
           loads = loads ();
           at = None;
         })
  in
  (bytes s, bytes p, bytes q)

(* The bytes of [chain]'s results and of [fused]'s over [x] and [y]. *)
let fused_maps x y = fused_maps_over (fun () -> (x, y))

let one_node_case =
  Gen.pair
    (Gen.of_list
       ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
       one_node_maps)
    case

let laws =
  group "Prim.eval and arithmetic"
    [
      prop "a one-node map gives the bits of its operation run" one_node_case
        (fun ((_, arity, maps), (s, la, lb, xs, ys)) ->
          cover "one operand" (arity = 1);
          cover "two operands" (arity = 2);
          cover "three operands" (arity = 3);
          cover "no element" (Array.exists (( = ) 0) s);
          cover "a strided operand" (la = Transposed || lb = Transposed);
          let x = on_host la D.Float32 s (Array.map f32 xs)
          and y = on_host lb D.Float32 s (Array.map f32 ys) in
          let c = on_host Plain D.Bool s (Array.map (fun x -> x > 0.) xs) in
          let fast, general = maps (x, y, c) in
          equal (array int) general fast);
      prop "a map of several nodes gives the bits of its nodes' functions"
        case (fun (s, la, lb, xs, ys) ->
          cover "no element" (Array.exists (( = ) 0) s);
          cover "a strided operand" (la = Transposed || lb = Transposed);
          let x = on_host la D.Float32 s (Array.map f32 xs)
          and y = on_host lb D.Float32 s (Array.map f32 ys) in
          let sum = Nx.add x y in
          let product = bytes (Nx.mul sum y) in
          equal
            (triple (array int) (array int) (array int))
            (bytes sum, product, product)
            (fused_maps x y));
      prop "a map gives the same bits over donated operands" case
        (fun (s, la, lb, xs, ys) ->
          let x = on_host la D.Float32 s (Array.map f32 xs)
          and y = on_host lb D.Float32 s (Array.map f32 ys) in
          let donated () = (Nx.donate (Nx.copy x), Nx.donate (Nx.copy y)) in
          equal
            (triple (array int) (array int) (array int))
            (fused_maps x y) (fused_maps_over donated));
      prop "add is float32 addition, each element rounded once" case
        (fun (s, la, lb, xs, ys) ->
          cover "no element" (Array.exists (( = ) 0) s);
          cover "a strided operand" (la = Transposed || lb = Transposed);
          let xs = Array.map f32 xs and ys = Array.map f32 ys in
          let z =
            Nx.add (on_host la D.Float32 s xs) (on_host lb D.Float32 s ys)
          in
          equal (array bits)
            (Array.map2 (fun a b -> f32 (a +. b)) (read la xs) (read lb ys))
            (elements z));
      prop "less orders floats, NaN below nothing" case
        (fun (s, la, lb, xs, ys) ->
          let xs = Array.map f32 xs and ys = Array.map f32 ys in
          let z =
            Nx.less (on_host la D.Float32 s xs) (on_host lb D.Float32 s ys)
          in
          equal (array bool)
            (Array.map2 (fun a b -> a < b) (read la xs) (read lb ys))
            (elements z));
      prop "where selects by its flag" case (fun (s, la, lb, xs, ys) ->
          let xs = Array.map f32 xs and ys = Array.map f32 ys in
          let c = Array.map (fun x -> x > 0.) xs in
          let z =
            Nx.where (on_host Plain D.Bool s c)
              (on_host la D.Float32 s xs)
              (on_host lb D.Float32 s ys)
          in
          equal (array bits)
            (Array.mapi
               (fun i c -> if c then (read la xs).(i) else (read lb ys).(i))
               c)
            (elements z));
      prop "int32 mul wraps" case (fun (s, la, lb, xs, ys) ->
          let i = Array.map (fun x -> Int32.of_float (Float.rem x 1e9)) in
          let xs = i (Array.map (fun x -> if Float.is_nan x then 0. else x) xs)
          and ys =
            i (Array.map (fun x -> if Float.is_nan x then 0. else x) ys)
          in
          let z = Nx.mul (on_host la D.Int32 s xs) (on_host lb D.Int32 s ys) in
          equal (array int32)
            (Array.map2 Int32.mul (read la xs) (read lb ys))
            (elements z));
      prop "a reshape and a copy keep the elements in C order" case
        (fun (s, la, _, xs, _) ->
          let n = Array.length xs in
          let x = on_host la D.Float64 s xs in
          equal (array bits) (read la xs) (elements (Nx.reshape [| n |] x));
          equal (array bits) (read la xs) (elements (Nx.copy x)));
      test "a cast to its own dtype is the value itself" (fun () ->
          let x = on_host Plain D.Float32 [| 2 |] [| 1.; 2. |] in
          satisfies ~claim:"x itself" pass
            (fun y -> y == x)
            (Nx.cast D.Float32 x));
      test "a cast stores each element in the dtype" (fun () ->
          let x = on_host Plain D.Float32 [| 3 |] [| 1.5; -2.5; 300. |] in
          equal (array int)
            (Array.map (D.of_float D.Uint8) [| 1.5; -2.5; 300. |])
            (elements (Nx.cast D.Uint8 x)));
    ]

let broadcasting =
  group "broadcasting"
    [
      (* Both operands narrower than the result: [[1]; [2]] beside [10; 20;
         30]. *)
      test "shapes broadcast from their last axes" (fun () ->
          let a = on_host Plain D.Float32 [| 2; 1 |] [| 1.; 2. |]
          and b = on_host Plain D.Float32 [| 3 |] [| 10.; 20.; 30. |] in
          equal (array bits)
            [| 11.; 21.; 31.; 12.; 22.; 32. |]
            (elements (Nx.add a b)));
      test "shapes that do not broadcast raise, naming both" (fun () ->
          raises
            (Invalid_argument "Nx.add: shapes [2; 3] and [4] do not broadcast")
            (fun () ->
              Nx.add
                (on_host Plain D.Float32 [| 2; 3 |] (Array.make 6 0.))
                (on_host Plain D.Float32 [| 4 |] (Array.make 4 0.))));
    ]

(* Counting nx.cpu, declining every map that adds, alone or among other nodes,
   as a library may decline a program it does not compute. *)
module Map_declining = struct
  include C

  let name = "nx.maps"

  let adds : Nx_kernel.Prog.node -> bool = function
    | Op2 (Binary Add, _, _) -> true
    | _ -> false

  let map (prog : Nx_kernel.Prog.t) ~dsts loads =
    if Iarray.exists adds prog.nodes then A.Declined else C.map prog ~dsts loads
end

module Maps = (val Nx.devices ~kernels:(module Map_declining) [ m 1 ])

let to_count x = Nx.place Count.on x

(* What [f ()] raises, or "no exception". *)
let raised f =
  match f () with _ -> "no exception" | exception Invalid_argument m -> m

let kernels =
  group "devices ~kernels"
    [
      test "an operation calls its set's kernels" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          ignore (Nx.add a a);
          at_least int ~than:1 (C.calls ()));
      (* nx checks that a call fits before it calls a kernel (nx_kernel.mli). *)
      cases "a call that does not fit raises before any kernel call"
        ~id:(fun (name, _) -> name)
        [
          ( "shapes that do not broadcast",
            fun () ->
              let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
              and b =
                to_count (on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |])
              in
              fun () -> ignore (Nx.add a b) );
          ( "a dtype the operation does not take, beside a creation",
            fun () ->
              let a =
                to_count (on_host Plain D.Bool [| 2 |] [| true; false |])
              in
              let c = Nx.zeros_like a in
              fun () -> ignore (Nx.add a c) );
        ]
        (fun (_, make) ->
          let call = make () in
          C.reset ();
          invalid ~by:"Nx.add" call;
          equal ~msg:"kernel calls" int 0 (C.calls ()));
      test "a dtype the operation does not take raises naming the function"
        (fun () ->
          let a = on_host Plain D.Bool [| 2 |] [| true; false |] in
          expect (raised (fun () -> Nx.add a a))
          @@ __POS_OF__ {|Nx.add: Add does not take bool, bool|});
      test "a case the kernels compute beside a declined one computes"
        (fun () ->
          let a =
            Nx.place Dec.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          equal (array bits) [| 1.; 4. |] (elements (Nx.mul a a)));
      test "a map of more loads than one run of the door claims computes"
        (fun () ->
          let n = A.max_operands + 2 in
          let x = on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
          let nodes =
            Iarray.init (n + 1) (fun k ->
                if k < n then Nx_kernel.Prog.In k
                else Op2 (Binary Add, 0, n - 1))
          in
          let prog =
            Nx_kernel.Prog.v
              ~ins:(Iarray.init n (fun _ -> D.Any D.Float32))
              nodes ~outs:[| n |]
          in
          let loads = Iarray.init n (fun _ : Nx.host Nx.Prim.load -> Plain x) in
          let op : ((float, D.float32_elt, Nx.host) Nx.t * unit) Nx.Prim.t =
            Map
              {
                layout = A.Layout.contiguous [| 3 |];
                prog;
                outs = [ D.Float32 ];
                loads;
                at = None;
              }
          in
          equal (array float_exact) [| 2.; 4.; 6. |]
            (elements (fst (Nx.Prim.eval ~by:"test" op))));
    ]

(* [f ()] raises for one decline by nx.refusing on m2 of [kind] over [dts],
   naming [by], the kernels, the device and [Nx.place], and no other kernel
   computes: nx raises after a decline and computes nothing in the library's
   place (nx_kernel.mli), so the library is asked once. *)
let declined_once by kind dts f =
  Atomic.set Refusing.declines 0;
  C.reset ();
  let m = raised f in
  starts_with ~msg:kind ~prefix:(by ^ ": ") m;
  List.iter
    (fun sub -> contains ~msg:kind ~sub m)
    [ "nx.refusing"; kind ^ " on " ^ dts; "(m2)"; "Nx.place" ];
  equal ~msg:(kind ^ ": declines") int 1 (Atomic.get Refusing.declines);
  equal ~msg:(kind ^ ": other kernel calls") int 0 (C.calls ())

let refused x = Nx.place Refuse.on (on_host Plain D.Float32 [| 2; 2 |] x)

let built_map ?(out = 2) x nodes =
  let f = D.Any D.Float32 in
  fst
    (Nx.Prim.eval ~by:"test"
       (Map
          {
            layout = A.Layout.contiguous (Iarray.of_array (Nx.shape x));
            prog = Nx_kernel.Prog.v ~ins:[| f; f |] nodes ~outs:[| out |];
            outs = [ D.Float32 ];
            loads = [| Plain x; Plain x |];
            at = None;
          }))

let declines =
  group "declines"
    [
      test
        "a decline raises naming the function, kernels, operation, dtypes, \
         device and place" (fun () ->
          let a =
            Nx.place Dec.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          expect (raised (fun () -> Nx.add a a))
          @@ __POS_OF__
               {|Nx.add: nx.test does not compute Add on float32, float32, float32 (m1); place its operands with Nx.place on a set whose kernels compute it|});
      cases "a declined operation raises once and computes nothing else"
        ~id:(fun (by, kind, _, _) -> by ^ ", " ^ kind)
        [
          ("Nx.neg", "Neg", "float32, float32", fun x _ -> Nx.neg x);
          ("Nx.add", "Add", "float32, float32, float32", fun x _ -> Nx.add x x);
          ( "Nx.where",
            "Where",
            "float32, bool, float32, float32",
            fun x flag -> Nx.where flag x x );
          ("Nx.sort", "Sort", "float32", fun x _ -> fst (Nx.sort x));
          ("Nx.matmul", "Contract", "float32, float32", fun x _ -> Nx.matmul x x);
          ( "Nx.logsumexp",
            "Reduce Logsumexp",
            "float32",
            fun x _ -> Nx.logsumexp ~axes:[ 1 ] x );
          ( "Nx.argmax",
            "Reduce Arg Max",
            "float32",
            fun x _ -> Nx.cast D.Float32 (Nx.argmax ~axis:1 x) );
          ( "Nx.reshape",
            "Copy",
            "float32",
            fun x _ -> Nx.reshape [| 4 |] (Nx.transpose x) );
          ( "test",
            "Add",
            "float32, float32, float32",
            fun x _ -> built_map x [| In 0; In 1; Op2 (Binary Add, 0, 1) |] );
        ]
        (fun (by, kind, dts, f) ->
          let x = refused [| 4.; 3.; 2.; 1. |] in
          let flag = Nx.equal x x in
          declined_once by kind dts (fun () -> f x flag));
      cases "a decline over devices raises once"
        ~id:(fun (name, _) -> name)
        [
          ( "a sort of a value cut along its rows",
            fun () ->
              let x =
                Nx.place (Refuse2.split ~axis:0)
                  (on_host Plain D.Float32 [| 2; 2 |] [| 4.; 3.; 2.; 1. |])
              in
              declined_once "Nx.sort" "Sort" "float32" (fun () ->
                  fst (Nx.sort x)) );
          ( "an argmax computed by partials",
            fun () ->
              let x =
                Nx.place (Refuse2.split ~axis:0)
                  (on_host Plain D.Float32 [| 4 |] [| 4.; 3.; 2.; 1. |])
              in
              declined_once "Nx.argmax" "Reduce Arg Max" "float32" (fun () ->
                  Nx.argmax ~axis:0 x) );
        ]
        (fun (_, check) -> check ());
      test "an operation over a cut value of no elements still declines"
        (fun () ->
          let x =
            Nx.place (Dec2.split ~axis:0) (on_host Plain D.Float32 [| 0 |] [||])
          in
          let m = raised (fun () -> Nx.add x x) in
          starts_with ~prefix:"Nx.add: nx.test does not compute Add" m);
      test "a declined map of several nodes names Map" (fun () ->
          let f = D.Any D.Float32 in
          let x =
            Nx.place Maps.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          let m =
            raised (fun () ->
                built_map ~out:3 x
                  [|
                    In 0; In 1; Op2 (Binary Add, 0, 1); Op1 (Unary Neg, f, 2);
                  |])
          in
          starts_with ~prefix:"test: nx.maps does not compute Map on" m);
      test "a declined creation raises at the call, naming it" (fun () ->
          let m = raised (fun () -> Nx.zeros_at Dec.on D.Float32 [| 2 |]) in
          starts_with ~prefix:"Nx.zeros_at: nx.test does not compute Fill" m);
      test "a declined operation leaves its donated operands dead" (fun () ->
          (* [x] borrows the host's memory, which no result takes. *)
          let data = [| 4.; 3.; 2.; 1. |] in
          let y = refused data in
          let dead by f =
            let x = refused data in
            invalid ~by (fun () -> ignore (f (Nx.donate x)));
            raises
              (Invalid_argument "Nx.copy: float32 [2; 2] was donated")
              (fun () -> Nx.copy x)
          in
          dead "Nx.sort" (fun h -> fst (Nx.sort h));
          dead "Nx.add" (fun h -> Nx.add h y);
          dead "Nx.less" (fun h -> Nx.less h y));
    ]

(* A value whose memory a take killed, as a donation's does: the value owns
   [a]'s memory, which a take then finds inside the claims. *)
let test_zeros_like_reads_nothing () =
  let a = A.to_device (m 0) (lay Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
  let x = Nx.Repr.unsafe_of_array Count.v a in
  let taken = Rig.Buffer.take ~why:"consumed by the test" [ A.buffer a ] in
  ignore (require_some ~msg:"a take" taken);
  invalid ~by:"Nx.add" (fun () -> Nx.add x x);
  C.reset ();
  let z = Nx.zeros_like x in
  equal ~msg:"kernel calls: one fill" int 1 (C.calls ());
  equal (array bits) [| 0.; 0. |] (elements z)

let creations =
  group "creations"
    [
      (* A creation holds its elements from the call on: the operations that
         read it compute only themselves. *)
      test "a creation computes at the call" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          let c = Nx.zeros_at Count.on D.Float32 [| 2 |] in
          equal ~msg:"kernel calls: the fill" int 1 (C.calls ());
          C.reset ();
          ignore (Nx.add a c);
          ignore (Nx.add a c);
          equal ~msg:"kernel calls: two adds" int 2 (C.calls ()));
      test "a read calls no kernel" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          let y = Nx.add a a in
          C.reset ();
          equal (array bits) [| 2.; 4. |] (elements y);
          equal ~msg:"kernel calls" int 0 (C.calls ()));
      test "scalar_like adds no kernel call to the operation reading it"
        (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          let c = Nx.scalar_like a 2. and full = Nx.full_like a 2. in
          let calls f =
            C.reset ();
            let y = f () in
            (y, C.calls ())
          in
          let _, beside_full = calls (fun () -> Nx.add a full) in
          let y, beside_scalar = calls (fun () -> Nx.add a c) in
          equal ~msg:"kernel calls beside a full operand" int beside_full
            beside_scalar;
          equal (array bits) [| 3.; 4. |] (elements y);
          let w = Nx.where (Nx.less a c) c a in
          equal (array bits) [| 2.; 2. |] (elements w));
      test "a scalar form is one kernel call" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          let y = Nx.add_s a 2. and z = Nx.rsub_s 2. a and l = Nx.less_s a 2. in
          equal ~msg:"kernel calls: one each" int 3 (C.calls ());
          equal (array bits) [| 3.; 4. |] (elements y);
          equal (array bits) [| 1.; 0. |] (elements z);
          equal (array bool) [| true; false |] (elements l));
      test "placing a host creation calls no kernel" (fun () ->
          let z = Nx.zeros D.Float32 [| 2 |] in
          let c = Nx.add z z in
          C.reset ();
          let y = Nx.place Count.on c in
          equal ~msg:"kernel calls" int 0 (C.calls ());
          equal (array bits) [| 0.; 0. |] (elements y));
      test "a creation of a sub-byte dtype holds its elements" (fun () ->
          let get x = A.to_array (Option.get (Nx.Repr.array ~by:"test" x)) in
          equal ~msg:"int4" (array int) [| -3; -3; -3 |]
            (get
               (Nx.add
                  (Nx.zeros D.Int4 [| 3 |])
                  (Nx.full D.Int4 [| 3 |] (-3))));
          equal ~msg:"uint4" (array int) [| 0; 0 |]
            (get (Nx.zeros D.Uint4 [| 2 |]));
          equal ~msg:"float4_e2m1fn" (array float_exact) [| 1.5 |]
            (get (Nx.reshape [| 1 |] (Nx.scalar D.Float4_e2m1fn 1.5)));
          equal ~msg:"bit" (array bool) [| true |]
            (get (Nx.reshape [| 1 |] (Nx.scalar D.Bit true))));
      test "a creation beside a split value is split with it" (fun () ->
          let x =
            Nx.place (Count2.split ~axis:0)
              (on_host Plain D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          let y = Nx.add x (Nx.zeros_like x) in
          equal (placement ()) (Count2.split ~axis:0) (Nx.placement y));
      test "a rule error in a creation raises at the call" (fun () ->
          invalid ~by:"Nx.zeros" (fun () -> Nx.zeros D.Float32 [| -1 |]));
      test "zeros_like reads none of its argument's elements"
        test_zeros_like_reads_nothing;
    ]

(* Every public function takes a value at any placement and gives what it
   gives on the host. *)
let every_function (type d) name (x : (float, D.float32_elt, d) Nx.t) =
  let msg f = name ^ ": " ^ f in
  let xs = elements x in
  let ys f = Array.map f xs in
  equal ~msg:(msg "shape") (array int) [| 2; 2 |] (Nx.shape x);
  equal ~msg:(msg "dtype") string "float32" (D.name (Nx.dtype x));
  equal ~msg:(msg "place") (array bits) xs
    (elements (Nx.place (Nx.placement x) x));
  equal ~msg:(msg "zeros_like") (array bits)
    (ys (fun _ -> 0.))
    (elements (Nx.zeros_like x));
  equal ~msg:(msg "add") (array bits)
    (ys (fun v -> v +. v))
    (elements (Nx.add x x));
  equal ~msg:(msg "mul") (array bits)
    (ys (fun v -> v *. v))
    (elements (Nx.mul x x));
  equal ~msg:(msg "add a scalar") (array bits)
    (ys (fun v -> v +. 1.))
    (elements (Nx.add x (Nx.scalar_like x 1.)));
  equal ~msg:(msg "add_s") (array bits)
    (ys (fun v -> v +. 1.))
    (elements (Nx.add_s x 1.));
  let less = Nx.less x (Nx.zeros_like x) in
  equal ~msg:(msg "less") (array bool) (ys (fun v -> v < 0.)) (elements less);
  equal ~msg:(msg "where") (array bits)
    (ys (fun v -> if v < 0. then v else 0.))
    (elements (Nx.where less x (Nx.zeros_like x)));
  equal ~msg:(msg "cast") (array bits) xs (elements (Nx.cast D.Float64 x));
  equal ~msg:(msg "bitcast") (array int32)
    (ys Int32.bits_of_float)
    (elements (Nx.bitcast D.Int32 x));
  (* A narrower dtype appends an axis of each element's bytes, in the host's
     byte order, and keeps [x]'s cuts. *)
  let bytes = Nx.bitcast D.Uint8 x in
  let byte v k =
    let k = if Sys.big_endian then 3 - k else k in
    let b = Int32.shift_right_logical (Int32.bits_of_float v) (8 * k) in
    Int32.to_int (Int32.logand b 0xffl)
  in
  equal ~msg:(msg "bitcast to bytes: shape") (array int) [| 2; 2; 4 |]
    (Nx.shape bytes);
  equal ~msg:(msg "bitcast to bytes") (array int)
    (Array.concat (Array.to_list (ys (fun v -> Array.init 4 (byte v)))))
    (elements bytes);
  equal ~msg:(msg "bitcast to bytes: placement") (placement ())
    (Nx.placement x) (Nx.placement bytes);
  equal ~msg:(msg "reshape") (array bits) xs (elements (Nx.reshape [| 4 |] x));
  equal ~msg:(msg "copy") (array bits) xs (elements (Nx.copy x));
  ignore (Nx.Repr.array ~by:"test" x);
  equal ~msg:(msg "shards") int
    (List.length (Nx.Devices.to_list (Nx.Placement.devices (Nx.placement x))))
    (Iarray.length (Nx.Repr.shards ~by:"test" x))

let test_every_placement () =
  let host = on_host Plain D.Float32 [| 2; 2 |] [| 1.; -2.; 3.; -4. |] in
  every_function "host" host;
  every_function "a host creation"
    (Nx.add
       (Nx.zeros D.Float32 [| 2; 2 |])
       (Nx.full D.Float32 [| 2; 2 |] (-1.5)));
  every_function "one software device" (Nx.place Count.on host);
  every_function "two, whole on each" (Nx.place Count2.on host);
  every_function "two, split" (Nx.place (Count2.split ~axis:0) host);
  every_function "a creation at a split"
    (Nx.full_at (Count2.split ~axis:0) D.Float32 [| 2; 2 |] 2.5)

(* A shape passed as an int array is read during the call only: [zeros],
   [zeros_at] and [reshape], over a creation and a value with bytes, give the
   same shape and elements with the argument overwritten after the call. *)
let law_arguments_owned (junk, s) =
  let overwrite a =
    Array.iteri (fun i _ -> a.(i) <- junk.(i mod Array.length junk)) a
  in
  let n = Array.fold_left ( * ) 1 s in
  let call name f =
    let arg = Array.copy s in
    let x = f arg in
    overwrite arg;
    equal ~msg:(name ^ ": shape") (array int)
      (Nx.shape (f (Array.copy s)))
      (Nx.shape x);
    equal ~msg:(name ^ ": elements") (array bits)
      (elements (f (Array.copy s)))
      (elements x)
  in
  let host = on_host Plain D.Float32 [| n |] (Array.init n Float.of_int) in
  call "zeros" (fun a -> Nx.zeros D.Float32 a);
  call "zeros_at" (fun a -> Nx.zeros_at Count2.on D.Float32 a);
  call "reshape of a creation" (fun a ->
      Nx.reshape a (Nx.add_s (Nx.zeros D.Float32 [| n |]) 1.));
  call "reshape" (fun a -> Nx.reshape a host)

let int_arrays =
  Gen.pair
    (Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range (-2) 5))
    (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 3))

let values =
  group "values"
    [
      prop "a shape overwritten after the call changes nothing" int_arrays
        law_arguments_owned;
    ]

(* A sub-byte result of an odd number of elements leaves the bits of its last
   byte past its last element zero, as fresh arrays do, on a device whose reused
   memory keeps the bits it held. *)
let test_sub_byte_tail () =
  for n = 1 to 32 do
    (* Memory of the result's size with every bit set, freed for reuse. *)
    let set = A.create (m 0) D.Uint8 [| n + 1 |] in
    for i = 0 to n do
      A.set set [| i |] 0xFF
    done;
    ignore (Sys.opaque_identity set);
    Gc.full_major ();
    let x =
      to_count
        (on_host Plain D.Int4 [| (2 * n) + 1 |] (Array.make ((2 * n) + 1) 1))
    in
    let y = Option.get (Nx.Repr.array ~by:"test" (Nx.add x x)) in
    let bytes =
      A.v D.Uint8 (Nx_array.Layout.contiguous [| n + 1 |]) (A.buffer y)
    in
    equal
      ~msg:(Printf.sprintf "the last byte of %d elements" ((2 * n) + 1))
      int 0x02 (A.get bytes [| n |])
  done

(* [f ()] under a handler that fails on any effect [f] performs. *)
let effectless f =
  let effc (type a) (e : a Effect.t) =
    ignore e;
    Some
      (fun (_ : (a, _) Effect.Deep.continuation) ->
        fail "an effect was performed")
  in
  Effect.Deep.match_with f () { retc = Fun.id; exnc = raise; effc }

(* [budget most f]: one call of [f] allocates at most [most] minor words, the
   words it allocated when the budget was set. The count depends on the compiler
   as well as the call: flambda inlines some allocations away. *)
let budget most f = at_most (float 0.5) ~than:most (words f)

let results =
  group "results"
    [
      test "a sub-byte result leaves its last byte's spare bits zero"
        test_sub_byte_tail;
    ]

let allocation =
  group "allocation"
    [
      test "an add of one-element host values allocates at most 52 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          at_most (float 0.5) ~than:52. (words (fun () -> Nx.add a a)));
      test "an add_s's words do not depend on the call before it" (fun () ->
          (* Its words after [before ()], beyond [before ()]'s own. *)
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          let after before =
            words (fun () ->
                ignore (Sys.opaque_identity (before ()));
                Nx.add_s a 1.)
            -. words before
          in
          expect
            (Printf.sprintf "%.0f %.0f %.0f words"
               (after (fun () -> Nx.add_s a 1.))
               (after (fun () -> Nx.add_s a 2.))
               (after (fun () -> Nx.mul_s a 1.)))
          @@ __POS_OF__ {| 70 70 70 words |});
      test "an add of host values performs no effect" (fun () ->
          let a = on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
          equal (array bits) [| 2.; 4.; 6. |]
            (elements (effectless (fun () -> Nx.add a a))));
      test "an add_s of a host value performs no effect" (fun () ->
          let a = on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
          equal (array bits) [| 2.; 3.; 4. |]
            (elements (effectless (fun () -> Nx.add_s a 1.))));
      test "a copy of a one-element host value allocates at most 88 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          budget 88. (fun () -> Nx.copy a));
      test "a matmul of 4x4 float32 host values allocates at most 163 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          budget 163. (fun () -> Nx.matmul a a));
      test "a contract of 4x4 float32 host values allocates at most 185 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          let p = Nx.Pattern.v "i k, k j -> i j | k" in
          budget 185. (fun () -> Nx.contract Nx.float32 p a a));
      test "a vdot of 16 float32 host elements allocates at most 154 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 16 |] (Array.make 16 1.) in
          budget 154. (fun () -> Nx.vdot a a));
      test "a slice of a 4x4 host value by ranges allocates at most 91 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          budget 91. (fun () -> Nx.slice Nx.[ R (1, 3); A ] a));
      test "a reshape of a one-element host value allocates at most 33 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          budget 33. (fun () -> Nx.reshape [| 1; 1 |] a));
      test "a sum of 16 float32 host elements allocates at most 129 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 16 |] (Array.make 16 1.) in
          budget 129. (fun () -> Nx.sum a));
      test "a cumsum of 16 float32 host elements allocates at most 116 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 16 |] (Array.make 16 1.) in
          budget 116. (fun () -> Nx.cumsum a));
      test "a take of 3 rows of a 4x4 host value allocates at most 175 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          let ids = on_host Plain D.Int64 [| 3 |] [| 0L; 2L; 3L |] in
          budget 175. (fun () -> Nx.take ~axis:0 ids a));
      test "a set of a row of a 4x4 host value allocates at most 341 words"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          let v = on_host Plain D.Float32 [| 4 |] (Array.make 4 2.) in
          budget 341. (fun () -> Nx.set Nx.[ I 1 ] v a));
    ]

(* A scatter's results lie at the union of its operands' cuts, as any
   operation's: positions and updates cut along columns beside a target whole on
   each device put them at that cut, each device scattering its columns, with
   the bits the host gives. *)
let test_scatter_union () =
  let x = on_host Plain D.Float32 [| 3; 4 |] (Array.init 12 Float.of_int) in
  let p =
    on_host Plain D.Int64 [| 2; 4 |] [| 0L; 1L; 2L; 0L; 2L; 2L; 0L; 1L |]
  in
  let u =
    on_host Plain D.Float32 [| 2; 4 |]
      (Array.init 8 (fun i -> 100. +. Float.of_int i))
  in
  let host = elements (Nx.scatter ~axis:0 p u x) in
  let split = Count2.split ~axis:1 in
  let y =
    Nx.scatter ~axis:0 (Nx.place split p) (Nx.place split u)
      (Nx.place Count2.on x)
  in
  equal ~msg:"split as its updates" (placement ()) split (Nx.placement y);
  equal (array bits) host (elements y)

(* A gather and a scatter over operands split along their first axis, and an
   assembly of pieces on one device of a set and whole on each, give the
   elements the host gives. *)
let test_index_split () =
  let split = Count2.split ~axis:0 in
  let at = Nx.place in
  let x = on_host Plain D.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let p = on_host Plain D.Int64 [| 2; 2 |] [| 2L; 0L; 1L; 1L |] in
  let u = on_host Plain D.Int32 [| 2; 2 |] [| 10l; 20l; 30l; 40l |] in
  let gather p x = Nx.take_along_axis ~axis:1 p x in
  let taken = [| 3l; 1l; 5l; 5l |] in
  equal ~msg:"gather" (array int32) taken (elements (gather p x));
  equal ~msg:"gather, split" (array int32) taken
    (elements (gather (at split p) (at split x)));
  let scatter p u x = Nx.scatter ~combine:Add ~axis:1 p u x in
  let added = [| 21l; 2l; 13l; 4l; 75l; 6l |] in
  equal ~msg:"scatter" (array int32) added (elements (scatter p u x));
  equal ~msg:"scatter, split" (array int32) added
    (elements (scatter (at split p) (at split u) (at split x)));
  let one = Nx.Placement.device Count2.v (m 1) in
  let row v = on_host Plain D.Int32 [| 1; 3 |] v in
  let top = at one (row [| 7l; 8l; 9l |]) in
  equal ~msg:"an assembly" (array int32)
    [| 7l; 8l; 9l; 1l; 2l; 3l |]
    (elements
       (Nx.concatenate ~axis:0 [ top; at Count2.on (row [| 1l; 2l; 3l |]) ]));
  equal ~msg:"an assembly with a fill" (array int32)
    [| -1l; -1l; -1l; 7l; 8l; 9l; -1l; -1l; -1l |]
    (elements (Nx.pad [| (1, 1); (0, 0) |] (-1l) top))

let routing =
  group "routing"
    [
      test "an operation over a split value computes on each device" (fun () ->
          let x =
            Nx.place (Count2.split ~axis:0)
              (on_host Plain D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          C.reset ();
          let y = Nx.add x x in
          equal int 2 (C.calls ());
          equal (array bits) [| 2.; 4.; 6.; 8. |] (elements y));
      test "a device whose window is empty computes nothing" (fun () ->
          let host = on_host Plain D.Float32 [| 2 |] [| 1.; 2. |] in
          let x = Nx.place (Count4.split ~axis:0) host in
          let mesh = Nx.Mesh.v Count4.v [ ("a", 2); ("b", 2) ] in
          let y =
            Nx.place (Nx.Placement.mesh mesh [ (0, [ "a"; "b" ]) ]) host
          in
          C.reset ();
          let sum = Nx.add x x
          and iota = Nx.arange_at (Count4.split ~axis:0) D.Int32 0 2 1 in
          equal ~msg:"split over four" int 4 (C.calls ());
          C.reset ();
          let product = Nx.mul y y in
          equal ~msg:"cut over a 2 x 2 mesh" int 2 (C.calls ());
          equal (array bits) [| 2.; 4. |] (elements sum);
          equal (array bits) [| 1.; 4. |] (elements product);
          equal (array int32) [| 0l; 1l |] (elements iota));
      test
        "solve and lstsq of a matrix whole on each device factor it once per \
         device" (fun () ->
          let a p =
            Nx.add (Nx.eye_at p D.Float64 3)
              (Nx.full_at p D.Float64 [| 3; 3 |] 0.5)
          in
          let data =
            on_host Plain D.Float64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
          in
          let solve a b = Array.concat [ elements (Nx.solve a b) ] in
          let lstsq a b =
            let x, residuals, rank, s = Nx.lstsq D.Float64 a b in
            Array.concat
              [
                elements x;
                elements residuals;
                Array.map Int64.to_float (elements rank);
                elements s;
              ]
          in
          let placed f = f (a Factor.on) (Nx.place Factor.on data) in
          let one = Nx.place Factor.on data in
          let check name ~factored want f b =
            Atomic.set Factoring.factored 0;
            let y = f (a (Nx.beside b)) b in
            equal
              ~msg:(name ^ ": factorizations")
              int factored
              (Atomic.get Factoring.factored);
            equal ~msg:(name ^ ": bits") (array bits) want y
          in
          (* Beside [b] on two devices, whole or cut, [a] lies whole on each,
             which factors it. A solve permutes [b]'s rows, a gather that
             computes over cut rows, and reads its columns whole, so a [b] cut
             along them raises. *)
          let both = Nx.place Factor2.on data in
          let rows = Nx.place (Factor2.split ~axis:0) data in
          let columns = Nx.place (Factor2.split ~axis:1) data in
          check "solve" ~factored:1 (placed solve) solve one;
          check "solve, on two devices" ~factored:2 (placed solve) solve both;
          check "lstsq" ~factored:1 (placed lstsq) lstsq one;
          check "lstsq, on two devices" ~factored:2 (placed lstsq) lstsq both;
          check "lstsq, rows cut" ~factored:2 (placed lstsq) lstsq rows;
          check "lstsq, columns cut" ~factored:2 (placed lstsq) lstsq columns;
          check "solve, rows cut" ~factored:2 (placed solve) solve rows;
          raises
            (Invalid_argument
               "Nx.solve: operand 1 lies at split ~axis:1 [m2; m3] and is read \
                at on [m2; m3]; Nx.place (Placement.on s) y") (fun () ->
              solve (a (Nx.beside columns)) columns));
      test "every function takes a value at any placement" test_every_placement;
      test "a scatter's updates cut beside a target whole on each device \
            scatter at their cut"
        test_scatter_union;
      test
        "a gather, scatter and assembly of split or placed operands give the \
         host's elements"
        test_index_split;
    ]

let () =
  exit
    (run "nx exec"
       [
         values;
         kernels;
         declines;
         routing;
         creations;
         broadcasting;
         laws;
         results;
         allocation;
       ])
