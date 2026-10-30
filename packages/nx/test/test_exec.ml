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
module Dec = (val Nx.devices ~kernels:(module Nx_support.Declining) [ m 1 ])

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
    if Array.exists declined rs then decline ()
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
let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

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
  let a = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)) in
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

(* The bytes of [chain]'s results and of [fused]'s over [x] and [y]. *)
let fused_maps x y =
  let layout = A.Layout.contiguous (Iarray.of_array (Nx.shape x)) in
  let loads : Nx.host Nx.Prim.load iarray = [| Plain x; Plain y |] in
  let s, (p, ()) =
    Nx.Prim.eval ~by:"test"
      (Map
         {
           layout;
           prog = chain;
           outs = [ D.Float32; D.Float32 ];
           loads;
           at = None;
         })
  in
  let q, () =
    Nx.Prim.eval ~by:"test"
      (Map { layout; prog = fused; outs = [ D.Float32 ]; loads; at = None })
  in
  (bytes s, bytes p, bytes q)

let one_node_case =
  Gen.pair
    (Gen.of_list
       ~pp:(fun ppf (name, _, _) -> Format.pp_print_string ppf name)
       one_node_maps)
    case

let laws =
  group "laws"
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
      prop "a map gives the same bits over operands offered for donation"
        case (fun (s, la, lb, xs, ys) ->
          let x = on_host la D.Float32 s (Array.map f32 xs)
          and y = on_host lb D.Float32 s (Array.map f32 ys) in
          let x' = Nx.copy x and y' = Nx.copy y in
          ignore (Nx.donate x');
          ignore (Nx.donate y');
          equal
            (triple (array int) (array int) (array int))
            (fused_maps x y) (fused_maps x' y'));
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
          equal bool true (Nx.cast D.Float32 x == x));
      test "a cast stores each element in the dtype" (fun () ->
          let x = on_host Plain D.Float32 [| 3 |] [| 1.5; -2.5; 300. |] in
          equal (array int)
            (Array.map (D.of_float D.Uint8) [| 1.5; -2.5; 300. |])
            (elements (Nx.cast D.Uint8 x)));
    ]

let broadcasting =
  group "broadcasting"
    [
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

(* Counting nx.cpu, declining every [Add] and [Less], alone or in a map. A
   call first runs what its domain left [pending], once: a call another domain
   makes after the operation checked its operands and before its kernel reads
   them or declines. *)
module Gated = struct
  include C

  let name = "nx.gated"
  let pending = Domain.DLS.new_key (fun () -> None)

  let first () =
    match Domain.DLS.get pending with
    | None -> ()
    | Some f ->
        Domain.DLS.set pending None;
        f ()

  let declined : Nx_kernel.Prog.node -> bool = function
    | Op2 ((Binary Add | Compare Less), _, _) -> true
    | _ -> false

  let contract ~batch ~contracting ~segments ~acc ~dst ~init a b =
    first ();
    C.contract ~batch ~contracting ~segments ~acc ~dst ~init a b

  let map (prog : Nx_kernel.Prog.t) ~dsts loads =
    first ();
    if Iarray.exists declined prog.nodes then A.Declined
    else C.map prog ~dsts loads
end

module Gate = (val Nx.devices ~kernels:(module Gated) [ m 1 ])

let to_count x = Nx.place Count.on x

(* [f ()] raises for one decline by nx.refusing on m2, of [kind] over [dts],
   naming [by]. *)
let refused_once by kind dts f =
  Atomic.set Refusing.declines 0;
  raises
    (Invalid_argument
       (by ^ ": nx.refusing does not compute " ^ kind ^ " on " ^ dts
      ^ " (m2); place its operands with Nx.place on a set whose kernels \
         compute it"))
    (fun () -> ignore (f ()));
  equal ~msg:(kind ^ ": declines") int 1 (Atomic.get Refusing.declines)

let refused x = Nx.place Refuse.on (on_host Plain D.Float32 [| 2; 2 |] x)

let declines =
  group "declines"
    [
      test "a declined reduction names its kinds" (fun () ->
          let x = refused [| 4.; 3.; 2.; 1. |] in
          refused_once "Nx.logsumexp" "Reduce Logsumexp" "float32" (fun () ->
              Nx.logsumexp ~axes:[ 1 ] x);
          refused_once "Nx.argmax" "Reduce Arg Max" "float32" (fun () ->
              Nx.argmax ~axis:1 x));
      test "a decline on a set cut over devices raises once" (fun () ->
          let x =
            Nx.place (Refuse2.split ~axis:0)
              (on_host Plain D.Float32 [| 2; 2 |] [| 4.; 3.; 2.; 1. |])
          in
          refused_once "Nx.sort" "Sort" "float32" (fun () -> fst (Nx.sort x)));
      test "a decline in a computation by partials raises once" (fun () ->
          let x =
            Nx.place (Refuse2.split ~axis:0)
              (on_host Plain D.Float32 [| 4 |] [| 4.; 3.; 2.; 1. |])
          in
          refused_once "Nx.argmax" "Reduce Arg Max" "float32" (fun () ->
              Nx.argmax ~axis:0 x));
      test "a built map of more arrays than the door's runs computes"
        (fun () ->
          let n = 20 in
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
      test "a declined built map names its node, or Map" (fun () ->
          let x = refused [| 4.; 3.; 2.; 1. |] in
          let f = D.Any D.Float32 in
          let prog nodes outs = Nx_kernel.Prog.v ~ins:[| f; f |] nodes ~outs in
          let layout = A.Layout.contiguous [| 2; 2 |] in
          let loads : Refuse.d Nx.Prim.load iarray = [| Plain x; Plain x |] in
          let eval p =
            fst
              (Nx.Prim.eval ~by:"test"
                 (Map
                    {
                      layout;
                      prog = p;
                      outs = [ D.Float32 ];
                      loads;
                      at = None;
                    }))
          in
          refused_once "test" "Add" "float32, float32, float32" (fun () ->
              eval (prog [| In 0; In 1; Op2 (Binary Add, 0, 1) |] [| 2 |]));
          let g = on_host Plain D.Float32 [| 2 |] [| 1.; 2. |] in
          let g = Nx.place Gate.on g in
          let nodes : Nx_kernel.Prog.node iarray =
            [| In 0; In 1; Op2 (Binary Add, 0, 1); Op1 (Unary Neg, f, 2) |]
          in
          raises
            (Invalid_argument
               "test: nx.gated does not compute Map on float32, float32 (m1); \
                place its operands with Nx.place on a set whose kernels \
                compute it")
            (fun () ->
              Nx.Prim.eval ~by:"test"
                (Map
                   {
                     layout = A.Layout.contiguous [| 2 |];
                     prog = prog nodes [| 3 |];
                     outs = [ D.Float32 ];
                     loads = [| Plain g; Plain g |];
                     at = None;
                   })));
      test "a declined copy raises once" (fun () ->
          let x = refused [| 4.; 3.; 2.; 1. |] in
          refused_once "Nx.reshape" "Copy" "float32" (fun () ->
              Nx.reshape [| 4 |] (Nx.transpose x)));
      test "a declined operation kills its donated operands" (fun () ->
          (* [x] borrows the host's memory, which no result takes. *)
          let data = [| 4.; 3.; 2.; 1. |] in
          let y = refused data in
          let dead by f =
            let x = refused data in
            invalid ~by (fun () -> ignore (f (Nx.donate x)));
            raises_match
              (Exn.invalid_arg
                 ~sub:("Nx.copy: float32 [2; 2] was donated to " ^ by))
              (fun () -> Nx.copy x)
          in
          dead "Nx.sort" (fun h -> fst (Nx.sort h));
          dead "Nx.add" (fun h -> Nx.add h y);
          dead "Nx.less" (fun h -> Nx.less h y));
    ]


(* [f ()] with [g] run on another domain inside its first call into nx.gated:
   after [f]'s operation checked its operands, before its kernel reads them. *)
let meanwhile g f =
  let other () = Domain.join (Domain.spawn g) in
  Domain.DLS.set Gated.pending (Some other);
  Fun.protect ~finally:(fun () -> Domain.DLS.set Gated.pending None) f

(* Donations of one value on nx.gated: handles, views and copies of handles,
   sums and comparisons that decline, on the fast path and off it, and a
   product and a sum that consume. The value and every handle of it die at the
   first consumer, named by it, and a decline kills them as a consumer does: it
   comes after the call's checks and claim. A refusal by the checks kills
   nothing. A movement kills nothing: the handle it read and the one it passed
   on die with their donor. A call made on another domain while a decline
   holds a value's handles is held there deterministically ([meanwhile]). *)
type donor = { xs : float array; mutable dead : string option }

(* A handle reads its donor's elements in [order]. It lives while its donor
   does. *)
type handle = { x : donor; order : int array }

let donor = abstract "x"
let handle = abstract "h"

let fours =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float a))))
    (Gen.array ~size:(Gen.int_range 4 4) (Gen.of_list [ 0.; 1.; -2.; 3.5 ]))

let on_gate xs = Nx.place Gate.on (on_host Plain D.Float32 [| 4 |] xs)
let twos () = on_gate (Array.make 4 2.)

(* Twos that a handle beside them broadcasts to. *)
let wide_twos () =
  Nx.place Gate.on (on_host Plain D.Float32 [| 2; 4 |] (Array.make 8 2.))

(* Computed, so a consumer may write into its memory. *)
let make_sys xs = Nx.mul (on_gate xs) (on_gate (Array.make 4 1.))
let make_ref xs = { xs; dead = None }

(* What a refusal says: a decline, the function a value went to, or that a
   kernel met memory a consumer in flight holds exclusive before it consumed
   it, naming no one ([taken]). *)
let named m =
  let has s i =
    i + String.length s <= String.length m
    && String.sub m i (String.length s) = s
  in
  let rec find s i =
    if i > String.length m then None
    else if has s i then Some i
    else find s (i + 1)
  in
  (* As an operand's check says it, or a kernel's refusal of a consumed
     buffer, which lists the operands after. *)
  let mark = "donated to " in
  match find mark 0 with
  | Some i ->
      let i = i + String.length mark in
      let j = Option.value (find " (" i) ~default:(String.length m) in
      String.sub m i (j - i)
  | None ->
      if Option.is_some (find "does not compute" 0) then "declined"
      else if Option.is_some (find "held exclusive" 0) then "taken"
      else "refused: " ^ m

let outcome f =
  match f () with ys -> Ok ys | exception Invalid_argument m -> Error (named m)

let outcomes = result (array float_exact) string
let elements_of h = Array.map (fun i -> h.x.xs.(i)) h.order

(* [h] and its donor known to live. *)
let lives h = if Option.is_some h.x.dead then fail "a dead handle lives"

(* The donor [x] seen dead to [f], or alive. *)
let seen x f = if x.dead <> f then fail "a donor seen in another state"

(* A call [by] on [h] that raised naming [e], its donor's consumer. *)
let lost by e h = if h.x.dead <> Some e then fail (by ^ " raised naming " ^ e)

(* As [lost], or, on two domains, a call that met [h]'s memory [taken] by a
   consumer in flight. *)
let lost_raced by e h =
  match (e, h.x.dead) with "taken", Some _ -> () | _ -> lost by e h

let donate_ref x =
  if Option.is_some x.dead then invalid_arg "dead";
  { x; order = [| 0; 1; 2; 3 |] }

(* A movement passes on a handle reading [order] of [h], killing nothing. *)
let moved order h =
  if Option.is_some h.x.dead then invalid_arg "dead";
  { x = h.x; order = Array.map (fun i -> h.order.(i)) order }

let view_ref = moved [| 0; 1; 2; 3 |]
let view_sys h = Nx.reshape [| 4 |] (Nx.reshape [| 2; 2 |] h)

(* A reshape of a transpose copies. *)
let copy_ref = moved [| 0; 2; 1; 3 |]
let copy_sys h = Nx.reshape [| 4 |] (Nx.transpose (Nx.reshape [| 2; 2 |] h))

(* A decline by [by] of [h], then [h] seen dead or alive ([after]): the
   decline kills [h] and its donor, naming [by]. *)
let declined by h = function
  | Error e -> raise e
  | Ok (Ok _, _) -> fail (by ^ " computed what its kernels decline")
  | Ok (Error "declined", after) ->
      lives h;
      if after <> Some by then fail (by ^ ": a declined donor lives");
      h.x.dead <- Some by
  | Ok (Error e, after) ->
      lost by e h;
      if after <> Some e then fail (by ^ ": a dead handle seen otherwise after")

(* [f h], then [h] seen dead or alive: [Nx.donate] raises for a dead value. *)
let then_seen f h =
  let out = outcome (fun () -> elements (f h)) in
  let after =
    match Nx.donate h with
    | _ -> None
    | exception Invalid_argument m -> Some (named m)
  in
  (out, after)

let seen_outcomes = pair outcomes (option string)
let add_sys = then_seen (fun h -> Nx.add h (twos ()))
let less_sys = then_seen (fun h -> Nx.cast D.Float32 (Nx.less h (twos ())))

(* Beside a scalar broadcast to its shape. *)
let add_scalar_sys = then_seen (fun h -> Nx.add h (Nx.scalar_like h 2.))

let less_scalar_sys =
  then_seen (fun h -> Nx.cast D.Float32 (Nx.less h (Nx.scalar_like h 2.)))

(* A kind refusing [h]'s dtype, on the fast path, where its result would fit
   [h]'s memory: it kills nothing. *)
let and_ref h = function
  | Error e -> raise e
  | Ok (Ok _, _) -> fail "Nx.bitwise_and computed floats"
  | Ok (Error e, after) ->
      equal string "refused: Nx.bitwise_and: And does not take float32, float32"
        e;
      cover "a live handle refused by dtype" (Option.is_none h.x.dead);
      if after <> h.x.dead then
        fail "Nx.bitwise_and: a refusal changed a handle's fate"

let and_sys = then_seen (fun h -> Nx.bitwise_and h (twos ()))

(* Broadcast beside wider twos: no result fits the handle's memory. *)
let add_wide_sys = then_seen (fun h -> Nx.add h (wide_twos ()))

let less_wide_sys =
  then_seen (fun h -> Nx.cast D.Float32 (Nx.less h (wide_twos ())))

let and_wide_sys = then_seen (fun h -> Nx.bitwise_and h (wide_twos ()))

(* Broadcast, a dead handle raises naming its consumer before the kind's
   refusal: the call checks its operands before it broadcasts them. *)
let and_wide_ref h = function
  | Ok (Error e, _) when Option.is_some h.x.dead -> lost "Nx.bitwise_and" e h
  | out -> and_ref h out

(* A consumer [by] of [h] whose result's elements are [f]'s, and whose
   refusal [lost] judges. *)
let consumer lost by f h = function
  | Error e -> raise e
  | Ok (Ok ys) ->
      lives h;
      equal (array float_exact) (f (elements_of h)) ys;
      h.x.dead <- Some by
  | Ok (Error e) -> lost by e h

let double = Array.map (fun v -> v *. 2.)
let total ys = [| Array.fold_left ( +. ) 0. ys |]
let mul_ref = consumer lost "Nx.mul" double
let mul_sys h = outcome (fun () -> elements (Nx.mul h (twos ())))

let mul_wide_ref =
  consumer lost "Nx.mul" (fun ys ->
      let doubled = double ys in
      Array.append doubled doubled)

let mul_wide_sys h = outcome (fun () -> elements (Nx.mul h (wide_twos ())))
let sum_ref = consumer lost "Nx.sum" total
let sum_sys h = outcome (fun () -> elements (Nx.sum h))

let read_ref x = function
  | Error e -> raise e
  | Ok (Ok ys) ->
      seen x None;
      equal outcomes (Ok x.xs) (Ok ys)
  | Ok (Error f) -> seen x (Some f)

let read_sys x = outcome (fun () -> elements x)

(* A decline [by] of [h] that runs [inner] on another domain meanwhile, whose
   outcome [met] is [None] where the decline raised before its kernel. While
   the decline runs, [h] and its donor are dead to it, so a call meanwhile that
   reads them loses to it. *)
let during by inner h = function
  | Error e -> raise e
  | Ok (out, met) -> (
      match (h.x.dead, met) with
      | None, Some r ->
          h.x.dead <- Some by;
          inner r;
          h.x.dead <- None;
          declined by h (Ok out)
      | None, None -> fail (by ^ ": nothing ran meanwhile")
      | Some _, None -> declined by h (Ok out)
      | Some _, Some _ -> fail (by ^ " ran a call meanwhile though refused"))

(* [f h], then [h] seen, with [g] run on another domain meanwhile. *)
let while_sys f g h =
  let met = ref None in
  let out = meanwhile (fun () -> met := Some (g ())) (fun () -> f h) in
  (out, !met)

let lost_in_flight by r =
  cover "a call loses to a decline in flight" (r = Error by)

let add_while_mul =
  command "add, meanwhile mul"
    (handle ^-> handle ^-> judges (pair seen_outcomes (option outcomes)))
    (fun h h' ->
      during "Nx.add"
        (fun r ->
          lost_in_flight "Nx.add" r;
          mul_ref h' (Ok r))
        h)
    (fun h h' -> while_sys add_sys (fun () -> mul_sys h') h)

let less_while_read =
  command "less, meanwhile read"
    (handle ^-> donor ^-> judges (pair seen_outcomes (option outcomes)))
    (fun h x ->
      during "Nx.less"
        (fun r ->
          lost_in_flight "Nx.less" r;
          read_ref x (Ok r))
        h)
    (fun h x -> while_sys less_sys (fun () -> read_sys x) h)

let donations =
  [
    command "make" (fours @-> makes donor) make_ref make_sys;
    command "donate" (donor ^-> makes handle) donate_ref Nx.donate;
    command "view" (handle ^-> makes handle) view_ref view_sys;
    command "copy" (handle ^-> makes handle) copy_ref copy_sys;
    command "add"
      (handle ^-> judges seen_outcomes)
      (declined "Nx.add") add_sys;
    command "add beside a scalar"
      (handle ^-> judges seen_outcomes)
      (declined "Nx.add") add_scalar_sys;
    command "less"
      (handle ^-> judges seen_outcomes)
      (declined "Nx.less") less_sys;
    command "less beside a scalar"
      (handle ^-> judges seen_outcomes)
      (declined "Nx.less") less_scalar_sys;
    command "bitwise_and" (handle ^-> judges seen_outcomes) and_ref and_sys;
    command "add broadcast"
      (handle ^-> judges seen_outcomes)
      (declined "Nx.add") add_wide_sys;
    command "less broadcast"
      (handle ^-> judges seen_outcomes)
      (declined "Nx.less") less_wide_sys;
    command "bitwise_and broadcast"
      (handle ^-> judges seen_outcomes)
      and_wide_ref and_wide_sys;
    command "mul" (handle ^-> judges outcomes) mul_ref mul_sys;
    command "mul broadcast" (handle ^-> judges outcomes) mul_wide_ref
      mul_wide_sys;
    command "sum" (handle ^-> judges outcomes) sum_ref sum_sys;
    command "read" (donor ^-> judges outcomes) read_ref read_sys;
    add_while_mul;
    less_while_read;
  ]

(* On two domains a decline kills its donor, as in sequence, and is not seen
   after. *)
let raced_decline by h = function
  | Error e -> raise e
  | Ok (Ok _) -> fail (by ^ " computed what its kernels decline")
  | Ok (Error "declined") ->
      lives h;
      h.x.dead <- Some by
  | Ok (Error e) -> lost_raced by e h

let raced_sys f h = outcome (fun () -> elements (f h))

(* A movement on two domains: it reads [h] live, killing nothing, or raises
   naming the consumer that took its donor. *)
let raced_moved h = function
  | Error e -> raise e
  | Ok (Ok _) -> lives h
  | Ok (Error e) -> lost_raced "Nx.reshape" e h

(* The movement's handle, which no command reads. *)
let moved_sys f h =
  outcome (fun () ->
      ignore (f h);
      [||])

(* A read on two domains may meet the donor's memory taken by a consumer in
   flight. *)
let read_judge x = function
  | Ok (Error "taken") ->
      if Option.is_none x.dead then fail "a read lost to no one"
  | out -> read_ref x out

let raced =
  [
    command "make" (fours @-> makes donor) make_ref make_sys;
    command "donate" (donor ^-> makes handle) donate_ref Nx.donate;
    command "view" (handle ^-> makes handle) view_ref view_sys;
    command "copy" (handle ^-> makes handle) copy_ref copy_sys;
    command "view racing" (handle ^-> judges outcomes) raced_moved
      (moved_sys view_sys);
    command "copy racing" (handle ^-> judges outcomes) raced_moved
      (moved_sys copy_sys);
    command "add"
      (handle ^-> judges outcomes)
      (raced_decline "Nx.add")
      (raced_sys (fun h -> Nx.add h (twos ())));
    command "add beside a scalar"
      (handle ^-> judges outcomes)
      (raced_decline "Nx.add")
      (raced_sys (fun h -> Nx.add h (Nx.scalar_like h 2.)));
    command "less"
      (handle ^-> judges outcomes)
      (raced_decline "Nx.less")
      (raced_sys (fun h -> Nx.cast D.Float32 (Nx.less h (twos ()))));
    command "less beside a scalar"
      (handle ^-> judges outcomes)
      (raced_decline "Nx.less")
      (raced_sys (fun h ->
           Nx.cast D.Float32 (Nx.less h (Nx.scalar_like h 2.))));
    command "mul"
      (handle ^-> judges outcomes)
      (consumer lost_raced "Nx.mul" double)
      mul_sys;
    command "sum"
      (handle ^-> judges outcomes)
      (consumer lost_raced "Nx.sum" total)
      sum_sys;
    command "read" (donor ^-> judges outcomes) read_judge read_sys;
  ]

(* [f ()] raises naming [by] and the consumer [consumer]. *)
let raises_by consumer by f =
  match f () with
  | _ -> fail (by ^ " raised nothing")
  | exception Invalid_argument m ->
      let n = Stdlib.min (String.length m) (String.length by) in
      equal ~msg:m (pair string string) (by, consumer)
        (String.sub m 0 n, named m)

let raises_mul = raises_by "Nx.mul"

(* A call that checked [x] live, and whose kernel then meets [x]'s memory
   consumed by an operation on another domain, raises naming that operation,
   on the fast path, off it, through a movement's copy of a handle, and where
   the operation computed into fresh memory and freed [x]'s. *)
let test_raced_read_names () =
  let consume h = ignore (Nx.mul h (twos ())) in
  let raced by f =
    let x = make_sys [| 1.; 2.; 3.; 4. |] in
    let h = Nx.donate x in
    raises_mul by (fun () ->
        meanwhile (fun () -> consume h) (fun () -> ignore (f x)))
  in
  (* An offered donor is read off the fast path. *)
  raced "Nx.neg" Nx.neg;
  raced "Nx.mul" (fun x -> Nx.mul x (twos ()));
  raced "Nx.reshape" (fun x -> copy_sys (Nx.donate x));
  let x = make_sys [| 1.; 2.; 3.; 4. |] in
  raises_mul "Nx.neg" (fun () ->
      meanwhile (fun () -> consume (Nx.donate x)) (fun () -> ignore (Nx.neg x)));
  let x = make_sys [| 1.; 2.; 3.; 4. |] in
  let h = Nx.donate x in
  raises_by "Nx.sum" "Nx.neg" (fun () ->
      meanwhile (fun () -> ignore (Nx.sum h)) (fun () -> ignore (Nx.neg x)))

(* A call on live values, one of which an operation on another domain donates
   and consumes after the call checked it and before its kernel reads it,
   raises naming that operation: on each fast path, beside a handle, and in a
   contraction. *)
let raced_live =
  let consume x = ignore (Nx.mul (Nx.donate x) (twos ())) in
  let fresh () = make_sys [| 1.; 2.; 3.; 4. |] in
  (* [f c h t x]: a mask [c], a handle [h], twos [t], made before the call. *)
  let case name by f =
    test name (fun () ->
        let c = Nx.equal (twos ()) (on_gate [| 2.; 0.; 2.; 0. |]) in
        let h = Nx.donate (fresh ()) and t = twos () and x = fresh () in
        raises_mul by (fun () ->
            meanwhile (fun () -> consume x) (fun () -> ignore (f c h t x))))
  in
  [
    case "a product" "Nx.mul" (fun _ _ t x -> Nx.mul x t);
    case "a product beside a handle" "Nx.mul" (fun _ h _ x -> Nx.mul h x);
    case "a selection" "Nx.where" (fun c _ t x -> Nx.where c x t);
    case "a selection beside a handle" "Nx.where" (fun c h _ x ->
        Nx.where c h x);
    case "a contraction" "Nx.contract" (fun _ _ t x ->
        Nx.contract Nx.float32 (Nx.Pattern.v "i, i -> | i") x t);
  ]

(* A call that broadcasts a handle and declines kills the donor: the broadcast
   passed on a new handle of the donor, which the decline claimed. One that
   its checks refuse for a handle another domain's consumer took kills
   nothing: the checks come before any claim. *)
let lost_kills_none =
  let at s xs = Nx.place Gate.on (on_host Plain D.Float32 s xs) in
  let computed s xs = Nx.mul (at s xs) (at s (Array.make (Array.length xs) 1.)) in
  let xs = [| 1.; 2.; 3.; 4. |] in
  let lives msg x h =
    equal ~msg:(msg ^ ": the donor") (array float_exact) xs (elements x);
    equal ~msg:(msg ^ ": its handle") (array float_exact) xs
      (elements (Nx.copy h))
  in
  (* [g ()] on another domain while [Nx.add hy t] holds [y], [hy]'s donor,
     inside its kernel, which declines: [g]'s outcome. *)
  let during_add hy g =
    let t = at (Nx.shape hy) (Array.make (Array.length xs) 2.) in
    let met = ref None in
    invalid ~by:"Nx.add" (fun () ->
        meanwhile (fun () -> met := Some (outcome g)) (fun () -> Nx.add hy t));
    Option.get !met
  in
  [
    test "a comparison declined beside a wider operand" (fun () ->
        let x = computed [| 4 |] xs in
        let h = Nx.donate x and t = at [| 2; 4 |] (Array.make 8 2.) in
        invalid ~by:"Nx.less" (fun () -> Nx.less h t);
        equal ~msg:"the donor" outcomes (Error "Nx.less")
          (outcome (fun () -> elements x)));
    test "a product refused for its second handle" (fun () ->
        (* The product checks [hy'] dead, its donor claimed by the sum, before
           it claims [hx]. *)
        let x = computed [| 4 |] xs in
        let hx = Nx.donate x in
        let y = computed [| 4 |] xs in
        let hy = Nx.donate y and hy' = Nx.donate y in
        equal outcomes (Error "Nx.add")
          (during_add hy (fun () -> elements (Nx.mul hx hy')));
        lives "after the refusal" x hx);
  ]

let donation =
  group "donation"
    [
      stateful "a value lives until a handle of it is consumed" donations;
      test "a read racing a consumer names it" test_raced_read_names;
      group "a live operand racing its consumer names it" raced_live;
      group "a declined call kills its donors, a refused one nothing"
        lost_kills_none;
      stateful ~domains:2
        "handles of one value declined and consumed on two domains" raced;
    ]

let kernels =
  group "kernels"
    [
      test "a refused operation calls no kernel" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          and b = to_count (on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |]) in
          C.reset ();
          invalid ~by:"Nx.add" (fun () -> Nx.add a b);
          equal int 0 (C.calls ()));
      test "a refused operation beside a creation calls no kernel" (fun () ->
          let a = to_count (on_host Plain D.Bool [| 2 |] [| true; false |]) in
          let c = Nx.zeros_like a in
          C.reset ();
          invalid ~by:"Nx.add" (fun () -> Nx.add a c);
          equal int 0 (C.calls ()));
      test "a kind refusing its operands' dtypes raises naming the function"
        (fun () ->
          let a = on_host Plain D.Bool [| 2 |] [| true; false |] in
          raises (Invalid_argument "Nx.add: Add does not take bool, bool")
            (fun () -> Nx.add a a));
      test "an operation computes with its set's kernels" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          ignore (Nx.add a a);
          equal int 1 (C.calls ()));
      test
        "a declined kind raises naming the kernels, kind, dtypes, device and \
         move" (fun () ->
          let a =
            Nx.place Dec.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          raises
            (Invalid_argument
               "Nx.add: nx.test does not compute Add on float32, float32, \
                float32 (m1); place its operands with Nx.place on a set whose \
                kernels compute it") (fun () -> Nx.add a a));
      test "a declined operation raises at its one kernel call" (fun () ->
          let x =
            Nx.place Refuse.on
              (on_host Plain D.Float32 [| 2; 2 |] [| 4.; 3.; 2.; 1. |])
          in
          let declined by kind dts f =
            C.reset ();
            refused_once by kind dts f;
            equal ~msg:(kind ^ ": kernel calls") int 0 (C.calls ())
          in
          declined "Nx.neg" "Neg" "float32, float32" (fun () -> Nx.neg x);
          declined "Nx.add" "Add" "float32, float32, float32" (fun () ->
              Nx.add x x);
          let flag = Nx.equal x x in
          declined "Nx.where" "Where" "float32, bool, float32, float32"
            (fun () -> Nx.where flag x x);
          declined "Nx.sort" "Sort" "float32" (fun () -> fst (Nx.sort x));
          declined "Nx.matmul" "Contract" "float32, float32" (fun () ->
              Nx.matmul x x));
      test "a kind the kernels compute beside a declined one runs" (fun () ->
          let a =
            Nx.place Dec.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          equal (array bits) [| 1.; 4. |] (elements (Nx.mul a a)));
      test "an operation over a split value computes on each device" (fun () ->
          let x =
            Nx.place (Count2.split ~axis:0)
              (on_host Plain D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          C.reset ();
          let y = Nx.add x x in
          equal int 2 (C.calls ());
          equal (array bits) [| 2.; 4.; 6.; 8. |] (elements y));
    ]

let creations =
  group "creations"
    [
      test
        "a creation and an operation compute once, at the call, and a read \
         computes nothing" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          let c = Nx.zeros_at Count.on D.Float32 [| 2 |] in
          equal ~msg:"kernel calls: the fill" int 1 (C.calls ());
          C.reset ();
          ignore (Nx.add a c);
          equal ~msg:"kernel calls: the add alone" int 1 (C.calls ());
          C.reset ();
          let y = Nx.add a c in
          equal ~msg:"kernel calls: the add alone, again" int 1 (C.calls ());
          C.reset ();
          equal (array bits) [| 1.; 2. |] (elements y);
          equal ~msg:"kernel calls: the read" int 0 (C.calls ()));
      test "a scalar beside a value broadcasts as a view" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          let c = Nx.scalar_like a 2. in
          C.reset ();
          let y = Nx.add a c in
          equal ~msg:"kernel calls: the add" int 1 (C.calls ());
          equal (array bits) [| 3.; 4. |] (elements y);
          let w = Nx.where (Nx.less a c) c a in
          equal (array bits) [| 2.; 2. |] (elements w));
      test "a scalar form computes in one kernel call" (fun () ->
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          let y = Nx.add_s a 2. and z = Nx.rsub_s 2. a and l = Nx.less_s a 2. in
          equal ~msg:"kernel calls: one each" int 3 (C.calls ());
          equal (array bits) [| 3.; 4. |] (elements y);
          equal (array bits) [| 1.; 0. |] (elements z);
          equal (array bool) [| true; false |] (elements l));
      test
        "solve and lstsq of a matrix whole on each device factor it once \
         per device" (fun () ->
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
            equal ~msg:(name ^ ": factorizations") int factored
              (Atomic.get Factoring.factored);
            equal ~msg:(name ^ ": bits") (array bits) want y
          in
          (* Beside [b] on two devices, whole or cut, [a] lies whole on
             each, which factors it. A solve permutes [b]'s rows, a gather
             that computes over cut rows, and reads its columns whole, so a
             [b] cut along them raises. *)
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
                at on [m2; m3]; Nx.place (Placement.on s) y")
            (fun () -> solve (a (Nx.beside columns)) columns));
      test "placing a host creation copies its bytes and computes nothing"
        (fun () ->
          let z = Nx.zeros D.Float32 [| 2 |] in
          let c = Nx.add z z in
          C.reset ();
          let y = Nx.place Count.on c in
          equal ~msg:"kernel calls" int 0 (C.calls ());
          equal (array bits) [| 0.; 0. |] (elements y));
      test "a creation of a sub-byte dtype holds its elements" (fun () ->
          let get x = A.to_array (Option.get (Nx.Repr.array x)) in
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
          equal bool true
            (Nx.Placement.equal (Count2.split ~axis:0) (Nx.placement y)));
      test "a rule error in a creation raises at the call" (fun () ->
          invalid ~by:"Nx.zeros" (fun () -> Nx.zeros D.Float32 [| -1 |]));
      test "a decline in a creation raises at the call, naming it" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Nx.zeros_at: nx.test does not compute Fill")
            (fun () -> Nx.zeros_at Dec.on D.Float32 [| 2 |]));
    ]

(* Kills [x]'s memory, as a donation consumes it. *)
let kill x =
  let b = A.buffer (Option.get (Nx.Repr.array x)) in
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"consumed by the test" b))

let test_zeros_like_reads_nothing () =
  let x = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
  kill x;
  invalid ~by:"Nx.add" (fun () -> Nx.add x x);
  C.reset ();
  let z = Nx.zeros_like x in
  equal ~msg:"kernel calls: one fill" int 1 (C.calls ());
  equal (array bits) [| 0.; 0. |] (elements z)

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
  ignore (Nx.Repr.array x);
  equal ~msg:(msg "shards") int
    (List.length (Nx.rigs (Nx.Placement.devices (Nx.placement x))))
    (Iarray.length (Nx.Repr.shards x))

let test_every_placement () =
  let host = on_host Plain D.Float32 [| 2; 2 |] [| 1.; -2.; 3.; -4. |] in
  every_function "host" host;
  every_function "a host creation"
    (Nx.add
       (Nx.zeros D.Float32 [| 2; 2 |])
       (Nx.full D.Float32 [| 2; 2 |] (-1.5)));
  every_function "one memory device" (Nx.place Count.on host);
  every_function "two, whole on each" (Nx.place Count2.on host);
  every_function "two, split" (Nx.place (Count2.split ~axis:0) host);
  every_function "a creation at a split"
    (Nx.full_at (Count2.split ~axis:0) D.Float32 [| 2; 2 |] 2.5)

(* An operation keeps nothing of its caller's arrays: every public function
   taking an int array, over a creation and a value with bytes, gives the
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

exception Failed of int array * float * int32

(* A failing check hands its failure the elements of its data at the first
   failing index, of each datum's dtype, whatever set the data lie on: at the
   call on the host, at the next wait on another set. *)
let test_check_elements () =
  let x = on_host Plain D.Float32 [| 3 |] [| 1.; -1.; -2. |] in
  let check x =
    Nx.Prim.eval ~by:"test"
      (Check
         {
           ok = Nx.less (Nx.zeros_like x) x;
           data = [ x; Nx.cast D.Int32 x ];
           fail = (fun i (v, (n, ())) -> Failed (i, v, n));
         })
  in
  let failed = Failed ([| 1 |], -1., -1l) in
  raises failed (fun () -> check x);
  let off_host x =
    check x;
    raises failed (fun () -> Nx.sync x)
  in
  off_host (Nx.place Count2.on x);
  off_host (Nx.place (Count2.split ~axis:0) x)

exception One of int array * float

(* A check of one element reduces nothing: its flag is read where it lies, on
   its device or on the one of a cut placement that holds it, and the check
   holds the flag's memory until a wait reads it, so a donation of the flag
   computes afresh. *)
let test_check_one () =
  let shape = [| 1; 1 |] in
  let host dt v =
    Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array shape) [| v |])
  in
  let check ok x =
    Nx.Prim.eval ~by:"test"
      (Check { ok; data = [ x ]; fail = (fun i (v, ()) -> One (i, v)) })
  in
  let failed = One ([| 0; 0 |], 5.) in
  let at (type d) (p : d Nx.Placement.t) =
    let flag b = Nx.place p (host D.Bool b) in
    let x = Nx.place p (host D.Float32 5.) in
    let ok = flag true and not_ok = flag false in
    C.reset ();
    check ok x;
    check not_ok x;
    equal ~msg:"kernel calls" int 0 (C.calls ());
    raises failed (fun () -> Nx.sync x)
  in
  at Count.on;
  at (Count2.split ~axis:0);
  at (Count2.split ~axis:1);
  let donated (type d) (p : d Nx.Placement.t) =
    let on dt v = Nx.place p (host dt v) in
    let x = on D.Float32 5. in
    (* Computed, so the memory is the flag's alone and a donation may write
       it. *)
    let ok = Nx.logical_not (on D.Bool true) in
    check ok x;
    ignore (Nx.logical_not (Nx.donate ok));
    raises failed (fun () -> Nx.sync x)
  in
  donated Count.on;
  donated (Count2.split ~axis:0);
  donated (Count2.split ~axis:1);
  let x = Nx.place Count.on (host D.Float32 5.) in
  let flag = Nx.equal_s (Nx.zeros_at Count.on D.Float32 shape) 1. in
  C.reset ();
  check flag x;
  equal ~msg:"a created flag's kernel calls" int 0 (C.calls ());
  raises failed (fun () -> Nx.sync x)

let sets =
  group "sets"
    [
      test "a check's failure receives its data's elements"
        test_check_elements;
      test "a check of one element reduces nothing" test_check_one;
      test "zeros_like reads none of its argument's elements"
        test_zeros_like_reads_nothing;
      test "every function takes a value at any placement" test_every_placement;
      prop "an int-array argument overwritten after the call changes nothing"
        int_arrays law_arguments_owned;
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
    let y = Option.get (Nx.Repr.array (Nx.add x x)) in
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

(* The minor words one call of [f] allocates, to the word: allocation depends on
   the call, not the machine, so a baseline records it. *)
let allocated f = Printf.sprintf "%.0f words" (words f)

let allocation =
  group "allocation"
    [
      test "a sub-byte result leaves its last byte's spare bits zero"
        test_sub_byte_tail;
      test "an add of one-element host values allocates its budget" (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          at_most (float 0.5) ~than:51. (words (fun () -> Nx.add a a)));
      test "an add_s of a one-element host value allocates its budget"
        (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          at_most (float 0.5) ~than:51. (words (fun () -> Nx.add_s a 1.)));
      test "an add of host values performs no effect" (fun () ->
          let a = on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
          equal (array bits) [| 2.; 4.; 6. |]
            (elements (effectless (fun () -> Nx.add a a))));
      test "an add_s of a host value performs no effect" (fun () ->
          let a = on_host Plain D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
          equal (array bits) [| 2.; 3.; 4. |]
            (elements (effectless (fun () -> Nx.add_s a 1.))));
      test "a copy of a one-element host value allocates its baseline"
        (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          expect (allocated (fun () -> Nx.copy a)) @@ __POS_OF__ {| 113 words |});
      test "a matmul of 4x4 float32 host values allocates its baseline"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          expect (allocated (fun () -> Nx.matmul a a))
          @@ __POS_OF__ {| 177 words |});
      test "a slice of a 4x4 host value by ranges allocates its baseline"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          expect (allocated (fun () -> Nx.slice Nx.[ R (1, 3); A ] a))
          @@ __POS_OF__ {| 225 words |});
      test "a set of a row of a 4x4 host value allocates its baseline"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          let v = on_host Plain D.Float32 [| 4 |] (Array.make 4 2.) in
          expect (allocated (fun () -> Nx.set Nx.[ I 1 ] v a))
          @@ __POS_OF__ {| 483 words |});
    ]

(* A scatter's results lie at the union of its operands' cuts, as any
   operation's: positions and updates cut along columns beside a target
   whole on each device put them at that cut, each device scattering its
   columns, with the bits the host gives. *)
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
  equal bool ~msg:"split as its updates" true
    (Nx.Placement.equal split (Nx.placement y));
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

let placements =
  group "placements"
    [
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
         laws;
         broadcasting;
         kernels;
         declines;
         donation;
         creations;
         sets;
         allocation;
         placements;
       ])
