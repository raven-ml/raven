(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Computing through a set's kernels, through Nx: each function against a
   reference over nx.array's elements, refusals before any kernel, declines,
   constants and allocation. *)

open Windtrap
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

  let contract ~order:_ ~batch:_ ~contracting:_ ~segments:_ ~acc:_ ~dst:_
      ~init:_ _ _ =
    decline ()
end

module Refuse = (val Nx.devices ~kernels:(module Refusing) [ m 2 ])
module Refuse2 = (val Nx.devices ~kernels:(module Refusing) [ m 2; m 3 ])

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

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
  | Plain -> A.of_array dt shape data
  | Transposed when r >= 2 && Array.length data > 0 ->
      let rev = Array.init r (fun i -> r - 1 - i) in
      let t = Array.map (fun a -> shape.(a)) rev in
      let base =
        A.of_array dt t
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
          A.set view idx v)
        data;
      view
  | Transposed -> A.of_array dt shape data
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
      if Array.length data = 0 then A.of_array dt shape data
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

(* [data] of shape [s] on the host, each NaN made signalling where it stays
   a NaN ({!Nx_support.signalled}): an OCaml float holds a NaN quiet. *)
let signalling (type v s) (dt : (v, s) D.t) s (data : v array) =
  Nx.Repr.of_array Nx.Host.v (Nx_support.signalled (A.of_array dt s data))

(* Whether [data] of [dt] holds a NaN {!signalling} makes signalling. *)
let signals (type v s) (dt : (v, s) D.t) (data : v array) =
  match D.kind dt with
  | D.Float | D.Complex when D.bits dt >= 8 ->
      let bytes a = A.to_array (Option.get (A.bitcast D.Uint8 a)) in
      let a = Nx_support.signalled (A.of_array dt [| Array.length data |] data) in
      bytes a <> bytes (Nx_support.quieted a)
  | _ -> false

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
   operation: the general path, which no fast path takes. *)
let run_map (type w r) node ins (dt : (w, r) D.t) shape loads =
  let prog = Nx_kernel.Prog.of_node ~ins node in
  let op : ((w, r, Nx.host) Nx.t * unit) Nx.Prim.t =
    Map { layout = A.Layout.contiguous shape; prog; outs = [ dt ]; loads }
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

(* The bytes [d] holds after four rounds of collection, each of which returns
   to the device what it collected. *)
let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create d 0);
    Rig.free_cache d
  done;
  Rig.allocated d

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
      test "a declined copy raises once" (fun () ->
          let x = refused [| 4.; 3.; 2.; 1. |] in
          refused_once "Nx.reshape" "Copy" "float32" (fun () ->
              Nx.reshape [| 4 |] (Nx.transpose x)));
      test "a declined operation kills no donated operand" (fun () ->
          (* [x] borrows the host's memory, which no result takes: a sort
             computes on the general path, a sum and a comparison on the fast
             one. *)
          let data = [| 4.; 3.; 2.; 1. |] in
          let x = refused data and y = refused data in
          let before = settled (m 2) in
          let kept by path f =
            invalid ~by (fun () -> ignore (f (Nx.donate x)));
            equal ~msg:path (array bits) data (elements x)
          in
          kept "Nx.sort" "a sort" (fun h -> fst (Nx.sort h));
          kept "Nx.add" "a sum into fresh memory" (fun h -> Nx.add h y);
          kept "Nx.less" "a comparison, of another dtype" (fun h ->
              Nx.less h y);
          equal ~msg:"bytes held" int before (settled (m 2)));
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

  let contract ~order ~batch ~contracting ~segments ~acc ~dst ~init a b =
    first ();
    C.contract ~order ~batch ~contracting ~segments ~acc ~dst ~init a b

  let map (prog : Nx_kernel.Prog.t) ~dsts loads =
    first ();
    if Iarray.exists declined prog.nodes then A.Declined
    else C.map prog ~dsts loads
end

module Gate = (val Nx.devices ~kernels:(module Gated) [ m 1 ])

(* [f ()] with [g] run on another domain inside its first call into nx.gated:
   after [f]'s operation checked its operands, before its kernel reads them. *)
let meanwhile g f =
  let other () = Domain.join (Domain.spawn g) in
  Domain.DLS.set Gated.pending (Some other);
  Fun.protect ~finally:(fun () -> Domain.DLS.set Gated.pending None) f

(* Donations of one value on nx.gated: handles, views and copies of handles,
   sums and comparisons that decline, on the fast path and off it, and a
   product and a sum that consume. The value and every handle of it die at the
   first consumer, named by it. A sum's decline may kill them, naming the sum,
   where its result would take a handle's memory: a copy's, or the value's
   while no value outside its handles shares it. Any other decline kills
   nothing: a comparison's result has another dtype. A movement kills nothing:
   the handle it read and the one it passed on die with their donor. A call
   made on another domain while a decline holds a value's handles is held there
   deterministically ([meanwhile]). *)
type donor = {
  xs : float array;
  mutable dead : string option list;  (* Every death it may have. *)
  mutable shared : bool;  (* Whether a read borrowed its memory. *)
}

(* A handle reads its donor's elements in [order], over its donor's memory or,
   [copied], a copy of it. It lives while its donor does. *)
type handle = { x : donor; order : int array; copied : bool }

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
let make_ref xs = { xs; dead = [ None ]; shared = false }

(* What a refusal says: a decline, or the function a value went to. *)
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
  let mark = "was donated to " in
  match find mark 0 with
  | Some i ->
      let i = i + String.length mark in
      String.sub m i (String.length m - i)
  | None ->
      if Option.is_some (find "does not compute" 0) then "declined"
      else "refused: " ^ m

let outcome f =
  match f () with ys -> Ok ys | exception Invalid_argument m -> Error (named m)

let outcomes = result (array float_exact) string

(* The deaths [h] may have: its donor's. *)
let deaths h = h.x.dead

let elements_of h = Array.map (fun i -> h.x.xs.(i)) h.order

(* [h] and its donor known to live. *)
let lives h =
  if not (List.mem None (deaths h)) then fail "a dead handle lives";
  h.x.dead <- [ None ]

(* The donor [x] seen dead to [f], or alive. *)
let seen x = function
  | None ->
      if not (List.mem None x.dead) then fail "a dead donor lives";
      x.dead <- [ None ]
  | Some f ->
      if not (List.mem (Some f) x.dead) then fail ("a donor died to " ^ f);
      x.dead <- [ Some f ]

(* A call [by] on [h] that raised naming [e], killing nothing. On two domains
   ([raced]) the donor may be held by a decline in flight, which then gives it
   back: a consumer of one handle claims nothing else, so it loses no later
   word. *)
let lost ~raced by e h =
  let ds = deaths h in
  let gives_back = List.mem e [ "Nx.add"; "Nx.less" ] in
  if not (raced && List.mem None ds && gives_back) then begin
    if not (List.mem (Some e) ds) then fail (by ^ " raised naming " ^ e);
    h.x.dead <- [ Some e ]
  end

let donate_ref x =
  if not (List.mem None x.dead) then invalid_arg "dead";
  { x; order = [| 0; 1; 2; 3 |]; copied = false }

(* A movement passes on a handle reading [order] of [h], killing nothing. *)
let moved ~copied order h =
  if deaths h <> [ None ] then invalid_arg "dead";
  {
    x = h.x;
    order = Array.map (fun i -> h.order.(i)) order;
    copied = copied || h.copied;
  }

let view_ref = moved ~copied:false [| 0; 1; 2; 3 |]
let view_sys h = Nx.reshape [| 4 |] (Nx.reshape [| 2; 2 |] h)

(* A reshape of a transpose copies. *)
let copy_ref = moved ~copied:true [| 0; 2; 1; 3 |]
let copy_sys h = Nx.reshape [| 4 |] (Nx.transpose (Nx.reshape [| 2; 2 |] h))

(* Whether a sum's decline on [h] may have killed it: its result could take
   [h]'s memory. *)
let in_place h = h.copied || not h.x.shared

(* A decline by [by] of [h], then [h] seen dead or alive ([after]): a sum's
   ([~sum]) may kill [h] and its donor, any other kills nothing. *)
let declined ~sum by h = function
  | Error e -> raise e
  | Ok (Ok _, _) -> fail (by ^ " computed what its kernels decline")
  | Ok (Error "declined", after) -> (
      lives h;
      match after with
      | None -> ()
      | Some f ->
          if not (sum && f = by && in_place h) then
            fail (by ^ " killed a donor it computed apart from");
          cover "a sum's decline kills its donor" true;
          h.x.dead <- [ Some by ])
  | Ok (Error e, after) -> (
      lost ~raced:false by e h;
      match (after, deaths h) with
      | Some f, [ Some f' ] when f = f' -> ()
      | _ -> fail (by ^ ": a dead handle seen otherwise after"))

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

(* Beside a constant, off the fast path. *)
let add_off_sys = then_seen (fun h -> Nx.add h (Nx.scalar Nx.float32 2.))

let less_off_sys =
  then_seen (fun h -> Nx.cast D.Float32 (Nx.less h (Nx.scalar Nx.float32 2.)))

(* A kind refusing [h]'s dtype, on the fast path, where its result would fit
   [h]'s memory: it kills nothing. *)
let and_ref h = function
  | Error e -> raise e
  | Ok (Ok _, _) -> fail "Nx.bitwise_and computed floats"
  | Ok (Error e, after) -> (
      equal string "refused: Nx.bitwise_and: And does not take float32, float32"
        e;
      cover "a live handle refused by dtype" (deaths h = [ None ]);
      match (deaths h, after) with
      | [ None ], None -> ()
      | [ Some f ], Some f' when f = f' -> ()
      | _ -> fail "Nx.bitwise_and: a refusal changed a handle's fate")

let and_sys = then_seen (fun h -> Nx.bitwise_and h (twos ()))

(* Broadcast beside wider twos: no result fits the handle's memory. *)
let add_wide_sys = then_seen (fun h -> Nx.add h (wide_twos ()))

let less_wide_sys =
  then_seen (fun h -> Nx.cast D.Float32 (Nx.less h (wide_twos ())))

let and_wide_sys = then_seen (fun h -> Nx.bitwise_and h (wide_twos ()))

(* Broadcast, a dead handle raises naming its consumer before the kind's
   refusal: the call checks its operands before it broadcasts them. *)
let and_wide_ref h = function
  | Ok (Error e, _) when deaths h <> [ None ] ->
      lost ~raced:false "Nx.bitwise_and" e h
  | out -> and_ref h out

(* A consumer [by] of [h] whose result's elements are [f]'s. *)
let consumer ~raced by f h = function
  | Error e -> raise e
  | Ok (Ok ys) ->
      lives h;
      equal (array float_exact) (f (elements_of h)) ys;
      h.x.dead <- [ Some by ]
  | Ok (Error e) -> lost ~raced by e h

let mul_ref ~raced = consumer ~raced "Nx.mul" (Array.map (fun v -> v *. 2.))
let mul_sys h = outcome (fun () -> elements (Nx.mul h (twos ())))

let mul_wide_ref ~raced =
  consumer ~raced "Nx.mul" (fun ys ->
      let doubled = Array.map (fun v -> v *. 2.) ys in
      Array.append doubled doubled)

let mul_wide_sys h = outcome (fun () -> elements (Nx.mul h (wide_twos ())))

let sum_ref ~raced =
  consumer ~raced "Nx.sum" (fun ys -> [| Array.fold_left ( +. ) 0. ys |])

let sum_sys h = outcome (fun () -> elements (Nx.sum h))

(* A read borrows the donor's memory for good. *)
let read_ref x = function
  | Error e -> raise e
  | Ok (Ok ys) ->
      seen x None;
      equal outcomes (Ok x.xs) (Ok ys);
      x.shared <- true
  | Ok (Error f) -> seen x (Some f)

let read_sys x = outcome (fun () -> elements x)

(* A decline [by] of [h] that runs [inner] on another domain meanwhile, whose
   outcome [met] is [None] where the decline raised before its kernel. While
   the decline holds [h] and its donor they are dead to it, so a call meanwhile
   that reads them loses to it; then the decline kills nothing, or, a sum's in
   a handle's memory, may kill them. *)
let during ~sum by inner h = function
  | Error e -> raise e
  | Ok (out, met) -> (
      match (deaths h, met) with
      | [ None ], Some r ->
          let dead = h.x.dead in
          h.x.dead <- [ Some by ];
          inner r;
          h.x.dead <- dead;
          declined ~sum by h (Ok out)
      | [ None ], None -> fail (by ^ ": nothing ran meanwhile")
      | _, None -> declined ~sum by h (Ok out)
      | _, Some _ -> fail (by ^ " ran a call meanwhile though refused"))

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
      during ~sum:true "Nx.add"
        (fun r ->
          lost_in_flight "Nx.add" r;
          mul_ref ~raced:false h' (Ok r))
        h)
    (fun h h' -> while_sys add_sys (fun () -> mul_sys h') h)

let less_while_read =
  command "less, meanwhile read"
    (handle ^-> donor ^-> judges (pair seen_outcomes (option outcomes)))
    (fun h x ->
      during ~sum:false "Nx.less"
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
      (declined ~sum:true "Nx.add")
      add_sys;
    command "add off the fast path"
      (handle ^-> judges seen_outcomes)
      (declined ~sum:true "Nx.add")
      add_off_sys;
    command "less"
      (handle ^-> judges seen_outcomes)
      (declined ~sum:false "Nx.less")
      less_sys;
    command "less off the fast path"
      (handle ^-> judges seen_outcomes)
      (declined ~sum:false "Nx.less")
      less_off_sys;
    command "bitwise_and" (handle ^-> judges seen_outcomes) and_ref and_sys;
    command "add broadcast"
      (handle ^-> judges seen_outcomes)
      (declined ~sum:false "Nx.add")
      add_wide_sys;
    command "less broadcast"
      (handle ^-> judges seen_outcomes)
      (declined ~sum:false "Nx.less")
      less_wide_sys;
    command "bitwise_and broadcast"
      (handle ^-> judges seen_outcomes)
      and_wide_ref and_wide_sys;
    command "mul" (handle ^-> judges outcomes) (mul_ref ~raced:false) mul_sys;
    command "mul broadcast"
      (handle ^-> judges outcomes)
      (mul_wide_ref ~raced:false) mul_wide_sys;
    command "sum" (handle ^-> judges outcomes) (sum_ref ~raced:false) sum_sys;
    command "read" (donor ^-> judges outcomes) read_ref read_sys;
    add_while_mul;
    less_while_read;
  ]

(* On two domains a sum's decline may or may not kill its donor, as in
   sequence, and is not seen after. *)
let raced_decline ~sum by h = function
  | Error e -> raise e
  | Ok (Ok _) -> fail (by ^ " computed what its kernels decline")
  | Ok (Error "declined") ->
      lives h;
      if sum then h.x.dead <- [ None; Some by ]
  | Ok (Error e) -> lost ~raced:true by e h

let raced_sys f h = outcome (fun () -> elements (f h))

(* A movement on two domains: it reads [h] live, killing nothing, or raises
   naming the consumer that held its donor. *)
let raced_moved h = function
  | Error e -> raise e
  | Ok (Ok _) -> lives h
  | Ok (Error e) -> lost ~raced:true "Nx.reshape" e h

(* The movement's handle, which no command reads. *)
let moved_sys f h =
  outcome (fun () ->
      ignore (f h);
      [||])

(* A read may see the donor dead to a decline in flight, which then gives it
   back. *)
let read_judge x = function
  | Ok (Error ("Nx.add" | "Nx.less")) when List.mem None x.dead -> ()
  | out -> read_ref x out

(* A donation and a movement are drawn where the fate is known. *)
let known h = List.length (deaths h) = 1
let known_donor x = List.length x.dead = 1

let raced =
  [
    command "make" (fours @-> makes donor) make_ref make_sys;
    command ~pre:known_donor "donate" (donor ^-> makes handle) donate_ref
      Nx.donate;
    command ~pre:known "view" (handle ^-> makes handle) view_ref view_sys;
    command ~pre:known "copy" (handle ^-> makes handle) copy_ref copy_sys;
    command "view racing" (handle ^-> judges outcomes) raced_moved
      (moved_sys view_sys);
    command "copy racing" (handle ^-> judges outcomes) raced_moved
      (moved_sys copy_sys);
    command "add"
      (handle ^-> judges outcomes)
      (raced_decline ~sum:true "Nx.add")
      (raced_sys (fun h -> Nx.add h (twos ())));
    command "add off the fast path"
      (handle ^-> judges outcomes)
      (raced_decline ~sum:true "Nx.add")
      (raced_sys (fun h -> Nx.add h (Nx.scalar Nx.float32 2.)));
    command "less"
      (handle ^-> judges outcomes)
      (raced_decline ~sum:false "Nx.less")
      (raced_sys (fun h -> Nx.cast D.Float32 (Nx.less h (twos ()))));
    command "less off the fast path"
      (handle ^-> judges outcomes)
      (raced_decline ~sum:false "Nx.less")
      (raced_sys (fun h ->
           Nx.cast D.Float32 (Nx.less h (Nx.scalar Nx.float32 2.))));
    command "mul" (handle ^-> judges outcomes) (mul_ref ~raced:true) mul_sys;
    command "sum" (handle ^-> judges outcomes) (sum_ref ~raced:true) sum_sys;
    command "read" (donor ^-> judges outcomes) read_judge read_sys;
  ]

(* A call that checked [x] live, and whose kernel then meets [x]'s memory
   consumed by an operation on another domain, raises naming that operation,
   on the fast path, off it, and through a movement's copy of a handle. *)
let test_raced_read_names () =
  let consume h = ignore (Nx.mul h (twos ())) in
  let raced by f =
    let x = make_sys [| 1.; 2.; 3.; 4. |] in
    let h = Nx.donate x in
    raises
      (Invalid_argument (by ^ ": operand 1 was donated to Nx.mul"))
      (fun () -> meanwhile (fun () -> consume h) (fun () -> ignore (f x)))
  in
  (* An offered donor is read off the fast path. *)
  raced "Nx.neg" Nx.neg;
  raced "Nx.mul" (fun x -> Nx.mul x (twos ()));
  raced "Nx.reshape" (fun x -> copy_sys (Nx.donate x));
  let x = make_sys [| 1.; 2.; 3.; 4. |] in
  raises
    (Invalid_argument "Nx.neg: operand 1 was donated to Nx.mul")
    (fun () ->
      meanwhile (fun () -> consume (Nx.donate x)) (fun () -> ignore (Nx.neg x)))

(* A call on live values, one of which an operation on another domain donates
   and consumes after the call checked it and before its kernel reads it,
   raises naming that operation: on each fast path, beside a handle, and in a
   contraction. *)
let raced_live =
  let consume x = ignore (Nx.mul (Nx.donate x) (twos ())) in
  let fresh () = make_sys [| 1.; 2.; 3.; 4. |] in
  (* [f c h t x]: a mask [c], a handle [h], twos [t], made before the call. *)
  let case name by i f =
    test name (fun () ->
        let c = Nx.equal (twos ()) (on_gate [| 2.; 0.; 2.; 0. |]) in
        let h = Nx.donate (fresh ()) and t = twos () and x = fresh () in
        raises
          (Invalid_argument
             (Printf.sprintf "%s: operand %d was donated to Nx.mul" by i))
          (fun () ->
            meanwhile (fun () -> consume x) (fun () -> ignore (f c h t x))))
  in
  [
    case "a product" "Nx.mul" 1 (fun _ _ t x -> Nx.mul x t);
    case "a product beside a handle" "Nx.mul" 2 (fun _ h _ x -> Nx.mul h x);
    case "a selection" "Nx.where" 2 (fun c _ t x -> Nx.where c x t);
    case "a selection beside a handle" "Nx.where" 3 (fun c h _ x ->
        Nx.where c h x);
    case "a contraction" "Nx.contract" 1 (fun _ _ t x ->
        Nx.contract Nx.float32 (Nx.Pattern.v "i, i -> | i") x t);
  ]

(* A call that broadcasts a handle, or loses another to a consumer on another
   domain, kills nothing: the broadcast passed on a new handle of the donor,
   and the loser gave back what it claimed. *)
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
        lives "after the decline" x h);
    test "a product losing its second handle" (fun () ->
        (* [x] is donated first: its key comes first, so the product claims
           [hx] before it meets [y] held. *)
        let x = computed [| 4 |] xs in
        let hx = Nx.donate x in
        let y = computed [| 4 |] xs in
        let hy = Nx.donate y and hy' = Nx.donate y in
        equal outcomes (Error "Nx.add")
          (during_add hy (fun () -> elements (Nx.mul hx hy')));
        lives "after the loss" x hx);
    test "a composite losing its second handle at its claim" (fun () ->
        (* [Nx.vdot] claims after its operations, the first of which runs the
           product consuming [y] on the other domain: the claim takes [hx],
           then loses [hy'] and gives [hx] back. *)
        let x = computed [| 4 |] xs in
        let hx = Nx.donate x in
        let y = computed [| 4 |] xs in
        let hy = Nx.donate y and hy' = Nx.donate y in
        let t = at [| 4 |] (Array.make 4 2.) in
        let met = ref None in
        raises
          (Invalid_argument "Nx.vdot: operand 2 was donated to Nx.mul")
          (fun () ->
            meanwhile
              (fun () -> met := Some (outcome (fun () -> elements (Nx.mul hy t))))
              (fun () -> Nx.vdot hx hy'));
        equal ~msg:"the product" (option outcomes)
          (Some (Ok [| 2.; 4.; 6.; 8. |]))
          !met;
        lives "after the loss" x hx);
    test "a composite reading the donor beside a copy of its handle"
      (fun () ->
        (* [h2], [h] copied by a movement, holds [x]'s key, and
           [Nx.array_equal] reads [x] beside it. A consumer of [h] on the
           other domain inside the comparison's kernel computes into fresh
           memory, so the comparison reads [x]'s elements, and the composite
           then raises as a loser. *)
        let xs = [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let x = computed [| 2; 3 |] xs in
        let h = Nx.donate x in
        let h2 = Nx.reshape [| 2; 3 |] (Nx.transpose h) in
        let met = ref None in
        raises
          (Invalid_argument "Nx.array_equal: operand 1 was donated to Nx.neg")
          (fun () ->
            meanwhile
              (fun () -> met := Some (outcome (fun () -> elements (Nx.neg h))))
              (fun () -> Nx.array_equal h2 x));
        equal ~msg:"the consumer" (option outcomes)
          (Some (Ok (Array.map Float.neg xs)))
          !met;
        raises
          (Invalid_argument "Nx.copy: operand 1 was donated to Nx.neg")
          (fun () -> Nx.copy x));
  ]

let donation =
  group "donation"
    [
      stateful "a value lives until a handle of it is consumed" donations;
      test "a read racing a consumer names it" test_raced_read_names;
      group "a live operand racing its consumer names it" raced_live;
      group "a refused or losing call kills nothing" lost_kills_none;
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
      test "a refused operation beside a constant calls no kernel" (fun () ->
          let a = to_count (on_host Plain D.Bool [| 2 |] [| true; false |]) in
          let c = Nx.zeros D.Bool [| 2 |] in
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

let constants =
  group "constants"
    [
      test "a constant takes the brand of what it meets" (fun () ->
          let c = Nx.scalar D.Float32 2. in
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          let h = on_host Plain D.Float32 [| 2 |] [| 1.; 2. |] in
          equal (array bits) [| 3.; 4. |] (elements (Nx.add a c));
          equal (array bits) [| 2.; 4. |] (elements (Nx.mul h c)));
      test "a constant computes on its set once, at its first use there"
        (fun () ->
          let c = Nx.zeros D.Float32 [| 2 |] in
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          C.reset ();
          ignore (Nx.add a c);
          let first = C.calls () in
          C.reset ();
          ignore (Nx.add a c);
          equal int 2 first;
          equal int 1 (C.calls ()));
      test "a constant read through a movement computes once, at its first use"
        (fun () ->
          let c = Nx.scalar D.Float32 2. in
          let a = to_count (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |]) in
          ignore (Nx.add a c);
          C.reset ();
          let y = Nx.add a c in
          equal ~msg:"kernel calls: the add alone" int 1 (C.calls ());
          equal (array bits) [| 3.; 4. |] (elements y);
          let b = Nx.less a c in
          C.reset ();
          let w = Nx.where b c a in
          equal ~msg:"kernel calls: the where alone" int 1 (C.calls ());
          equal (array bits) [| 2.; 2. |] (elements w));
      test "placing a constant computes it at the placement, each time"
        (fun () ->
          let z = Nx.zeros D.Float32 [| 2 |] in
          let c = Nx.add z z in
          equal (array bits) [| 0.; 0. |] (elements (Nx.place Nx.Host.on c));
          C.reset ();
          let y = Nx.place Count.on c in
          equal ~msg:"kernel calls: the zeros, then the add" int 2 (C.calls ());
          ignore (Nx.place Count.on c);
          equal ~msg:"kernel calls: both again, into memory of its own" int 4
            (C.calls ());
          equal (array bits) [| 0.; 0. |] (elements y));
      test "a constant of a sub-byte dtype computes on the host" (fun () ->
          let host x = Nx.place Nx.Host.on x in
          let get x = A.to_array (Option.get (Nx.Repr.array (host x))) in
          equal ~msg:"int4" (array int) [| -3; -3; -3 |]
            (get (Nx.add (Nx.zeros D.Int4 [| 3 |]) (Nx.scalar D.Int4 (-3))));
          equal ~msg:"uint4" (array int) [| 0; 0 |]
            (get (Nx.zeros D.Uint4 [| 2 |]));
          equal ~msg:"float4_e2m1fn" (array float_exact) [| 1.5 |]
            (get (Nx.reshape [| 1 |] (Nx.scalar D.Float4_e2m1fn 1.5)));
          equal ~msg:"bit" (array bool) [| true |]
            (get (Nx.reshape [| 1 |] (Nx.scalar D.Bit true))));
      test "a constant beside a split value is split with it" (fun () ->
          let x =
            Nx.place (Count2.split ~axis:0)
              (on_host Plain D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          let y = Nx.add x (Nx.zeros D.Float32 [| 4 |]) in
          equal bool true
            (Nx.Placement.equal (Count2.split ~axis:0)
               (Option.get (Nx.placement y))));
      test "a rule error in a constant raises at the call" (fun () ->
          invalid ~by:"Nx.zeros" (fun () -> Nx.zeros D.Float32 [| -1 |]));
      test "a decline in a constant raises at its first use, naming its maker"
        (fun () ->
          let c = Nx.zeros D.Float32 [| 2 |] in
          let a =
            Nx.place Dec.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          raises_match
            (Exn.invalid_arg
               ~substring:"Nx.zeros: nx.test does not compute Fill") (fun () ->
              Nx.mul a c));
      test "zeros_like is zeros where its argument lies" (fun () ->
          let x =
            Nx.place (Count2.split ~axis:0)
              (on_host Plain D.Float32 [| 4 |] [| 1.; 2.; 3.; 4. |])
          in
          let z = Nx.zeros_like x in
          equal bool true
            (Nx.Placement.equal
               (Option.get (Nx.placement x))
               (Option.get (Nx.placement z)));
          equal (array bits) [| 0.; 0.; 0.; 0. |] (elements z));
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

(* Every public function takes a value of every set, a constant included, and
   gives what it gives on the host. *)
let every_function (type d) ?(every_set = false) name
    (x : (float, D.float32_elt, d) Nx.t) =
  let msg f = name ^ ": " ^ f in
  let xs = elements x in
  let ys f = Array.map f xs in
  equal ~msg:(msg "shape") (array int) [| 2; 2 |] (Nx.shape x);
  equal ~msg:(msg "dtype") string "float32" (D.name (Nx.dtype x));
  equal ~msg:(msg "placement") bool every_set (Nx.placement x = None);
  Option.iter
    (fun p ->
      equal ~msg:(msg "place") (array bits) xs (elements (Nx.place p x)))
    (Nx.placement x);
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
    (elements (Nx.add x (Nx.scalar D.Float32 1.)));
  let less = Nx.less x (Nx.zeros_like x) in
  equal ~msg:(msg "less") (array bool) (ys (fun v -> v < 0.)) (elements less);
  equal ~msg:(msg "where") (array bits)
    (ys (fun v -> if v < 0. then v else 0.))
    (elements (Nx.where less x (Nx.zeros_like x)));
  equal ~msg:(msg "cast") (array bits) xs (elements (Nx.cast D.Float64 x));
  equal ~msg:(msg "reshape") (array bits) xs (elements (Nx.reshape [| 4 |] x));
  equal ~msg:(msg "copy") (array bits) xs (elements (Nx.copy x));
  ignore (Nx.Repr.array x);
  equal ~msg:(msg "shards") bool (not every_set) (Nx.Repr.shards x <> None)

let test_every_set () =
  let host = on_host Plain D.Float32 [| 2; 2 |] [| 1.; -2.; 3.; -4. |] in
  every_function "host" host;
  every_function "one memory device" (Nx.place Count.on host);
  every_function "two, whole on each" (Nx.place Count2.on host);
  every_function "two, split" (Nx.place (Count2.split ~axis:0) host);
  every_function ~every_set:true "a value of every set"
    (Nx.add (Nx.zeros D.Float32 [| 2; 2 |]) (Nx.scalar D.Float32 (-1.5)))

(* An operation keeps nothing of its caller's arrays: overwriting an int-array
   argument after the call changes neither the value's shape nor its elements
   (the external review's program, for zeros). *)
let test_zeros_shape_owned () =
  let shape = [| 2 |] in
  let x = Nx.zeros D.Float32 shape in
  shape.(0) <- 3;
  equal ~msg:"shape" (array int) [| 2 |] (Nx.shape x);
  equal ~msg:"placed" (array int) [| 2 |] (Nx.shape (Nx.place Nx.Host.on x));
  equal ~msg:"read" (array bits) [| 0.; 0. |]
    (elements (Nx.add x (Nx.place Nx.Host.on (Nx.zeros D.Float32 [| 2 |]))))

(* Every public function taking an int array, over a value of every set and a
   value with bytes, the argument overwritten after the call. *)
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
  call "reshape of a formula" (fun a ->
      Nx.reshape a
        (Nx.add (Nx.zeros D.Float32 [| n |]) (Nx.scalar D.Float32 1.)));
  call "reshape" (fun a -> Nx.reshape a host)

let int_arrays =
  Gen.pair
    (Gen.array ~size:(Gen.int_range 1 3) (Gen.int_range (-2) 5))
    (Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 3))

(* A placed value of every set is memory of its own: writing it reaches neither
   another placement of the formula nor a later operation on it. *)
let test_placed_owns () =
  let c = Nx.add (Nx.zeros D.Float32 [| 2 |]) (Nx.scalar D.Float32 1.) in
  let first = Option.get (Nx.Repr.array (Nx.place Nx.Host.on c)) in
  let second = Option.get (Nx.Repr.array (Nx.place Nx.Host.on c)) in
  equal ~msg:"two placements share no memory" bool false
    (Rig.Buffer.overlaps (A.buffer first) (A.buffer second));
  A.set first [| 0 |] 42.;
  equal ~msg:"a later use" (array bits) [| 2.; 2. |] (elements (Nx.add c c))

(* A movement of a formula placed after an operation read the formula: memory of
   its own, which the formula's later uses never see. *)
let test_placed_view_owns () =
  let c = Nx.add (Nx.zeros D.Float32 [| 2 |]) (Nx.scalar D.Float32 1.) in
  let h = on_host Plain D.Float32 [| 2 |] [| 1.; 1. |] in
  equal ~msg:"a first use" (array bits) [| 2.; 2. |] (elements (Nx.add h c));
  let v =
    Option.get (Nx.Repr.array (Nx.place Nx.Host.on (Nx.reshape [| 2; 1 |] c)))
  in
  A.set v [| 0; 0 |] 42.;
  equal ~msg:"a later use" (array bits) [| 2.; 2. |] (elements (Nx.add h c))

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
   computes afresh. A flag of every set is computed there first, as alone. *)
let test_check_one () =
  let shape = [| 1; 1 |] in
  let check ok x =
    Nx.Prim.eval ~by:"test"
      (Check { ok; data = [ x ]; fail = (fun i (v, ()) -> One (i, v)) })
  in
  let failed = One ([| 0; 0 |], 5.) in
  let at (type d) (p : d Nx.Placement.t) =
    let flag b =
      Nx.place p (Nx.Repr.of_array Nx.Host.v (A.of_array D.Bool shape [| b |]))
    in
    let x =
      Nx.place p
        (Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 shape [| 5. |]))
    in
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
    let on dt v =
      Nx.place p (Nx.Repr.of_array Nx.Host.v (A.of_array dt shape [| v |]))
    in
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
  let on_count dt v =
    Nx.place Count.on (Nx.Repr.of_array Nx.Host.v (A.of_array dt shape [| v |]))
  in
  let x = on_count D.Float32 5. in
  let formula () = Nx.equal_s (Nx.zeros D.Float32 shape) 1. in
  C.reset ();
  ignore (Nx.place Count.on (formula ()));
  let alone = C.calls () in
  C.reset ();
  check (formula ()) x;
  equal ~msg:"a formula flag's kernel calls" int alone (C.calls ());
  raises failed (fun () -> Nx.sync x)

let sets =
  group "sets"
    [
      test "a check's failure receives its data's elements"
        test_check_elements;
      test "a check of one element reduces nothing" test_check_one;
      test "zeros_like reads none of its argument's elements"
        test_zeros_like_reads_nothing;
      test "every function takes a value of every set" test_every_set;
      test "zeros keeps nothing of its shape argument" test_zeros_shape_owned;
      test "a placed formula is memory of its own" test_placed_owns;
      test "a placed movement of a formula is memory of its own"
        test_placed_view_owns;
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

(* The minor words one call of [f] allocates, over a thousand after one. *)
let words f =
  ignore (f ());
  let n = 1000 in
  let before = Gc.minor_words () in
  for _ = 1 to n do
    ignore (Sys.opaque_identity (f ()))
  done;
  (Gc.minor_words () -. before) /. Float.of_int n

let allocation =
  group "allocation"
    [
      test "a sub-byte result leaves its last byte's spare bits zero"
        test_sub_byte_tail;
      test "an add of one-element host values allocates its budget" (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          at_most (float 0.5) ~than:51. (words (fun () -> Nx.add a a)));
      test "a copy of a one-element host value allocates its budget"
        (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          at_most (float 0.5) ~than:130. (words (fun () -> Nx.copy a)));
      test "a matmul of 4x4 float32 host values allocates its budget"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          at_most (float 0.5) ~than:179. (words (fun () -> Nx.matmul a a)));
      test "a slice of a 4x4 host value by ranges allocates its budget"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          at_most (float 0.5) ~than:294.
            (words (fun () -> Nx.slice Nx.[ R (1, 3); A ] a)));
      test "a set of a row of a 4x4 host value allocates its budget"
        (fun () ->
          let a = on_host Plain D.Float32 [| 4; 4 |] (Array.make 16 1.) in
          let v = on_host Plain D.Float32 [| 4 |] (Array.make 4 2.) in
          at_most (float 0.5) ~than:599.
            (words (fun () -> Nx.set Nx.[ I 1 ] v a)));
    ]

(* Placements *)

module Set3 = (val Nx.devices [ m 0; m 1; m 2 ])
module Set4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let mesh4 = Nx.Mesh.v Set4.v [ ("a", 2); ("b", 2) ]

(* Where a value lies: split over three or four devices; a 2 x 2 mesh cutting
   one axis over both its axes or two axes over one each; or one axis over its
   minor axis, the major holding copies. *)
type where = Split3 of int | Split4 of int | Cuts of int * int | Minor of int

let pp_where ppf = function
  | Split3 a -> Format.fprintf ppf "split %d over 3" a
  | Split4 a -> Format.fprintf ppf "split %d over 4" a
  | Cuts (x, y) when x = y -> Format.fprintf ppf "cut %d over 2x2" x
  | Cuts (x, y) -> Format.fprintf ppf "cuts %d/a %d/b" x y
  | Minor x -> Format.fprintf ppf "cut %d/b" x

(* The axes [w] cuts, with their tiles. *)
let tiles = function
  | Split3 a -> [ (a, 3) ]
  | Split4 a -> [ (a, 4) ]
  | Cuts (x, y) when x = y -> [ (x, 4) ]
  | Cuts (x, y) -> [ (x, 2); (y, 2) ]
  | Minor x -> [ (x, 2) ]

let where_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.one_of
    [
      Gen.map (fun a -> Split3 a) axis;
      Gen.map (fun a -> Split4 a) axis;
      Gen.map (fun (x, y) -> Cuts (x, y)) (Gen.pair axis axis);
      Gen.map (fun x -> Minor x) axis;
    ]

(* A dtype, the elements drawn for it, how its elements' bits print, and a
   map and a fold it takes. *)
type ('v, 's) kind = {
  dtype : ('v, 's) D.t;
  values : 'v list;
  show : 'v -> string;
  combine : 'd. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t;
  fold : 'd. int -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t;
}

type any_kind = Kind : ('v, 's) kind -> any_kind

let float_bits v = Printf.sprintf "%Lx" (Int64.bits_of_float v)

(* Floats whose sums depend on their association, signed zeros, infinities and
   NaNs of both signs and two payloads. *)
let floats =
  [
    1e7; -1e7; 1.; 0.1; 3.3; 0.; -0.; infinity; neg_infinity; nan; -.nan;
    Int32.float_of_bits 0x7fc00123l; Int32.float_of_bits 0xffa00001l;
  ]

let float_kind dtype =
  Kind
    {
      dtype;
      values = floats;
      show = float_bits;
      combine = Nx.add;
      fold = (fun a x -> Nx.sum ~axes:[ a ] x);
    }

let kinds =
  [
    float_kind D.Float32;
    float_kind D.Float16;
    float_kind D.Bfloat16;
    Kind
      {
        dtype = D.Complex64;
        values =
          List.map2
            (fun re im -> { Complex.re; im })
            floats (List.rev floats);
        show = (fun c -> float_bits c.Complex.re ^ "," ^ float_bits c.im);
        combine = Nx.add;
        fold = (fun a x -> Nx.sum ~axes:[ a ] x);
      };
    Kind
      {
        dtype = D.Int32;
        values = [ 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int ];
        show = Int32.to_string;
        combine = Nx.add;
        fold = (fun a x -> Nx.sum ~axes:[ a ] x);
      };
    Kind
      {
        dtype = D.Int8;
        values = [ 0; 1; -1; 5; 127; -128 ];
        show = string_of_int;
        combine = Nx.add;
        fold = (fun a x -> Nx.sum ~axes:[ a ] x);
      };
    Kind
      {
        dtype = D.Bool;
        values = [ true; false ];
        show = string_of_bool;
        combine = Nx.logical_xor;
        fold = (fun a x -> Nx.any ~axes:[ a ] x);
      };
  ]

(* A movement that keeps every axis of a value in order. *)
type movement = Unit_axis | Reversed | Leading | Sliced

(* An operation over [x]: [combine x x]; a fold along an axis; a movement,
   then the moved value combined with itself; and [x] combined with a constant
   of its shape: an arange's cumulative sum along an axis, cast. *)
type op = Combine | Fold of int | Moved of movement | Beside of int

let pp_op ppf = function
  | Combine -> Format.pp_print_string ppf "combined"
  | Fold a -> Format.fprintf ppf "folded along %d" a
  | Moved Unit_axis -> Format.pp_print_string ppf "given a unit axis"
  | Moved Reversed -> Format.pp_print_string ppf "transposed"
  | Moved Leading -> Format.pp_print_string ppf "broadcast along a new axis"
  | Moved Sliced -> Format.pp_print_string ppf "sliced along its last axis"
  | Beside a -> Format.fprintf ppf "beside a constant scanned along %d" a

let apply (type v s d) (k : (v, s) kind) op (x : (v, s, d) Nx.t) :
    (v, s, d) Nx.t =
  let s = Nx.shape x in
  let r = Array.length s in
  match op with
  | Combine -> k.combine x x
  | Fold a -> k.fold a x
  | Moved mv ->
      let y =
        match mv with
        | Unit_axis -> Nx.reshape (Array.append s [| 1 |]) x
        | Reversed -> Nx.transpose x
        | Leading -> Nx.broadcast_to (Array.append [| 2 |] s) x
        | Sliced ->
            let last i = if i = r - 1 then Nx.R (1, s.(i)) else Nx.A in
            Nx.slice (List.init r last) x
      in
      k.combine y y
  | Beside a ->
      let n = Array.fold_left ( * ) 1 s in
      let c = Nx.cumsum ~axis:a (Nx.reshape s (Nx.arange D.Int32 0 n 1)) in
      k.combine x (Nx.cast k.dtype c)

type case =
  | Case : ('v, 's) kind * int array * 'v array * where * op -> case

let pp_case ppf (Case (k, s, data, w, op)) =
  Format.fprintf ppf "%s %a [%s] at %a, %a" (D.name k.dtype) pp_ints s
    (String.concat "; " (Array.to_list (Array.map k.show data)))
    pp_where w pp_op op

let placed_case =
  let open Gen in
  with_pp pp_case
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let axis = int_range 0 (rank - 1) in
     let* op =
       one_of
         [
           constant Combine;
           map (fun a -> Fold a) axis;
           map (fun mv -> Moved mv)
             (of_list [ Unit_axis; Reversed; Leading; Sliced ]);
           map (fun a -> Beside a) axis;
         ]
     in
     let* (Kind k) = of_list kinds in
     let n = Array.fold_left ( * ) 1 s in
     let+ data = array ~size:(constant n) (of_list k.values) in
     Case (k, s, data, w, op))

(* [y]'s elements' bits, read on the host. *)
let shown (type v s d) (k : (v, s) kind) (y : (v, s, d) Nx.t) =
  Array.map k.show (elements y)

let law_placed (Case (k, s, data, w, op)) =
  let ts = tiles w in
  let cut = List.map fst ts in
  let extent a = s.(a) in
  cover "an uneven cut" (List.exists (fun (a, t) -> extent a mod t <> 0) ts);
  cover "an empty window"
    (List.exists (fun (a, t) -> extent a > 0 && extent a < t) ts);
  cover "a cut of an empty axis" (List.exists (fun a -> extent a = 0) cut);
  cover "a cut over several mesh axes"
    (match w with Cuts (x, y) -> x = y | _ -> false);
  (match op with
  | Fold a -> cover "a fold along an axis no cut names" (not (List.mem a cut))
  | Moved Sliced ->
      cover "a slice that keeps a cut axis"
        (not (List.mem (Array.length s - 1) cut))
  | Moved _ -> cover "a movement of a cut value" true
  | Beside a ->
      cover "a constant scanned along a cut axis" (List.mem a cut);
      cover "a constant computed whole along an axis, beside another cut"
        (List.mem a cut && List.length cut > 1)
  | Combine -> ());
  cover "a signalling NaN" (signals k.dtype data);
  let x = signalling k.dtype s data in
  let expected = shown k (apply k op x) in
  let actual =
    match w with
    | Split3 a -> shown k (apply k op (Nx.place (Set3.split ~axis:a) x))
    | Split4 a -> shown k (apply k op (Nx.place (Set4.split ~axis:a) x))
    | Cuts (a, b) ->
        let cuts =
          if a = b then [ (a, [ "a"; "b" ]) ]
          else [ (a, [ "a" ]); (b, [ "b" ]) ]
        in
        shown k (apply k op (Nx.place (Nx.Placement.mesh mesh4 cuts) x))
    | Minor a ->
        shown k
          (apply k op (Nx.place (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) x))
  in
  equal (array string) expected actual

(* Reductions over cut axes *)

(* A dtype, the elements drawn for it, how their bits print, and which are
   NaNs. *)
type ('v, 's) reduced = {
  rdtype : ('v, 's) D.t;
  rvalues : 'v list;
  rshow : 'v -> string;
  is_nan : 'v -> bool;
}

type any_reduced = Reduced : ('v, 's) reduced -> any_reduced

let complexes =
  List.map2 (fun re im -> { Complex.re; im }) floats (List.rev floats)

let reduced_kinds =
  let never _ = false in
  let float dt =
    Reduced
      {
        rdtype = dt;
        rvalues = floats;
        rshow = float_bits;
        is_nan = Float.is_nan;
      }
  in
  let small dt values =
    Reduced
      {
        rdtype = dt;
        rvalues = values;
        rshow = string_of_int;
        is_nan = never;
      }
  in
  let truth dt =
    Reduced
      {
        rdtype = dt;
        rvalues = [ true; false ];
        rshow = string_of_bool;
        is_nan = never;
      }
  in
  [
    float D.Float32; float D.Float64; float D.Float16; float D.Bfloat16;
    Reduced
      {
        rdtype = D.Complex64;
        rvalues = complexes;
        rshow = (fun c -> float_bits c.Complex.re ^ "," ^ float_bits c.im);
        is_nan = (fun c -> Float.is_nan c.Complex.re || Float.is_nan c.im);
      };
    Reduced
      {
        rdtype = D.Complex128;
        rvalues = complexes;
        rshow = (fun c -> float_bits c.Complex.re ^ "," ^ float_bits c.im);
        is_nan = (fun c -> Float.is_nan c.Complex.re || Float.is_nan c.im);
      };
    Reduced
      {
        rdtype = D.Int32;
        rvalues = [ 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int ];
        rshow = Int32.to_string;
        is_nan = never;
      };
    small D.Int8 [ 0; 1; -1; 5; 127; -128 ];
    small D.Uint8 [ 0; 1; 3; 200; 255 ];
    small D.Int4 [ 0; 1; -1; 3; 7; -8 ];
    small D.Uint4 [ 0; 1; 5; 15 ];
    truth D.Bool; truth D.Bit;
  ]

type reduction =
  | Sum
  | Prod
  | Max
  | Min
  | Argmax
  | Argmin
  | All
  | Any
  | Count
  | Mean
  | Var
  | Std
  | Logsumexp
  | Logmeanexp

let reduction_name = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"
  | Argmax -> "argmax"
  | Argmin -> "argmin"
  | All -> "all"
  | Any -> "any"
  | Count -> "count"
  | Mean -> "mean"
  | Var -> "var"
  | Std -> "std"
  | Logsumexp -> "logsumexp"
  | Logmeanexp -> "logmeanexp"

(* The reductions of floats alone. *)
let of_floats = [ Var; Std; Logsumexp; Logmeanexp ]

(* The reductions [dt] takes beside the extremes, positions and truths. *)
let folds (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Float -> [ Sum; Prod; Mean ] @ of_floats
  | D.Complex -> [ Sum; Prod; Mean ]
  | D.Signed | D.Unsigned -> [ Sum; Prod ]
  | D.Boolean -> []

type reduced_case =
  | Reduced_case :
      ('v, 's) reduced
      * int array
      * 'v array
      * where
      * int list option
      * reduction
      -> reduced_case

let pp_reduced_case ppf (Reduced_case (k, s, data, w, axes, r)) =
  Format.fprintf ppf "%s along %s of %s %a [%s] at %a" (reduction_name r)
    (match axes with
    | None -> "every axis"
    | Some l -> String.concat "," (List.map string_of_int l))
    (D.name k.rdtype) pp_ints s
    (String.concat "; " (Array.to_list (Array.map k.rshow data)))
    pp_where w

(* [r] of the floats [x] along [axes], for [r] a reduction of floats alone. *)
let of_float (type s d) r axes (x : (float, s, d) Nx.t) =
  match r with
  | Var -> Nx.var ?axes x
  | Std -> Nx.std ?axes x
  | Logsumexp -> Nx.logsumexp ?axes x
  | Logmeanexp -> Nx.logmeanexp ?axes x
  | _ -> invalid_arg "of_float: not a reduction of floats alone"

(* Each result's bits as strings, [rshow] printing an element's, or the
   message of the refusal. *)
let reduce (type v s d) (rshow : v -> string) r axes (x : (v, s, d) Nx.t) =
  let show y = Array.map rshow (elements y) in
  let int64s y = Array.map Int64.to_string (elements y) in
  let bools y = Array.map string_of_bool (elements y) in
  let axis = Option.map List.hd axes in
  match
    match r with
    | Sum -> show (Nx.sum ?axes x)
    | Prod -> show (Nx.prod ?axes x)
    | Max -> show (Nx.max ?axes x)
    | Min -> show (Nx.min ?axes x)
    | Argmax -> int64s (Nx.argmax ?axis x)
    | Argmin -> int64s (Nx.argmin ?axis x)
    | All -> bools (Nx.all ?axes x)
    | Any -> bools (Nx.any ?axes x)
    | Count -> int64s (Nx.count ?axes (Nx.cast D.Bool x))
    | Mean -> show (Nx.mean ?axes x)
    | Var | Std | Logsumexp | Logmeanexp -> (
        match D.kind (Nx.dtype x) with
        | D.Float -> show (of_float r axes x)
        | D.Complex | D.Signed | D.Unsigned | D.Boolean ->
            invalid_arg "reduce: a reduction of floats alone")
  with
  | v -> Ok v
  | exception Invalid_argument m -> Error m

let printed = function
  | Ok v -> String.concat " " (Array.to_list v)
  | Error m -> "raises " ^ m

(* {!reduce} of [x] placed as [w]. *)
let reduce_at (type v s) w (rshow : v -> string) r axes
    (x : (v, s, Nx.Host.d) Nx.t) =
  match w with
  | Split3 a -> reduce rshow r axes (Nx.place (Set3.split ~axis:a) x)
  | Split4 a -> reduce rshow r axes (Nx.place (Set4.split ~axis:a) x)
  | Cuts (a, b) ->
      let cuts =
        if a = b then [ (a, [ "a"; "b" ]) ] else [ (a, [ "a" ]); (b, [ "b" ]) ]
      in
      reduce rshow r axes (Nx.place (Nx.Placement.mesh mesh4 cuts) x)
  | Minor a ->
      reduce rshow r axes
        (Nx.place (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) x)

let reduced_case =
  let open Gen in
  with_pp pp_reduced_case
    (let* rank = int_range 1 3 in
     let* s = array ~size:(constant rank) (of_list [ 0; 1; 2; 3; 5; 7 ]) in
     let* w = where_of rank in
     let* (Reduced k) = of_list reduced_kinds in
     let* r =
       of_list
         (folds k.rdtype @ [ Max; Min; Argmax; Argmin; All; Any; Count ])
     in
     (* Axes along the cut ones most often; one for a position. *)
     let cut = List.map fst (tiles w) in
     let* axes =
       match r with
       | Argmax | Argmin ->
           frequency
             [
               (3, map (fun a -> Some [ a ]) (of_list cut));
               (1, map (fun a -> Some [ a ]) (int_range 0 (rank - 1)));
               (1, constant None);
             ]
       | _ ->
           frequency
             [
               ( 4,
                 map
                   (fun keep ->
                     let l =
                       List.filter (fun a -> keep.(a)) (List.init rank Fun.id)
                     in
                     Some (if l = [] then cut else l))
                   (array ~size:(constant rank) bool) );
               (1, constant None);
             ]
     in
     let n = Array.fold_left ( * ) 1 s in
     let+ data = array ~size:(constant n) (of_list k.rvalues) in
     Reduced_case (k, s, data, w, axes, r))

let law_reduced (Reduced_case (k, s, data, w, axes, r)) =
  let ts = tiles w and rank = Array.length s in
  let along = match axes with None -> List.init rank Fun.id | Some l -> l in
  let cut_along = List.filter (fun (a, _) -> List.mem a along) ts in
  let over = cut_along <> [] in
  cover "a reduction over a cut axis" over;
  cover ("a reduction over a cut axis: " ^ reduction_name r) over;
  cover "a cut over a reduced axis after the first reduced one"
    (List.exists (fun (a, _) -> a > List.hd along) cut_along);
  cover "an uneven cut over a reduced axis"
    (List.exists (fun (a, t) -> s.(a) mod t <> 0) cut_along);
  cover "an empty window over a reduced axis"
    (List.exists (fun (a, t) -> s.(a) > 0 && s.(a) < t) cut_along);
  cover "a cut reduced axis of no element"
    (List.exists (fun (a, _) -> s.(a) = 0) cut_along);
  cover "a reduced axis cut over two mesh axes"
    (over && match w with Cuts (x, y) -> x = y | _ -> false);
  cover "a reduced axis cut beside a kept one"
    (over && List.length cut_along < List.length ts);
  cover "a reduced axis cut, the other mesh axis holding copies"
    (over && match w with Minor _ -> true | _ -> false);
  cover "a position over a cut axis"
    (over && (r = Argmax || r = Argmin));
  cover "a NaN term over a cut axis" (over && Array.exists k.is_nan data);
  cover "a signalling NaN term over a cut axis" (over && signals k.rdtype data);
  cover "a sub-byte dtype over a cut axis" (over && D.bits k.rdtype < 8);
  let x = signalling k.rdtype s data in
  equal string
    (printed (reduce k.rshow r axes x))
    (printed (reduce_at w k.rshow r axes x))

(* nx.cpu, counting its loop positions, over three memory devices. *)
module Count3 = (val Nx.devices ~kernels:(module C) [ m 0; m 1; m 2 ])

(* The loop positions [f] computes over [x] split along [axis] over three
   devices, after [x] is placed. *)
let positions axis f x =
  let x = Nx.place (Count3.split ~axis) (Nx.Repr.of_array Nx.Host.v x) in
  C.reset ();
  let y = f x in
  let n = C.positions () in
  ignore (elements y);
  n

let test_partials_move () =
  (* 240 terms over three devices, 4 outputs: each device reduces its window
     of the terms, then combines three partials per output. Reading the
     operand whole, each would reduce all 240 terms. *)
  let ints s = A.of_array D.Int32 s (Array.init 240 Int32.of_int) in
  let floats s = A.of_array D.Float32 s (Array.init 240 Float.of_int) in
  let fewer what got = less int ~msg:what ~than:(2 * 240) got in
  let rows = [| 60; 4 |] and columns = [| 4; 60 |] in
  fewer "an integer sum"
    (positions 0 (fun x -> Nx.sum ~axes:[ 0 ] x) (ints rows));
  fewer "a maximum" (positions 0 (fun x -> Nx.max ~axes:[ 0 ] x) (floats rows));
  fewer "positions"
    (positions 1 (fun x -> Nx.argmax ~axis:1 x) (floats columns));
  fewer "a reduction of every axis"
    (positions 1 (fun x -> Nx.min x) (ints columns))

(* Float reductions over cut axes of several of lane order's blocks *)

(* A float or complex dtype and how its elements' bits print. *)
type lane_kind = Lane_kind : ('v, 's) D.t * ('v -> string) -> lane_kind

let complex_bits c = float_bits c.Complex.re ^ "," ^ float_bits c.im

let lane_kinds =
  [
    Lane_kind (D.Float32, float_bits);
    Lane_kind (D.Float64, float_bits);
    Lane_kind (D.Float16, float_bits);
    Lane_kind (D.Bfloat16, float_bits);
    Lane_kind (D.Complex64, complex_bits);
    Lane_kind (D.Complex128, complex_bits);
  ]

(* Terms that decide a result's bits: NaNs of both signs and with a payload,
   infinities and [-0.]. *)
type special = Nan of float | Inf | Neg_inf | Neg_zero

let specials =
  List.map (fun x -> Nan x) [ nan; -.nan; Int32.float_of_bits 0x7fc00123l ]
  @ [ Inf; Neg_inf; Neg_zero ]

let special_value = function
  | Nan x -> x
  | Inf -> infinity
  | Neg_inf -> neg_infinity
  | Neg_zero -> -0.

type lane_case = {
  lkind : lane_kind;
  lshape : int array;
  laxes : int list option;
  lwhere : where;
  lr : reduction;
  seed : int;
  marks : (float * special) list;
      (* Specials at fractions of the terms, in C order. *)
}

let pp_lane_case ppf c =
  let (Lane_kind (dt, _)) = c.lkind in
  Format.fprintf ppf "%s along %s of %s %a at %a, seed %d, specials %s"
    (reduction_name c.lr)
    (match c.laxes with
    | None -> "every axis"
    | Some l -> String.concat "," (List.map string_of_int l))
    (D.name dt) pp_ints c.lshape pp_where c.lwhere c.seed
    (String.concat "; "
       (List.map
          (fun (f, s) -> Printf.sprintf "%g:%h" f (special_value s))
          c.marks))

(* A finite term for [r] from [st]: spread over 2^±20 for a sum, near 1 for a
   product, below 10 for a logsumexp. *)
let finite r st =
  match r with
  | Prod ->
      (if Random.State.bool st then 1. else -1.)
      *. (0.999 +. Random.State.float st 0.002)
  | Logsumexp | Logmeanexp -> Random.State.float st 210. -. 200.
  | _ ->
      Float.ldexp (Random.State.float st 2. -. 1.) (Random.State.int st 41 - 20)

(* The case's terms, of [dt]: finite ones drawn from its seed, then its
   specials, one part of a complex number each. *)
let lane_data (type v s) (dt : (v, s) D.t) c : v array =
  let st = Random.State.make [| c.seed |] in
  let n = Array.fold_left ( * ) 1 c.lshape in
  let floats = Array.init n (fun _ -> finite c.lr st) in
  let parts = Array.init n (fun _ -> 0.01 *. finite c.lr st) in
  let at f = Stdlib.min (n - 1) (int_of_float (f *. Float.of_int n)) in
  let complex = D.is D.Complex dt in
  List.iter
    (fun (f, sp) ->
      if n > 0 then
        if complex && Random.State.bool st then parts.(at f) <- special_value sp
        else floats.(at f) <- special_value sp)
    c.marks;
  match D.kind dt with
  | D.Float -> floats
  | D.Complex -> Array.map2 (fun re im -> { Complex.re; im }) floats parts
  | D.Signed | D.Unsigned | D.Boolean -> invalid_arg "lane_data: not a float"

let lane_case =
  let open Gen in
  with_pp pp_lane_case
    (let* rank = int_range 1 3 in
     (* One axis of many terms, the others few. *)
     let* big = int_range 0 (rank - 1) in
     let* lshape = array ~size:(constant rank) (of_list [ 1; 2; 3; 5 ]) in
     let* many =
       if rank = 1 then
         of_list [ 0; 2; 3; 1023; 1024; 1025; 2049; 3000; 4096; 4097 ]
       else of_list [ 1000; 1024; 1025; 2048; 2049 ]
     in
     lshape.(big) <- many;
     let* lwhere = where_of rank in
     let* (Lane_kind (dt, _) as lkind) = of_list lane_kinds in
     let* lr =
       of_list
         (match D.kind dt with
         | D.Float -> [ Sum; Prod; Mean ] @ of_floats
         | _ -> [ Sum; Prod; Mean ])
     in
     let cut = List.map fst (tiles lwhere) in
     let* laxes =
       frequency
         [
           ( 4,
             map
               (fun keep ->
                 let l =
                   List.filter (fun a -> keep.(a)) (List.init rank Fun.id)
                 in
                 Some (List.sort_uniq compare (cut @ l)))
               (array ~size:(constant rank) bool) );
           (1, constant None);
         ]
     in
     let* seed = int in
     (* Infinities of both signs, whose sum is a NaN made from numbers, early or
        late, and a NaN term the other side or none. *)
     let early = float_range 0. 0.3 and late = float_range 0.6 1. in
     let made a b = [ (a, Inf); (b, Neg_inf) ] in
     let nan_term = map (fun x -> Nan x) (of_list [ nan; -.nan ]) in
     let+ marks =
       frequency
         [
           (1, constant []);
           ( 2,
             list ~size:(int_range 1 3)
               (pair (float_range 0. 1.) (of_list specials)) );
           ( 1,
             let+ a = early and+ b = early and+ c = late and+ x = nan_term in
             made a b @ [ (c, x) ] );
           ( 1,
             let+ a = late and+ b = late and+ c = early and+ x = nan_term in
             (c, x) :: made a b );
           ( 1,
             let+ a = float_range 0. 1. and+ b = float_range 0. 1. in
             made a b );
         ]
     in
     { lkind; lshape; laxes; lwhere; lr; seed; marks })

(* For the first output of [shape] reduced along [axes], the device window
   holding each term, numbered by the cuts [ts] of [axes]: a term's window along
   each cut reduced axis. *)
let owners shape axes ts =
  let axes = Array.of_list axes in
  let whole = Array.map (fun a -> shape.(a)) axes in
  let n = Array.fold_left ( * ) 1 whole in
  let window e t i =
    (* Balanced windows: the first [e mod t] hold one more. *)
    let q = e / t and r = e mod t in
    let j = ref 0 in
    while !j + 1 < t && ((!j + 1) * q) + Stdlib.min (!j + 1) r <= i do
      incr j
    done;
    !j
  in
  Array.init n (fun term ->
      let idx = Array.make (Array.length axes) 0 and k = ref term in
      for i = Array.length axes - 1 downto 0 do
        idx.(i) <- !k mod whole.(i);
        k := !k / whole.(i)
      done;
      List.map
        (fun (a, t) ->
          match Array.find_index (( = ) a) axes with
          | Some i -> window whole.(i) t idx.(i)
          | None -> 0)
        ts)

(* The number of windows holding terms of each block of 1024. *)
let block_owners owners =
  let n = Array.length owners in
  List.init
    ((n + 1023) / 1024)
    (fun b ->
      let seen = ref [] in
      for t = b * 1024 to Stdlib.min n ((b + 1) * 1024) - 1 do
        if not (List.mem owners.(t) !seen) then seen := owners.(t) :: !seen
      done;
      List.length !seen)

let law_lanes c =
  let (Lane_kind (dt, show)) = c.lkind in
  let rank = Array.length c.lshape in
  let along =
    match c.laxes with None -> List.init rank Fun.id | Some l -> l
  in
  let ts = tiles c.lwhere in
  let cut_along = List.filter (fun (a, _) -> List.mem a along) ts in
  let data = lane_data dt c in
  let owners = owners c.lshape along cut_along in
  let blocks = block_owners owners in
  let n = Array.length owners in
  cover ("a reduction of " ^ D.name dt) true;
  cover ("a reduction by " ^ reduction_name c.lr) true;
  cover "no term" (n = 0);
  cover "an empty window along a reduced axis"
    (List.exists
       (fun (a, t) -> c.lshape.(a) > 0 && c.lshape.(a) < t)
       cut_along);
  cover "an uneven cut along a reduced axis"
    (List.exists (fun (a, t) -> c.lshape.(a) mod t <> 0) cut_along);
  cover "a cut reduced axis after the first reduced one"
    (List.exists (fun (a, _) -> a > List.hd along) cut_along);
  cover "two reduced axes cut" (List.length cut_along = 2);
  cover "a reduced axis cut over two mesh axes"
    (match c.lwhere with Cuts (a, b) -> a = b | _ -> false);
  cover "a block whose terms lie in three windows or more"
    (List.exists (fun k -> k >= 3) blocks);
  cover "a short last block whose terms lie in several windows"
    (n mod 1024 <> 0
    && match List.rev blocks with k :: _ -> k >= 2 | [] -> false);
  cover "several blocks, each in one window"
    (List.length blocks > 1 && List.for_all (( = ) 1) blocks);
  let nan_at =
    List.filter_map (function f, Nan _ -> Some f | _ -> None) c.marks
  in
  let inf_at s =
    List.filter_map (fun (f, s') -> if s' = s then Some f else None) c.marks
  in
  let infs = (inf_at Inf, inf_at Neg_inf) in
  let first l = List.fold_left Float.min 1. l in
  let both =
    List.length along = rank
    && match infs with [], _ | _, [] -> false | _ -> true
  in
  cover "a NaN term after infinities of both signs"
    (both && nan_at <> []
    && first nan_at > Float.max (first (fst infs)) (first (snd infs)));
  cover "a NaN term before infinities of both signs"
    (both && nan_at <> []
    && first nan_at < Float.min (first (fst infs)) (first (snd infs)));
  cover "infinities of both signs and no NaN term" (both && nan_at = []);
  cover "a signalling NaN term" (signals dt data);
  let x = signalling dt c.lshape data in
  equal string
    (printed (reduce show c.lr c.laxes x))
    (printed (reduce_at c.lwhere show c.lr c.laxes x))

(* nx.cpu, counting its loop positions, over four memory devices. *)
module Count4 = (val Nx.devices ~kernels:(module C) [ m 0; m 1; m 2; m 3 ])

let test_float_partials_move () =
  (* Read whole, each of four devices would compute every term: [4 n] positions.
     Each device's window of 2048 terms is a node of lane order's tree; of 6000,
     windows of 1500 leave blocks straddling two. *)
  let counted (type v s) (x : (v, s) A.t) f =
    let x = Nx.place (Count4.split ~axis:0) (Nx.Repr.of_array Nx.Host.v x) in
    C.reset ();
    let y = f x in
    let n = C.positions () in
    ignore (elements y);
    n
  in
  let floats dt n =
    A.of_array dt [| n |] (Array.init n (fun i -> Float.of_int (i mod 7) /. 7.))
  in
  let complexes n =
    A.of_array D.Complex64 [| n |]
      (Array.init n (fun i -> { Complex.re = Float.of_int (i mod 5); im = 1. }))
  in
  let under what n limit got =
    less int ~msg:(Printf.sprintf "%s of %d terms" what n) ~than:limit got
  in
  List.iter
    (fun n ->
      under "a sum" n (2 * n) (counted (floats D.Float32 n) Nx.sum);
      under "a product" n (2 * n) (counted (floats D.Float64 n) Nx.prod);
      under "a float16 mean" n (2 * n) (counted (floats D.Float16 n) Nx.mean);
      (* A complex sum reduces each part as a float. *)
      under "a complex sum" n (3 * n) (counted (complexes n) Nx.sum);
      under "a logsumexp" n (4 * n) (counted (floats D.Float32 n) Nx.logsumexp))
    [ 8192; 6000 ];
  under "a variance" 8192 (4 * 8192) (counted (floats D.Float32 8192) Nx.var)

(* Contractions over cut contracting axes *)

(* An operand dtype, the accumulator a contraction of it takes, how its
   elements' bits print and a draw of one. *)
type ckind =
  | Ckind : {
      cdtype : ('v, 's) D.t;
      acc : D.any;
      cshow : 'v -> string;
      draw : Random.State.t -> 'v;
    }
      -> ckind

(* A finite float spread over 2^±20, so that sums depend on their
   association, or [-0.]. *)
let spread st =
  if Random.State.int st 16 = 0 then -0.
  else
    Float.ldexp (Random.State.float st 2. -. 1.) (Random.State.int st 41 - 20)

(* A contraction's bits, every NaN alike: a contraction leaves a NaN's sign
   and payload unspecified, and windows keep the bits of the rest
   ({!Nx_kernel.S}). Where a float16 operand overflows to infinities, their
   sums are NaNs. *)
let product_bits v = if Float.is_nan v then "nan" else float_bits v
let complex_product_bits c = product_bits c.Complex.re ^ "," ^ product_bits c.im

let ckinds =
  let float dt acc =
    Ckind { cdtype = dt; acc = D.Any acc; cshow = product_bits; draw = spread }
  in
  let complex dt =
    Ckind
      {
        cdtype = dt;
        acc = D.Any dt;
        cshow = complex_product_bits;
        draw = (fun st -> { Complex.re = spread st; im = spread st });
      }
  in
  let int32s = [| 0l; 1l; -1l; 7l; Int32.max_int; Int32.min_int |] in
  [
    float D.Float32 D.Float32;
    float D.Float64 D.Float64;
    float D.Float16 D.Float32;
    float D.Bfloat16 D.Float32;
    complex D.Complex64;
    complex D.Complex128;
    Ckind
      {
        cdtype = D.Int32;
        acc = D.Any D.Int32;
        cshow = Int32.to_string;
        draw = (fun st -> int32s.(Random.State.int st (Array.length int32s)));
      };
    Ckind
      {
        cdtype = D.Int8;
        acc = D.Any D.Int32;
        cshow = string_of_int;
        draw = (fun st -> Random.State.int st 256 - 128);
      };
  ]

(* Where the operands lie, the contracting pair [p] cut alike in both: over
   three or four devices; over both mesh axes; over the minor one, beside
   [a]'s rows over the major one; over the minor one, beside the batch over
   the major one in both; or pair 0 over the major axis and pair 1 over the
   minor one. Or [b]'s pair [p] alone cut, over the minor axis, and [a]'s
   batch over the major one and rows over the minor one. *)
type ccut =
  | Pair3 of int
  | Pair4 of int
  | Pair_mesh of int
  | Pair_rows of int
  | Pair_batch of int
  | Pairs
  | Batch_rows of int

(* Where [init] lies: at the result's placement, or whole on every device,
   which differs where the result is cut. *)
type iwhere = Init_as_result | Init_everywhere

type ccase = {
  ck : ckind;
  nb : int option;  (* The batch's extent, if there is one. *)
  rows : int;
  cols : int;
  ks : int array;  (* The contracting pairs' extents. *)
  ccut : ccut;
  order : Nx_kernel.order;
  init : bool;
  iwhere : iwhere;
  cseed : int;
}

let pp_ccase ppf c =
  let (Ckind k) = c.ck in
  let cut =
    match c.ccut with
    | Pair3 p -> Printf.sprintf "pair %d over 3" p
    | Pair4 p -> Printf.sprintf "pair %d over 4" p
    | Pair_mesh p -> Printf.sprintf "pair %d over 2x2" p
    | Pair_rows p -> Printf.sprintf "pair %d/b rows/a" p
    | Pair_batch p -> Printf.sprintf "pair %d/b batch/a" p
    | Pairs -> "pair 0/a pair 1/b"
    | Batch_rows p -> Printf.sprintf "batch/a rows/b, b's pair %d/b" p
  in
  Format.fprintf ppf "%s %s%d x %d by %a, %s, %s order%s, seed %d"
    (D.name k.cdtype)
    (match c.nb with Some b -> Printf.sprintf "%d of " b | None -> "")
    c.rows c.cols pp_ints c.ks cut
    (match c.order with Lane -> "lane" | Chain -> "chain")
    (match (c.init, c.iwhere) with
    | false, _ -> ""
    | true, Init_as_result -> " from init"
    | true, Init_everywhere -> " from init on every device")
    c.cseed

(* The axis of [a] and of [b] that the contracting pair [p] names: [a] is
   [[batch; rows; pairs]], [b] [[batch; pairs; columns]]. *)
let pair_axes c p =
  let ob = if Option.is_some c.nb then 1 else 0 in
  (ob + 1 + p, ob + p)

(* The first index of each of [t] tiles of [extent], then [extent]: the first
   [extent mod t] tiles hold one more. *)
let tile_starts extent t =
  List.init (t + 1) (fun j -> (j * (extent / t)) + Stdlib.min j (extent mod t))

(* Extents of one contracting pair either side of lane order's blocks of
   1024 terms, 512 complex products, and of a contraction's blocks of 4096
   products, and of several of those. *)
let pair_extents =
  [
    0; 1; 2; 3; 5; 511; 512; 513; 1023; 1024; 1025; 1500; 2047; 2048; 2049;
    4095; 4096; 4097; 8191; 8192; 8193; 12289; 20481;
  ]

(* Cases over one contracting pair of [extents], or, where [pairs], over
   two a quarter of the time. *)
let ccase_of ~pairs ~extents =
  let open Gen in
  with_pp pp_ccase
    (let* ck = of_list ckinds in
     let* two =
       if pairs then frequency [ (3, constant false); (1, constant true) ]
       else constant false
     in
     let* ccut =
       if two then
         of_list
           [ Pair3 1; Pair4 1; Pair_mesh 1; Pair_rows 1; Pair_mesh 0; Pairs ]
       else
         of_list
           [
             Pair3 0; Pair4 0; Pair_mesh 0; Pair_rows 0; Pair_batch 0;
             Batch_rows 0;
           ]
     in
     let* nb =
       match ccut with
       | Pair_batch _ | Batch_rows _ -> map Option.some (of_list [ 1; 2; 3 ])
       | _ -> frequency [ (3, constant None); (1, constant (Some 2)) ]
     in
     let* rows = int_range 1 3 in
     let* cols = int_range 1 3 in
     let* ks =
       if two then
         (* A quarter of them of several blocks, whose edges fall inside
            the first pair's rows. *)
         let* many = frequency [ (3, constant false); (1, constant true) ] in
         let+ k0 = of_list (if many then [ 3; 5 ] else [ 1; 2; 3; 5 ])
         and+ k1 =
           of_list
             (if many then [ 1500; 1700 ] else [ 0; 1; 3; 300; 511; 700; 1700 ])
         in
         [| k0; k1 |]
       else
         let+ k = of_list extents in
         [| k |]
     in
     let* order = of_list [ Nx_kernel.Lane; Chain ] in
     let* init = bool in
     let* iwhere = of_list [ Init_as_result; Init_everywhere ] in
     let+ cseed = int in
     { ck; nb; rows; cols; ks; ccut; order; init; iwhere; cseed })

let ccase = ccase_of ~pairs:true ~extents:pair_extents

(* Contractions over one pair either side of a contraction's blocks, for a
   library that sums them its own way ({!Nx_support.Blocked}), and over
   enough blocks that one window holds several. *)
let bcase =
  ccase_of ~pairs:false
    ~extents:
      [
        1; 2; 1023; 1024; 1025; 2047; 2048; 2049; 4095; 4096; 4097; 8191; 8192;
        8193; 12289; 16384; 20481; 36865;
      ]

(* The case's operands and its contraction's [batch] and [contracting]
   pairs. *)
let ccase_values (type v s) (dt : (v, s) D.t) (draw : Random.State.t -> v) c =
  let st = Random.State.make [| c.cseed |] in
  let lead = match c.nb with Some b -> [| b |] | None -> [||] in
  let sa = Array.concat [ lead; [| c.rows |]; c.ks ] in
  let sb = Array.concat [ lead; c.ks; [| c.cols |] ] in
  let so = Array.concat [ lead; [| c.rows; c.cols |] ] in
  let host s =
    let n = Array.fold_left ( * ) 1 s in
    let data = Array.init n (fun _ -> draw st) in
    Nx.Repr.of_array Nx.Host.v (A.of_array dt s data)
  in
  let a = host sa and b = host sb and i = host so in
  let batch = match c.nb with Some _ -> [| (0, 0) |] | None -> [||] in
  let contracting = Array.init (Array.length c.ks) (pair_axes c) in
  (a, b, i, batch, contracting)

(* The contraction of the case over [a], [b] and [init], its bits. *)
let ccontract (type v s d) (cshow : v -> string) c acc batch contracting
    (a : (v, s, d) Nx.t) (b : (v, s, d) Nx.t) (init : (v, s, d) Nx.t) =
  let y =
    Nx.Prim.eval ~by:"test"
      (Contract
         {
           order = c.order;
           batch = Iarray.of_array batch;
           contracting = Iarray.of_array contracting;
           acc;
           out = Nx.dtype a;
           a;
           b;
           init = (if c.init then Some init else None);
           a_scale = None;
           b_scale = None;
           segments = None;
         })
  in
  Array.map cshow (elements y)

(* One kernel library's sets of memory devices: one, three and four, and a
   2 x 2 mesh of the four. *)
module type Sets = sig
  module One : Nx.Devices
  module Three : Nx.Devices
  module Four : Nx.Devices

  val mesh : Four.d Nx.Mesh.t
end

module Cpu_sets = struct
  module One = Nx.Host
  module Three = Set3
  module Four = Set4

  let mesh = mesh4
end

module Blocked_sets = struct
  let on ds = Nx.devices ~kernels:(module Nx_support.Blocked) (List.map m ds)

  module One = (val on [ 0 ])
  module Three = (val on [ 0; 1; 2 ])
  module Four = (val on [ 0; 1; 2; 3 ])

  let mesh = Nx.Mesh.v Four.v [ ("a", 2); ("b", 2) ]
end

(* [pairs] where cases may have no product or two pairs. *)
let law_cut ~pairs (module S : Sets) c =
  let (Ckind k) = c.ck in
  let a, b, init, batch, contracting = ccase_values k.cdtype k.draw c in
  let cut_pair, tiles =
    match c.ccut with
    | Pair3 p -> (p, 3)
    | Pair4 p | Pair_mesh p -> (p, 4)
    | Pair_rows p | Pair_batch p | Batch_rows p -> (p, 2)
    | Pairs -> (1, 2)
  in
  let kc = c.ks.(cut_pair) in
  let terms = Array.fold_left ( * ) 1 c.ks in
  let one_pair = Array.length c.ks = 1 in
  (* Products per block of a contraction, and where windows of one pair
     end. *)
  let block = 4096 in
  let edges = List.filter (fun s -> s > 0 && s < kc) (tile_starts kc tiles) in
  cover ("a contraction of " ^ D.name k.cdtype) true;
  cover "lane order" (c.order = Lane);
  cover "chain order" (c.order = Chain);
  cover "from init" c.init;
  cover "from init on every device beside a cut result"
    (c.init && c.iwhere = Init_everywhere
    &&
    match c.ccut with
    | Pair_rows _ | Pair_batch _ | Batch_rows _ -> true
    | _ -> false);
  if pairs then begin
    cover "no product" (terms = 0);
    cover "a cut on a pair after the first" (cut_pair > 0);
    cover "two pairs cut" (c.ccut = Pairs);
    cover "several blocks over two pairs, a block's edge inside a row"
      ((not one_pair) && terms > block && block mod c.ks.(1) <> 0)
  end
  else begin
    (* Whether one window holds two blocks, the second perhaps short, that
       one unit sums: lane order's tree pairs blocks from an even one,
       chain order any two in a row. *)
    let starts = tile_starts kc tiles in
    let rec within lo hi = function
      | s :: (e :: _ as rest) -> (s <= lo && hi <= e) || within lo hi rest
      | _ -> false
    in
    let step = match c.order with Lane -> 2 | Chain -> 1 in
    let rec several j =
      (j + 1) * block < kc
      && (within (j * block) (Int.min kc ((j + 2) * block)) starts
         || several (j + step))
    in
    cover "a lane unit of several blocks on one device"
      (c.order = Lane && one_pair && several 0);
    cover "a chain unit of several blocks on one device"
      (c.order = Chain && one_pair && several 0)
  end;
  cover "an empty window along the cut pair" (kc > 0 && kc < tiles);
  cover "an uneven cut" (kc mod tiles <> 0);
  cover "a pair cut over both mesh axes"
    (match c.ccut with Pair_mesh _ -> true | _ -> false);
  cover "a pair cut beside rows"
    (match c.ccut with Pair_rows _ -> true | _ -> false);
  cover "a pair cut beside the batch"
    (match c.ccut with Pair_batch _ -> true | _ -> false);
  cover "a pair cut in b alone beside a's batch and rows"
    (match c.ccut with Batch_rows _ -> true | _ -> false);
  let split = one_pair && List.exists (fun e -> e mod block <> 0) edges in
  cover "a lane block that window edges split" (c.order = Lane && split);
  cover "a chain block that window edges split" (c.order = Chain && split);
  cover "several blocks, each in one window"
    (one_pair && terms > block
    && List.for_all (fun e -> e mod block = 0) edges);
  cover "several blocks, some split by window edges"
    (one_pair && terms > 2 * block && split);
  let expected =
    ccontract k.cshow c k.acc batch contracting (Nx.place S.One.on a)
      (Nx.place S.One.on b) (Nx.place S.One.on init)
  in
  let at (type d) (pa : d Nx.Placement.t) (pb : d Nx.Placement.t)
      (pi : d Nx.Placement.t) =
    ccontract k.cshow c k.acc batch contracting (Nx.place pa a) (Nx.place pb b)
      (Nx.place pi init)
  in
  let mesh cuts = Nx.Placement.mesh S.mesh cuts in
  let result cuts =
    match c.iwhere with
    | Init_as_result -> mesh cuts
    | Init_everywhere -> S.Four.on
  in
  let pa p = fst (pair_axes c p) and pb p = snd (pair_axes c p) in
  let actual =
    match c.ccut with
    | Pair3 p ->
        at (S.Three.split ~axis:(pa p)) (S.Three.split ~axis:(pb p)) S.Three.on
    | Pair4 p ->
        at (S.Four.split ~axis:(pa p)) (S.Four.split ~axis:(pb p)) S.Four.on
    | Pair_mesh p ->
        let both axis = mesh [ (axis, [ "a"; "b" ]) ] in
        at (both (pa p)) (both (pb p)) S.Four.on
    | Pair_rows p ->
        let rows = if Option.is_some c.nb then 1 else 0 in
        at
          (mesh [ (rows, [ "a" ]); (pa p, [ "b" ]) ])
          (mesh [ (pb p, [ "b" ]) ])
          (result [ (rows, [ "a" ]) ])
    | Pair_batch p ->
        at
          (mesh [ (0, [ "a" ]); (pa p, [ "b" ]) ])
          (mesh [ (0, [ "a" ]); (pb p, [ "b" ]) ])
          (result [ (0, [ "a" ]) ])
    | Batch_rows p ->
        let batch_rows = [ (0, [ "a" ]); (1, [ "b" ]) ] in
        at (mesh batch_rows) (mesh [ (pb p, [ "b" ]) ]) (result batch_rows)
    | Pairs ->
        at
          (mesh [ (pa 0, [ "a" ]); (pa 1, [ "b" ]) ])
          (mesh [ (pb 0, [ "a" ]); (pb 1, [ "b" ]) ])
          S.Four.on
  in
  equal (array string) expected actual

(* A matrix product through Nx over operands cut along its contracting axis:
   [matmul], [dot] of a matrix and a vector, and [einsum]. *)
type nx_product = Matmul | Dot_vector | Einsum

type nx_case = {
  nk : ckind;
  product : nx_product;
  nrows : int;
  ncols : int;
  k : int;
  nwhere : ccut;
  nseed : int;
}

let pp_nx_case ppf c =
  let (Ckind k) = c.nk in
  Format.fprintf ppf "%s %s %d x %d by %d, %s, seed %d"
    (match c.product with
    | Matmul -> "matmul"
    | Dot_vector -> "dot"
    | Einsum -> "einsum")
    (D.name k.cdtype) c.nrows c.ncols c.k
    (match c.nwhere with
    | Pair3 _ -> "over 3"
    | Pair4 _ -> "over 4"
    | Pair_mesh _ -> "over 2x2"
    | _ -> "beside rows")
    c.nseed

let nx_case =
  let open Gen in
  with_pp pp_nx_case
    (let* nk = of_list ckinds in
     (* Few rows or columns sum in lane order, more in chain order: half the
        cases have many of both. *)
     let* many = bool in
     let* product =
       of_list
         (if many then [ Matmul; Einsum ] else [ Matmul; Dot_vector; Einsum ])
     in
     let* nrows = of_list (if many then [ 9; 12 ] else [ 1; 2; 5; 9 ]) in
     let* ncols = of_list (if many then [ 9; 12 ] else [ 1; 2; 5; 9 ]) in
     let* k = of_list [ 0; 1; 2; 3; 7; 1025; 1500; 2049 ] in
     let* nwhere = of_list [ Pair3 0; Pair4 0; Pair_mesh 0; Pair_rows 0 ] in
     let+ nseed = int in
     { nk; product; nrows; ncols; k; nwhere; nseed })

let law_nx_case c =
  let (Ckind k) = c.nk in
  let st = Random.State.make [| c.nseed |] in
  let host s =
    let n = Array.fold_left ( * ) 1 s in
    Nx.Repr.of_array Nx.Host.v
      (A.of_array k.cdtype s (Array.init n (fun _ -> k.draw st)))
  in
  let a = host [| c.nrows; c.k |] in
  let b =
    match c.product with
    | Dot_vector -> host [| c.k |]
    | Matmul | Einsum -> host [| c.k; c.ncols |]
  in
  let order =
    Nx.Prim.contract_order ~batch:[||] ~contracting:[| (1, 0) |] a b
  in
  cover "lane order" (order = Lane);
  cover "chain order" (order = Chain);
  cover "an empty window" (c.k > 0 && c.k < 3);
  let product (type d) (a : (_, _, d) Nx.t) (b : (_, _, d) Nx.t) =
    let y =
      match c.product with
      | Matmul -> Nx.matmul a b
      | Dot_vector -> Nx.dot a b
      | Einsum -> Nx.einsum (Nx.Pattern.v "i j, j k -> i k | j") [ a; b ]
    in
    Array.map k.cshow (elements y)
  in
  let mesh cuts = Nx.Placement.mesh mesh4 cuts in
  let actual =
    match c.nwhere with
    | Pair3 _ ->
        product
          (Nx.place (Set3.split ~axis:1) a)
          (Nx.place (Set3.split ~axis:0) b)
    | Pair4 _ ->
        product
          (Nx.place (Set4.split ~axis:1) a)
          (Nx.place (Set4.split ~axis:0) b)
    | Pair_mesh _ ->
        product
          (Nx.place (mesh [ (1, [ "a"; "b" ]) ]) a)
          (Nx.place (mesh [ (0, [ "a"; "b" ]) ]) b)
    | Pair_rows _ | Pair_batch _ | Pairs | Batch_rows _ ->
        product
          (Nx.place (mesh [ (0, [ "a" ]); (1, [ "b" ]) ]) a)
          (Nx.place (mesh [ (0, [ "b" ]) ]) b)
  in
  equal (array string) (product a b) actual

let test_contractions_move () =
  (* Read whole, each of four devices would contract all of the [k] terms of
     each output: [4 m n k] products. Cut, each contracts its window: 8 x 8
     sums in chain order, 2 x 2 in lane order; windows of 1500 leave lane
     blocks straddling two devices. *)
  let counted m n k =
    let floats s =
      A.of_array D.Float32 s
        (Array.init (Array.fold_left ( * ) 1 s) (fun i ->
             Float.of_int (i mod 7) /. 7.))
    in
    let place axis x =
      Nx.place (Count4.split ~axis) (Nx.Repr.of_array Nx.Host.v x)
    in
    let a = place 1 (floats [| m; k |]) and b = place 0 (floats [| k; n |]) in
    C.reset ();
    let y = Nx.matmul a b in
    let got = C.products () in
    ignore (elements y);
    less int
      ~msg:(Printf.sprintf "%d x %d by %d" m n k)
      ~than:(2 * m * n * k) got
  in
  counted 8 8 4096;
  counted 2 2 4096;
  counted 2 2 6000;
  counted 8 8 6000

(* Scans along cut axes *)

type scan = Cumsum | Cumprod | Cummax | Cummin

let scan_name = function
  | Cumsum -> "cumsum"
  | Cumprod -> "cumprod"
  | Cummax -> "cummax"
  | Cummin -> "cummin"

(* Each result's bits as strings, [show] printing an element's, or the message
   of the refusal. *)
let scan_bits (type v s d) (show : v -> string) r axis (x : (v, s, d) Nx.t) =
  match
    match r with
    | Cumsum -> Nx.cumsum ~axis x
    | Cumprod -> Nx.cumprod ~axis x
    | Cummax -> Nx.cummax ~axis x
    | Cummin -> Nx.cummin ~axis x
  with
  | y -> Ok (Array.map show (elements y))
  | exception Invalid_argument m -> Error m

(* {!scan_bits} of [x] placed as [w]. *)
let scan_at (type v s) w (show : v -> string) r axis
    (x : (v, s, Nx.Host.d) Nx.t) =
  match w with
  | Split3 a -> scan_bits show r axis (Nx.place (Set3.split ~axis:a) x)
  | Split4 a -> scan_bits show r axis (Nx.place (Set4.split ~axis:a) x)
  | Cuts (a, b) ->
      let cuts =
        if a = b then [ (a, [ "a"; "b" ]) ] else [ (a, [ "a" ]); (b, [ "b" ]) ]
      in
      scan_bits show r axis (Nx.place (Nx.Placement.mesh mesh4 cuts) x)
  | Minor a ->
      scan_bits show r axis
        (Nx.place (Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]) x)

type scan_case = {
  skind : any_reduced;
  sshape : int array;
  saxis : int;
  swhere : where;
  sr : scan;
  sseed : int;
  smarks : (float * special) list;
}

let pp_scan_case ppf c =
  let (Reduced k) = c.skind in
  Format.fprintf ppf "%s along %d of %s %a at %a, seed %d, specials %s"
    (scan_name c.sr) c.saxis (D.name k.rdtype) pp_ints c.sshape pp_where
    c.swhere c.sseed
    (String.concat "; "
       (List.map
          (fun (f, s) -> Printf.sprintf "%g:%h" f (special_value s))
          c.smarks))

(* The case's terms: a float's or a complex's as {!lane_data} draws a sum's or
   a product's, another dtype's drawn from its values. *)
let scan_data (type v s) (k : (v, s) reduced) c : v array =
  match D.kind k.rdtype with
  | D.Float | D.Complex ->
      lane_data k.rdtype
        {
          lkind = Lane_kind (k.rdtype, k.rshow);
          lshape = c.sshape;
          laxes = None;
          lwhere = c.swhere;
          lr = (if c.sr = Cumprod then Prod else Sum);
          seed = c.sseed;
          marks = c.smarks;
        }
  | D.Signed | D.Unsigned | D.Boolean ->
      let st = Random.State.make [| c.sseed |] in
      let values = Array.of_list k.rvalues in
      Array.init
        (Array.fold_left ( * ) 1 c.sshape)
        (fun _ -> values.(Random.State.int st (Array.length values)))

let scan_case =
  let open Gen in
  with_pp pp_scan_case
    (let* rank = int_range 1 3 in
     let* swhere = where_of rank in
     let cut = List.map fst (tiles swhere) in
     (* Along a cut axis most often. *)
     let* saxis =
       frequency [ (5, of_list cut); (1, int_range 0 (rank - 1)) ]
     in
     let* many =
       frequency
         [
           (1, of_list [ 0; 1; 2; 3; 5; 7 ]);
           ( 3,
             of_list
               [ 4095; 4096; 4097; 6000; 8192; 9001; 12289; 16384; 20000 ] );
         ]
     in
     let* sshape = array ~size:(constant rank) (of_list [ 1; 2; 3 ]) in
     sshape.(saxis) <- many;
     let* (Reduced k as skind) = of_list reduced_kinds in
     let* sr =
       of_list
         (match D.kind k.rdtype with
         | D.Boolean -> [ Cummax; Cummin ]
         | _ -> [ Cumsum; Cumprod; Cummax; Cummin ])
     in
     let* sseed = int in
     let early = float_range 0. 0.3 and late = float_range 0.6 1. in
     let made a b = [ (a, Inf); (b, Neg_inf) ] in
     let nan_term = map (fun x -> Nan x) (of_list [ nan; -.nan ]) in
     let+ smarks =
       frequency
         [
           (1, constant []);
           ( 2,
             list ~size:(int_range 1 3)
               (pair (float_range 0. 1.) (of_list specials)) );
           ( 1,
             let+ a = early and+ b = early and+ c = late and+ x = nan_term in
             made a b @ [ (c, x) ] );
           ( 1,
             let+ a = late and+ b = late and+ c = early and+ x = nan_term in
             (c, x) :: made a b );
         ]
     in
     { skind; sshape; saxis; swhere; sr; sseed; smarks })

(* Balanced windows' starts along an axis of [e] over [t] tiles. *)
let starts e t =
  let q = e / t and r = e mod t in
  List.init t (fun j -> (j * q) + Stdlib.min j r)

let law_scans c =
  let (Reduced k) = c.skind in
  let n = c.sshape.(c.saxis) in
  let cut =
    List.filter_map
      (fun (a, t) -> if a = c.saxis then Some t else None)
      (tiles c.swhere)
  in
  let along = cut <> [] in
  let ss = List.concat_map (fun t -> starts n t) cut in
  let chunk = 4096 in
  cover "a scan along a cut axis" along;
  cover ("a scan along a cut axis: " ^ scan_name c.sr) along;
  cover ("a scan along a cut axis of " ^ D.name k.rdtype) along;
  cover "chunks across windows" (along && n > chunk);
  cover "a window starting inside a chunk past the first"
    (List.exists (fun s -> s > chunk && s mod chunk <> 0) ss);
  cover "a window starting at a chunk's start"
    (List.exists (fun s -> s > 0 && s mod chunk = 0) ss);
  cover "windows that share a chunk"
    (List.exists
       (fun s ->
         s mod chunk <> 0
         && List.exists
              (fun s' -> s' < s && s' / chunk = s / chunk && s' mod chunk <> 0)
              ss)
       ss);
  cover "an empty window along the scanned axis"
    (List.exists (fun t -> n > 0 && n < t) cut);
  cover "an uneven cut along the scanned axis"
    (List.exists (fun t -> n mod t <> 0) cut);
  cover "a scanned axis of no term" (along && n = 0);
  cover "a scanned axis cut over two mesh axes"
    (along && match c.swhere with Cuts (a, b) -> a = b | _ -> false);
  cover "a scanned axis cut, the other mesh axis holding copies"
    (along && match c.swhere with Minor _ -> true | _ -> false);
  cover "a scanned axis cut beside another"
    (along && List.length (tiles c.swhere) > 1);
  let floats =
    match D.kind k.rdtype with
    | D.Float | D.Complex -> true
    | D.Signed | D.Unsigned | D.Boolean -> false
  in
  let nans = List.exists (function _, Nan _ -> true | _ -> false) c.smarks in
  let infs =
    List.mem Inf (List.map snd c.smarks)
    && List.mem Neg_inf (List.map snd c.smarks)
  in
  cover "a NaN term along a cut axis" (along && floats && nans);
  cover "infinities of both signs along a cut axis" (along && floats && infs);
  let data = scan_data k c in
  cover "a signalling NaN term along a cut axis"
    (along && signals k.rdtype data);
  let x = signalling k.rdtype c.sshape data in
  equal string
    (printed (scan_bits k.rshow c.sr c.saxis x))
    (printed (scan_at c.swhere k.rshow c.sr c.saxis x))

(* A prefix of a cut float32 value along its cut axis, of [n] terms and [m]
   kept. *)
let prefix_case =
  let open Gen in
  with_pp
    (fun ppf (n, m, seed, split4) ->
      Format.fprintf ppf "%d of [%d] split over %d, seed %d" m n
        (if split4 then 4 else 3)
        seed)
    (let* n = of_list [ 7; 4097; 9001; 12289 ] in
     let* m = int_range 0 n in
     let* seed = int in
     let+ split4 = bool in
     (n, m, seed, split4))

let law_prefix (n, m, seed, split4) =
  cover "a prefix ending inside a later window" (m > n / 2);
  let st = Random.State.make [| seed |] in
  let data =
    Array.init n (fun _ ->
        Float.ldexp
          (Random.State.float st 2. -. 1.)
          (Random.State.int st 41 - 20))
  in
  let x = Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] data) in
  let bits y = Array.map float_bits (elements y) in
  let laws (type d) (x : (float, D.float32_elt, d) Nx.t) =
    let prefix y = Nx.slice [ Nx.R (0, m) ] y in
    equal (array string)
      (bits (prefix (Nx.cumsum ~axis:0 x)))
      (bits (Nx.cumsum ~axis:0 (prefix x)))
  in
  if split4 then laws (Nx.place (Set4.split ~axis:0) x)
  else laws (Nx.place (Set3.split ~axis:0) x)

let test_scans_move () =
  (* Read whole, each of four devices would scan every term: [4 n] positions.
     Cut, each device scans its window: an integer scan adds one combination
     per result, a float scan the chunk totals, the units of the chunks
     across windows and the NaN fill, and a float16 one, where windows end
     inside chunks, scans each window's last piece again into float32. *)
  let counted (type v s) (x : (v, s) A.t) f =
    let x = Nx.place (Count4.split ~axis:0) (Nx.Repr.of_array Nx.Host.v x) in
    C.reset ();
    let y = f x in
    let n = C.positions () in
    ignore (elements y);
    n
  in
  let under what n limit got =
    less int ~msg:(Printf.sprintf "%s of %d terms" what n) ~than:limit got
  in
  List.iter
    (fun n ->
      let ints = A.of_array D.Int32 [| n |] (Array.init n Int32.of_int) in
      let floats dt =
        A.of_array dt [| n |]
          (Array.init n (fun i -> Float.of_int (i mod 7) /. 7.))
      in
      under "an integer cumsum" n (2 * n) (counted ints (Nx.cumsum ~axis:0));
      under "a float cummax" n (2 * n)
        (counted (floats D.Float32) (Nx.cummax ~axis:0));
      under "a float cumsum" n (3 * n)
        (counted (floats D.Float32) (Nx.cumsum ~axis:0));
      under "a float16 cumprod" n (4 * n)
        (counted (floats D.Float16) (Nx.cumprod ~axis:0)))
    [ 16384; 9001 ]

let placements =
  group "placements"
    [
      prop ~count:1000
        "an operation over a value at any placement gives its bits on one \
         device"
        placed_case law_placed;
      prop ~count:2000
        "a reduction over cut axes gives its bits on one device" reduced_case
        law_reduced;
      test "a reduction over a cut axis moves partials, not the operand"
        test_partials_move;
      prop ~count:300
        "a float reduction over cut axes of many terms gives its bits on one \
         device"
        lane_case law_lanes;
      test "a float reduction over a cut axis moves partials, not the operand"
        test_float_partials_move;
      prop ~count:400
        "a contraction over cut contracting axes gives its bits on one device"
        ccase
        (law_cut ~pairs:true (module Cpu_sets));
      prop ~count:300
        "a contraction cut over its contracted extent gives its bits on one \
         device, for a library that sums blocks its own way"
        bcase
        (law_cut ~pairs:false (module Blocked_sets));
      prop ~count:200
        "a matrix product over a cut contracting axis gives its bits on one \
         device"
        nx_case law_nx_case;
      test "a contraction over a cut axis contracts each window, not the \
            operands"
        test_contractions_move;
      prop ~count:500 "a scan along cut axes gives its bits on one device"
        scan_case law_scans;
      prop ~count:100 "a scan of a cut value's prefix is its scan's prefix"
        prefix_case law_prefix;
      test "a scan along a cut axis scans each window, not the operand"
        test_scans_move;
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
         constants;
         sets;
         allocation;
         placements;
       ])
