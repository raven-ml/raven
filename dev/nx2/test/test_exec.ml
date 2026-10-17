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
      let view = Option.get (A.move (M.Permute rev) base) in
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
      Option.get (A.move (M.Reshape shape) tail)
  | Broadcast ->
      if Array.length data = 0 then A.of_array dt shape data
      else
        let one = A.of_array dt [||] [| data.(0) |] in
        Option.get (A.move (M.Broadcast shape) one)

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

let laws =
  group "laws"
    [
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
        "a declined core kind raises naming the kernels, kind, dtypes, device \
         and move" (fun () ->
          let a =
            Nx.place Dec.on (on_host Plain D.Float32 [| 2 |] [| 1.; 2. |])
          in
          raises
            (Invalid_argument
               "Nx.add: nx.test does not compute Add on float32, float32, \
                float32 (m1); place its operands with Nx.place on a set whose \
                kernels compute it") (fun () -> Nx.add a a));
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

let sets =
  group "sets"
    [
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

let allocation =
  group "allocation"
    [
      test "a sub-byte result leaves its last byte's spare bits zero"
        test_sub_byte_tail;
      test "an add of one-element host values allocates its budget" (fun () ->
          let a = on_host Plain D.Float32 [| 1 |] [| 1. |] in
          ignore (Nx.add a a);
          let n = 1000 in
          let before = Gc.minor_words () in
          for _ = 1 to n do
            ignore (Sys.opaque_identity (Nx.add a a))
          done;
          let per = (Gc.minor_words () -. before) /. Float.of_int n in
          at_most (float 0.5) ~than:51. per);
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
      cover "a constant computed whole along an axis, beside another cut"
        (List.mem a cut && List.length cut > 1)
  | Combine -> ());
  let x = Nx.Repr.of_array Nx.Host.v (A.of_array k.dtype s data) in
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
  cover "a sub-byte dtype over a cut axis" (over && D.bits k.rdtype < 8);
  let x = Nx.Repr.of_array Nx.Host.v (A.of_array k.rdtype s data) in
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
        late, and a NaN term the other side. *)
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
  let x = Nx.Repr.of_array Nx.Host.v (A.of_array dt c.lshape data) in
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

(* Two vectors split over three devices, their contracting axis. *)
let split_vectors =
  let open Gen in
  let* n = int_range 0 9 in
  let+ xs = array ~size:(constant n) (of_list floats)
  and+ ys = array ~size:(constant n) (of_list floats) in
  (xs, ys)

let law_split_dot (xs, ys) =
  let n = Array.length xs in
  cover "an empty window" (n > 0 && n < 3);
  cover "an uneven cut" (n mod 3 <> 0);
  let host v = Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] v) in
  let x = host xs and y = host ys in
  let split v = Nx.place (Set3.split ~axis:0) v in
  let bits v = Array.map float_bits (elements v) in
  equal (array string) (bits (Nx.dot x y)) (bits (Nx.dot (split x) (split y)))

let placements =
  group "placements"
    [
      prop "a contraction of operands cut along its contracting axis gives \
            its bits on one device"
        split_vectors law_split_dot;
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
    ]

let () =
  exit
    (run "nx exec"
       [ laws; broadcasting; kernels; constants; sets; allocation; placements ])
