(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Placing values, through Nx, against a model of which window each device
   holds. *)

open Windtrap
module A = Nx_array
module M = Nx_array.Move

let m = Nx_support.memory

module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])
module One = (val Nx.devices [ m 2 ])

let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]
let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

type value = V : (float, A.Dtype.float32_elt, 'd) Nx.t -> value

(* Placements, and the window each of their devices holds, in order: the device,
   then the first index and extent per axis. [One] is a set of S4's third device
   alone; the others are over S4. *)
type where = Host | One | On | Split of int | Cuts of int * int | Minor of int

let pp_where ppf = function
  | Host -> Format.pp_print_string ppf "host"
  | One -> Format.pp_print_string ppf "one"
  | On -> Format.pp_print_string ppf "on"
  | Split a -> Format.fprintf ppf "split %d" a
  | Cuts (x, y) -> Format.fprintf ppf "cuts %d/a %d/b" x y
  | Minor x -> Format.fprintf ppf "cut %d/b" x

let place w (V x) =
  match w with
  | Host -> V (Nx.place Nx.Host.on x)
  | One -> V (Nx.place One.on x)
  | On -> V (Nx.place S4.on x)
  | Split a -> V (Nx.place (S4.split ~axis:a) x)
  | Cuts (x', y) when x' = y ->
      V (Nx.place (Nx.Placement.mesh mesh [ (x', [ "a"; "b" ]) ]) x)
  | Cuts (x', y) ->
      V (Nx.place (Nx.Placement.mesh mesh [ (x', [ "a" ]); (y, [ "b" ]) ]) x)
  | Minor x' -> V (Nx.place (Nx.Placement.mesh mesh [ (x', [ "b" ]) ]) x)

let windows w shape =
  let whole = Array.map (fun n -> (0, n)) shape in
  let cut axis tiles j (b : (int * int) array) =
    let size = shape.(axis) / tiles in
    b.(axis) <- (j * size, size);
    b
  in
  match w with
  | Host -> [ (Rig.host, whole) ]
  | One -> [ (m 2, whole) ]
  | On -> List.init 4 (fun d -> (m d, whole))
  | Split a -> List.init 4 (fun d -> (m d, cut a 4 d (Array.copy whole)))
  | Cuts (x, y) when x = y ->
      List.init 4 (fun d -> (m d, cut x 4 d (Array.copy whole)))
  | Cuts (x, y) ->
      List.init 4 (fun d ->
          (m d, cut y 2 (d mod 2) (cut x 2 (d / 2) (Array.copy whole))))
  | Minor x ->
      List.init 4 (fun d -> (m d, cut x 2 (d mod 2) (Array.copy whole)))

let where_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.with_pp pp_where
    (Gen.one_of
       [
         Gen.of_list [ Host; One; On ];
         Gen.map (fun a -> Split a) axis;
         Gen.map (fun (x, y) -> Cuts (x, y)) (Gen.pair axis axis);
         Gen.map (fun x -> Minor x) axis;
       ])

(* Host values whose arrays lie in C order, transposed, or broadcast along axis
   0. *)
type layout = Contiguous | Transposed | Broadcast

let pp_layout ppf l =
  Format.pp_print_string ppf
    (match l with
    | Contiguous -> "contiguous"
    | Transposed -> "transposed"
    | Broadcast -> "broadcast")

let floats shape = Array.init (Array.fold_left ( * ) 1 shape) float_of_int

let source layout shape =
  let r = Array.length shape in
  let of_shape s = A.of_array A.Dtype.Float32 s (floats s) in
  let view m a = Option.get (A.move m a) in
  match layout with
  | Contiguous -> of_shape shape
  | Transposed ->
      let rev a = Array.init r (fun i -> a.(r - 1 - i)) in
      view (M.Permute (rev (Array.init r Fun.id))) (of_shape (rev shape))
  | Broadcast ->
      let row = Array.mapi (fun i n -> if i = 0 then 1 else n) shape in
      view (M.Broadcast shape) (of_shape row)

let host shape =
  V
    (Nx.Repr.of_array Nx.Host.v
       (A.of_array A.Dtype.Float32 shape (floats shape)))

(* The elements of window [b] of [shape] in C order, from [xs], the whole's
   elements in C order. *)
let expected xs shape b =
  let r = Array.length shape in
  let out = ref [] in
  let rec go a off =
    if a = r then out := xs.(off) :: !out
    else
      let start, n = b.(a) in
      for i = start to start + n - 1 do
        go (a + 1) ((off * shape.(a)) + i)
      done
  in
  go 0 0;
  List.rev !out

let shards (V x) = Iarray.to_array (require_some (Nx.Repr.shards x))

(* [x] holds, on each device of [w], that device's window of [xs]. *)
let holds xs shape w x =
  let arrays = shards x in
  let ws = windows w shape in
  equal int ~msg:"one array per device" (List.length ws) (Array.length arrays);
  List.iteri
    (fun j (d, b) ->
      equal string ~msg:"its device" (Rig.name d)
        (Rig.name (A.device arrays.(j)));
      equal (list float_exact) ~msg:"its window" (expected xs shape b)
        (Array.to_list (A.to_array arrays.(j))))
    ws

(* Each array of [y], at [w], shares the memory of an array of [x] or holds no
   more than its window's bytes. *)
let allocates_its_window shape x w y =
  let sources = shards x in
  List.iteri
    (fun j (_, b) ->
      let a = (shards y).(j) in
      let shared =
        Array.exists
          (fun s -> Rig.Buffer.overlaps (A.buffer s) (A.buffer a))
          sources
      in
      if not shared then begin
        cover "a destination that copies" true;
        let n = Array.fold_left (fun n (_, k) -> n * k) 1 b in
        at_most int ~msg:"its buffer's bytes"
          ~than:(A.Dtype.bytes A.Dtype.Float32 n)
          (Rig.Buffer.length (A.buffer a))
      end)
    (windows w shape)

let shape_of rank =
  Gen.array ~size:(Gen.constant rank) (Gen.of_list [ 0; 4; 8 ])

let case =
  Gen.with_pp
    (fun ppf (s, l, w, w') ->
      Format.fprintf ppf "%a [%s] at %a then %a" pp_layout l
        (String.concat "; " (Array.to_list (Array.map string_of_int s)))
        pp_where w pp_where w')
    (Gen.bind (Gen.int_range 1 3) (fun rank ->
         Gen.map
           (fun ((s, l), (w, w')) -> (s, l, w, w'))
           (Gen.pair
              (Gen.pair (shape_of rank)
                 (Gen.of_list [ Contiguous; Transposed; Broadcast ]))
              (Gen.pair (where_of rank) (where_of rank)))))

let split = function
  | Split _ | Cuts _ | Minor _ -> true
  | Host | One | On -> false

let laws =
  group "laws"
    [
      prop "each device holds its window of a host value placed" case
        (fun (s, l, w, _) ->
          cover "split" (split w);
          let a = source l s in
          let x = V (Nx.Repr.of_array Nx.Host.v a) in
          holds (A.to_array a) s w (place w x));
      prop ~count:500 "each device holds its window after a second placement"
        case (fun (s, l, w, w') ->
          cover "a change of arrangement" (w <> w');
          cover "split to split on another axis"
            (match (w, w') with Split a, Split b -> a <> b | _ -> false);
          cover "a coarser split of one axis"
            (match (w, w') with Split a, Minor b -> a = b | _ -> false);
          cover "a strided source" (l <> Contiguous && split w');
          let a = source l s in
          let x = V (Nx.Repr.of_array Nx.Host.v a) in
          holds (A.to_array a) s w' (place w' (place w x)));
      prop ~count:500 "each device allocates no more than its window" case
        (fun (s, l, w, w') ->
          cover "split to split on another axis"
            (match (w, w') with Split a, Split b -> a <> b | _ -> false);
          let x = V (Nx.Repr.of_array Nx.Host.v (source l s)) in
          let y = place w x in
          allocates_its_window s x w y;
          allocates_its_window s y w' (place w' y));
      prop "a value placed back on the host has its elements" case
        (fun (s, l, w, _) ->
          let a = source l s in
          let (V back) =
            place Host (place w (V (Nx.Repr.of_array Nx.Host.v a)))
          in
          equal (list float_exact)
            (Array.to_list (A.to_array a))
            (Array.to_list (A.to_array (require_some (Nx.Repr.array back)))));
    ]

(* A narrow dtype across a change of arrangement, by runs where whole bytes copy
   and element by element where a window's column is half a byte. *)
let narrow (type v s) name (dt : (v, s) A.Dtype.t) (of_int : int -> v) =
  let keeps shape =
    let xs =
      Array.init (Array.fold_left ( * ) 1 shape) (fun i -> of_int (i mod 7))
    in
    let x = Nx.Repr.of_array Nx.Host.v (A.of_array dt shape xs) in
    let y = Nx.place (S4.split ~axis:1) (Nx.place (S4.split ~axis:0) x) in
    let back = Nx.place Nx.Host.on y in
    equal bool true (A.to_array (require_some (Nx.Repr.array back)) = xs)
  in
  test (name ^ " keeps its elements across a change of arrangement") (fun () ->
      keeps [| 4; 8 |];
      keeps [| 4; 4 |])

let sharing =
  group "sharing"
    [
      test "a value already at a placement keeps its arrays" (fun () ->
          let (V x) = place On (host [| 4 |]) in
          let a = require_some (Nx.Repr.shards x) in
          let b = require_some (Nx.Repr.shards (Nx.place S4.on x)) in
          equal bool true (Iarray.for_all2 ( == ) a b));
      test "a set over the host's device reads the host's array" (fun () ->
          let module Fast =
            (val Nx.devices ~kernels:(module Nx_cpu) [ Rig.host ])
          in
          let (V x) = host [| 4 |] in
          let a = require_some (Nx.Repr.array x) in
          let b = require_some (Nx.Repr.array (Nx.place Fast.on x)) in
          equal bool true (a == b));
      test "a device that maps the host's memory borrows it" (fun () ->
          let (V x) = host [| 1024 |] in
          let a = require_some (Nx.Repr.array x) in
          let b = require_some (Nx.Repr.shards (Nx.place S4.on x)) in
          equal bool true
            (Rig.Buffer.overlaps (A.buffer a) (A.buffer (Iarray.get b 1))));
      narrow "int4" A.Dtype.Int4 Fun.id;
      narrow "int16" A.Dtype.Int16 Fun.id;
      narrow "bool" A.Dtype.Bool (fun i -> i mod 2 = 0);
    ]

let refusals =
  group "refusals"
    [
      test "a split that does not divide the shape raises" (fun () ->
          let (V x) = host [| 6 |] in
          invalid ~by:"Nx.place" (fun () -> Nx.place (S4.split ~axis:0) x));
      test "a lost device raises when placed from, and its facts answer"
        (fun () ->
          let d =
            match Rig.memory_device "nx-place-lost" with
            | Ok d -> d
            | Error e -> failwith e
          in
          let module L = (val Nx.devices [ d ]) in
          let x = Nx.Repr.of_array L.v (A.create d A.Dtype.Float32 [| 4 |]) in
          Rig.close d;
          raises_match
            (function Rig.Lost _ -> true | _ -> false)
            (fun () -> Nx.place Nx.Host.on x);
          equal (array int) [| 4 |] (Nx.shape x);
          equal string "nx-place-lost"
            (Format.asprintf "%a" Nx.Placement.pp (Option.get (Nx.placement x))));
    ]

let () = exit (run "nx place" [ laws; sharing; refusals ])
