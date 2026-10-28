(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Placing values, through Nx, against a model of which window each device
   holds. *)

open Windtrap
module A = Nx_array
module M = Nx_array.Move

(* Three memory devices, whose memory is the host's, and an io device, whose
   memory the host does not address: its windows copy through
   Rig.Buffer.copy. *)
let m d = if d = 3 then Nx_array_support.io_device () else Nx_support.memory d

module S4 = (val Nx.devices [ m 0; m 1; m 2; m 3 ])
module One = (val Nx.devices [ m 2 ])

let mesh = Nx.Mesh.v S4.v [ ("a", 2); ("b", 2) ]
let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

type value = V : (float, A.Dtype.float32_elt, 'd) Nx.t -> value

(* Placements, and the window each of their devices holds, in order: the device,
   then the first index and extent per axis. [One] is a set of S4's third device
   alone; the others are over S4. *)
type where =
  | Host
  | One
  | On
  | Split of int
  | Cuts of int * int
  | Major of int
  | Minor of int

let pp_where ppf = function
  | Host -> Format.pp_print_string ppf "host"
  | One -> Format.pp_print_string ppf "one"
  | On -> Format.pp_print_string ppf "on"
  | Split a -> Format.fprintf ppf "split %d" a
  | Cuts (x, y) -> Format.fprintf ppf "cuts %d/a %d/b" x y
  | Major x -> Format.fprintf ppf "cut %d/a" x
  | Minor x -> Format.fprintf ppf "cut %d/b" x

(* A placement over the set [v], whose devices [mesh] lays out. *)
let over (type d) (v : d Nx.devices) mesh w : d Nx.Placement.t =
  match w with
  | Host | One -> invalid_arg "over: not a placement of the set"
  | On -> Nx.Placement.on v
  | Split a -> Nx.Placement.split ~axis:a v
  | Cuts (x, y) when x = y -> Nx.Placement.mesh mesh [ (x, [ "a"; "b" ]) ]
  | Cuts (x, y) -> Nx.Placement.mesh mesh [ (x, [ "a" ]); (y, [ "b" ]) ]
  | Major x -> Nx.Placement.mesh mesh [ (x, [ "a" ]) ]
  | Minor x -> Nx.Placement.mesh mesh [ (x, [ "b" ]) ]

let at = over S4.v mesh

let place w (V x) =
  match w with
  | Host -> V (Nx.place Nx.Host.on x)
  | One -> V (Nx.place One.on x)
  | On | Split _ | Cuts _ | Major _ | Minor _ -> V (Nx.place (at w) x)

let windows ?(dev = m) w shape =
  let whole = Array.map (fun n -> (0, n)) shape in
  (* Tile [j] of [tiles]: [n / tiles] elements, one more where
     [j < n mod tiles], after those of the tiles before it. *)
  let cut axis tiles j (b : (int * int) array) =
    let n = shape.(axis) in
    let q = n / tiles and r = n mod tiles in
    b.(axis) <- ((j * q) + Int.min j r, q + if j < r then 1 else 0);
    b
  in
  match w with
  | Host -> [ (Rig.host, whole) ]
  | One -> [ (m 2, whole) ]
  | On -> List.init 4 (fun d -> (dev d, whole))
  | Split a -> List.init 4 (fun d -> (dev d, cut a 4 d (Array.copy whole)))
  | Cuts (x, y) when x = y ->
      List.init 4 (fun d -> (dev d, cut x 4 d (Array.copy whole)))
  | Cuts (x, y) ->
      List.init 4 (fun d ->
          (dev d, cut y 2 (d mod 2) (cut x 2 (d / 2) (Array.copy whole))))
  | Major x ->
      List.init 4 (fun d -> (dev d, cut x 2 (d / 2) (Array.copy whole)))
  | Minor x ->
      List.init 4 (fun d -> (dev d, cut x 2 (d mod 2) (Array.copy whole)))

let where_of rank =
  let axis = Gen.int_range 0 (rank - 1) in
  Gen.with_pp pp_where
    (Gen.one_of
       [
         Gen.of_list [ Host; One; On ];
         Gen.map (fun a -> Split a) axis;
         Gen.map (fun (x, y) -> Cuts (x, y)) (Gen.pair axis axis);
         Gen.map (fun x -> Major x) axis;
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
  let of_shape s = A.of_array A.Dtype.Float32 (Iarray.of_array s) (floats s) in
  let view m a = Option.get (A.move m a) in
  match layout with
  | Contiguous -> of_shape shape
  | Transposed ->
      let rev a = Array.init r (fun i -> a.(r - 1 - i)) in
      view
        (M.Permute (Iarray.of_array (rev (Array.init r Fun.id))))
        (of_shape (rev shape))
  | Broadcast ->
      let row = Array.mapi (fun i n -> if i = 0 then 1 else n) shape in
      view (M.Broadcast (Iarray.of_array shape)) (of_shape row)

let host shape =
  V
    (Nx.Repr.of_array Nx.Host.v
       (A.of_array A.Dtype.Float32 (Iarray.of_array shape) (floats shape)))

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

(* [a]'s elements, read on the host. *)
let elements a =
  if Rig.reaches Rig.host (A.device a) then A.to_array a
  else A.to_array (A.to_device Rig.host a)

(* [x] holds, on each device of [w], that device's window of [xs]. *)
let holds ?dev xs shape w x =
  let arrays = shards x in
  let ws = windows ?dev w shape in
  equal int ~msg:"one array per device" (List.length ws) (Array.length arrays);
  List.iteri
    (fun j (d, b) ->
      equal string ~msg:"its device" (Rig.name d)
        (Rig.name (A.device arrays.(j)));
      equal (list float_exact) ~msg:"its window" (expected xs shape b)
        (Array.to_list (elements arrays.(j))))
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
  Gen.array ~size:(Gen.constant rank) (Gen.of_list [ 0; 1; 3; 4; 5; 8 ])

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
  | Split _ | Cuts _ | Major _ | Minor _ -> true
  | Host | One | On -> false

(* Whether the io device's window of [a], of shape [s], at [w] reaches more of
   [a]'s elements than it holds: placing then packs its elements into memory the
   host does not address. *)
let packed a s w =
  match windows w s with
  | [ _; _; _; (_, b) ] ->
      let range (start, count) = { M.start; count; step = 1 } in
      let v =
        Option.get (A.move (M.Slice (Iarray.of_array (Array.map range b))) a)
      in
      let lo, hi = A.Layout.span (A.layout v) in
      let n = Array.fold_left (fun n (_, k) -> n * k) 1 b in
      n > 0 && hi - lo > n
  | _ -> false

let laws =
  group "laws"
    [
      prop "each device holds its window of a host value placed" case
        (fun (s, l, w, _) ->
          cover "split" (split w);
          let counts a = List.map (fun (_, b) -> snd b.(a)) (windows w s) in
          let axes = List.init (Array.length s) Fun.id in
          cover "windows of two extents along an axis"
            (List.exists
               (fun a -> List.length (List.sort_uniq compare (counts a)) > 1)
               axes);
          cover "an empty window of a non-empty axis"
            (List.exists
               (fun a -> s.(a) > 0 && List.mem 0 (counts a))
               axes);
          let a = source l s in
          cover "a window packed into memory the host does not address"
            (packed a s w);
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
    let x =
      Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array shape) xs)
    in
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
      test "an empty window holds no memory of the value's" (fun () ->
          let (V x) = host [| 1 |] in
          let a = require_some (Nx.Repr.array x) in
          let b =
            require_some (Nx.Repr.shards (Nx.place (S4.split ~axis:0) x))
          in
          for j = 1 to 3 do
            let e = A.buffer (Iarray.get b j) in
            equal bool ~msg:"shares" false (Rig.Buffer.overlaps (A.buffer a) e);
            equal int ~msg:"its bytes" 0 (Rig.Buffer.length e)
          done);
      narrow "int4" A.Dtype.Int4 Fun.id;
      narrow "int16" A.Dtype.Int16 Fun.id;
      narrow "bool" A.Dtype.Bool (fun i -> i mod 2 = 0);
    ]

(* An operation of a kind, on a value of any set. *)
type kind = { run : 'd. (float, A.Dtype.float32_elt, 'd) Nx.t -> unit }

let refusals =
  group "refusals"
    [
      test "a split of an axis the value lacks raises" (fun () ->
          let (V x) = host [| 6 |] in
          invalid ~by:"Nx.place" (fun () -> Nx.place (S4.split ~axis:1) x));
      test "a lost device raises when placed from, and its facts answer"
        (fun () ->
          let d = Result.get_ok (Rig.memory_device "nx-place-lost") in
          let module L = (val Nx.devices [ d ]) in
          let x = Nx.Repr.of_array L.v (A.create d A.Dtype.Float32 [| 4 |]) in
          Rig.close d;
          raises_match
            (function Rig.Lost _ -> true | _ -> false)
            (fun () -> Nx.place Nx.Host.on x);
          equal (array int) [| 4 |] (Nx.shape x);
          equal string "nx-place-lost"
            (Format.asprintf "%a" Nx.Placement.pp (Option.get (Nx.placement x))));
      test "every kind of operation raises a lost device, and a live one answers"
        (fun () ->
          let d = Result.get_ok (Rig.memory_device "nx-place-lost-ops") in
          let e = Result.get_ok (Rig.memory_device "nx-place-live-ops") in
          let module L = (val Nx.devices [ d ]) in
          let module E = (val Nx.devices [ e ]) in
          let square on =
            Nx.place on (Nx.create Nx.float32 [| 2; 2 |] [| 4.; 1.; 2.; 3. |])
          in
          let x = square L.on and y = square E.on in
          Rig.close d;
          let lost = function Rig.Lost (t, _) -> Rig.equal t d | _ -> false in
          let kinds =
            [
              ("map", { run = (fun x -> ignore (Nx.neg x)) });
              ("reduce", { run = (fun x -> ignore (Nx.sum x)) });
              ("contract", { run = (fun x -> ignore (Nx.matmul x x)) });
              ("sort", { run = (fun x -> ignore (Nx.sort x)) });
              ("fft", { run = (fun x -> ignore (Nx.rfft Nx.complex64 x)) });
              ("lu", { run = (fun x -> ignore (Nx.lu x)) });
            ]
          in
          List.iter
            (fun (name, k) ->
              raises_match ~msg:name lost (fun () -> k.run x);
              k.run y)
            kinds);
    ]

(* Reshapes of a cut value. A reshape pairs the axes of one product of the
   extents before them in order, each with the result's next axis of that
   product and of its extent, and keeps a cut of an axis on its pair. It keeps
   a cut of an unpaired axis [a] of [s], of more than one element after a
   positive product, in [t] tiles, on the result's axis of more than one
   element of that product, where [t] divides both extents: each window then
   holds whole rows on both sides. Otherwise it reads the value whole along
   that axis, which raises where the value has elements, since a device then
   needs rows it does not hold, and the result is whole along it. *)
let before s a = Array.fold_left ( * ) 1 (Array.sub s 0 a)

let lands s s' a t =
  let p = before s a in
  let run s =
    List.filter (fun x -> before s x = p) (List.init (Array.length s) Fun.id)
  in
  (* Each axis of [s]'s run with its pair, in order. *)
  let rec pairs xs ys =
    match xs with
    | [] -> []
    | x :: xs -> (
        let rec drop = function
          | [] -> None
          | y :: ys -> if s'.(y) = s.(x) then Some (y, ys) else drop ys
        in
        match drop ys with
        | Some (y, ys) -> (x, y) :: pairs xs ys
        | None -> pairs xs ys)
  in
  match List.assoc_opt a (pairs (run s) (run s')) with
  | Some a' -> Some a'
  | None when p > 0 && s.(a) > 1 -> (
      match List.find_opt (fun a' -> s'.(a') > 1) (run s') with
      | Some a' when s.(a) mod t = 0 && s'.(a') mod t = 0 -> Some a'
      | Some _ | None -> None)
  | None -> None

let reshaped s s' w =
  match w with
  | Host | One | On -> w
  | Split a -> ( match lands s s' a 4 with Some a' -> Split a' | None -> On)
  | Cuts (x, y) when x = y -> (
      match lands s s' x 4 with Some x' -> Cuts (x', x') | None -> On)
  | Cuts (x, y) -> (
      match (lands s s' x 2, lands s s' y 2) with
      | Some x', Some y' -> Cuts (x', y')
      | Some x', None -> Major x'
      | None, Some y' -> Minor y'
      | None, None -> On)
  | Major x -> ( match lands s s' x 2 with Some x' -> Major x' | None -> On)
  | Minor y -> ( match lands s s' y 2 with Some y' -> Minor y' | None -> On)

(* The placement a reshape of [s] to [s'] reads a value at [w] at: [w] with
   each cut that keeps no axis undone. *)
let read s s' w =
  let kept a t = lands s s' a t <> None in
  match w with
  | Host | One | On -> w
  | Split a -> if kept a 4 then w else On
  | Cuts (x, y) when x = y -> if kept x 4 then w else On
  | Cuts (x, y) -> (
      match (kept x 2, kept y 2) with
      | true, true -> w
      | true, false -> Major x
      | false, true -> Minor y
      | false, false -> On)
  | Major x -> if kept x 2 then w else On
  | Minor y -> if kept y 2 then w else On

(* A reshape of [s] to [s'] that merges or factors the axis [cut] of [s], past a
   unit axis or not, of a value laid out as [layout] at [w]; [wanted] is a
   placement of its result. *)
type reshape = {
  s : int array;
  s' : int array;
  merge : bool;
  unit : bool;
  cut : int;
  layout : layout;
  w : where;
  wanted : where;
}

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_reshape ppf c =
  Format.fprintf ppf "%a %a at %a to %a, wanted at %a" pp_layout c.layout
    pp_ints c.s pp_where c.w pp_ints c.s' pp_where c.wanted

(* [b] blocks of [r] elements between leading and trailing axes: one axis of
   [b × r] factored into [b; r], a unit axis before them or not, or merged
   from them. The cut is mostly the axis of [b × r], or of [b]. *)
let reshape_case =
  let open Gen in
  let axes = list ~size:(int_range 0 1) (of_list [ 1; 2; 3 ]) in
  let* pre = axes
  and+ post = axes
  and+ b = of_list [ 0; 1; 2; 3; 4; 6; 8; 12 ]
  and+ r = of_list [ 1; 2; 3; 4 ]
  and+ merge = bool
  and+ unit = bool in
  let pre = Array.of_list pre and post = Array.of_list post in
  let whole = Array.concat [ pre; [| b * r |]; post ] in
  let parts =
    Array.concat [ pre; (if unit then [| 1; b; r |] else [| b; r |]); post ]
  in
  let s, s' = if merge then (parts, whole) else (whole, parts) in
  let cut = Array.length pre + if merge && unit then 1 else 0 in
  let into = Array.length pre + if (not merge) && unit then 1 else 0 in
  (* A placement of a value of [s], mostly cutting its axis [major]. *)
  let where s major =
    let axis =
      frequency [ (3, constant major); (1, int_range 0 (Array.length s - 1)) ]
    in
    frequency
      [
        (2, map (fun a -> Split a) axis);
        (1, map (fun a -> Cuts (a, a)) axis);
        (2, map (fun (a, a') -> Cuts (a, a')) (pair axis axis));
        (1, map (fun (a, a') -> Cuts (a', a)) (pair axis axis));
        (1, map (fun a -> Major a) axis);
        (1, map (fun a -> Minor a) axis);
      ]
  in
  let+ w = where s cut
  and+ wanted = where s' into
  and+ layout = of_list [ Contiguous; Transposed; Broadcast ] in
  { s; s'; merge; unit; cut; layout; w; wanted }

let reshape_case = Gen.with_pp pp_reshape reshape_case

(* Four memory devices, computed by nx.cpu: a reshape of a strided window
   copies it. *)
module Mem = (val Nx.devices (List.init 4 Nx_support.memory))

let on_mem = over Mem.v (Nx.Mesh.v Mem.v [ ("a", 2); ("b", 2) ])
let placement = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* Each cut axis of [w] with its number of tiles. *)
let cuts = function
  | Host | One | On -> []
  | Split a -> [ (a, 4) ]
  | Cuts (x, y) when x = y -> [ (x, 4) ]
  | Cuts (x, y) -> [ (x, 2); (y, 2) ]
  | Major x | Minor x -> [ (x, 2) ]

let law_reshape c =
  (* The result axis the factored or merged axis regroups into. *)
  let into =
    match lands c.s c.s' c.cut 1 with
    | Some a' when c.s'.(a') <> c.s.(c.cut) -> Some a'
    | Some _ | None -> None
  in
  let w' = reshaped c.s c.s' c.w in
  let kept =
    into <> None
    && List.exists
         (fun (a, t) -> a = c.cut && lands c.s c.s' a t <> None)
         (cuts c.w)
  in
  cover "a factoring keeps its cut" (kept && not c.merge);
  cover "a merge keeps its cut" (kept && c.merge);
  cover "a factoring past a unit axis keeps its cut"
    (kept && c.unit && not c.merge);
  cover "a cut of a strided source kept" (kept && c.layout <> Contiguous);
  cover "tiles that divide the merged extent alone raise"
    (c.merge && c.w = Split c.cut
    &&
    match into with
    | Some a' -> c.s.(c.cut) mod 4 <> 0 && c.s'.(a') mod 4 = 0
    | None -> false);
  cover "an uneven cut kept"
    (List.exists
       (fun (a, t) -> c.s.(a) mod t <> 0 && lands c.s c.s' a t <> None)
       (cuts c.w));
  cover "two cuts kept" (match w' with Cuts (x, y) -> x <> y | _ -> false);
  cover "no element" (Array.mem 0 c.s);
  let r = read c.s c.s' c.w in
  let refused = r <> c.w && not (Array.mem 0 c.s) in
  cover "a cut kept beside one that raises"
    (refused && match r with Major _ | Minor _ -> true | _ -> false);
  cover "a cut that keeps no axis raises" refused;
  let a = source c.layout c.s in
  let x = Nx.place (on_mem c.w) (Nx.Repr.of_array Nx.Host.v a) in
  let x =
    if not refused then x
    else begin
      let prefix =
        Format.asprintf
          "Nx.reshape: operand 0 lies at %a and is read at %a; Nx.place ("
          Nx.Placement.pp (on_mem c.w) Nx.Placement.pp (on_mem r)
      in
      raises_match
        (Exn.invalid_arg ~substring:prefix)
        (fun () -> Nx.reshape c.s' x);
      Nx.place (on_mem r) x
    end
  in
  let y = Nx.reshape c.s' x in
  equal placement (on_mem w') (Option.get (Nx.placement y));
  holds ~dev:Nx_support.memory (A.to_array a) c.s' w' (V y)

(* A reshape of a value of every set, wanted at a placement: each device
   computes its window there. *)
let law_wanted c =
  (* The wanted cuts of an axis of [s'] that [s]'s regroups into: in tiles
     that divide both extents, kept, or else undone. *)
  let regrouped kept =
    List.exists
      (fun (a', t) ->
        match lands c.s' c.s a' 1 with
        | Some a when c.s.(a) <> c.s'.(a') ->
            lands c.s' c.s a' t <> None = kept
        | Some _ | None -> false)
      (cuts c.wanted)
  in
  cover "a wanted cut kept along a regrouped axis" (regrouped true);
  cover "a wanted cut across rows undone" (regrouped false);
  let n = Array.fold_left ( * ) 1 c.s in
  let x = Nx.reshape c.s (Nx.arange Nx.float32 0 n 1) in
  let y = Nx.place (on_mem c.wanted) (Nx.reshape c.s' x) in
  equal placement (on_mem c.wanted) (Option.get (Nx.placement y));
  holds ~dev:Nx_support.memory (floats c.s) c.s' c.wanted (V y)

(* gpt-oss's 2880 inputs are 90 blocks of 32: four devices divide the
   elements but not the blocks. *)
let blocks n =
  let s = [| 2; n; 32 |] in
  let x =
    Nx.Repr.of_array Nx.Host.v
      (A.of_array A.Dtype.Float32 (Iarray.of_array s) (floats s))
  in
  let y = Nx.reshape [| 2; n * 32 |] (Nx.place (Mem.split ~axis:1) x) in
  (Option.get (Nx.placement y), y)

let reshapes =
  group "reshapes"
    [
      prop ~count:1000 "a reshape keeps a cut that holds whole rows"
        reshape_case law_reshape;
      prop "a reshape of a value of every set is computed where it is wanted"
        reshape_case law_wanted;
      (* A unit axis the result drops keeps no cut: a device past the first
         tile holds none of its rows, so the reshape raises, and read where
         it names, the result keeps the paired axes' cuts. *)
      cases "unit axes cut by a mesh pair with the result's in order"
        ~name:(fun (s, w, s', _, _) ->
          Format.asprintf "%a at %a to %a" pp_ints s pp_where w pp_ints s')
        [
          ([| 1; 1 |], Cuts (0, 1), [| 1 |], Major 0, Major 0);
          ([| 1; 1 |], Cuts (0, 1), [| 1; 1; 1 |], Cuts (0, 1), Cuts (0, 1));
          ([| 1; 8 |], Cuts (0, 1), [| 8 |], Minor 1, Minor 0);
          ([| 1; 1; 4 |], Cuts (0, 1), [| 1; 4 |], Major 0, Major 0);
        ]
        (fun (s, w, s', r, w') ->
          let x =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array A.Dtype.Float32 (Iarray.of_array s) (floats s))
          in
          let x = Nx.place (on_mem w) x in
          if r <> w then
            raises_match
              (Exn.invalid_arg
                 ~substring:
                   (Format.asprintf "is read at %a;" Nx.Placement.pp
                      (on_mem r)))
              (fun () -> Nx.reshape s' x);
          let y = Nx.reshape s' (Nx.place (on_mem r) x) in
          equal placement (on_mem w') (Option.get (Nx.placement y));
          holds ~dev:Nx_support.memory (floats s) s' w' (V y));
      test "blocks that four devices divide stay cut when merged" (fun () ->
          let p, y = blocks 92 in
          equal placement (Mem.split ~axis:1) p;
          holds ~dev:Nx_support.memory
            (floats [| 2; 92; 32 |])
            [| 2; 2944 |] (Split 1) (V y));
      test "2880 elements of 90 blocks over four devices raise" (fun () ->
          raises
            (Invalid_argument
               "Nx.reshape: operand 0 lies at split ~axis:1 [m0; m1; m2; m3] \
                and is read at on [m0; m1; m2; m3]; Nx.place (Placement.on s) \
                x")
            (fun () -> blocks 90));
    ]

(* Leading axes: a placement's cuts move up by one with an axis in front, and
   down without the first, whose cuts become copies. *)

let lead = function
  | (Host | One | On) as w -> w
  | Split a -> Split (a + 1)
  | Cuts (x, y) -> Cuts (x + 1, y + 1)
  | Major x -> Major (x + 1)
  | Minor x -> Minor (x + 1)

let unlead = function
  | (Host | One | On) as w -> w
  | Split 0 | Major 0 | Minor 0 | Cuts (0, 0) -> On
  | Split a -> Split (a - 1)
  | Cuts (0, y) -> Minor (y - 1)
  | Cuts (x, 0) -> Major (x - 1)
  | Cuts (x, y) -> Cuts (x - 1, y - 1)
  | Major x -> Major (x - 1)
  | Minor x -> Minor (x - 1)

let over_s4 = Gen.bind (Gen.int_range 1 3) where_of

let placed : S4.d Nx.Placement.t testable =
  Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let leading =
  group "leading axes"
    [
      prop "an axis in front moves each cut up by one" over_s4 (fun w ->
          cover "a cut" (split w);
          match w with
          | Host | One -> ()
          | _ ->
              equal placed (at (lead w))
                (Nx.Placement.with_leading_axis (at w)));
      prop "each device holds the new axis whole by its window" case
        (fun (s, l, w, _) ->
          cover "a cut" (split w);
          match w with
          | Host | One -> ()
          | _ ->
              let n = 2 in
              let s' = Array.append [| n |] s in
              let a = source l s' in
              let p = Nx.Placement.with_leading_axis (at w) in
              let x = Nx.place p (Nx.Repr.of_array Nx.Host.v a) in
              holds (A.to_array a) s' (lead w) (V x));
      prop "without the first axis its cuts become copies" over_s4 (fun w ->
          cover "a cut of the first axis"
            (match w with
            | Split 0 | Major 0 | Minor 0 | Cuts (0, _) | Cuts (_, 0) -> true
            | _ -> false);
          cover "a cut of another axis"
            (match w with
            | Split a | Major a | Minor a -> a > 0
            | Cuts (x, y) -> x > 0 || y > 0
            | Host | One | On -> false);
          match w with
          | Host | One -> ()
          | _ ->
              equal placed (at (unlead w))
                (Nx.Placement.without_leading_axis (at w)));
      test "a placement that cuts no axis is itself" (fun () ->
          let p = S4.on and q = One.on in
          equal bool true (Nx.Placement.with_leading_axis p == p);
          equal bool true (Nx.Placement.without_leading_axis p == p);
          equal bool true (Nx.Placement.with_leading_axis q == q);
          equal bool true (Nx.Placement.without_leading_axis q == q));
      test "a cut of the last axis a layout has refuses an axis in front"
        (fun () ->
          let last = A.Layout.max_rank - 1 in
          equal placed ~msg:"the axis before it moves up" (S4.split ~axis:last)
            (Nx.Placement.with_leading_axis (S4.split ~axis:(last - 1)));
          invalid ~by:"Nx.Placement.with_leading_axis" (fun () ->
              Nx.Placement.with_leading_axis (S4.split ~axis:last)));
    ]

let () =
  exit (run "nx place" [ laws; reshapes; sharing; refusals; leading ])
