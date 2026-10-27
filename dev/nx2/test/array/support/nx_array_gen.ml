(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Layouts and movements drawn for nx2's suites, their positions by brute force,
   and the witnesses and references the array suites share. *)

open Windtrap
module L = Nx_array.Layout
module M = Nx_array.Move

(* Indices and positions, by brute force *)

(* Every index of shape [s], in C order. *)
let indices s =
  let r = Array.length s in
  let rec from i =
    if i = r then [ [] ]
    else
      let rest = from (i + 1) in
      List.concat_map
        (fun j -> List.map (fun idx -> j :: idx) rest)
        (List.init s.(i) Fun.id)
  in
  List.map Array.of_list (from 0)

let position l idx =
  let p = ref (L.offset l) in
  Array.iteri (fun i j -> p := !p + (j * L.stride l i)) idx;
  !p

let positions l = List.map (position l) (indices (Iarray.to_array (L.shape l)))
(* Generators *)

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_move ppf = function
  | M.Reshape s -> Format.fprintf ppf "Reshape %a" pp_ints (Iarray.to_array s)
  | M.Broadcast s ->
      Format.fprintf ppf "Broadcast %a" pp_ints (Iarray.to_array s)
  | M.Permute p -> Format.fprintf ppf "Permute %a" pp_ints (Iarray.to_array p)
  | M.Slice rs ->
      Format.fprintf ppf "Slice [%s]"
        (String.concat "; "
           (Iarray.to_list
              (Iarray.map
                 (fun (x : M.range) ->
                   Printf.sprintf "%d,%d,%d" x.start x.count x.step)
                 rs)))
  | M.Window ws ->
      Format.fprintf ppf "Window [%s]"
        (String.concat "; "
           (Iarray.to_list
              (Iarray.map
                 (fun (w : M.window) ->
                   Printf.sprintf "axis %d size %d step %d dilation %d" w.axis
                     w.size w.step w.dilation)
                 ws)))

let const n = Gen.constant ~pp:Format.pp_print_int n
let ints_of l = Gen.of_list ~pp:Format.pp_print_int l
let extent = Gen.frequency [ (1, Gen.int_range 0 0); (6, Gen.int_range 1 4) ]
let shape = Gen.array ~size:(Gen.int_range 0 4) extent

(* [o] raised by the reach of [s]'s negative strides, so that [o] is the least
   position: positions are non-negative. *)
let lift s strides o =
  let o = ref o in
  Array.iteri
    (fun i d ->
      if d > 1 && strides.(i) < 0 then o := !o - ((d - 1) * strides.(i)))
    s;
  !o

(* Layouts over arbitrary strides, their least position at an arbitrary
   offset. *)
let strided_of s =
  let open Gen in
  let+ strides = array ~size:(const (Array.length s)) (int_range (-5) 5)
  and+ least = int_range 0 40 in
  L.v ~offset:(lift s strides least) ~strides:(Iarray.of_array strides)
    (Iarray.of_array s)

let strided = Gen.bind shape strided_of

let range d =
  let open Gen in
  if d = 0 then
    let+ step = ints_of [ -2; -1; 1; 2 ] in
    { M.start = 0; count = 0; step }
  else
    let* step = ints_of [ -3; -2; -1; 1; 2; 3 ] in
    let* start = int_range 0 (d - 1) in
    let room = if step > 0 then (d - 1 - start) / step else start / -step in
    let+ count = int_range 0 (room + 1) in
    { M.start; count; step }

(* Windows on some axes of [s]; [apart] keeps windows from overlapping. *)
let windows ~apart s =
  let open Gen in
  let r = Array.length s in
  let axes = List.filter (fun i -> s.(i) >= 1) (List.init r Fun.id) in
  let* axes = subsequence ~pp:Format.pp_print_int axes in
  (* Each window appends an axis: keep the result within the most axes. *)
  let axes = List.filteri (fun i _ -> r + i < L.max_rank) axes in
  let window axis =
    let d = s.(axis) in
    let* size = int_range 1 d in
    let* dilation =
      if size = 1 then int_range 1 3 else int_range 1 ((d - 1) / (size - 1))
    in
    let reach = (dilation * (size - 1)) + 1 in
    let+ step = if apart then int_range reach (reach + 2) else int_range 1 3 in
    { M.axis; size; step; dilation }
  in
  let rec all = function
    | [] -> constant []
    | a :: rest ->
        let+ w = window a and+ ws = all rest in
        w :: ws
  in
  let+ ws = all axes in
  M.Window (Iarray.of_list ws)

(* Shapes of [n] elements, of rank up to 4, with extents of 1 among them. *)
let reshape_target s =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  if n = 0 then (
    let+ s' = array ~size:(int_range 1 4) (int_range 0 3) in
    let s' = Array.copy s' in
    s'.(0) <- 0;
    M.Reshape (Iarray.of_array s'))
  else
    (* Deal [n]'s prime factors to four axes, then drop or keep ones. *)
    let rec primes n p =
      if n = 1 then []
      else if n mod p = 0 then p :: primes (n / p) p
      else primes n (p + 1)
    in
    let ps = primes n 2 in
    let+ slots = list ~size:(const (List.length ps)) (int_range 0 3)
    and+ keep = array ~size:(const 4) bool in
    let s' = Array.make 4 1 in
    List.iter2 (fun p k -> s'.(k) <- s'.(k) * p) ps slots;
    M.Reshape
      (Iarray.of_list
         (List.filteri (fun i d -> d > 1 || keep.(i)) (Array.to_list s')))

(* Broadcasts of [s]: up to two new leading axes, and each extent-1 axis to an
   extent up to 3. *)
let broadcast_of s =
  let open Gen in
  let* extra = array ~size:(int_range 0 2) (int_range 0 3) in
  let+ fill = array ~size:(const (Array.length s)) (int_range 0 3) in
  M.Broadcast
    (Iarray.of_array
       (Array.append extra
          (Array.mapi (fun i d -> if d = 1 then fill.(i) else d) s)))

(* Permutations of [s]'s axes. *)
let permute_of s =
  let open Gen in
  let+ p =
    permutation ~pp:Format.pp_print_int (List.init (Array.length s) Fun.id)
  in
  M.Permute (Iarray.of_list p)

let movement ~apart s =
  let open Gen in
  let r = Array.length s in
  let slice =
    let rec all i =
      if i = r then constant []
      else
        let+ x = range s.(i) and+ xs = all (i + 1) in
        x :: xs
    in
    let+ rs = all 0 in
    M.Slice (Iarray.of_list rs)
  in
  let views = [ permute_of s; slice; windows ~apart s ] in
  if apart then one_of views
  else one_of (reshape_target s :: broadcast_of s :: views)

(* A layout [contiguous] reaches by up to three movements, views that keep
   positions apart if [apart]. *)
let reached ~apart =
  let open Gen in
  let* s = shape in
  let rec go l k =
    if k = 0 then constant l
    else
      let* m = movement ~apart (Iarray.to_array (L.shape l)) in
      match L.move m l with Some l' -> go l' (k - 1) | None -> go l (k - 1)
  in
  let* k = int_range 0 3 in
  go (L.contiguous (Iarray.of_array s)) k

let any_layout = Gen.with_pp L.pp (Gen.one_of [ strided; reached ~apart:false ])

(* Movements of [s] drawn about the bounds of their preconditions, most of them
   refused. *)
let near_bounds s =
  let open Gen in
  let r = Array.length s in
  let small = int_range (-1) 5 in
  let reshape =
    let+ s' = array ~size:(int_range 0 4) small in
    M.Reshape (Iarray.of_array s')
  in
  let broadcast =
    let+ s' = array ~size:(int_range (max 0 (r - 1)) (r + 2)) small in
    M.Broadcast (Iarray.of_array s')
  in
  let permute =
    let+ p =
      array ~size:(int_range (max 0 (r - 1)) (r + 1)) (int_range (-1) r)
    in
    M.Permute (Iarray.of_array p)
  in
  let range d =
    let+ start = int_range (-1) (d + 1)
    and+ count = int_range (-1) (d + 1)
    and+ step = int_range (-3) 3 in
    { M.start; count; step }
  in
  let slice =
    let* n = int_range (max 0 (r - 1)) (r + 1) in
    let rec all i =
      if i = n then constant []
      else
        let+ x = range (if i < r then s.(i) else 2) and+ xs = all (i + 1) in
        x :: xs
    in
    let+ rs = all 0 in
    M.Slice (Iarray.of_list rs)
  in
  let window =
    let+ axis = int_range (-1) r
    and+ size = int_range 0 4
    and+ step = int_range 0 3
    and+ dilation = int_range 0 3 in
    { M.axis; size; step; dilation }
  in
  let windows =
    let+ ws = list ~size:(int_range 0 3) window in
    M.Window (Iarray.of_list ws)
  in
  (* One window whose reach passes its axis, so the axis gives no window: the
     random windows above are rarely all valid and that long. *)
  let longer =
    match List.filter (fun i -> s.(i) >= 1) (List.init r Fun.id) with
    | [] -> windows
    | axes ->
        let* axis = ints_of axes in
        let+ dilation = int_range 1 3
        and+ past = int_range 1 3
        and+ step = int_range 1 3 in
        (* The least size whose reach, [dilation·(size - 1) + 1], is at least
           [s.(axis) + past]. *)
        let size = 1 + ((s.(axis) + past - 1 + dilation - 1) / dilation) in
        M.Window [| { M.axis; size; step; dilation } |]
  in
  one_of [ reshape; broadcast; permute; slice; windows; longer ]

(* Shapes of up to four axes, an extent among them now and then negative, with
   movements valid or not. *)
let shape_and_move =
  let open Gen in
  with_pp
    (fun ppf (s, m) -> Format.fprintf ppf "%a, %a" pp_ints s pp_move m)
    (let* s =
       frequency
         [ (8, shape); (1, array ~size:(int_range 1 4) (int_range (-1) 3)) ]
     in
     let valid = Array.for_all (fun d -> d >= 0) s in
     let+ m =
       if valid then one_of [ movement ~apart:false s; near_bounds s ]
       else near_bounds s
     in
     (s, m))

(* Layouts with movements of their shape, valid or not. *)
let layout_and_move =
  let open Gen in
  with_pp
    (fun ppf (l, m) -> Format.fprintf ppf "%a, %a" L.pp l pp_move m)
    (let* l = one_of [ strided; reached ~apart:false ] in
     let s = Iarray.to_array (L.shape l) in
     let+ m = frequency [ (3, movement ~apart:false s); (1, near_bounds s) ] in
     (l, m))

(* Witnesses and references *)

(* Layouts, equal by {!L.equal}. *)
let layout = Testable.make ~pp:L.pp ~equal:L.equal

(* Arrays of ints, as dims and descriptors are. *)
let ints = array int

(* Formats an iarray of ints as {!pp_ints} does. *)
let pp_iints ppf s = pp_ints ppf (Iarray.to_array s)

(* Iarrays of ints, as shapes and strides are, equal element by element. *)
let iints = Testable.make ~pp:pp_iints ~equal:(Iarray.equal Int.equal)

(* A witness of [dt]'s values that prints them as [dt] does and compares floats
   bit for bit. *)
let value : type v s. (v, s) Nx_array.Dtype.t -> v testable =
 fun dt ->
  let module D = Nx_array.Dtype in
  let float = Testable.equal float_exact in
  match D.kind dt with
  | D.Float -> Testable.make ~pp:(D.pp_value dt) ~equal:float
  | D.Complex ->
      Testable.make ~pp:(D.pp_value dt) ~equal:(fun (a : Complex.t) b ->
          float a.re b.re && float a.im b.im)
  | D.Boolean -> bool
  | D.Signed | D.Unsigned -> Testable.make ~pp:(D.pp_value dt) ~equal:( = )

(* The product of [s]'s extents, [None] if the product of those other than 0
   passes {!L.max_numel}. *)
let numel s =
  let nonzero =
    Iarray.fold_left
      (fun n d ->
        match n with
        | Some n when d = 0 -> Some n
        | Some n when d <= L.max_numel / n -> Some (n * d)
        | _ -> None)
      (Some 1) s
  in
  Option.map (fun n -> if Iarray.mem 0 s then 0 else n) nonzero

(* An array of [dt]'s values, each compared as {!value} does. *)
let values dt = array (value dt)

(* A host float32 array of shape [s] holding [xs] in C order. *)
let floats32 s xs = Nx_array.of_array Nx_array.Dtype.Float32 s xs

(* Claims and answers *)

(* Kills [b]: a donation consumes its memory, giving [why]. *)
let kill ?(why = "consumed by the test") b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c ->
      ignore (Rig.Claim.consume c ~why b))

(* Whether no claim holds [b]'s memory: a donation of it is exclusive. *)
let unclaimed b =
  Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c -> Rig.Claim.exclusive c b)

(* Answers, by their names. *)
let answer = Testable.make ~pp:Nx_array_support.pp_answer ~equal:( = )

(* What a function given to a door or a read raises. *)
exception Raised

(* Late devices *)

(* Late devices opened so far, which number the next. *)
let lates = Atomic.make 0

(* A fresh Late device and its driver. *)
let late () =
  Nx_array_support.Late.open_
    (Printf.sprintf "nx2-test-late:%d" (Atomic.fetch_and_add lates 1))

(* The little-endian binary32 bytes of [xs]. *)
let f32_bytes xs =
  let b = Bytes.create (4 * Array.length xs) in
  Array.iteri
    (fun i x -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float x))
    xs;
  Bytes.to_string b

(* Work on [a]'s device that stores [xs] into [a], unreached. *)
let pending a xs = Nx_array_support.write (Nx_array.buffer a) (f32_bytes xs)

(* An array of [xs] on the Late device [d], its work reached. *)
let on_late d xs =
  let a = Nx_array.create d Nx_array.Dtype.Float32 [| Array.length xs |] in
  pending a xs;
  Rig.Buffer.wait (Nx_array.buffer a) Rig.Read_write;
  a

(* Whether no claim holds [a]'s memory: a donation of it is held exclusive. *)
let no_claim a =
  let b = Nx_array.buffer a in
  equal ~msg:"no claim left" bool true
    (Rig.Claim.with_ ~read:[] ~donate:[ b ] (fun c -> Rig.Claim.exclusive c b))

(* Whether [e] is a device's loss. *)
let lost = function Rig.Lost _ -> true | _ -> false

(* Arrays to place *)

(* A host array of [dt] of shape [s], its values stores of drawn floats. *)
let filled dt s =
  let open Gen in
  let n = Array.fold_left ( * ) 1 s in
  let+ xs = array ~size:(const n) (float_range (-300.) 300.) in
  Nx_array.of_array dt (Iarray.of_array s)
    (Array.map (Nx_array.Dtype.of_float dt) xs)

(* Any dtype, those narrower than a byte one draw in four. *)
let placed_dtype =
  let module D = Nx_array.Dtype in
  let pp ppf (D.Any dt) = D.pp ppf dt in
  let narrow, wide = List.partition (fun (D.Any dt) -> D.bits dt < 8) D.all in
  Gen.frequency [ (1, Gen.of_list ~pp narrow); (3, Gen.of_list ~pp wide) ]

(* An array and a movement whose span holds no more bytes than its elements,
   which [to_device] keeps: none, a permutation, a broadcast, a reshape, or a
   run of the first axis' rows, either way. A sub-byte array's extents are
   mostly odd, so that its last element ends inside a byte. *)
let moved =
  let open Gen in
  let* (Nx_array.Dtype.Any dt) = placed_dtype in
  let odd = array ~size:(int_range 1 3) (ints_of [ 1; 3; 5 ]) in
  let* s =
    if Nx_array.Dtype.bits dt < 8 then frequency [ (7, odd); (1, shape) ]
    else shape
  in
  let* a = filled dt s in
  let rows =
    let d = if s = [||] then 0 else s.(0) in
    let* step = ints_of [ 1; -1 ] in
    let* start = int_range 0 (max 0 (d - 1)) in
    let room = if d = 0 then 0 else if step > 0 then d - start else start + 1 in
    let+ count = int_range 0 room in
    M.Slice
      (Iarray.init (Array.length s) (fun i ->
           if i = 0 then { M.start; count; step }
           else { M.start = 0; count = s.(i); step = 1 }))
  in
  let moves = [ permute_of s; broadcast_of s; reshape_target s ] in
  let moves = if s = [||] then moves else rows :: moves in
  let+ m = option (one_of moves) in
  (Nx_array.Any a, m)

(* A matrix transposed or its rows reversed: views [to_device] keeps, strided. A
   sub-byte matrix's extents are odd. *)
let transposed =
  let open Gen in
  let* (Nx_array.Dtype.Any dt) = placed_dtype in
  let extent =
    if Nx_array.Dtype.bits dt < 8 then ints_of [ 3; 5 ] else int_range 2 4
  in
  let* rows = extent in
  let* cols = extent in
  let* a = filled dt [| rows; cols |] in
  let reversed =
    M.Slice
      [|
        { M.start = rows - 1; count = rows; step = -1 };
        { M.start = 0; count = cols; step = 1 };
      |]
  in
  let+ m = of_list ~pp:pp_move [ M.Permute [| 1; 0 |]; reversed ] in
  (Nx_array.Any a, Some m)

(* An array and a movement whose span [to_device] keeps. *)
let kept = Gen.one_of [ moved; transposed ]

(* An array and a movement whose span holds more bytes than its elements, which
   [to_device] packs: a column of a matrix whose rows hold two bytes or more,
   that column broadcast along a new axis, or a slice whose steps pass two bytes
   or more. *)
let packed =
  let open Gen in
  let* (Nx_array.Dtype.Any dt) = placed_dtype in
  let bits = Nx_array.Dtype.bits dt in
  (* The elements in two bytes, at least two. *)
  let wide = max 2 ((16 + bits - 1) / bits) in
  let column =
    let* rows = int_range 2 4 in
    let* cols = int_range wide (wide + 2) in
    let* col = int_range 0 (cols - 1) in
    let+ a = filled dt [| rows; cols |] in
    let take =
      M.Slice
        [|
          { M.start = 0; count = rows; step = 1 };
          { M.start = col; count = 1; step = 1 };
        |]
    in
    (a, take, rows)
  in
  let one_column =
    let+ a, take, _ = column in
    (Nx_array.Any a, Some take)
  in
  let broadcast_column =
    let* a, take, rows = column in
    let+ k = int_range 2 3 in
    let c = Option.get (Nx_array.move take a) in
    (Nx_array.Any c, Some (M.Broadcast [| rows; k |]))
  in
  let stepped =
    let* count = int_range 2 4 in
    let* step = int_range wide (wide + 2) in
    let* extra = int_range 0 2 in
    let n = ((count - 1) * step) + 1 + extra in
    let* first = int_range 0 extra in
    let* reversed = bool in
    let+ a = filled dt [| n |] in
    let start, step =
      if reversed then (n - 1 - first, -step) else (first, step)
    in
    (Nx_array.Any a, Some (M.Slice [| { M.start; count; step } |]))
  in
  one_of [ one_column; broadcast_column; stepped ]

(* An array and a movement, as often kept by [to_device] as packed. *)
let to_device_case = Gen.one_of [ kept; packed ]
