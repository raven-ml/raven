(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Indexing and functional updates through Nx: slice and set against a reference
   that reads and writes OCaml arrays entry by entry, their round trip, take,
   take_along_axis and scatter against their definitions, views, every dtype and
   the refusals. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))
let numel s = Array.fold_left ( * ) 1 s

let index s k =
  let r = Array.length s in
  let idx = Array.make r 0 and k = ref k in
  for a = r - 1 downto 0 do
    idx.(a) <- !k mod s.(a);
    k := !k / s.(a)
  done;
  idx

let position s idx =
  let k = ref 0 in
  Array.iteri (fun a i -> k := (!k * s.(a)) + i) idx;
  !k

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let host dt s data = Nx.Repr.of_array Nx.Host.v (A.of_array dt s data)
let positions s ps = host D.Int64 s (Array.map Int64.of_int ps)

(* The reference *)

(* An entry as the test draws it: positions held in data are an OCaml array of a
   shape. *)
type entry =
  | I of int
  | L of int list
  | T of int array * int array
  | R of int * int
  | Rs of int * int * int
  | A
  | N
  | D of int * int

let pp_entry ppf = function
  | I p -> Format.fprintf ppf "I %d" p
  | L ps -> Format.fprintf ppf "L %a" pp_ints (Array.of_list ps)
  | T (s, ps) -> Format.fprintf ppf "T %a %a" pp_ints s pp_ints ps
  | R (a, b) -> Format.fprintf ppf "R (%d, %d)" a b
  | Rs (a, b, c) -> Format.fprintf ppf "Rs (%d, %d, %d)" a b c
  | A -> Format.fprintf ppf "A"
  | N -> Format.fprintf ppf "N"
  | D (p, n) -> Format.fprintf ppf "D (%d, %d)" p n

let to_index : entry -> Nx.host Nx.index = function
  | I p -> I p
  | L ps -> L ps
  | T (s, ps) -> T (positions s ps)
  | R (a, b) -> R (a, b)
  | Rs (a, b, c) -> Rs (a, b, c)
  | A -> A
  | N -> N
  | D (p, n) -> D (positions [||] [| p |], n)

(* The positions a range written in the program keeps of an axis of extent [d]:
   Python's slice rule. *)
let range_positions d start stop step =
  let clip lo hi e =
    let e = if e < 0 then e + d else e in
    max lo (min hi e)
  in
  let out = ref [] in
  if step > 0 then begin
    let i = ref (clip 0 d start) and stop = clip 0 d stop in
    while !i < stop do
      out := !i :: !out;
      i := !i + step
    done
  end
  else begin
    let i = ref (clip (-1) (d - 1) start) and stop = clip (-1) (d - 1) stop in
    while !i > stop do
      out := !i :: !out;
      i := !i + step
    done
  end;
  Array.of_list (List.rev !out)

(* For each entry, the selection axes it gives and the position it reads at a
   selection index along them, [None] outside its axis. *)
let axes_of d = function
  | I _ -> [||]
  | L ps -> [| List.length ps |]
  | T (s, _) -> s
  | R (a, b) -> [| Array.length (range_positions d a b 1) |]
  | Rs (a, b, c) -> [| Array.length (range_positions d a b c) |]
  | A -> [| d |]
  | N -> [| 1 |]
  | D (_, n) -> [| n |]

let read_at d e (j : int array) =
  let norm p = if p < 0 then p + d else p in
  match e with
  | I p -> Some (norm p)
  | L ps -> Some (norm (List.nth ps j.(0)))
  | T (s, ps) ->
      let p = ps.(position s j) in
      if p < 0 || p >= d then None else Some p
  | R (a, b) -> Some (range_positions d a b 1).(j.(0))
  | Rs (a, b, c) -> Some (range_positions d a b c).(j.(0))
  | A -> Some j.(0)
  | N -> None
  | D (p, n) -> Some (max 0 (min (d - n) p) + j.(0))

(* The selection's shape, and the operand index each selection index reads:
   [None] where a position held in data lies outside its axis. *)
let selection s entries =
  let r = Array.length s in
  let addressed = List.length (List.filter (( <> ) N) entries) in
  let entries = entries @ List.init (r - addressed) (fun _ -> A) in
  let axis = ref 0 in
  let parts =
    List.map
      (fun e ->
        let d = if e = N then 0 else s.(!axis) in
        let a = if e = N then -1 else !axis in
        if e <> N then incr axis;
        (e, a, d, axes_of d e))
      entries
  in
  let sel = Array.concat (List.map (fun (_, _, _, ax) -> ax) parts) in
  let read i =
    let ix = Array.make r 0 and ok = ref true and j = ref 0 in
    List.iter
      (fun (e, a, d, ax) ->
        let k = Array.length ax in
        let sub = Array.sub i !j k in
        j := !j + k;
        if a >= 0 then
          match read_at d e sub with
          | Some p -> ix.(a) <- p
          | None -> ok := false)
      parts;
    if !ok then Some ix else None
  in
  (sel, read)

(* Drawing *)

type case = { s : int array; entries : entry list }

let pp_case ppf c =
  Format.fprintf ppf "%a [%a]" pp_ints c.s
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       pp_entry)
    c.entries

let entry d =
  let open Gen in
  let pos = int_range (-d) (d - 1) in
  let held =
    let* r = int_range 0 2 in
    let* s = array ~size:(constant r) (int_range 0 2) in
    let+ ps = array ~size:(constant (numel s)) (int_range (-1) d) in
    T (s, ps)
  in
  let ends = int_range (-d - 1) (d + 1) in
  let any =
    [
      held;
      (let+ a = ends and+ b = ends in
       R (a, b));
      (let+ a = ends and+ b = ends and+ c = of_list [ -2; -1; 1; 2 ] in
       Rs (a, b, c));
      constant A;
      (let* n = int_range 0 d in
       let+ p = int_range (-1) (d + 1) in
       D (p, n));
    ]
  in
  if d = 0 then one_of any
  else
    one_of
      ((let+ p = pos in
        I p)
      :: (let* k = int_range 0 3 in
          let+ ps = list ~size:(constant k) pos in
          L ps)
      :: any)

let case =
  let open Gen in
  with_pp pp_case
    (let* r = int_range 0 3 in
     let* s = array ~size:(constant r) (int_range 0 3) in
     let* m = int_range 0 r in
     let rec entries a =
       if a = m then constant []
       else
         let* e = entry s.(a) in
         let* news = int_range 0 1 in
         let+ rest = entries (a + 1) in
         (if news = 1 then [ N; e ] else [ e ]) @ rest
     in
     let+ entries = entries 0 in
     { s; entries })

let operand s =
  let data = Array.init (numel s) (fun k -> Int32.of_int (k + 1)) in
  (host D.Int32 s data, data)

let covers c =
  let has f = List.exists f c.entries in
  cover "held positions" (has (function T _ -> true | _ -> false));
  cover "a window" (has (function D _ -> true | _ -> false));
  cover "written positions" (has (function L _ -> true | _ -> false));
  cover "a negative step" (has (function Rs (_, _, c) -> c < 0 | _ -> false));
  cover "a new axis" (has (( = ) N))

(* Whether no list of written positions names a position twice, which [set]
   refuses. *)
let distinct_lists c =
  let axis = ref 0 in
  List.for_all
    (fun e ->
      match e with
      | N -> true
      | L ps ->
          let d = c.s.(!axis) in
          incr axis;
          let ps = List.map (fun p -> if p < 0 then p + d else p) ps in
          List.length (List.sort_uniq compare ps) = List.length ps
      | I _ | T _ | R _ | Rs _ | A | D _ ->
          incr axis;
          true)
    c.entries

let laws =
  group "laws"
    [
      prop "slice reads each entry's positions, zero outside" case (fun c ->
          covers c;
          let x, data = operand c.s in
          let sel, read = selection c.s c.entries in
          let y = Nx.slice (List.map to_index c.entries) x in
          equal ~msg:"shape" (array int) sel (Nx.shape y);
          equal ~msg:"elements" (array int32)
            (Array.init (numel sel) (fun k ->
                 match read (index sel k) with
                 | Some ix -> data.(position c.s ix)
                 | None -> 0l))
            (elements y));
      prop "set writes each entry's positions, the last write winning" case
        (fun c ->
          covers c;
          let data_entry = function T _ | L _ | D _ -> true | _ -> false in
          cover "positions written in the program alone"
            (not (List.exists data_entry c.entries));
          cover "one axis of positions, every other whole"
            (List.length (List.filter data_entry c.entries) = 1
            && List.for_all (fun e -> data_entry e || e = A || e = N) c.entries
            );
          let x, data = operand c.s in
          let sel, read = selection c.s c.entries in
          assume (distinct_lists c);
          let v = Array.init (numel sel) (fun k -> Int32.of_int (-k - 1)) in
          let expected = Array.copy data in
          for k = 0 to numel sel - 1 do
            match read (index sel k) with
            | Some ix -> expected.(position c.s ix) <- v.(k)
            | None -> ()
          done;
          let y = Nx.set (List.map to_index c.entries) (host D.Int32 sel v) x in
          equal ~msg:"shape" (array int) c.s (Nx.shape y);
          equal ~msg:"elements" (array int32) expected (elements y);
          equal ~msg:"x unchanged" (array int32) data (elements x));
      prop "set of a slice is the identity" case (fun c ->
          let x, data = operand c.s in
          let idx = List.map to_index c.entries in
          assume (distinct_lists c);
          equal (array int32) data (elements (Nx.set idx (Nx.slice idx x) x)));
    ]

(* take, take_along_axis and scatter against their definitions. *)
let definitions =
  group "definitions"
    [
      prop "take replaces an axis by the positions' axes"
        Gen.(
          let* r = int_range 1 3 in
          let* s = array ~size:(constant r) (int_range 1 3) in
          let* a = int_range 0 (r - 1) in
          let* pr = int_range 0 2 in
          let* ps = array ~size:(constant pr) (int_range 0 3) in
          let+ p = array ~size:(constant (numel ps)) (int_range (-1) s.(a)) in
          (s, a, ps, p))
        (fun (s, a, ps, p) ->
          let x, data = operand s in
          let y = Nx.take ~axis:a (positions ps p) x in
          let r = Array.length s and pr = Array.length ps in
          let s' =
            Array.concat
              [ Array.sub s 0 a; ps; Array.sub s (a + 1) (r - a - 1) ]
          in
          equal (array int) s' (Nx.shape y);
          equal (array int32)
            (Array.init (numel s') (fun k ->
                 let i = index s' k in
                 let q = p.(position ps (Array.sub i a pr)) in
                 if q < 0 || q >= s.(a) then 0l
                 else
                   let ix =
                     Array.init r (fun d ->
                         if d < a then i.(d)
                         else if d = a then q
                         else i.(d + pr - 1))
                   in
                   data.(position s ix)))
            (elements y));
      prop "take_along_axis pairs positions with indices"
        Gen.(
          let* r = int_range 1 3 in
          let* s = array ~size:(constant r) (int_range 1 3) in
          let* a = int_range 0 (r - 1) in
          let* k = int_range 0 3 in
          let ps = Array.mapi (fun i d -> if i = a then k else d) s in
          let+ p = array ~size:(constant (numel ps)) (int_range (-1) s.(a)) in
          (s, a, ps, p))
        (fun (s, a, ps, p) ->
          let x, data = operand s in
          let y = Nx.take_along_axis ~axis:a (positions ps p) x in
          equal (array int) ps (Nx.shape y);
          equal (array int32)
            (Array.init (numel ps) (fun k ->
                 let i = index ps k in
                 let q = p.(k) in
                 if q < 0 || q >= s.(a) then 0l
                 else
                   let ix = Array.copy i in
                   ix.(a) <- q;
                   data.(position s ix)))
            (elements y));
      prop "scatter combines updates in C order"
        Gen.(
          let* r = int_range 1 3 in
          let* s = array ~size:(constant r) (int_range 1 3) in
          let* a = int_range 0 (r - 1) in
          let* k = int_range 0 4 in
          let us = Array.mapi (fun i d -> if i = a then k else d) s in
          let* p = array ~size:(constant (numel us)) (int_range (-1) s.(a)) in
          let+ combine = of_list [ `Set; `Add; `Max; `Min ] in
          (s, a, us, p, combine))
        (fun (s, a, us, p, combine) ->
          let x, data = operand s in
          let u =
            Array.init (numel us) (fun k -> Int32.of_int ((k * 7 mod 11) - 5))
          in
          let f, c =
            match combine with
            | `Set -> ((fun _ v -> v), Nx.Set)
            | `Add -> (Int32.add, Nx.Add)
            | `Max ->
                ((fun a b -> if Int32.compare a b >= 0 then a else b), Nx.Max)
            | `Min ->
                ((fun a b -> if Int32.compare a b <= 0 then a else b), Nx.Min)
          in
          let expected = Array.copy data in
          for k = 0 to numel us - 1 do
            let i = index us k in
            let q = p.(k) in
            if q >= 0 && q < s.(a) then begin
              let ix = Array.copy i in
              ix.(a) <- q;
              let t = position s ix in
              expected.(t) <- f expected.(t) u.(k)
            end
          done;
          let y =
            Nx.scatter ~combine:c ~axis:a (positions us p) (host D.Int32 us u) x
          in
          equal (array int32) expected (elements y));
    ]

let shares x y =
  let buffer v = A.buffer (Option.get (Nx.Repr.array v)) in
  Rig.Buffer.overlaps (buffer x) (buffer y)

let views =
  group "views"
    [
      test "positions written in the program make a view" (fun () ->
          let x, _ = operand [| 3; 4 |] in
          let y = Nx.slice Nx.[ I 1; Rs (3, 0, -2); N ] x in
          equal (array int) [| 2; 1 |] (Nx.shape y);
          equal (array int32) [| 8l; 6l |] (elements y);
          equal bool true (shares x y);
          let z = Nx.get [ -1 ] x in
          equal (array int32) [| 9l; 10l; 11l; 12l |] (elements z);
          equal bool true (shares x z));
      test "set consumes a donated value, naming itself" (fun () ->
          let x =
            Nx.add (fst (operand [| 2; 3 |])) (Nx.zeros Nx.int32 [| 2; 3 |])
          in
          let y =
            Nx.set Nx.[ I 0; A ] (Nx.zeros Nx.int32 [| 3 |]) (Nx.donate x)
          in
          equal (array int32) [| 0l; 0l; 0l; 4l; 5l; 6l |] (elements y);
          raises_match (Exn.invalid_arg ~substring:"was donated to Nx.set")
            (fun () -> Nx.copy x));
      test "a decode step writes one row of a cache" (fun () ->
          let cache = Nx.zeros Nx.float32 [| 1; 2; 4; 3 |] in
          let row =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Float32 [| 1; 2; 1; 3 |]
                 (Array.init 6 Float.of_int))
          in
          let at pos = Nx.[ A; A; D (positions [||] [| pos |], 1); A ] in
          let c = Nx.set (at 2) row (Nx.place Nx.Host.on cache) in
          let got = elements (Nx.slice Nx.[ A; A; I 2; A ] c) in
          equal (array float_exact) [| 0.; 1.; 2.; 3.; 4.; 5. |] got;
          let past = Nx.set (at 9) row (Nx.place Nx.Host.on cache) in
          equal ~msg:"a start past capacity writes the last row"
            (array float_exact)
            [| 0.; 1.; 2.; 3.; 4.; 5. |]
            (elements (Nx.slice Nx.[ A; A; I 3; A ] past)));
    ]

(* A gather and a scatter of every dtype. *)
let dtypes =
  cases
    ~name:(fun (D.Any dt) -> D.name dt)
    "every dtype" D.all
    (fun (D.Any dt) ->
      let data =
        Array.init 6 (fun k -> D.of_float dt (Float.of_int (k mod 7)))
      in
      let x = host dt [| 2; 3 |] data in
      let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
      let p = positions [| 2 |] [| 2; 0 |] in
      equal (array w)
        [| data.(2); data.(0); data.(5); data.(3) |]
        (elements (Nx.slice Nx.[ A; T p ] x));
      equal (array w)
        [| data.(0); data.(1); data.(1); data.(3); data.(4); data.(4) |]
        (elements (Nx.set Nx.[ A; T p ] (Nx.slice Nx.[ A; L [ 1; 0 ] ] x) x)))

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let refusals =
  let x, _ = operand [| 2; 3 |] in
  group "refusals"
    [
      test "messages name the function and the operand" (fun () ->
          raises
            (Invalid_argument
               "Nx.slice: position 3 is outside axis 1 of int32 [2; 3]")
            (fun () -> Nx.slice Nx.[ A; I 3 ] x);
          raises
            (Invalid_argument
               "Nx.slice: 3 entries address 3 axes; the operand is int32 [2; 3]")
            (fun () -> Nx.slice Nx.[ A; A; A ] x);
          raises
            (Invalid_argument
               "Nx.set: int32 [4] does not broadcast to the selection [2; 3]")
            (fun () -> Nx.set Nx.[ A ] (Nx.zeros Nx.int32 [| 4 |]) x));
      test "each refusal raises before anything is computed" (fun () ->
          invalid ~by:"Nx.slice" (fun () -> Nx.slice Nx.[ Rs (0, 2, 0) ] x);
          invalid ~by:"Nx.slice" (fun () ->
              Nx.slice Nx.[ D (positions [| 1 |] [| 0 |], 1) ] x);
          invalid ~by:"Nx.slice" (fun () ->
              Nx.slice Nx.[ D (positions [||] [| 0 |], 3) ] x);
          invalid ~by:"Nx.slice" (fun () -> Nx.slice Nx.[ L [ 0; 2 ] ] x);
          invalid ~by:"Nx.set" (fun () -> Nx.set Nx.[ L [ 1; 1 ] ] x x);
          invalid ~by:"Nx.get" (fun () -> Nx.get [ 0; 0; 0 ] x);
          invalid ~by:"Nx.take" (fun () ->
              Nx.take ~axis:2 (positions [| 1 |] [| 0 |]) x);
          invalid ~by:"Nx.take_along_axis" (fun () ->
              Nx.take_along_axis ~axis:0 (positions [| 1 |] [| 0 |]) x);
          invalid ~by:"Nx.scatter" (fun () ->
              Nx.scatter ~axis:0 (positions [| 2 |] [| 0; 1 |]) x x);
          invalid ~by:"Nx.scatter" (fun () ->
              let b = Nx.zeros Nx.bool [| 2 |] in
              Nx.scatter ~combine:Add ~axis:0 (positions [| 2 |] [| 0; 1 |]) b b));
    ]

let () = exit (run "nx indexing" [ laws; definitions; views; dtypes; refusals ])
