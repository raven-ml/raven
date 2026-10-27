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

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let host dt s data =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) data)
let held s ps = host D.Int64 s (Array.map Int64.of_int ps)

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
  | T (s, ps) -> T (held s ps)
  | R (a, b) -> R (a, b)
  | Rs (a, b, c) -> Rs (a, b, c)
  | A -> A
  | N -> N
  | D (p, n) -> D (held [||] [| p |], n)

(* The positions a range written in the program keeps of an axis of extent [d]:
   Python's slice rule, each next position formed only if it lies before
   [stop], so that no step overflows. *)
let range_positions d start stop step =
  let clip lo hi e =
    let e = if e < 0 then e + d else e in
    max lo (min hi e)
  in
  let out = ref [] in
  if step > 0 then begin
    let i = ref (clip 0 d start) and stop = clip 0 d stop and go = ref true in
    while !go && !i < stop do
      out := !i :: !out;
      if step >= stop - !i then go := false else i := !i + step
    done
  end
  else begin
    let i = ref (clip (-1) (d - 1) start)
    and stop = clip (-1) (d - 1) stop
    and go = ref true in
    while !go && !i > stop do
      out := !i :: !out;
      if step <= stop - !i then go := false else i := !i + step
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

(* How a list of written positions is drawn: [Repeats] may name a position
   twice, which [slice] reads twice and [set] refuses; [Distinct] names each
   at most once, in any order, each from the start or the end. *)
type lists = Repeats | Distinct

let entry lists d =
  let open Gen in
  let pos = int_range (-d) (d - 1) in
  let written =
    match lists with
    | Repeats ->
        let* k = int_range 0 3 in
        list ~size:(constant k) pos
    | Distinct ->
        let* k = int_range 0 d in
        let* keys = array ~size:(constant d) (int_range 0 999) in
        let+ ends = array ~size:(constant d) bool in
        let order =
          List.sort (fun i j -> compare keys.(i) keys.(j)) (List.init d Fun.id)
        in
        List.filteri (fun n _ -> n < k) order
        |> List.map (fun p -> if ends.(p) then p - d else p)
  in
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
      (let+ a = ends
       and+ b = ends
       and+ c =
         frequency
           [
             (8, of_list [ -2; -1; 1; 2 ]);
             (1, of_list [ max_int; max_int - 1; min_int; min_int + 1 ]);
           ]
       in
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
      :: (let+ ps = written in
          L ps)
      :: any)

let case lists =
  let open Gen in
  with_pp pp_case
    (let* r = int_range 0 3 in
     let* s = array ~size:(constant r) (int_range 0 3) in
     (* The number of axes selected along: one case in five selects along
        none, so that each kind of entry turns up in a run. *)
     let* m =
       if r = 0 then constant 0
       else frequency [ (1, constant 0); (4, int_range 1 r) ]
     in
     let rec entries a =
       if a = m then constant []
       else
         let* e = entry lists s.(a) in
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

(* The cases a law that [covers] runs. Its rarest label, a negative step, marks
   about one case in sixteen: 100 cases miss it once in 500 runs, 300 once in
   10^8. *)
let count = 300

(* [Nx.set] of [c] writes each selection index's value at the position the
   reference reads, the last write winning, and leaves the operand as it
   was. *)
let check_set c =
  let x, data = operand c.s in
  let sel, read = selection c.s c.entries in
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
  equal ~msg:"x unchanged" (array int32) data (elements x)

let laws =
  group "laws"
    [
      prop ~count "slice reads each entry's positions, zero outside"
        (case Repeats) (fun c ->
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
      prop ~count "set writes each entry's positions, the last write winning"
        (case Distinct)
        (fun c ->
          covers c;
          let data_entry = function T _ | L _ | D _ -> true | _ -> false in
          cover "positions written in the program alone"
            (not (List.exists data_entry c.entries));
          cover "one axis of positions, every other whole"
            (List.length (List.filter data_entry c.entries) = 1
            && List.for_all (fun e -> data_entry e || e = A || e = N) c.entries
            );
          check_set c);
      prop "set of a slice is the identity" (case Distinct) (fun c ->
          let x, data = operand c.s in
          let idx = List.map to_index c.entries in
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
          let y = Nx.take ~axis:a (held ps p) x in
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
          let y = Nx.take_along_axis ~axis:a (held ps p) x in
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
            Nx.scatter ~combine:c ~axis:a (held us p) (host D.Int32 us u) x
          in
          equal (array int32) expected (elements y));
    ]

(* Masks *)

(* A selection with a mask over the [k] axes from [at], the other entries
   drawn as [case] draws them. Its reference is the selection of the operand
   with those axes merged into one, the mask by the positions where it
   holds, in C order: the operand's elements in C order are the merged
   one's. *)
type mask_case = {
  ms : int array;
  before : entry list;
  at : int;
  k : int;
  mask : bool array;
  after : entry list;
}

let pp_mask_case ppf c =
  Format.fprintf ppf "%a [%a; M %a; %a]" pp_ints c.ms
    (Format.pp_print_list pp_entry) c.before pp_ints
    (Array.sub c.ms c.at c.k)
    (Format.pp_print_list pp_entry) c.after

let mask_case lists =
  let open Gen in
  with_pp pp_mask_case
    (let* r = int_range 0 3 in
     let* ms = array ~size:(constant r) (int_range 0 3) in
     (* One mask in five is 0-d, and two in five cover two axes or more
        where the operand has them. *)
     let* k =
       frequency
         ((1, constant 0)
         :: (if r >= 1 then [ (2, constant 1) ] else [])
         @ if r >= 2 then [ (2, int_range 2 r) ] else [])
     in
     let* at = int_range 0 (r - k) in
     let rec entries a stop =
       if a = stop then constant []
       else
         let* e = entry lists ms.(a) in
         let+ rest = entries (a + 1) stop in
         e :: rest
     in
     let* before = entries 0 at in
     let* stop = int_range (at + k) r in
     let* after = entries (at + k) stop in
     let+ mask =
       array
         ~size:(constant (numel (Array.sub ms at k)))
         (frequency [ (2, constant true); (1, constant false) ])
     in
     { ms; before; at; k; mask; after })

let held_positions mask =
  List.filter (fun i -> mask.(i)) (List.init (Array.length mask) Fun.id)
  |> Array.of_list

(* The merged shape and the reference's entries. *)
let merged c =
  let r = Array.length c.ms in
  let s =
    Array.concat
      [
        Array.sub c.ms 0 c.at;
        [| Array.length c.mask |];
        Array.sub c.ms (c.at + c.k) (r - c.at - c.k);
      ]
  in
  let ps = held_positions c.mask in
  (s, c.before @ [ T ([| Array.length ps |], ps) ] @ c.after)

let mask_index c =
  List.map to_index c.before
  @ [ Nx.M (host D.Bool (Array.sub c.ms c.at c.k) c.mask) ]
  @ List.map to_index c.after

let mask_covers c =
  cover "a 0-d mask" (c.k = 0);
  cover "a mask over two axes or more" (c.k >= 2);
  cover "entries after the mask" (c.after <> []);
  cover "a mask holding nowhere" (not (Array.exists Fun.id c.mask))

let law_mask_slice c =
  mask_covers c;
  let x, data = operand c.ms in
  let s, entries = merged c in
  let sel, read = selection s entries in
  let y = Nx.slice (mask_index c) x in
  equal ~msg:"shape" (array int) sel (Nx.shape y);
  equal ~msg:"elements" (array int32)
    (Array.init (numel sel) (fun k ->
         match read (index sel k) with
         | Some ix -> data.(position s ix)
         | None -> 0l))
    (elements y)

let law_mask_set c =
  mask_covers c;
  let x, data = operand c.ms in
  let s, entries = merged c in
  let sel, read = selection s entries in
  let v = Array.init (numel sel) (fun k -> Int32.of_int (-k - 1)) in
  let expected = Array.copy data in
  for k = 0 to numel sel - 1 do
    match read (index sel k) with
    | Some ix -> expected.(position s ix) <- v.(k)
    | None -> ()
  done;
  let y = Nx.set (mask_index c) (host D.Int32 sel v) x in
  equal ~msg:"shape" (array int) c.ms (Nx.shape y);
  equal ~msg:"elements" (array int32) expected (elements y)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let mark i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

let test_mask_cases () =
  let x, _ = operand [| 2; 3 |] in
  let m s bs = host D.Bool s bs in
  equal (array int32) [| 2l; 3l; 5l |]
    (elements
       (Nx.slice
          Nx.[ M (m [| 2; 3 |] [| false; true; true; false; true; false |]) ]
          x));
  equal (array int32) [| 4l; 5l; 6l |]
    (elements (Nx.slice Nx.[ M (m [| 2 |] [| false; true |]) ] x));
  equal (array int32) [| 1l; 3l; 4l; 6l |]
    (elements (Nx.slice Nx.[ A; M (m [| 3 |] [| true; false; true |]) ] x));
  equal (array int) [| 1; 2; 3 |]
    (Nx.shape (Nx.slice Nx.[ M (m [||] [| true |]) ] x));
  equal (array int) [| 0; 2; 3 |]
    (Nx.shape (Nx.slice Nx.[ M (m [||] [| false |]) ] x));
  equal (array int32) [| 1l; 0l; 0l; 4l; 0l; 0l |]
    (elements
       (Nx.set Nx.[ A; M (m [| 3 |] [| false; true; true |]) ]
          (Nx.zeros Nx.int32 [| 1 |])
          x));
  (* Two masks, and a mask beside positions, each select along their own
     axes: the outer product of their selections. *)
  let two =
    Nx.slice
      Nx.[ M (m [| 2 |] [| true; true |]); M (m [| 3 |] [| true; false; true |]) ]
      x
  in
  equal (array int) [| 2; 2 |] (Nx.shape two);
  equal (array int32) [| 1l; 3l; 4l; 6l |] (elements two);
  let beside =
    Nx.slice Nx.[ M (m [| 2 |] [| false; true |]); T (held [| 2 |] [| 2; 0 |]) ] x
  in
  equal (array int) [| 1; 2 |] (Nx.shape beside);
  equal (array int32) [| 6l; 4l |] (elements beside);
  equal (array int32) [| 0l; 2l; 0l; 0l; 5l; 0l |]
    (elements
       (Nx.set
          Nx.[ M (m [| 2 |] [| true; true |]); M (m [| 3 |] [| true; false; true |]) ]
          (Nx.zeros Nx.int32 [| 2; 1 |])
          x));
  invalid ~by:"Nx.slice" (fun () ->
      Nx.slice Nx.[ M (m [| 3 |] [| true; true; true |]) ] x);
  invalid ~by:"Nx.slice" (fun () ->
      Nx.slice Nx.[ A; A; M (m [| 1 |] [| true |]) ] x);
  invalid ~by:"Nx.set" (fun () ->
      Nx.set Nx.[ M (m [| 2 |] [| true; true |]) ] (Nx.zeros Nx.int32 [| 3; 3 |]) x);
  (* Under a trace a mask's count is unknown: the selection raises, and the
     positions of a fixed size select. *)
  let mask = m [| 2 |] [| false; true |] in
  Nx.Prim.interpret ~name:"test.mask" Values mark (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form mask) Mark in
      raises
        (Invalid_argument
           "Nx.slice: a mask's count is data, which test.mask cannot read; \
            select with T (positions ~size m) or compute with where")
        (fun () -> Nx.slice Nx.[ M t ] x);
      invalid ~by:"Nx.set" (fun () -> Nx.set Nx.[ M t ] x x);
      ignore (Nx.slice Nx.[ T (Nx.positions ~size:1 t) ] x))

(* Coordinates *)

let law_unravel (s, ks) =
  let n = numel s in
  cover "an index outside" (Array.exists (fun k -> k < 0 || k >= n) ks);
  cover "an empty shape" (n = 0);
  let i = held [| Array.length ks |] ks in
  let cs = Nx.unravel_index i s in
  equal ~msg:"axes" int (Array.length s) (List.length cs);
  List.iteri
    (fun j c ->
      equal ~msg:"coordinates" (array int64)
        (Array.map
           (fun k ->
             Int64.of_int (if k < 0 || k >= n then s.(j) else (index s k).(j)))
           ks)
        (elements c))
    cs

let test_unravel_cases () =
  match Nx.unravel_index (held [| 2; 2 |] [| 0; 5; 6; -1 |]) [| 2; 3 |] with
  | [ r; c ] ->
      equal (array int) [| 2; 2 |] (Nx.shape r);
      equal (array int64) [| 0L; 1L; 2L; 2L |] (elements r);
      equal (array int64) [| 0L; 2L; 3L; 3L |] (elements c);
      equal int 0 (List.length (Nx.unravel_index (held [| 1 |] [| 0 |]) [||]));
      invalid ~by:"Nx.unravel_index" (fun () ->
          Nx.unravel_index (held [| 1 |] [| 0 |]) [| -1 |])
  | _ -> fail "two axes"

let masks =
  group "masks and coordinates"
    [
      prop ~count "slice through a mask reads its positions in C order"
        (mask_case Repeats) law_mask_slice;
      prop ~count "set through a mask writes its positions"
        (mask_case Distinct) law_mask_set;
      test "masks at the values their definition gives" test_mask_cases;
      prop "unravel_index gives each index's coordinates"
        Gen.(
          let* r = int_range 0 3 in
          let* s = array ~size:(constant r) (int_range 0 3) in
          let n = numel s in
          let+ ks = array ~size:(int_range 0 6) (int_range (-2) (n + 1)) in
          (s, ks))
        law_unravel;
      test "unravel_index at the values its definition gives"
        test_unravel_cases;
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
      test "a selection of the most axes passes through no more" (fun () ->
          let s = Array.init 32 (fun i -> if i < 2 then 2 else 1) in
          let x, _ = operand s in
          let y = Nx.slice Nx.[ I 1; T (held [| 1; 1 |] [| 1 |]) ] x in
          equal (array int) (Array.make 32 1) (Nx.shape y);
          equal (array int32) [| 4l |] (elements y));
      test "a step past the axis keeps one position" (fun () ->
          let x, _ = operand [| 3 |] in
          List.iter
            (fun (start, stop, step, want) ->
              equal
                ~msg:(Printf.sprintf "Rs (%d, %d, %d)" start stop step)
                (array int32) want
                (elements (Nx.slice Nx.[ Rs (start, stop, step) ] x)))
            [
              (0, 3, max_int, [| 1l |]);
              (1, 3, max_int - 1, [| 2l |]);
              (2, -4, min_int, [| 3l |]);
              (2, 0, min_int + 1, [| 3l |]);
              (min_int, max_int, max_int, [| 1l |]);
              (max_int, min_int, min_int, [| 3l |]);
              (3, 0, max_int, [||]);
              (0, 2, min_int, [||]);
            ]);
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
          let at pos = Nx.[ A; A; D (held [||] [| pos |], 1); A ] in
          let c = Nx.set (at 2) row (Nx.place Nx.Host.on cache) in
          let got = elements (Nx.slice Nx.[ A; A; I 2; A ] c) in
          equal (array float_exact) [| 0.; 1.; 2.; 3.; 4.; 5. |] got;
          let past = Nx.set (at 9) row (Nx.place Nx.Host.on cache) in
          equal ~msg:"a start past capacity writes the last row"
            (array float_exact)
            [| 0.; 1.; 2.; 3.; 4.; 5. |]
            (elements (Nx.slice Nx.[ A; A; I 3; A ] past)));
    ]

(* nx.cpu, declining every scatter. *)
module No_scatters = struct
  include Nx_cpu

  let name = "nx.no-scatters"
  let scatter _ ~unique:_ ~axis:_ ~dst:_ ~into:_ _ _ = A.Declined
end

module None_ =
  (val Nx.devices ~kernels:(module No_scatters) [ Nx_support.memory 2 ])

let test_declined_scatter_raises () =
  let on dt shape data = Nx.place None_.on (host dt shape data) in
  let p = on D.Int64 [| 3 |] [| 1L; 1L; 0L |] in
  let u = on D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
  let x = on D.Float32 [| 2 |] [| 0.; 0. |] in
  raises
    (Invalid_argument
       "Nx.scatter: nx.no-scatters does not compute Scatter on int64, float32, \
        float32 (m2); place its operands with Nx.place on a set whose kernels \
        compute it") (fun () -> Nx.scatter ~combine:Nx.Add ~axis:0 p u x)

let declines =
  group "declines"
    [
      test "a scatter its device's kernels decline raises naming the move"
        test_declined_scatter_raises;
    ]

(* An update through more positions held in data than one program reads:
   [n] scalar positions into a value of [n] unit axes, the [k]th (if any)
   outside its axis. *)
let test_many_held n k () =
  let x = Nx.zeros Nx.float32 (Array.make n 1) in
  let at i = Nx.T (Nx.scalar Nx.int64 (if i = k then 1L else 0L)) in
  let y = Nx.set (List.init n at) (Nx.scalar Nx.float32 5.) x in
  equal (array float_exact) [| (if k < n then 0. else 5.) |] (elements y)

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
      let p = held [| 2 |] [| 2; 0 |] in
      equal (array w)
        [| data.(2); data.(0); data.(5); data.(3) |]
        (elements (Nx.slice Nx.[ A; T p ] x));
      equal (array w)
        [| data.(0); data.(1); data.(1); data.(3); data.(4); data.(4) |]
        (elements (Nx.set Nx.[ A; T p ] (Nx.slice Nx.[ A; L [ 1; 0 ] ] x) x)))

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
              Nx.slice Nx.[ D (held [| 1 |] [| 0 |], 1) ] x);
          invalid ~by:"Nx.slice" (fun () ->
              Nx.slice Nx.[ D (held [||] [| 0 |], 3) ] x);
          invalid ~by:"Nx.slice" (fun () -> Nx.slice Nx.[ L [ 0; 2 ] ] x);
          invalid ~by:"Nx.set" (fun () -> Nx.set Nx.[ L [ 1; 1 ] ] x x);
          invalid ~by:"Nx.get" (fun () -> Nx.get [ 0; 0; 0 ] x);
          invalid ~by:"Nx.take" (fun () ->
              Nx.take ~axis:2 (held [| 1 |] [| 0 |]) x);
          invalid ~by:"Nx.take_along_axis" (fun () ->
              Nx.take_along_axis ~axis:0 (held [| 1 |] [| 0 |]) x);
          invalid ~by:"Nx.scatter" (fun () ->
              Nx.scatter ~axis:0 (held [| 2 |] [| 0; 1 |]) x x);
          invalid ~by:"Nx.scatter" (fun () ->
              let b = Nx.zeros Nx.bool [| 2 |] in
              Nx.scatter ~combine:Add ~axis:0 (held [| 2 |] [| 0; 1 |]) b b));
    ]

(* An update at a window whose start is donated: the start is read once. *)
let test_donated_start () =
  let x = Nx.zeros Nx.float32 [| 4 |] in
  let start = Nx.place Nx.Host.on (Nx.scalar Nx.int64 1L) in
  let y = Nx.set [ Nx.D (Nx.donate start, 2) ] (Nx.ones Nx.float32 [| 2 |]) x in
  equal (array float_exact) [| 0.; 1.; 1.; 0. |] (elements y)

let many =
  group "many positions"
    [
      test "an update at a donated window start" test_donated_start;
      cases "an update through positions held in data" ~name:fst
        [
          ("15", (15, 15));
          ("16", (16, 16));
          ("32", (32, 32));
          ("32, the 31st outside", (32, 30));
          ("16, the first outside", (16, 0));
        ]
        (fun (_, (n, k)) -> test_many_held n k ());
      (* Written positions, positions held in data and a window over three
         axes: past two of them the address so far carries into the next
         map, so a position outside its axis read before the carry still
         keeps its index from being written, and a window's start read
         after it is still clamped. *)
      cases "set mixes written, held and window positions"
        ~name:(Format.asprintf "%a" pp_case)
        [
          {
            s = [| 3; 4; 5 |];
            entries = [ L [ 2; 0 ]; T ([| 2 |], [| 1; 4 |]); D (4, 2) ];
          };
          {
            s = [| 4; 3; 4 |];
            entries = [ D (-1, 2); L [ 1; -1 ]; T ([| 2 |], [| 3; -1 |]) ];
          };
          {
            s = [| 4; 4; 3 |];
            entries = [ T ([| 2 |], [| 0; 4 |]); D (9, 1); L [ -2; 0 ] ];
          };
        ]
        check_set;
    ]

let () =
  exit
    (run "nx indexing"
       [ laws; definitions; masks; views; dtypes; declines; many; refusals ])
