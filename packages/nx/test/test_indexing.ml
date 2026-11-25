(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.mli's Indexing and slicing and Functional updates: slice and set against a
   reference that reads and writes OCaml arrays entry by entry, over operands of
   every layout Nx_support.laid makes; take, take_along_axis, unravel_index and
   scatter against their definitions; views, every dtype and the raises. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype

let elements = Nx.to_array
let numel s = Array.fold_left ( * ) 1 s

let position s idx =
  let k = ref 0 in
  Array.iteri (fun a i -> k := (!k * s.(a)) + i) idx;
  !k

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

let host dt s data =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) data)

let held s ps = host D.Int64 s (Array.map Int64.of_int ps)

(* [by]'s raises, each row a reason and a call. *)
let raises_naming by rows =
  cases "raises naming itself" ~id:fst rows (fun (_, f) -> invalid ~by f)

let r reason f = (reason, fun () -> ignore (f ()))

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
   Python's slice rule, each next position formed only if it lies before [stop],
   so that no step overflows. *)
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
   twice, which [slice] reads twice and [set] refuses; [Distinct] names each at
   most once, in any order, each from the start or the end. *)
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
     (* The number of axes selected along: one case in five selects along none,
        so that each kind of entry turns up in a run. *)
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

(* Distinct elements of shape [s] laid out as [l], and the elements the value
   reads. *)
let operand ?(l = C_order) s =
  laid l D.Int32 s (Array.init (numel s) (fun k -> Int32.of_int (k + 1)))

let covers (c, l) =
  let has f = List.exists f c.entries in
  cover "held positions" (has (function T _ -> true | _ -> false));
  cover "a window" (has (function D _ -> true | _ -> false));
  cover "written positions" (has (function L _ -> true | _ -> false));
  cover "a negative step" (has (function Rs (_, _, c) -> c < 0 | _ -> false));
  cover "a new axis" (has (( = ) N));
  cover "a strided operand"
    (l = Reversed || l = Axes_reversed || l = Every_other)

(* The cases a law that [covers] runs. Its rarest label, a negative step, marks
   about one case in sixteen, so a named example holds it; the count keeps the
   other labels' reach. *)
let count = 300

(* [Nx.slice] of [c] reads each selection index at the position the reference
   reads, zero outside. *)
let law_slice (c, l) =
  covers (c, l);
  let x, data = operand ~l c.s in
  let sel, read = selection c.s c.entries in
  let y = Nx.slice (List.map to_index c.entries) x in
  equal ~msg:"shape" (array int) sel (Nx.shape y);
  equal ~msg:"elements" (array int32)
    (Array.init (numel sel) (fun k ->
         match read (index sel k) with
         | Some ix -> data.(position c.s ix)
         | None -> 0l))
    (elements y)

(* [Nx.set] of [c] writes each selection index's value at the position the
   reference reads, the last write winning, and leaves the operand as it was. *)
let check_set (c, l) =
  let x, data = operand ~l c.s in
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

let law_set (c, l) =
  covers (c, l);
  let data_entry = function T _ | L _ | D _ -> true | _ -> false in
  cover "positions written in the program alone"
    (not (List.exists data_entry c.entries));
  cover "one axis of positions, every other whole"
    (List.length (List.filter data_entry c.entries) = 1
    && List.for_all (fun e -> data_entry e || e = A || e = N) c.entries);
  check_set (c, l)

(* A negative step, and written positions, positions held in data and a window
   over three axes in one update: a position outside its axis keeps its index
   from being written whichever entry holds it, and a window's start is
   clamped. *)
let set_examples =
  [
    ({ s = [| 3; 2 |]; entries = [ Rs (2, 0, -1) ] }, Reversed);
    ( {
        s = [| 3; 4; 5 |];
        entries = [ L [ 2; 0 ]; T ([| 2 |], [| 1; 4 |]); D (4, 2) ];
      },
      C_order );
    ( {
        s = [| 4; 3; 4 |];
        entries = [ D (-1, 2); L [ 1; -1 ]; T ([| 2 |], [| 3; -1 |]) ];
      },
      Every_other );
    ( {
        s = [| 4; 4; 3 |];
        entries = [ T ([| 2 |], [| 0; 4 |]); D (9, 1); L [ -2; 0 ] ];
      },
      Axes_reversed );
  ]

let pp_laid_case ppf (c, l) = Format.fprintf ppf "%a, %a" pp_case c pp_layout l
let laid_case lists = Gen.(with_pp pp_laid_case (pair (case lists) layouts))

(* Masks *)

(* A selection with a mask over the [k] axes from [at], the other entries drawn
   as [case] draws them. Its reference is the selection of the operand with
   those axes merged into one, the mask by the positions where it holds, in C
   order: the operand's elements in C order are the merged one's. *)
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
    (Format.pp_print_list pp_entry)
    c.before pp_ints (Array.sub c.ms c.at c.k)
    (Format.pp_print_list pp_entry)
    c.after

let mask_case lists =
  let open Gen in
  with_pp pp_mask_case
    (let* r = int_range 0 3 in
     let* ms = array ~size:(constant r) (int_range 0 3) in
     (* One mask in five is 0-d, and two in five cover two axes or more where
        the operand has them. *)
     let* k =
       frequency
         (((1, constant 0) :: (if r >= 1 then [ (2, constant 1) ] else []))
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
let mask s bs = host D.Bool s bs

(* A mask's selections at values its definition gives, over [[1 2 3]; [4 5
   6]]. *)
let mask_rows =
  [
    ( "a mask over both axes",
      ( Nx.[ M (mask [| 2; 3 |] [| false; true; true; false; true; false |]) ],
        [| 3 |],
        [| 2l; 3l; 5l |] ) );
    ( "a mask over the rows",
      (Nx.[ M (mask [| 2 |] [| false; true |]) ], [| 1; 3 |], [| 4l; 5l; 6l |])
    );
    ( "a mask over the columns",
      ( Nx.[ A; M (mask [| 3 |] [| true; false; true |]) ],
        [| 2; 2 |],
        [| 1l; 3l; 4l; 6l |] ) );
    ( "a 0-d mask that holds",
      ( Nx.[ M (mask [||] [| true |]) ],
        [| 1; 2; 3 |],
        [| 1l; 2l; 3l; 4l; 5l; 6l |] ) );
    ( "a 0-d mask that does not",
      (Nx.[ M (mask [||] [| false |]) ], [| 0; 2; 3 |], [||]) );
    ( "two masks, each along its own axes",
      ( Nx.
          [
            M (mask [| 2 |] [| true; true |]);
            M (mask [| 3 |] [| true; false; true |]);
          ],
        [| 2; 2 |],
        [| 1l; 3l; 4l; 6l |] ) );
    ( "a mask beside positions",
      ( Nx.[ M (mask [| 2 |] [| false; true |]); T (held [| 2 |] [| 2; 0 |]) ],
        [| 1; 2 |],
        [| 6l; 4l |] ) );
  ]

let mask_set_rows =
  [
    ( "the columns where a mask holds",
      ( Nx.[ A; M (mask [| 3 |] [| false; true; true |]) ],
        [| 1 |],
        [| 1l; 0l; 0l; 4l; 0l; 0l |] ) );
    ( "the outer product of two masks",
      ( Nx.
          [
            M (mask [| 2 |] [| true; true |]);
            M (mask [| 3 |] [| true; false; true |]);
          ],
        [| 2; 1 |],
        [| 0l; 2l; 0l; 0l; 5l; 0l |] ) );
  ]

(* Under a trace a mask's count is unknown: the selection raises, and the
   positions of a fixed size select. *)
let test_traced_mask () =
  let x = fst (operand [| 2; 3 |]) in
  let m = mask [| 2 |] [| false; true |] in
  Nx.Prim.interpret ~name:"test.mask" Values mark (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form m) Mark in
      raises
        (Invalid_argument
           "Nx.slice: a mask's count is data, which test.mask cannot read; \
            select with T (positions ~size m) or compute with where") (fun () ->
          Nx.slice Nx.[ M t ] x);
      invalid ~by:"Nx.set" (fun () -> Nx.set Nx.[ M t ] x x);
      equal (array int) [| 1; 3 |]
        (Nx.shape (Nx.slice Nx.[ T (Nx.positions ~size:1 t) ] x)))

module Counted2 =
  (val Nx.devices
         ~kernels:(module Nx_support.Counting)
         [ Nx_support.memory 0; Nx_support.memory 1 ])

(* A gather at written positions on two devices: each device's row is the one
   its position selects, by as many kernel calls as a single position takes. *)
let law_written (n, ps) =
  cover "no position" (ps = []);
  cover "one position" (List.length ps = 1);
  cover "more positions than the axis" (List.length ps > n);
  let data = Array.init (2 * n) Int32.of_int in
  let x = Nx.place Counted2.on (host D.Int32 [| n; 2 |] data) in
  let gather ps =
    Nx_support.Counting.reset ();
    let y = Nx.slice Nx.[ L ps ] x in
    (y, Nx_support.Counting.calls ())
  in
  let y, calls = gather ps in
  equal ~msg:"shape" (array int) [| List.length ps; 2 |] (Nx.shape y);
  let row p =
    let p = if p < 0 then p + n else p in
    [| data.(2 * p); data.((2 * p) + 1) |]
  in
  Nx.sync y;
  let copies = Iarray.to_list (Nx.Repr.shards ~by:"test" y) in
  equal ~msg:"copies" int 2 (List.length copies);
  List.iter
    (fun a ->
      equal ~msg:"rows" (array int32)
        (Array.concat (List.map row ps))
        (A.to_array a))
    copies;
  match ps with
  | [] -> ()
  | p :: _ ->
      let _, one = gather [ p ] in
      equal ~msg:"kernel calls of one position" int one calls

let held_values =
  Testable.make
    ~pp:(fun ppf v ->
      A.Layout.pp ppf (A.layout (Option.get (Nx.Repr.array ~by:"test" v))))
    ~equal:( == )

let shares x y =
  let buffer v = A.buffer (Option.get (Nx.Repr.array ~by:"test" v)) in
  Rig.Buffer.overlaps (buffer x) (buffer y)

let x23 () = fst (operand [| 2; 3 |])

let slice =
  group "slice"
    [
      prop ~count "reads each entry's positions, zero outside"
        ~examples:
          [ ({ s = [| 3; 2 |]; entries = [ Rs (2, 0, -1) ] }, Every_other) ]
        (laid_case Repeats) law_slice;
      prop ~count "reads a mask's positions in C order" (mask_case Repeats)
        law_mask_slice;
      cases "reads masks at the values their definition gives" ~id:fst mask_rows
        (fun (_, (idx, shape, want)) ->
          let y = Nx.slice idx (x23 ()) in
          equal ~msg:"shape" (array int) shape (Nx.shape y);
          equal ~msg:"elements" (array int32) want (elements y));
      test "raises on a traced mask, where positions of a size select"
        test_traced_mask;
      test "makes a view of positions written in the program" (fun () ->
          let x = fst (operand [| 3; 4 |]) in
          let y = Nx.slice Nx.[ I 1; Rs (3, 0, -2); N ] x in
          equal (array int) [| 2; 1 |] (Nx.shape y);
          equal (array int32) [| 8l; 6l |] (elements y);
          satisfies ~claim:"a view over the operand's memory" held_values
            (shares x) y);
      test "passes a selection of the most axes through no more" (fun () ->
          let s = Array.init 32 (fun i -> if i < 2 then 2 else 1) in
          let x = fst (operand s) in
          let y = Nx.slice Nx.[ I 1; T (held [| 1; 1 |] [| 1 |]) ] x in
          equal (array int) (Array.make 32 1) (Nx.shape y);
          equal (array int32) [| 4l |] (elements y));
      cases "keeps one position where Rs's step passes the axis"
        ~id:(fun (a, b, c, _) -> Printf.sprintf "Rs (%d, %d, %d)" a b c)
        [
          (0, 3, max_int, [| 1l |]);
          (1, 3, max_int - 1, [| 2l |]);
          (2, -4, min_int, [| 3l |]);
          (2, 0, min_int + 1, [| 3l |]);
          (min_int, max_int, max_int, [| 1l |]);
          (max_int, min_int, min_int, [| 3l |]);
          (3, 0, max_int, [||]);
          (0, 2, min_int, [||]);
        ]
        (fun (start, stop, step, want) ->
          equal (array int32) want
            (elements
               (Nx.slice Nx.[ Rs (start, stop, step) ] (fst (operand [| 3 |])))));
      prop "gathers written positions by one copy to each device"
        Gen.(
          let* n = int_range 1 6 in
          let size =
            frequency [ (1, constant 0); (1, constant 1); (2, int_range 2 40) ]
          in
          let+ ps = list ~size (int_range (-n) (n - 1)) in
          (n, ps))
        law_written;
      raises_naming "Nx.slice"
        [
          r "a written position past its axis" (fun () ->
              Nx.slice Nx.[ A; I 3 ] (x23 ()));
          r "a list position past its axis" (fun () ->
              Nx.slice Nx.[ A; L [ 0; 3 ] ] (x23 ()));
          r "a list position past a short axis" (fun () ->
              Nx.slice Nx.[ L [ 0; 2 ] ] (x23 ()));
          r "more entries than axes" (fun () ->
              Nx.slice Nx.[ A; A; A ] (x23 ()));
          r "a step of 0" (fun () -> Nx.slice Nx.[ Rs (0, 2, 0) ] (x23 ()));
          r "a window past its axis" (fun () ->
              Nx.slice Nx.[ D (held [||] [| 0 |], 3) ] (x23 ()));
          r "a window of negative length" (fun () ->
              Nx.slice Nx.[ D (held [||] [| 0 |], -1) ] (x23 ()));
          r "a window's start not 0-d" (fun () ->
              Nx.slice Nx.[ D (held [| 1 |] [| 0 |], 1) ] (x23 ()));
          r "a mask of another shape" (fun () ->
              Nx.slice Nx.[ M (mask [| 3 |] [| true; true; true |]) ] (x23 ()));
          r "a mask past the axes" (fun () ->
              Nx.slice Nx.[ A; A; M (mask [| 1 |] [| true |]) ] (x23 ()));
        ];
    ]

let get =
  group "get"
    [
      test "is the view of slice's I entries" (fun () ->
          let x = fst (operand [| 3; 4 |]) in
          let z = Nx.get [ -1 ] x in
          equal (array int32) [| 9l; 10l; 11l; 12l |] (elements z);
          satisfies ~claim:"a view over the operand's memory" held_values
            (shares x) z);
      raises_naming "Nx.get"
        [ r "more positions than axes" (fun () -> Nx.get [ 0; 0; 0 ] (x23 ())) ];
    ]

(* take, take_along_axis and scatter against their definitions. *)

let law_take (s, axis, ps, p, l) =
  let x, data = operand ~l s in
  cover "no axis" (axis = None);
  let r = Array.length s and pr = Array.length ps in
  let s, a, data =
    match axis with Some a -> (s, a, data) | None -> ([| numel s |], 0, data)
  in
  let r = if axis = None then 1 else r in
  let y =
    match axis with
    | Some a -> Nx.take ~axis:a (held ps p) x
    | None -> Nx.take (held ps p) x
  in
  let s' =
    Array.concat [ Array.sub s 0 a; ps; Array.sub s (a + 1) (r - a - 1) ]
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
                 if d < a then i.(d) else if d = a then q else i.(d + pr - 1))
           in
           data.(position s ix)))
    (elements y)

let take =
  group "take"
    [
      prop "replaces an axis by the positions' axes, the flat value without one"
        ~examples:[ ([| 2; 3 |], None, [| 2 |], [| 5; -1 |], Every_other) ]
        Gen.(
          let* r = int_range 1 3 in
          let* s = array ~size:(constant r) (int_range 1 3) in
          let* axis = option (int_range 0 (r - 1)) in
          let d = match axis with Some a -> s.(a) | None -> numel s in
          let* pr = int_range 0 2 in
          let* ps = array ~size:(constant pr) (int_range 0 3) in
          let* p = array ~size:(constant (numel ps)) (int_range (-1) d) in
          let+ l = layouts in
          (s, axis, ps, p, l))
        law_take;
      raises_naming "Nx.take"
        [
          r "an axis past the value's" (fun () ->
              Nx.take ~axis:2 (held [| 1 |] [| 0 |]) (x23 ()));
        ];
    ]

(* [p] of [ps]'s shape and [x] of [s], broadcast off [axis]. *)
let law_take_along (xs, a, ps, p, l) =
  let off f =
    List.exists (fun i -> i <> a && f i) (List.init (Array.length xs) Fun.id)
  in
  cover "positions broadcast" (off (fun i -> ps.(i) = 1 && xs.(i) > 1));
  cover "the value broadcast" (off (fun i -> xs.(i) = 1 && ps.(i) > 1));
  let x, data = operand ~l xs in
  let y = Nx.take_along_axis ~axis:a (held ps p) x in
  let out =
    Array.mapi (fun i e -> if i = a then e else Stdlib.max e xs.(i)) ps
  in
  equal (array int) out (Nx.shape y);
  equal (array int32)
    (Array.init (numel out) (fun k ->
         let i = index out k in
         let q =
           p.(position ps
                (Array.mapi (fun d j -> if ps.(d) = 1 then 0 else j) i))
         in
         if q < 0 || q >= xs.(a) then 0l
         else
           let ix =
             Array.mapi
               (fun d j -> if d = a then q else if xs.(d) = 1 then 0 else j)
               i
           in
           data.(position xs ix)))
    (elements y)

let take_along_axis =
  group "take_along_axis"
    [
      prop "pairs positions with indices, broadcast off its axis"
        ~examples:[ ([| 2; 3 |], 1, [| 1; 2 |], [| 0; 2 |], C_order) ]
        Gen.(
          let* r = int_range 1 3 in
          let* s = array ~size:(constant r) (int_range 1 3) in
          let* a = int_range 0 (r - 1) in
          let* k = int_range 0 3 in
          (* Off the axis, one extent in three of the positions or of the value
             is 1, which broadcasts. *)
          let* units = array ~size:(constant r) (int_range 0 2) in
          let ps =
            Array.mapi
              (fun i d -> if i = a then k else if units.(i) = 1 then 1 else d)
              s
          in
          let xs =
            Array.mapi (fun i d -> if i <> a && units.(i) = 2 then 1 else d) s
          in
          let* p = array ~size:(constant (numel ps)) (int_range (-1) xs.(a)) in
          let+ l = layouts in
          (xs, a, ps, p, l))
        law_take_along;
      raises_naming "Nx.take_along_axis"
        [
          r "positions of another rank" (fun () ->
              Nx.take_along_axis ~axis:0 (held [| 1 |] [| 0 |]) (x23 ()));
          r "shapes that do not broadcast off the axis" (fun () ->
              Nx.take_along_axis ~axis:0 (held [| 1; 2 |] [| 0; 0 |]) (x23 ()));
          r "an axis past the value's" (fun () ->
              Nx.take_along_axis ~axis:2 (held [| 1; 1 |] [| 0 |]) (x23 ()));
        ];
    ]

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

let unravel_index =
  group "unravel_index"
    [
      prop "gives each index's coordinates, each axis' extent outside"
        Gen.(
          let* r = int_range 0 3 in
          let* s = array ~size:(constant r) (int_range 0 3) in
          let n = numel s in
          let+ ks = array ~size:(int_range 0 6) (int_range (-2) (n + 1)) in
          (s, ks))
        law_unravel;
      test "gives the coordinates its definition gives" (fun () ->
          equal
            (list (array int64))
            [ [| 0L; 1L; 2L; 2L |]; [| 0L; 2L; 3L; 3L |] ]
            (List.map elements
               (Nx.unravel_index (held [| 2; 2 |] [| 0; 5; 6; -1 |]) [| 2; 3 |]));
          equal ~msg:"a 0-d shape" int 0
            (List.length (Nx.unravel_index (held [| 1 |] [| 0 |]) [||])));
      raises_naming "Nx.unravel_index"
        [
          r "a negative extent" (fun () ->
              Nx.unravel_index (held [| 1 |] [| 0 |]) [| -1 |]);
        ];
    ]

(* set *)

(* An update through [n] positions held in data: [n] scalar positions into a
   value of [n] unit axes, the [k]th (if any) outside its axis. *)
let test_many_held (n, k) =
  let x = Nx.zeros Nx.float32 (Array.make n 1) in
  let at i = Nx.T (Nx.scalar Nx.int64 (if i = k then 1L else 0L)) in
  let y = Nx.set (List.init n at) (Nx.scalar Nx.float32 5.) x in
  equal (array float_exact) [| (if k < n then 0. else 5.) |] (elements y)

(* An update of [4; 4; 4] at two positions held in data along each axis, the
   first entry's split over two software devices and the others' at [cuts]. Each
   entry's positions, broadcast to the selection [2; 2; 2], are read beside the
   others' at the union of their cuts: an entry cut along its own axis of the
   selection beside the first is refused, named as the operand of its rank among
   the entries, and computes once placed whole on each device. *)
module Two = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

let set_cut (second, third) =
  let at p = Nx.T (Nx.place p (held [| 2 |] [| 0; 3 |])) in
  let x = Nx.place Two.on (Nx.zeros Nx.int32 [| 4; 4; 4 |]) in
  let v = Nx.place Two.on (Nx.scalar Nx.int32 1l) in
  Nx.set [ at (Two.split ~axis:0); at second; at third ] v x

let set =
  group "set"
    [
      prop ~count "writes each entry's positions, the last write winning"
        ~examples:set_examples (laid_case Distinct) law_set;
      prop "of a slice is the identity" (laid_case Distinct) (fun (c, l) ->
          let x, data = operand ~l c.s in
          let idx = List.map to_index c.entries in
          equal (array int32) data (elements (Nx.set idx (Nx.slice idx x) x)));
      prop ~count "writes a mask's positions" (mask_case Distinct) law_mask_set;
      cases "writes masks at the values their definition gives" ~id:fst
        mask_set_rows (fun (_, (idx, vs, want)) ->
          equal (array int32) want
            (elements (Nx.set idx (Nx.zeros Nx.int32 vs) (x23 ()))));
      test "consumes a donated value, naming itself" (fun () ->
          let x = Nx.add (x23 ()) (Nx.zeros Nx.int32 [| 2; 3 |]) in
          let h = Nx.donate x in
          let y = Nx.set Nx.[ I 0; A ] (Nx.zeros Nx.int32 [| 3 |]) h in
          equal (array int32) [| 0l; 0l; 0l; 4l; 5l; 6l |] (elements y);
          raises_match (Exn.invalid_arg ~sub:"was donated to Nx.set") (fun () ->
              Nx.copy h));
      test "writes one row of a cache at a decode step, the last past capacity"
        (fun () ->
          let cache = Nx.zeros Nx.float32 [| 1; 2; 4; 3 |] in
          let row =
            Nx.Repr.of_array Nx.Host.v
              (A.of_array D.Float32 [| 1; 2; 1; 3 |]
                 (Array.init 6 Float.of_int))
          in
          let at pos = Nx.[ A; A; D (held [||] [| pos |], 1); A ] in
          let c = Nx.set (at 2) row cache in
          equal ~msg:"at 2" (array float_exact)
            [| 0.; 1.; 2.; 3.; 4.; 5. |]
            (elements (Nx.slice Nx.[ A; A; I 2; A ] c));
          let past = Nx.set (at 9) row cache in
          equal ~msg:"a start past capacity writes the last row"
            (array float_exact)
            [| 0.; 1.; 2.; 3.; 4.; 5. |]
            (elements (Nx.slice Nx.[ A; A; I 3; A ] past)));
      test "reads a donated window start once" (fun () ->
          let x = Nx.zeros Nx.float32 [| 4 |] in
          let start = Nx.scalar Nx.int64 1L in
          let y =
            Nx.set [ Nx.D (Nx.donate start, 2) ] (Nx.ones Nx.float32 [| 2 |]) x
          in
          equal (array float_exact) [| 0.; 1.; 1.; 0. |] (elements y));
      cases
        "refuses positions cut along their own selection axis, naming the \
         entry's rank"
        ~id:(fun (n, _, _) -> n)
        [
          ("the second cut beside the first", (Two.split ~axis:0, Two.on), 1);
          ("the third cut beside the first", (Two.on, Two.split ~axis:0), 2);
        ]
        (fun (_, cuts, k) ->
          match set_cut cuts with
          | _ -> fail "Nx.set raised nothing"
          | exception Invalid_argument m ->
              starts_with
                ~prefix:(Printf.sprintf "Nx.set: operand %d lies at " k)
                m);
      test
        "computes once the refused entry's positions are whole on each device"
        (fun () ->
          let y = set_cut (Two.on, Two.on) in
          let want =
            Array.init 64 (fun k ->
                let i = index [| 4; 4; 4 |] k in
                if Array.for_all (fun p -> p = 0 || p = 3) i then 1l else 0l)
          in
          equal (array int32) want (elements y));
      cases "writes through many positions held in data, none outside"
        ~id:(fun (n, k) ->
          if k < n then Printf.sprintf "%d, position %d outside" n k
          else string_of_int n)
        [ (15, 15); (16, 16); (32, 32); (32, 30); (16, 0) ]
        test_many_held;
      raises_naming "Nx.set"
        [
          r "a list naming a position twice" (fun () ->
              Nx.set Nx.[ L [ 1; 1 ] ] (x23 ()) (x23 ()));
          r "a list naming a position twice, once from the end" (fun () ->
              Nx.set
                Nx.[ A; L [ 1; -2 ] ]
                (Nx.zeros Nx.int32 [| 2; 2 |])
                (x23 ()));
          r "a value that does not broadcast to the selection" (fun () ->
              Nx.set Nx.[ A ] (Nx.zeros Nx.int32 [| 4 |]) (x23 ()));
          r "a value that fits but for the mask's count" (fun () ->
              Nx.set
                Nx.[ M (mask [| 2 |] [| true; true |]) ]
                (Nx.zeros Nx.int32 [| 3; 3 |])
                (x23 ()));
          r "a written position past its axis" (fun () ->
              Nx.set Nx.[ I 2 ] (Nx.zeros Nx.int32 [| 3 |]) (x23 ()));
        ];
    ]

(* scatter *)

let law_scatter_ints (s, a, us, p, combine) =
  let x, data = operand s in
  let u = Array.init (numel us) (fun k -> Int32.of_int ((k * 7 mod 11) - 5)) in
  let f, c =
    match combine with
    | `Set -> ((fun _ v -> v), Nx.Set)
    | `Add -> (Int32.add, Nx.Add)
    | `Max -> ((fun a b -> if Int32.compare a b >= 0 then a else b), Nx.Max)
    | `Min -> ((fun a b -> if Int32.compare a b <= 0 then a else b), Nx.Min)
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
  equal (array int32) expected
    (elements (Nx.scatter ~combine:c ~axis:a (held us p) (host D.Int32 us u) x))

let scatter_case =
  Gen.(
    let* r = int_range 1 3 in
    let* s = array ~size:(constant r) (int_range 1 3) in
    let* a = int_range 0 (r - 1) in
    let* k = int_range 0 4 in
    let us = Array.mapi (fun i d -> if i = a then k else d) s in
    let* p = array ~size:(constant (numel us)) (int_range (-1) s.(a)) in
    let+ combine = of_list [ `Set; `Add; `Max; `Min ] in
    (s, a, us, p, combine))

(* Floats a scatter combines: small integers, so that every sum is exact in
   float32 and in the dtype, signed zeros where the combine reads them by their
   sign (Set and Add), and NaN. *)
type float_dt = F : (float, 's) D.t -> float_dt

let pp_float_dt ppf (F dt) = D.pp ppf dt

let float_case =
  Gen.(
    let* f =
      of_list ~pp:pp_float_dt [ F D.Float32; F D.Float64; F D.Bfloat16 ]
    in
    let* c = scatter_case in
    let+ picks =
      pair
        (array ~size:(int_range 1 8) (int_range 0 5))
        (array ~size:(int_range 1 8) (int_range 0 5))
    in
    (f, c, picks))

let law_scatter_floats (F dt, (s, a, us, p, combine), (xi, ui)) =
  let pool =
    match combine with
    | `Set | `Add -> [| 0.; -0.; 1.; -2.; 3.; nan |]
    | `Max | `Min -> [| 1.; -2.; 3.; nan; 4.; -5. |]
  in
  let pick picks k = pool.(picks.(k mod Array.length picks)) in
  let xs = Array.init (numel s) (pick xi)
  and u = Array.init (numel us) (pick ui) in
  let larger a b =
    if Float.is_nan a || Float.is_nan b then nan else Float.max a b
  in
  let smaller a b =
    if Float.is_nan a || Float.is_nan b then nan else Float.min a b
  in
  let f, c =
    match combine with
    | `Set -> ((fun _ v -> v), Nx.Set)
    | `Add -> (( +. ), Nx.Add)
    | `Max -> (larger, Nx.Max)
    | `Min -> (smaller, Nx.Min)
  in
  let expected = Array.copy xs in
  let landed = ref false in
  for k = 0 to numel us - 1 do
    let q = p.(k) in
    if q >= 0 && q < s.(a) then begin
      let ix = index us k in
      ix.(a) <- q;
      let t = position s ix in
      landed := true;
      expected.(t) <- f expected.(t) u.(k)
    end
  done;
  cover "an update lands" !landed;
  cover "a narrow float" (D.bits dt < 32);
  let y =
    Nx.scatter ~combine:c ~axis:a (held us p) (host dt us u) (host dt s xs)
  in
  equal (Nx_support.elements dt)
    (Array.to_list (Array.map (D.of_float dt) expected))
    (Array.to_list (Nx.to_array y))

(* A float32 [-0.] target and [-0.] updates added: [-0.], no [+0] added. *)
let add_of_zeros =
  (F D.Float32, ([| 2 |], 0, [| 2 |], [| 0; 0 |], `Add), ([| 1 |], [| 1 |]))

(* An element no update lands on keeps its bits, a signalling NaN's. *)
let test_untouched (_, c) =
  let nan_bits = 0x7fc00005l in
  let x =
    signalling D.Float32 [| 3 |] [| Int32.float_of_bits nan_bits; 1.; 2. |]
  in
  let bits v = (Nx.to_array (Nx.bitcast Nx.int32 v)).(0) in
  let y =
    Nx.scatter ~combine:c ~axis:0 (held [| 1 |] [| 2 |])
      (host D.Float32 [| 1 |] [| 5. |])
      x
  in
  equal int32 (bits x) (bits y)

(* nx.cpu, declining every scatter. *)
module No_scatters = struct
  include Nx_cpu

  let name = "nx.no-scatters"
  let scatter _ ~unique:_ ~axis:_ ~dst:_ ~into:_ _ _ = A.Declined
end

module None_ =
  (val Nx.devices ~kernels:(module No_scatters) [ Nx_support.memory 2 ])

(* A decline raises naming the function, the kernels, the operation, its dtypes,
   the device and the place to write. *)
let test_declined () =
  let on dt shape data = Nx.place None_.on (host dt shape data) in
  let p = on D.Int64 [| 3 |] [| 1L; 1L; 0L |] in
  let u = on D.Float32 [| 3 |] [| 1.; 2.; 3. |] in
  let x = on D.Float32 [| 2 |] [| 0.; 0. |] in
  match Nx.scatter ~combine:Nx.Add ~axis:0 p u x with
  | _ -> fail "Nx.scatter raised nothing"
  | exception Invalid_argument m ->
      starts_with ~prefix:"Nx.scatter: " m;
      List.iter
        (fun sub -> contains ~sub m)
        [
          "nx.no-scatters";
          "Scatter";
          "int64, float32, float32";
          "m2";
          "Nx.place";
        ]

let scatter =
  group "scatter"
    [
      prop "combines integer updates in C order" scatter_case law_scatter_ints;
      prop
        "combines float updates as its combine says, -0., NaN and narrow \
         floats included"
        ~examples:[ add_of_zeros ] float_case law_scatter_floats;
      cases
        "keeps the bits of an element no update lands on, a signalling NaN's"
        ~id:fst
        [ ("Set", Nx.Set); ("Add", Nx.Add); ("Max", Nx.Max); ("Min", Nx.Min) ]
        test_untouched;
      test "raises naming a decline's parts" test_declined;
      raises_naming "Nx.scatter"
        [
          r "positions and updates of other shapes" (fun () ->
              Nx.scatter ~axis:0 (held [| 2 |] [| 0; 1 |]) (x23 ()) (x23 ()));
          r "Add of booleans" (fun () ->
              let b = Nx.zeros Nx.bool [| 2 |] in
              Nx.scatter ~combine:Add ~axis:0 (held [| 2 |] [| 0; 1 |]) b b);
          r "an axis past the value's" (fun () ->
              Nx.scatter ~axis:2
                (held [| 2; 3 |] [| 0; 0; 0; 0; 0; 0 |])
                (x23 ()) (x23 ()));
          r "updates whose shape differs off the axis" (fun () ->
              Nx.scatter ~axis:0
                (held [| 1; 2 |] [| 0; 0 |])
                (Nx.zeros Nx.int32 [| 1; 2 |])
                (x23 ()));
        ];
    ]

(* A gather, a set and a scatter of every dtype. *)
let dtypes =
  cases "every dtype gathers, sets and scatters"
    ~id:(fun (D.Any dt) -> D.name dt)
    D.all
    (fun (D.Any dt) ->
      let data =
        Array.init 6 (fun k -> D.of_float dt (Float.of_int (k mod 7)))
      in
      let x = host dt [| 2; 3 |] data in
      let w = Testable.make ~pp:(D.pp_value dt) ~equal:( = ) in
      let p = held [| 2 |] [| 2; 0 |] in
      equal ~msg:"slice" (array w)
        [| data.(2); data.(0); data.(5); data.(3) |]
        (elements (Nx.slice Nx.[ A; T p ] x));
      equal ~msg:"set" (array w)
        [| data.(0); data.(1); data.(1); data.(3); data.(4); data.(4) |]
        (elements (Nx.set Nx.[ A; T p ] (Nx.slice Nx.[ A; L [ 1; 0 ] ] x) x));
      let rows = held [| 1; 3 |] [| 0; 1; 0 |]
      and u = Nx.slice Nx.[ I 1; N ] x in
      equal ~msg:"scatter Set" (array w)
        [| data.(3); data.(1); data.(5); data.(3); data.(4); data.(5) |]
        (elements (Nx.scatter ~axis:0 rows u x));
      if D.is D.Boolean dt then
        invalid ~by:"Nx.scatter" (fun () ->
            Nx.scatter ~combine:Add ~axis:0 rows u x))

(* The words of a few raises: the function, the reason and the operand. *)
let messages =
  test "raises read as their reason and operand" (fun () ->
      let x = x23 () in
      let said f =
        match ignore (f ()) with
        | () -> "raised nothing"
        | exception Invalid_argument m -> m
      in
      expect
        (String.concat "\n"
           [
             said (fun () -> Nx.slice Nx.[ A; I 3 ] x);
             said (fun () -> Nx.slice Nx.[ A; A; A ] x);
             said (fun () -> Nx.slice Nx.[ Rs (0, 2, 0) ] x);
             said (fun () -> Nx.slice Nx.[ A; L [ 0; 3 ] ] x);
             said (fun () -> Nx.slice Nx.[ D (held [||] [| 0 |], 3) ] x);
             said (fun () -> Nx.slice Nx.[ D (held [| 1 |] [| 0 |], 1) ] x);
             said (fun () -> Nx.take ~axis:2 (held [| 1 |] [| 0 |]) x);
             said (fun () -> Nx.set Nx.[ A ] (Nx.zeros Nx.int32 [| 4 |]) x);
           ])
      @@ __POS_OF__
           {|
        Nx.slice: position 3 is outside axis 1 of int32 [2; 3]
        Nx.slice: 3 entries address 3 axes; the operand is int32 [2; 3]
        Nx.slice: a step of 0
        Nx.slice: position 3 is outside axis 1 of int32 [2; 3]
        Nx.slice: a window of 3 on axis 0 of int32 [2; 3]
        Nx.slice: a window's start int64 [1] is not 0-d
        Nx.take: 2 is not an axis of int32 [2; 3]
        Nx.set: int32 [4] does not broadcast to the selection [2; 3]
        |})

let () =
  exit
    (run "nx indexing"
       [
         slice;
         get;
         take;
         take_along_axis;
         unravel_index;
         set;
         scatter;
         dtypes;
         messages;
       ])
