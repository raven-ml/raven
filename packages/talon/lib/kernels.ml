(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type checked = Column.t * (int * Error.t) option

(* Columns *)

let tensor dt c =
  match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false

let int64s c = tensor Nx.int64 c
let both a = function None -> a | Some b -> Nx.logical_and a b

(* [holding c] is where [c] holds a value, as a condition. *)
let holding c = Option.map (Nx.cast Nx.bool) (Column.validity c)

(* [valid cs] is where each of the columns [cs], of one row or of the rows they
   broadcast to, holds a value, [None] where none has a validity. *)
let valid cs =
  let n =
    List.fold_left
      (fun n c -> if Column.length c = 1 then n else Column.length c)
      1 cs
  in
  let add v c =
    match Column.validity c with
    | None -> v
    | Some m -> Some (both (Nx.broadcast_to [| n |] m) v)
  in
  List.fold_left add None cs

(* [make ty valid x] is the column of [ty] stored as [x], null where [valid] is
   clear. *)
let make ty valid x =
  let length = Nx.dim 0 x in
  let validity = Option.map (Nx.broadcast_to [| length |]) valid in
  Column.make ty ?validity ~length (Fixed (P x))

(* [checked ty cs ~ok x why] is [make ty] of [x], computed over the columns
   [cs], and the failure [why row] at the first row where [ok] does not hold,
   which is null; rows outside [live] and null rows do not fail. *)
let checked ty cs ?(live = Nx.scalar Nx.bool true) ~ok x why =
  let valid = valid cs in
  let bad =
    both
      (Nx.logical_and live (Nx.logical_not ok))
      (Option.map (Nx.cast Nx.bool) valid)
  in
  let bad = Nx.broadcast_to [| Nx.dim 0 x |] bad in
  let first () = Int64.to_int (Nx.item [] (Nx.argmax (Nx.cast Nx.uint8 bad))) in
  match if Nx.dim 0 x = 0 then None else Some (first ()) with
  | Some i when Nx.item [ i ] bad ->
      (make ty (Some (both (Nx.cast Nx.bit ok) valid)) x, Some (i, why i))
  | _ -> (make ty valid x, None)

(* [cell c row] writes [c]'s value at the frame's row [row]. *)
let cell c row = Form.pp_cell c (if Column.length c = 1 then 0 else row)
let error fmt = Format.kasprintf (fun m -> Error.v (m ^ ".")) fmt

(* [parsed c p why] is [p c], or where [p] fails at a row, [p] of [c]'s rows
   before it, null from it on, and the failure [why row reason]. *)
let parsed c p why =
  match p c with
  | Ok r -> (r, None)
  | Error (row, reason) ->
      let rows = Nx.arange Nx.int64 0 (Column.length c) 1 in
      let null = Nx.full_like rows (-1L) in
      let kept = Nx.less_s rows (Int64.of_int row) in
      let before = Nx.where kept rows null in
      let found = Column.mask (Nx.cast Nx.bit kept) in
      ( Result.get_ok (p (Column.restrict found (Column.gather before c))),
        Some (row, why row reason) )

let read_text c p =
  let why row reason =
    let text = Column.values Kind.string (Column.sub c ~offset:row ~length:1) in
    Error.v ~text:text.(0) (reason ^ ".")
  in
  parsed c p why

(* Numbers

   An integer type is its dtype and its range [\[lo; hi)] as floats, which are
   exact; [bool] is the integers [0] and [1]. *)

type number =
  | Integer : ('a, 'b) Nx.dtype * float * float -> number
  | Floating : ('a, 'b) Nx.dtype -> number

let number : type a. a Type.t -> number option = function
  | Bool -> Some (Integer (Nx.bit, 0., 2.))
  | Int8 -> Some (Integer (Nx.int8, -0x1p7, 0x1p7))
  | Int16 -> Some (Integer (Nx.int16, -0x1p15, 0x1p15))
  | Int32 -> Some (Integer (Nx.int32, -0x1p31, 0x1p31))
  | Int64 -> Some (Integer (Nx.int64, -0x1p63, 0x1p63))
  | Uint8 -> Some (Integer (Nx.uint8, 0., 0x1p8))
  | Uint16 -> Some (Integer (Nx.uint16, 0., 0x1p16))
  | Uint32 -> Some (Integer (Nx.uint32, 0., 0x1p32))
  | Uint64 -> Some (Integer (Nx.uint64, 0., 0x1p64))
  | Float16 -> Some (Floating Nx.float16)
  | Float32 -> Some (Floating Nx.float32)
  | Float64 -> Some (Floating Nx.float64)
  | _ -> None

let negative x = Nx.less x (Nx.zeros_like x)

(* [within bound x] is where [-bound < x < bound]. *)
let within bound x =
  Nx.logical_and (Nx.greater_s x (Int64.neg bound)) (Nx.less_s x bound)

(* [numeric from into c] is [c]'s values, of [from], converted to [into], and
   where they are exact. *)
let numeric from into c : Nx.host Nx.packed * Nx.host Nx.bool_t =
  match (from, into) with
  | Integer (df, _, _), Integer (dt, _, _) ->
      let x = tensor df c in
      let y = Nx.cast dt x in
      let back = Nx.equal (Nx.cast df y) x in
      (P y, Nx.logical_and back (Nx.equal (negative x) (negative y)))
  | Integer (df, _, _), Floating dt ->
      let y = Nx.cast dt (tensor df c) in
      (P y, Nx.isfinite y)
  | Floating _, Integer (dt, lo, hi) ->
      let f = tensor Nx.float64 c in
      let range = Nx.logical_and (Nx.greater_equal_s f lo) (Nx.less_s f hi) in
      let ok = Nx.logical_and range (Nx.equal (Nx.floor f) f) in
      (P (Nx.cast dt (Nx.where ok f (Nx.zeros_like f))), ok)
  | Floating df, Floating dt ->
      let x = tensor df c in
      let y = Nx.cast dt x in
      (P y, Nx.logical_or (Nx.logical_not (Nx.isfinite x)) (Nx.isfinite y))

(* Ticks *)

let floor_mod x k =
  let r = Nx.mod_s x k in
  Nx.where (Nx.less_s r 0L) (Nx.add_s r k) r

(* [floor_div x k] is [x / k] rounded toward negative infinity, for [k > 0]: the
   truncated quotient, less one where the remainder is negative. It never
   overflows, as [x - floor_mod x k] does near [Int64.min_int]. *)
let floor_div x k =
  let q = Nx.div_s x k in
  Nx.where (Nx.less_s (Nx.mod_s x k) 0L) (Nx.sub_s q 1L) q

let ns_per_day = 86_400_000_000_000L
let ticks_per_day u = Int64.div ns_per_day (Type.ns_per_unit u)

let in_int32 x =
  Nx.logical_and
    (Nx.greater_equal_s x (-0x8000_0000L))
    (Nx.less_s x 0x8000_0000L)

let in_day u x =
  Nx.logical_and (Nx.greater_equal_s x 0L) (Nx.less_s x (ticks_per_day u))

(* Casts *)

let unit_of : type a. a Type.t -> Type.unit_ option = function
  | Datetime { unit_; _ } -> Some unit_
  | Duration u | Clock u -> Some u
  | _ -> None

(* [ns_per_tick ty] is the nanoseconds of a tick of the temporal type [ty]: a
   date's tick is a day. *)
let ns_per_tick : type a. a Type.t -> int64 option = function
  | Date -> Some ns_per_day
  | Datetime { unit_; _ } | Duration unit_ | Clock unit_ ->
      Some (Type.ns_per_unit unit_)
  | _ -> None

(* [temporal from into n0 n1] casts the values of [from], ticks of [n0]
   nanoseconds, to [into], ticks of [n1]. A finer tick multiplies and fails on
   overflow; a coarser one divides and fails on a remainder, except that a
   datetime becomes the day it falls on, rounding toward the past. The result
   fails outside [into]'s range: int32 days for a date, one day for a clock. *)
let temporal (Type.Any from) (Type.Any into as t) n0 n1 :
    live:Nx.host Nx.bool_t -> Column.t -> checked =
  let cannot c row = error "cannot cast %t to %a" (cell c row) Type.pp into in
  let range r =
    match into with
    | Date -> Some (in_int32 r)
    | Clock u -> Some (in_day u r)
    | _ -> None
  in
  fun ~live c ->
    let x = int64s c in
    let r, exact =
      if Int64.equal n0 n1 then (x, None)
      else if n0 > n1 then
        let k = Int64.div n0 n1 in
        (Nx.mul_s x k, Some (within (Int64.succ (Int64.div Int64.max_int k)) x))
      else
        let k = Int64.div n1 n0 in
        match (from, into) with
        | Datetime _, Date -> (floor_div x k, None)
        | _ -> (Nx.div_s x k, Some (Nx.equal_s (Nx.mod_s x k) 0L))
    in
    match List.filter_map Fun.id [ exact; range r ] with
    | [] -> (Column.with_data t (Column.data c) c, None)
    | ok :: oks -> (
        let ok = List.fold_left Nx.logical_and ok oks in
        match into with
        | Date -> checked t [ c ] ~live ~ok (Nx.cast Nx.int32 r) (cannot c)
        | _ -> checked t [ c ] ~live ~ok r (cannot c))

let is_prefix d0 d1 =
  let n = Iarray.length d0 in
  n <= Iarray.length d1
  && Iarray.equal String.equal d0 (Iarray.sub d1 ~pos:0 ~len:n)

(* [to_text c] is the text of [c], a [string] or categorical column. *)
let to_text c =
  match Column.type_ c with
  | Any (Categorical dict) ->
      (* A code under a null is any: the gather reads an empty row outside the
         dictionary, and the column keeps [c]'s nulls. *)
      let words = Column.v Type.string (Iarray.to_array dict) in
      let text = Column.gather (int64s c) words in
      Column.with_data (Any Type.string) (Column.data text) c
  | _ -> c

(* [rows_of offsets m] is the row of each of the [m] elements of a list column
   of [offsets], clamped to a row, and where an element is in a row. *)
let rows_of offsets m =
  let n = Nx.dim 0 offsets - 1 in
  let elements = Nx.arange Nx.int64 0 m 1 in
  let ends = Nx.slice [ Nx.R (1, n + 1) ] offsets in
  let rows = Nx.searchsorted ~side:`Right ends elements in
  let first = Nx.item [ 0 ] offsets and stop = Nx.item [ n ] offsets in
  let inside =
    Nx.logical_and (Nx.greater_equal_s elements first) (Nx.less_s elements stop)
  in
  (Nx.minimum_s rows (Int64.of_int (Int.max 0 (n - 1))), inside)

let earliest f g =
  match (f, g) with
  | Some (r, _), Some (r', _) when r' < r -> g
  | None, g -> g
  | f, _ -> f

(* [convert from into ~live c] casts [c], whose rows outside [live] do not fail:
   list elements and record fields under a null. *)
let rec convert (Type.Any from) (Type.Any into as t) :
    live:Nx.host Nx.bool_t -> Column.t -> checked =
  let cannot c row = error "cannot cast %t to %a" (cell c row) Type.pp into in
  match (from, into, number from, number into) with
  | _ when Type.equal from into -> fun ~live:_ c -> (c, None)
  | _, _, Some f, Some i ->
      fun ~live c ->
        let P x, ok = numeric f i c in
        checked t [ c ] ~live ~ok x (cannot c)
  | (String | Categorical _), String, _, _ -> fun ~live:_ c -> (to_text c, None)
  | Categorical d0, Categorical d1, _, _ when is_prefix d0 d1 ->
      fun ~live:_ c -> (Column.with_data t (Column.data c) c, None)
  | (String | Categorical _), Categorical _, _, _ ->
      fun ~live:_ c ->
        let s = to_text c in
        parsed s (Form.parse t) (fun row _ -> cannot s row)
  | Tensor _, Tensor (dt, _), _, _ ->
      fun ~live:_ c -> (Column.with_data t (Fixed (P (tensor dt c))) c, None)
  | List e0, List e1, _, _ ->
      let element = convert (Any e0) (Any e1) in
      fun ~live c ->
        let offsets, child =
          match Column.data c with
          | List { offsets; child } -> (offsets, child)
          | _ -> assert false
        in
        let rows, inside = rows_of offsets (Column.length child) in
        let held = Nx.broadcast_to [| Column.length c |] live in
        let held = both held (holding c) in
        let live = Nx.logical_and inside (Nx.take rows held) in
        let child, f = element ~live child in
        let row (j, e) = (Int64.to_int (Nx.item [ j ] rows), e) in
        (Column.with_data t (List { offsets; child }) c, Option.map row f)
  | Record fs0, Record fs1, _, _ ->
      let fields = List.map2 (fun (_, t0) (_, t1) -> convert t0 t1) fs0 fs1 in
      fun ~live c ->
        let cs =
          match Column.data c with Fields cs -> cs | _ -> assert false
        in
        let live = Nx.broadcast_to [| Column.length c |] live in
        let live = both live (holding c) in
        let cs = List.map2 (fun f c -> f ~live c) fields cs in
        let f = List.fold_left (fun f (_, g) -> earliest f g) None cs in
        (Column.with_data t (Fields (List.map fst cs)) c, f)
  | _ -> (
      match (ns_per_tick from, ns_per_tick into) with
      | Some n0, Some n1 -> temporal (Any from) t n0 n1
      | _ ->
          invalid_arg
            (Format.asprintf "Kernels.cast: binding does not cast %a to %a"
               Type.pp from Type.pp into))

let cast from into =
  let k = convert from into in
  k ~live:(Nx.scalar Nx.bool true)

(* Text *)

let text : type a. a Expr.text_op -> Column.t -> checked =
 fun op ->
  let kept ty c data = (Column.with_data ty data c, None) in
  let bytes c = match Column.data c with Bytes r -> r | _ -> assert false in
  match op with
  | Length ->
      fun c ->
        let by = "Str.length" in
        let n = Strings.length ~by ?mask:(Column.validity c) (bytes c) in
        kept (Any Type.int64) c (Fixed (P n))
  | Slice { offset; length } ->
      fun c ->
        let by = "Str.slice" and mask = Column.validity c in
        let r = Strings.slice ~by ?mask ~offset ~length (bytes c) in
        kept (Any Type.string) c (Bytes r)
  | Matches p ->
      fun c ->
        let by = "Str.matches" in
        let m = Strings.matches ~by ?mask:(Column.validity c) p (bytes c) in
        kept (Any Type.bool) c (Fixed (P m))
  | Split sep ->
      fun c ->
        let by = "Str.split" and mask = Column.validity c in
        let offsets, pieces = Strings.split ~by ?mask sep (bytes c) in
        let child =
          Column.make (Any Type.string) ~length:(Nx_ragged.length pieces)
            (Bytes pieces)
        in
        kept (Any (Type.list Type.string)) c (List { offsets; child })
  | Trim ->
      fun c ->
        let by = "Str.trim" and mask = Column.validity c in
        kept (Any Type.string) c (Bytes (Strings.trim ~by ?mask (bytes c)))
  | Replace { sub; by = into } ->
      fun c ->
        let by = "Str.replace" and mask = Column.validity c in
        let r = Strings.replace ~by ?mask ~sub ~into (bytes c) in
        kept (Any Type.string) c (Bytes r)
  | Parse ty -> fun c -> read_text c (Form.parse (Any ty))

let concat sep cs =
  let length =
    List.fold_left
      (fun n c -> if Column.length c = 1 then n else Column.length c)
      1 cs
  in
  let bytes c = match Column.data c with Bytes r -> r | _ -> assert false in
  let validity = valid cs in
  let r =
    Strings.concat ~by:"Str.concat" ?mask:validity ~length sep
      (List.map bytes cs)
  in
  Column.make (Any Type.string) ?validity ~length (Bytes r)

(* Zones

   Offsets change at whole seconds, so a tick reads a second by rounding down
   and keeps its fraction across the conversion. These kernels loop over the
   rows: [Tz] answers a row in O(log n), and an ordered resolution reads its
   frames' rows in order anyway. *)

module A1 = Bigarray.Array1

let per_second u = Int64.div (ticks_per_day u) 86_400L

let floor_s t per =
  let q = Int64.div t per in
  if Int64.compare (Int64.rem t per) 0L < 0 then Int64.pred q else q

(* A row's outcome: its value, null, or why it fails. *)
let row_ok = 0
let row_null = 1
let out_of_range = 2
let repeats = 3
let skipped = 4
let backward = 5

(* [resolved ty z c x codes] is the column of the ticks [x] whose rows [codes]
   rule on. *)
let resolved ty z c x codes =
  let name = Tz.name z in
  let why row =
    let t = cell c row in
    match Nx.item [ row ] codes with
    | code when code = out_of_range -> error "%t in %s is out of range" t name
    | code when code = repeats -> error "%t repeats in %s" t name
    | code when code = skipped -> error "%t is skipped in %s" t name
    | _ -> error "%t goes backwards from the row before it" t
  in
  let keep = Nx.cast Nx.bit (Nx.not_equal_s codes row_null) in
  let validity = both keep (Column.validity c) in
  let length = Column.length c in
  let held = Column.make (Column.type_ c) ~validity ~length (Column.data c) in
  checked ty [ held ] ~ok:(Nx.less_s codes out_of_range) x why

(* [writes n f] is the [n] ticks and codes that [f] writes through [set i t off
   per], which stores [t], ticks of [per] a second, moved [off] seconds back,
   and [k], the codes. *)
let writes n f =
  let out = Strings.fresh Nx.int64 n and codes = Strings.fresh Nx.uint8 n in
  Strings.fill out (fun o ->
      Strings.fill codes (fun k ->
          A1.fill o 0L;
          A1.fill k row_ok;
          let set i t off per =
            let d = Int64.mul (Int64.of_int off) per in
            let r = Int64.sub t d in
            if Bool.equal (Int64.compare d 0L >= 0) (Int64.compare r t <= 0)
            then A1.unsafe_set o i r
            else A1.unsafe_set k i out_of_range
          in
          f set o k));
  (Strings.value out, Strings.value codes)

let unit_of_any (Type.Any ty) = Option.get (unit_of ty)

let wall z t =
  let u = unit_of_any t in
  let per = per_second u and ty = Type.Any (Type.wall u) in
  fun c ->
    let n = Column.length c in
    let x, codes =
      Strings.reading ~by:"Temporal.wall" (int64s c) Bigarray.int64
      @@ fun v ->
      writes n @@ fun set _ _ ->
      for i = 0 to n - 1 do
        let t = A1.unsafe_get v i in
        set i t (-Tz.offset_s z (floor_s t per)) per
      done
    in
    resolved ty z c x codes

(* [fold_run z per v ~live rows q last ~before ~after s] is the rows of a frame
   from position [q] (exclusive bound [last]) that repeat the readings of the
   fold holding the wall second [s], of offsets [before] and [after], that is
   consecutive live rows; and the position after them. A wall second falls in a
   given fold only within the fold's length of another. *)
let fold_run z per v ~live rows q last ~before ~after s =
  let span = Int64.of_int (before - after) in
  let rec go acc q =
    if q >= last then (List.rev acc, q)
    else
      let j = Int64.to_int rows.(q) in
      if not (live j) then go acc (q + 1)
      else
        let w = floor_s (A1.unsafe_get v j) per in
        let near = Int64.compare (Int64.abs (Int64.sub w s)) span < 0 in
        match Tz.local z w with
        | Ambiguous { before = b; after = a }
          when b = before && a = after && near ->
            go (j :: acc) (q + 1)
        | _ -> (List.rev acc, q)
  in
  go [] q

(* [second_pass ~tie_repeats ticks] is the position in the rows of a fold of the
   first one read in the second pass: the first strictly below its predecessor,
   or the second of two equal rows that are the fold's only rows, when
   [tie_repeats] of their reading holds. [None] when the rows show neither, so
   they say nothing about their pass. *)
let second_pass ~tie_repeats ticks =
  let rec strict i = function
    | a :: (b :: _ as rest) ->
        if Int64.compare b a < 0 then Some (i + 1) else strict (i + 1) rest
    | _ -> None
  in
  match (strict 0 ticks, ticks) with
  | (Some _ as pass), _ -> pass
  | None, [ a; b ] when Int64.equal a b && tie_repeats a -> Some 1
  | None, _ -> None

(* Two equal rows are the two passes of a logger whose cadence is the fold's
   length [span]: the rows on both sides lie [span] away. A row at another
   distance, or a frame cut beside them, leaves them a possible duplicate. *)
let at_fold_cadence ~span ~left ~right a =
  let away t = Int64.equal (Int64.abs (Int64.sub t a)) span in
  match (left, right) with
  | Some l, Some r -> away l && away r
  | _ -> false

let localize_with ~order z ~ambiguous ~gap t =
  let u = unit_of_any t in
  let per = per_second u and ty = Type.Any (Type.instant u) in
  fun c ->
    let n = Column.length c in
    let live =
      match Column.validity c with
      | None -> fun _ -> true
      | Some m ->
          let m = Nx.to_array m in
          fun i -> m.(i)
    in
    let x, codes =
      Strings.reading ~by:"Temporal.localize" (int64s c) Bigarray.int64
      @@ fun v ->
      writes n @@ fun set o k ->
      let decide (p : Expr.Temporal.policy) i t ~first ~second code =
        match p with
        | `Earlier -> set i t first per
        | `Later -> set i t second per
        | `Null -> A1.unsafe_set k i row_null
        | `Fail -> A1.unsafe_set k i code
      in
      let row i =
        let t = A1.unsafe_get v i in
        match Tz.local z (floor_s t per) with
        | Unique off -> set i t off per
        | Ambiguous { before; after } ->
            decide ambiguous i t ~first:before ~second:after repeats
        | Gap { before; after } ->
            decide gap i t ~first:after ~second:before skipped
      in
      match order with
      | None ->
          for i = 0 to n - 1 do
            row i
          done
      | Some order ->
          let rows = Nx.to_array (Nx_ragged.values order) in
          let offsets = Nx.to_array (Nx_ragged.offsets order) in
          (* [settle prev i] checks the row [i] against its frame's previous
             instant, and moves the state to it: a null or a failed row has no
             instant and leaves it. *)
          let settle prev i =
            if A1.unsafe_get k i = row_ok then
              let r = A1.unsafe_get o i in
              match !prev with
              | Some p when Int64.compare r p < 0 ->
                  A1.unsafe_set k i backward
              | _ -> prev := Some r
          in
          for f = 0 to Array.length offsets - 2 do
            let last = Int64.to_int offsets.(f + 1) in
            let prev = ref None and q = ref (Int64.to_int offsets.(f)) in
            (* The ticks of the live row before [!q]. *)
            let left = ref None in
            let next_live from =
              let rec go q =
                if q >= last then None
                else
                  let j = Int64.to_int rows.(q) in
                  if live j then Some (A1.unsafe_get v j) else go (q + 1)
              in
              go from
            in
            while !q < last do
              let i = Int64.to_int rows.(!q) in
              if not (live i) then incr q
              else
                let s = floor_s (A1.unsafe_get v i) per in
                match Tz.local z s with
                | Ambiguous { before; after } ->
                    let members, next =
                      fold_run z per v ~live rows !q last ~before ~after s
                    in
                    let ticks = List.map (A1.unsafe_get v) members in
                    let tie_repeats =
                      at_fold_cadence
                        ~span:(Int64.mul (Int64.of_int (before - after)) per)
                        ~left:!left ~right:(next_live next)
                    in
                    let pass = second_pass ~tie_repeats ticks in
                    let resolve pos j =
                      let t = A1.unsafe_get v j in
                      (match pass with
                      | Some p -> set j t (if pos < p then before else after) per
                      | None ->
                          decide ambiguous j t ~first:before ~second:after
                            repeats);
                      settle prev j
                    in
                    List.iteri resolve members;
                    left := Some (List.hd (List.rev ticks));
                    q := next
                | Unique _ ->
                    row i;
                    settle prev i;
                    left := Some (A1.unsafe_get v i);
                    incr q
                | Gap _ ->
                    (* A row moved by [gap] leaves the state. *)
                    row i;
                    left := Some (A1.unsafe_get v i);
                    incr q
            done
          done
    in
    resolved ty z c x codes

let localize z ~ambiguous ~gap t = localize_with ~order:None z ~ambiguous ~gap t

let localize_ordered z ~ambiguous ~gap t ~order c =
  localize_with ~order:(Some order) z ~ambiguous ~gap t c

let parse_with fmt ty c = read_text c (Form.parse_with fmt ty)

(* Calendar

   Dates are days and other temporal values ticks, as [int64]. The civil
   calendar is Howard Hinnant's algorithms, as nx arithmetic. *)

(* [civil days] is the year, month and day of [days] since 1970-01-01, and the
   day of the year counted from March 1. *)
let civil days =
  let z = Nx.add_s days 719_468L in
  let era = floor_div z 146_097L in
  let doe = Nx.sub z (Nx.mul_s era 146_097L) in
  let yoe =
    Nx.(
      div_s
        (sub
           (add (sub doe (div_s doe 1460L)) (div_s doe 36_524L))
           (div_s doe 146_096L))
        365L)
  in
  let doy =
    Nx.(sub doe (sub (add (mul_s yoe 365L) (div_s yoe 4L)) (div_s yoe 100L)))
  in
  let mp = Nx.div_s (Nx.add_s (Nx.mul_s doy 5L) 2L) 153L in
  let d = Nx.sub doy (Nx.div_s (Nx.add_s (Nx.mul_s mp 153L) 2L) 5L) in
  let march = Nx.less_s mp 10L in
  let m = Nx.where march (Nx.add_s mp 3L) (Nx.sub_s mp 9L) in
  let y = Nx.add yoe (Nx.mul_s era 400L) in
  (Nx.where march y (Nx.add_s y 1L), m, Nx.add_s d 1L, doy)

(* [days_of_civil y m d] is the days since 1970-01-01 of the day [d] of the
   month [m] of the year [y]. *)
let days_of_civil y m d =
  let y = Nx.where (Nx.less_equal_s m 2L) (Nx.sub_s y 1L) y in
  let era = floor_div y 400L in
  let yoe = Nx.sub y (Nx.mul_s era 400L) in
  let mp = Nx.mod_s (Nx.add_s m 9L) 12L in
  let doy = Nx.add (Nx.div_s (Nx.add_s (Nx.mul_s mp 153L) 2L) 5L) d in
  let doe =
    Nx.(add (sub (add (mul_s yoe 365L) (div_s yoe 4L)) (div_s yoe 100L)) doy)
  in
  Nx.add_s (Nx.add (Nx.mul_s era 146_097L) doe) (-719_469L)

let is_leap y =
  let divides k = Nx.equal_s (Nx.mod_s y k) 0L in
  Nx.logical_or
    (Nx.logical_and (divides 4L) (Nx.logical_not (divides 100L)))
    (divides 400L)

let month_days =
  Nx.create Nx.int64 [| 12 |]
    [| 31L; 28L; 31L; 30L; 31L; 30L; 31L; 31L; 30L; 31L; 30L; 31L |]

let days_in_month y m =
  let d = Nx.take (Nx.sub_s m 1L) month_days in
  let leap_day = Nx.logical_and (Nx.equal_s m 2L) (is_leap y) in
  Nx.add d (Nx.cast Nx.int64 leap_day)

(* [split ty x] is the days of the values [x] of [ty], their ticks into the day,
   and the ticks of a day. *)
let split (Type.Any ty) x =
  match ty with
  | Date -> (x, Nx.zeros_like x, 1L)
  | Clock u -> (Nx.zeros_like x, x, ticks_per_day u)
  | Datetime { unit_; _ } ->
      let day = ticks_per_day unit_ in
      (floor_div x day, floor_mod x day, day)
  | _ -> assert false (* Binding checked the type. *)

let field f t c =
  let days, tod, day = split t (int64s c) in
  let second = Int64.div day 86_400L in
  let x =
    match f with
    | (`Year | `Month | `Day | `Yearday) as f -> (
        let y, m, d, doy = civil days in
        match f with
        | `Year -> y
        | `Month -> m
        | `Day -> d
        | `Yearday ->
            (* January and February end the year that [doy] counts. *)
            let leap = Nx.cast Nx.int64 (is_leap y) in
            let after_feb = Nx.add (Nx.add_s doy 60L) leap in
            Nx.where (Nx.greater_equal_s doy 306L) (Nx.sub_s doy 305L) after_feb
        )
    | `Hour -> Nx.div_s tod (Int64.mul second 3600L)
    | `Minute -> Nx.mod_s (Nx.div_s tod (Int64.mul second 60L)) 60L
    | `Second -> Nx.mod_s (Nx.div_s tod second) 60L
    | `Nanosecond ->
        Nx.mul_s (Nx.mod_s tod second) (Int64.div 1_000_000_000L second)
    | `Weekday -> Nx.add_s (floor_mod (Nx.add_s days 3L) 7L) 1L
  in
  make (Any Type.int64) (valid [ c ]) x

(* [plus x k] is [x + k] and where it does not overflow. *)
let plus x k =
  let r = Nx.add x k in
  (r, Nx.equal (Nx.greater_equal_s k 0L) (Nx.greater_equal r x))

let add (Type.Any ta as t) (Type.Any td) =
  let d_unit = match td with Duration u -> u | _ -> assert false in
  let n_d = Type.ns_per_unit d_unit in
  fun a d ->
    let x = int64s a and k = int64s d in
    let range row =
      error "%t plus %t is out of range" (cell a row) (cell d row)
    in
    match ta with
    | Date ->
        let day = ticks_per_day d_unit in
        let whole = Nx.equal_s (Nx.mod_s k day) 0L in
        let r = Nx.add x (Nx.div_s k day) in
        let why row =
          let at = if Nx.dim 0 whole = 1 then 0 else row in
          if Nx.item [ at ] whole then range row
          else error "%t is not whole days" (cell d row)
        in
        let ok = Nx.logical_and whole (in_int32 r) in
        checked t [ a; d ] ~ok (Nx.cast Nx.int32 r) why
    | _ ->
        let u = Option.get (unit_of ta) in
        let factor = Int64.div n_d (Type.ns_per_unit u) in
        let r, ok = plus x (Nx.mul_s k factor) in
        (* A span of the operand's unit is its ticks already; a coarser one must
           not overflow when scaled to them. *)
        let ok =
          if factor = 1L then ok
          else
            let fits = within (Int64.succ (Int64.div Int64.max_int factor)) k in
            Nx.logical_and fits ok
        in
        let ok =
          match ta with
          | Clock u ->
              let day = ticks_per_day u in
              Nx.logical_and ok
                (Nx.logical_and (Nx.greater_equal_s r 0L) (Nx.less_s r day))
          | _ -> ok
        in
        checked t [ a; d ] ~ok r range

let diff (Type.Any ty) =
  let u = match ty with Date -> Type.S | _ -> Option.get (unit_of ty) in
  let t = Type.Any (Type.duration u) in
  fun a b ->
    let x = int64s a and y = int64s b in
    match ty with
    | Date -> (make t (valid [ a; b ]) (Nx.mul_s (Nx.sub x y) 86_400L), None)
    | _ ->
        let r = Nx.sub x y in
        let ok = Nx.equal (Nx.less_equal_s y 0L) (Nx.greater_equal r x) in
        let range row =
          error "%t minus %t is out of range" (cell a row) (cell b row)
        in
        checked t [ a; b ] ~ok r range

(* [exact ty s] is the ticks of [ty]'s unit in the span [s], which binding
   checked is whole ticks. *)
let exact (Type.Any ty) s =
  Int64.div (Time.Span.to_ns s) (Type.ns_per_unit (Option.get (unit_of ty)))

(* [moved t c days tod why] is the values of the column [c] of [t] moved to the
   days [days] at the times of day [tod], failing with [why] out of range. The
   caller gives the times of day: computed from a midnight outside the ticks,
   they would wrap and hide an overflow of [days]. *)
let moved t c days tod why =
  match t with
  | Type.Any Date ->
      checked t [ c ] ~ok:(in_int32 days) (Nx.cast Nx.int32 days) why
  | Any (Datetime { unit_; _ }) ->
      let day = ticks_per_day unit_ in
      let r = Nx.add (Nx.mul_s days day) tod in
      checked t [ c ] ~ok:(Nx.equal (floor_div r day) days) r why
  | _ -> assert false (* Binding checked the type. *)

let floor (step : Time.step) t =
  let what = Format.asprintf "%a" Time.pp_step step in
  let floor_days days =
    match step with
    | Days n -> Nx.sub days (floor_mod days (Int64.of_int n))
    | Weeks n ->
        (* Weeks start on Monday 1970-01-05. *)
        Nx.sub days (floor_mod (Nx.sub_s days 4L) (Int64.of_int (7 * n)))
    | Months n ->
        let y, m, _, _ = civil days in
        let months = Nx.add (Nx.mul_s (Nx.sub_s y 1970L) 12L) (Nx.sub_s m 1L) in
        let months = Nx.sub months (floor_mod months (Int64.of_int n)) in
        let y = Nx.add_s (floor_div months 12L) 1970L in
        let m = Nx.add_s (floor_mod months 12L) 1L in
        days_of_civil y m (Nx.ones_like y)
    | Exact _ -> assert false
  in
  let why c row =
    error "the %s period of %t is out of range" what (cell c row)
  in
  match step with
  | Exact s ->
      let k = exact t s in
      fun c ->
        let x = int64s c in
        let r = Nx.sub x (floor_mod x k) in
        checked t [ c ] ~ok:(Nx.less_equal r x) r (why c)
  | _ ->
      fun c ->
        let x = int64s c in
        let days, _, _ = split t x in
        moved t c (floor_days days) (Nx.zeros_like x) (why c)

let offset (step : Time.step) t =
  let move_days days =
    match step with
    | Days n -> Nx.add_s days (Int64.of_int n)
    | Weeks n -> Nx.add_s days (Int64.of_int (7 * n))
    | Months n ->
        let y, m, d, _ = civil days in
        let months = Nx.add (Nx.mul_s y 12L) (Nx.sub_s m 1L) in
        let months = Nx.add_s months (Int64.of_int n) in
        let y = floor_div months 12L
        and m = Nx.add_s (floor_mod months 12L) 1L in
        days_of_civil y m (Nx.minimum d (days_in_month y m))
    | Exact _ -> assert false
  in
  let why c row =
    error "%t moved by %a is out of range" (cell c row) Time.pp_step step
  in
  match step with
  | Exact s ->
      let k = Nx.scalar Nx.int64 (exact t s) in
      fun c ->
        let r, ok = plus (int64s c) k in
        checked t [ c ] ~ok r (why c)
  | _ ->
      fun c ->
        let x = int64s c in
        let days, tod, _ = split t x in
        moved t c (move_days days) tod (why c)
