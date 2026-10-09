(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

(* A descriptor is its family's struct of nx_spec.h: int32 fields in the host's
   byte order, its arrays as long as its counts say. *)
type 'f t = string
type contract

let max_rank = L.max_rank
let int32 s at = Int32.to_int (String.get_int32_ne s at)
let set b at x = Bytes.set_int32_ne b at (Int32.of_int x)

(* The dtype of each code, shared: an accessor allocates none. *)
let dtypes = Array.of_list D.all

(* nx_spec_contract's fields. *)

let family_contract = 1
let at_family = 0
let at_acc = 4
let at_out = 8
let at_init = 12
let at_nbatch = 16
let at_ncontracting = 20

(* The pairs, batch then contracting, two int32 each. *)
let at_pairs = 24

(* Pair [k] of the pairs from [at]: its axis of [a] (side 0) or of [b]. *)
let pair_axis s at k side = int32 s (at + (8 * k) + (4 * side))
let nbatch s = int32 s at_nbatch
let ncontracting s = int32 s at_ncontracting
let at_batch = at_pairs
let at_contracting s = at_pairs + (8 * nbatch s)
let batch_axis s k side = pair_axis s at_batch k side
let contracting_axis s k side = pair_axis s (at_contracting s) k side

let pp_shape ppf s =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_list s)

(* Loads and monoids *)

type pad = {
  lo : int array;
  hi : int array;
  interior : int array;
  windows : Nx_array.Move.window array;
}

type load = Plain | Padded of { fill : string; pad : pad }
type monoid = Sum | Prod | Max | Min | Logsumexp
type extreme = Max | Min
type combine = Set | Add | Max | Min

(* Maps. nx_spec_map: the family, the load count, the program's byte offset
   and length, then one int32 per load, the byte offset of its record or 0
   for a plain load. The program follows, then each padded load's record:
   its rank and window count, sixteen bytes of fill, then int64 [lo], [hi]
   and [interior] by axis and each window's axis, size, step and dilation.
   Every part starts on 8 bytes. *)

type map

let family_map = 2
let at_nloads = 4
let at_prog = 8
let at_prog_len = 12
let at_loads = 16
let at_fill = 8
let at_geometry = 24
let align8 n = (n + 7) land lnot 7
let get64 s at = Int64.to_int (String.get_int64_ne s at)
let set64 b at x = Bytes.set_int64_ne b at (Int64.of_int x)

(* Whether [b] is the bits of an element of [dt]. *)
let is_element (D.Any dt) b =
  String.length b = D.bytes dt 1
  &&
  match D.bits dt with
  | 1 -> Char.code b.[0] <= 1
  | 4 -> Char.code b.[0] < 16
  | 8 when D.equal dt D.Bool -> Char.code b.[0] <= 1
  | _ -> true

let check_pad fn k (p : pad) =
  let r = Array.length p.lo in
  if Array.length p.hi <> r || Array.length p.interior <> r then
    invalid_argf "%s: load %d's lo, hi and interior differ in length" fn k;
  if Array.exists (fun i -> i < 0) p.interior then
    invalid_argf "%s: load %d's interior padding is negative" fn k;
  Array.iteri
    (fun w (x : Nx_array.Move.window) ->
      let prev = if w = 0 then -1 else p.windows.(w - 1).axis in
      if x.axis <= prev || x.axis >= r then
        invalid_argf "%s: load %d's window %d is on axis %d" fn k w x.axis;
      if x.size < 1 || x.step < 1 || x.dilation < 1 then
        invalid_argf "%s: load %d's window %d is empty" fn k w)
    p.windows

let record_bytes (p : pad) =
  at_geometry + (8 * 3 * Array.length p.lo) + (8 * 4 * Array.length p.windows)

let map p ~loads =
  let fn = "Nx_kernel.Spec.map" in
  let ins = Prog.ins p in
  if Array.length loads <> Array.length ins then
    invalid_argf "%s: %d loads for %d operands" fn (Array.length loads)
      (Array.length ins);
  Array.iteri
    (fun k l ->
      match l with
      | Plain -> ()
      | Padded { fill; pad } ->
          if not (is_element ins.(k) fill) then
            invalid_argf "%s: load %d's fill is no element of its dtype" fn k;
          check_pad fn k pad)
    loads;
  let n = Array.length loads in
  let at_p = align8 (at_loads + (4 * n)) in
  let p = (p :> string) in
  let len = String.length p in
  let next = ref (align8 (at_p + len)) in
  let ats =
    Array.map
      (function
        | Plain -> 0
        | Padded { pad; _ } ->
            let at = !next in
            next := align8 (at + record_bytes pad);
            at)
      loads
  in
  let b = Bytes.make !next '\000' in
  set b at_family family_map;
  set b at_nloads n;
  set b at_prog at_p;
  set b at_prog_len len;
  Bytes.blit_string p 0 b at_p len;
  Array.iteri
    (fun k l ->
      set b (at_loads + (4 * k)) ats.(k);
      match l with
      | Plain -> ()
      | Padded { fill; pad } ->
          let at = ats.(k) and r = Array.length pad.lo in
          set b at r;
          set b (at + 4) (Array.length pad.windows);
          Bytes.blit_string fill 0 b (at + at_fill) (String.length fill);
          let g = at + at_geometry in
          for i = 0 to r - 1 do
            set64 b (g + (8 * i)) pad.lo.(i);
            set64 b (g + (8 * (r + i))) pad.hi.(i);
            set64 b (g + (8 * ((2 * r) + i))) pad.interior.(i)
          done;
          Array.iteri
            (fun w (x : Nx_array.Move.window) ->
              let at = g + (8 * 3 * r) + (32 * w) in
              set64 b at x.axis;
              set64 b (at + 8) x.size;
              set64 b (at + 16) x.step;
              set64 b (at + 24) x.dilation)
            pad.windows)
    loads;
  Bytes.unsafe_to_string b

(* The program's bytes, made by Prog.v when the map was. *)
let prog s =
  Option.get (Prog.of_string (String.sub s (int32 s at_prog) (int32 s at_prog_len)))

let load s k =
  let at = int32 s (at_loads + (4 * k)) in
  if at = 0 then Plain
  else
    let r = int32 s at and nw = int32 s (at + 4) in
    let (D.Any dt) = (Prog.ins (prog s)).(k) in
    let fill = String.sub s (at + at_fill) (D.bytes dt 1) in
    let g = at + at_geometry in
    let axis o = Array.init r (fun i -> get64 s (g + (8 * ((o * r) + i)))) in
    let windows =
      Array.init nw (fun w ->
          let at = g + (8 * 3 * r) + (32 * w) in
          {
            Nx_array.Move.axis = get64 s at;
            size = get64 s (at + 8);
            step = get64 s (at + 16);
            dilation = get64 s (at + 24);
          })
    in
    Padded { fill; pad = { lo = axis 0; hi = axis 1; interior = axis 2; windows } }

let loads s = Array.init (int32 s at_nloads) (load s)

(* The shape operand [k] of shape [x] has once loaded, or why it has none. *)
let loaded s k x =
  match load s k with
  | Plain -> Ok x
  | Padded { pad; _ } -> (
      let r = Array.length pad.lo in
      if Array.length x <> r then
        Error (Printf.sprintf "operand %d has rank %d, its pad %d" k
                 (Array.length x) r)
      else
        let padded =
          Array.mapi
            (fun i d ->
              pad.lo.(i) + pad.hi.(i) + d
              + if d > 0 then pad.interior.(i) * (d - 1) else 0)
            x
        in
        match Array.find_index (fun d -> d < 0) padded with
        | Some i ->
            Error (Printf.sprintf "operand %d's padded axis %d is negative" k i)
        | None -> (
            if pad.windows = [||] then Ok padded
            else
              match Nx_array.Move.shape (Window pad.windows) padded with
              | y -> Ok y
              | exception Invalid_argument msg ->
                  Error (Printf.sprintf "operand %d's windows: %s" k msg)))

let map_shapes s ins =
  let n = int32 s at_nloads in
  if Array.length ins <> n then
    Error (Printf.sprintf "%d operands, not %d" (Array.length ins) n)
  else if n = 0 then Error "a map with no operand has no shape of its own"
  else
    let outs = Array.length (Prog.outs (prog s)) in
    let rec go k first =
      if k = n then Ok (Array.make outs first)
      else
        match loaded s k ins.(k) with
        | Error _ as e -> e
        | Ok y when k > 0 && y <> first ->
            Error
              (Format.asprintf "operand %d loads as %a, operand 0 as %a" k
                 pp_shape y pp_shape first)
        | Ok y -> go (k + 1) (if k = 0 then y else first)
    in
    go 0 [||]

(* Contractions *)

let narrow_acc (D.Any dt) =
  match dt with
  | D.Bool | D.Bit | D.Float16 | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2
  | D.Float4_e2m1fn ->
      true
  | _ -> false

let contract ~batch ~contracting ~acc ~out ~init =
  let fn = "Nx_kernel.Spec.contract" in
  if narrow_acc acc then begin
    let (D.Any dt) = acc in
    invalid_argf "%s: an accumulator of %a" fn D.pp dt
  end;
  let used = Array.make_matrix 2 max_rank false in
  let check what pairs =
    Array.iteri
      (fun k pair ->
        let axes = [| fst pair; snd pair |] in
        Array.iteri
          (fun side ax ->
            let name = if side = 0 then "a" else "b" in
            if ax < 0 || ax >= max_rank then
              invalid_argf "%s: %s pair %d names axis %d of %s" fn what k ax
                name;
            if used.(side).(ax) then
              invalid_argf "%s: axis %d of %s is in two pairs" fn ax name;
            used.(side).(ax) <- true)
          axes)
      pairs
  in
  check "batch" batch;
  check "contracting" contracting;
  let nb = Array.length batch and nc = Array.length contracting in
  let b = Bytes.make (at_pairs + (8 * (nb + nc))) '\000' in
  let code (D.Any dt) = D.code dt in
  set b at_family family_contract;
  set b at_acc (code acc);
  set b at_out (code out);
  set b at_init (Bool.to_int init);
  set b at_nbatch nb;
  set b at_ncontracting nc;
  let pairs at =
    Array.iteri (fun k (i, j) ->
        set b (at + (8 * k)) i;
        set b (at + (8 * k) + 4) j)
  in
  pairs at_pairs batch;
  pairs (at_pairs + (8 * nb)) contracting;
  Bytes.unsafe_to_string b

let pairs s n at =
  Array.init n (fun k -> (pair_axis s at k 0, pair_axis s at k 1))

let batch s = pairs s (nbatch s) at_batch
let contracting s = pairs s (ncontracting s) (at_contracting s)
let acc s = dtypes.(int32 s at_acc)
let out s = dtypes.(int32 s at_out)
let init s = int32 s at_init <> 0

(* The axes of [a] (side 0) or [b] of rank [r] that no pair names, in axis
   order. *)
let free s side r =
  let named ax =
    let rec any at n k =
      k < n && (pair_axis s at k side = ax || any at n (k + 1))
    in
    any at_batch (nbatch s) 0 || any (at_contracting s) (ncontracting s) 0
  in
  List.filter (fun ax -> not (named ax)) (List.init r Fun.id)

let contract_shapes s ins =
  let nb = nbatch s and nc = ncontracting s in
  let n = 2 + Bool.to_int (init s) in
  if Array.length ins <> n then
    Error (Printf.sprintf "%d operands, not %d" (Array.length ins) n)
  else
    let a = ins.(0) and b = ins.(1) in
    let ra = Array.length a and rb = Array.length b in
    let bad = ref None in
    let fail msg = if !bad = None then bad := Some msg in
    let pair what at k =
      let i = pair_axis s at k 0 and j = pair_axis s at k 1 in
      if i >= ra then
        fail (Printf.sprintf "%s pair %d: a has rank %d" what k ra)
      else if j >= rb then
        fail (Printf.sprintf "%s pair %d: b has rank %d" what k rb)
      else if a.(i) <> b.(j) then
        fail (Printf.sprintf "%s pair %d: extents %d and %d" what k a.(i) b.(j))
    in
    for k = 0 to nb - 1 do
      pair "batch" at_batch k
    done;
    for k = 0 to nc - 1 do
      pair "contracting" (at_contracting s) k
    done;
    match !bad with
    | Some msg -> Error msg
    | None -> (
        let y =
          Array.concat
            [
              Array.init nb (fun k -> a.(batch_axis s k 0));
              Array.of_list (List.map (fun ax -> a.(ax)) (free s 0 ra));
              Array.of_list (List.map (fun ax -> b.(ax)) (free s 1 rb));
            ]
        in
        if Array.length y > max_rank then
          Error (Printf.sprintf "a result of rank %d" (Array.length y))
        else
          match init s with
          | true when ins.(2) <> y ->
              Error
                (Format.asprintf "init of shape %a, not %a" pp_shape ins.(2)
                   pp_shape y)
          | _ -> Ok [| y |])

let shapes s ins =
  let f = int32 s at_family in
  if f = family_contract then contract_shapes s ins
  else if f = family_map then map_shapes s ins
  else invalid_argf "Nx_kernel.Spec.shapes: family %d" f

(* Views *)

module Contract_view = struct
  type operand = A | B | Init | Dst
  type axis = Batch | Row | Column | Contracted

  (* A view is nx_spec.h's nx_contract_view, int64 fields in the host's byte
     order, followed by [fill]'s scratch. Operands and axes by index: A 0, B 1,
     Init 2, Dst 3; Batch 0, Row 1, Column 2, Contracted 3. The scratch holds
     the coalescer's extents and strides, each operand's place in them, the
     free axes of [a] then of [b], and three numbers. *)
  type t = Bytes.t

  let at_extent = 0
  let at_offset = 32
  let at_strides = 64
  let view_bytes = 192
  let at_ext = view_bytes
  let at_st = at_ext + (8 * max_rank)
  let at_slot = at_st + (8 * 4 * max_rank)
  let at_free = at_slot + (8 * 4)
  let at_has_init = at_free + (8 * 2 * max_rank)
  let at_nb = at_has_init + 8
  let at_fa = at_nb + 8
  let size = at_fa + 8
  let get v at = Int64.to_int (Bytes.get_int64_ne v at)
  let put v at x = Bytes.set_int64_ne v at (Int64.of_int x)
  let make () = Bytes.make size '\000'

  let operand_index = function A -> 0 | B -> 1 | Init -> 2 | Dst -> 3

  let axis_index = function
    | Batch -> 0
    | Row -> 1
    | Column -> 2
    | Contracted -> 3

  (* nx_array.h's coalescer over [n] operands of the [r] extents at [at_ext]
     of [v], operand k's strides from [at_st] + 8·k·max_rank, in place;
     answers the merged rank. *)
  external coalesce :
    (int[@untagged]) -> (int[@untagged]) -> t -> (int[@untagged]) ->
    (int[@untagged]) -> (int[@untagged])
    = "nx_kernel_coalesce_byte" "nx_kernel_coalesce"
  [@@noalloc]

  (* The operands of each group, by index. *)
  let members = [| [| 0; 1; 2; 3 |]; [| 0; 2; 3 |]; [| 1; 2; 3 |]; [| 0; 1 |] |]

  let layout ops dst o =
    let (Nx_array.Any a) = if o = 3 then dst else ops.(o) in
    Nx_array.layout a

  (* Whether a pair names axis [ax] of [a] (side 0) or [b]. *)
  let named s side ax =
    let found = ref false in
    for k = 0 to nbatch s - 1 do
      if batch_axis s k side = ax then found := true
    done;
    for k = 0 to ncontracting s - 1 do
      if contracting_axis s k side = ax then found := true
    done;
    !found

  (* Writes the free axes of a side of rank [r] from free axis [at]; answers
     their count. *)
  let fill_free v s side r at =
    let n = ref 0 in
    for ax = 0 to r - 1 do
      if not (named s side ax) then begin
        put v (at_free + (8 * (at + !n))) ax;
        incr n
      end
    done;
    !n

  let has_init v = get v at_has_init <> 0

  (* The axis of operand [o]'s layout that is the [k]th of group [g]. *)
  let source v s o g k =
    match o with
    | 0 -> (
        match g with
        | 0 -> batch_axis s k 0
        | 1 -> get v (at_free + (8 * k))
        | _ -> contracting_axis s k 0)
    | 1 -> (
        match g with
        | 0 -> batch_axis s k 1
        | 2 -> get v (at_free + (8 * (max_rank + k)))
        | _ -> contracting_axis s k 1)
    | _ -> (
        match g with
        | 0 -> k
        | 1 -> get v at_nb + k
        | _ -> get v at_nb + get v at_fa + k)

  let misfit what = invalid_arg ("Nx_kernel.Spec.Contract_view.fill: " ^ what)

  (* Raises unless the operands of group [g] agree on its [count] extents. *)
  let fits v s ops dst g count =
    let ms = members.(g) and l0 = layout ops dst members.(g).(0) in
    for p = 1 to Array.length ms - 1 do
      let o = ms.(p) in
      if o <> 2 || has_init v then begin
        let l = layout ops dst o in
        for k = 0 to count - 1 do
          if L.dim l (source v s o g k) <> L.dim l0 (source v s ms.(0) g k)
          then misfit "extents differ within a group"
        done
      end
    done

  let at_stride o x = at_strides + (8 * ((4 * o) + x))

  (* Groups the [count] axes of group [g] into one, or is [false]. Its
     operands agree on the extents ({!fits}). *)
  let group v s ops dst g count =
    let ms = members.(g) in
    let n = ref 0 in
    for p = 0 to Array.length ms - 1 do
      let o = ms.(p) in
      if o <> 2 || has_init v then begin
        let l = layout ops dst o in
        for k = 0 to count - 1 do
          let ax = source v s o g k in
          put v (at_ext + (8 * k)) (L.dim l ax);
          put v (at_st + (8 * ((!n * max_rank) + k))) (L.stride l ax)
        done;
        put v (at_slot + (8 * o)) !n;
        incr n
      end
    done;
    coalesce !n count v at_ext at_st = 1
    && begin
      put v (at_extent + (8 * g)) (get v at_ext);
      for p = 0 to Array.length ms - 1 do
        let o = ms.(p) in
        if o <> 2 || has_init v then
          put v (at_stride o g)
            (get v (at_st + (8 * get v (at_slot + (8 * o)) * max_rank)))
      done;
      true
    end

  let rank ops dst o = L.rank (layout ops dst o)

  let fill v s ~dst ops =
    let init = int32 s at_init <> 0 in
    if Array.length ops <> 2 + Bool.to_int init then
      misfit "another number of operands";
    let nb = nbatch s and nc = ncontracting s in
    let ra = rank ops dst 0 and rb = rank ops dst 1 in
    let fa = ra - nb - nc and fb = rb - nb - nc in
    let ry = nb + fa + fb in
    (* A pair past an operand's rank leaves more free axes than [fa]. *)
    if
      fa < 0 || fb < 0
      || rank ops dst 3 <> ry
      || (init && rank ops dst 2 <> ry)
      || fill_free v s 0 ra 0 <> fa
      || fill_free v s 1 rb max_rank <> fb
    then misfit "ranks the pairs do not fit";
    put v at_has_init (Bool.to_int init);
    put v at_nb nb;
    put v at_fa fa;
    (* What an operand lacks reads 0: its axes, and an absent init. *)
    Bytes.fill v at_offset (view_bytes - at_offset) '\000';
    for o = 0 to 3 do
      if o <> 2 || init then
        put v (at_offset + (8 * o)) (L.offset (layout ops dst o))
    done;
    (* Every group fits before any merges: a misfit raises even behind a
       group that does not merge. *)
    fits v s ops dst 0 nb;
    fits v s ops dst 1 fa;
    fits v s ops dst 2 fb;
    fits v s ops dst 3 nc;
    group v s ops dst 0 nb && group v s ops dst 1 fa && group v s ops dst 2 fb
    && group v s ops dst 3 nc

  let extent v x = get v (at_extent + (8 * axis_index x))

  let check_init fn v o =
    if o = Init && not (has_init v) then
      invalid_arg
        ("Nx_kernel.Spec.Contract_view." ^ fn ^ ": the view has no init")

  let offset v o =
    check_init "offset" v o;
    get v (at_offset + (8 * operand_index o))

  let stride v o x =
    check_init "stride" v o;
    (match (o, x) with
    | A, Column | B, Row | (Init | Dst), Contracted ->
        invalid_arg
          "Nx_kernel.Spec.Contract_view.stride: an axis the operand does not \
           have"
    | _ -> ());
    get v (at_stride (operand_index o) (axis_index x))
end
