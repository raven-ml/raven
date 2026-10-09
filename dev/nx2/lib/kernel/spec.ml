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

let pp_shape ppf s =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_list s)

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
  else invalid_argf "Nx_kernel.Spec.shapes: family %d" f

(* Views *)

module Contract_view = struct
  type operand = A | B | Init | Dst
  type axis = Batch | Row | Column | Contracted

  (* Operands and axes by index: A 0, B 1, Init 2, Dst 3; Batch 0, Row 1, Column
     2, Contracted 3. [stride] is by operand then axis. [ext] and [st] are the
     coalescer's arrays, [slot] each operand's place in them, and [free] the
     free axes of [a] then of [b]. *)
  type t = {
    extent : int array;
    offset : int array;
    stride : int array;
    mutable init : bool;
    mutable nb : int;
    mutable fa : int;
    ext : int array;
    st : int array;
    slot : int array;
    free : int array;
  }

  let make () =
    {
      extent = Array.make 4 0;
      offset = Array.make 4 0;
      stride = Array.make 16 0;
      init = false;
      nb = 0;
      fa = 0;
      ext = Array.make max_rank 0;
      st = Array.make (4 * max_rank) 0;
      slot = Array.make 4 0;
      free = Array.make (2 * max_rank) 0;
    }

  let operand_index = function A -> 0 | B -> 1 | Init -> 2 | Dst -> 3

  let axis_index = function
    | Batch -> 0
    | Row -> 1
    | Column -> 2
    | Contracted -> 3

  (* nx_array.h's coalescer over [n] operands of the [r] extents [ext], operand
     k's strides from [st] + k·max_rank, in place; answers the merged rank. *)
  external coalesce :
    (int[@untagged]) -> (int[@untagged]) -> int array -> int array ->
    (int[@untagged]) = "nx_kernel_coalesce_byte" "nx_kernel_coalesce"
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

  (* Writes the free axes of a side of rank [r] from [v.free.(at)]; answers
     their count. *)
  let fill_free v s side r at =
    let n = ref 0 in
    for ax = 0 to r - 1 do
      if not (named s side ax) then begin
        v.free.(at + !n) <- ax;
        incr n
      end
    done;
    !n

  (* The axis of operand [o]'s layout that is the [k]th of group [g]. *)
  let source v s o g k =
    match o with
    | 0 -> (
        match g with
        | 0 -> batch_axis s k 0
        | 1 -> v.free.(k)
        | _ -> contracting_axis s k 0)
    | 1 -> (
        match g with
        | 0 -> batch_axis s k 1
        | 2 -> v.free.(max_rank + k)
        | _ -> contracting_axis s k 1)
    | _ -> ( match g with 0 -> k | 1 -> v.nb + k | _ -> v.nb + v.fa + k)

  (* Groups the [count] axes of group [g] into one, or is [false]. *)
  let group v s ops dst g count =
    let ms = members.(g) in
    let n = ref 0 and fits = ref true in
    for p = 0 to Array.length ms - 1 do
      let o = ms.(p) in
      if o <> 2 || v.init then begin
        let l = layout ops dst o in
        for k = 0 to count - 1 do
          let ax = source v s o g k in
          let d = L.dim l ax in
          if !n = 0 then v.ext.(k) <- d
          else if v.ext.(k) <> d then fits := false;
          v.st.((!n * max_rank) + k) <- L.stride l ax
        done;
        v.slot.(o) <- !n;
        incr n
      end
    done;
    !fits
    && coalesce !n count v.ext v.st = 1
    && begin
      v.extent.(g) <- v.ext.(0);
      for p = 0 to Array.length ms - 1 do
        let o = ms.(p) in
        if o <> 2 || v.init then
          v.stride.((4 * o) + g) <- v.st.(v.slot.(o) * max_rank)
      done;
      true
    end

  let rank ops dst o = L.rank (layout ops dst o)

  let fill v s ~dst ops =
    let init = int32 s at_init <> 0 in
    Array.length ops = 2 + Bool.to_int init
    &&
    let nb = nbatch s and nc = ncontracting s in
    let ra = rank ops dst 0 and rb = rank ops dst 1 in
    let fa = ra - nb - nc and fb = rb - nb - nc in
    let ry = nb + fa + fb in
    fa >= 0 && fb >= 0
    && rank ops dst 3 = ry
    && ((not init) || rank ops dst 2 = ry)
    (* A pair past an operand's rank leaves more free axes than [fa]. *)
    && fill_free v s 0 ra 0 = fa
    && fill_free v s 1 rb max_rank = fb
    && begin
      v.init <- init;
      v.nb <- nb;
      v.fa <- fa;
      for o = 0 to 3 do
        if o <> 2 || init then v.offset.(o) <- L.offset (layout ops dst o)
      done;
      group v s ops dst 0 nb && group v s ops dst 1 fa && group v s ops dst 2 fb
      && group v s ops dst 3 nc
    end

  let extent v x = v.extent.(axis_index x)

  let check_init fn v o =
    if o = Init && not v.init then
      invalid_arg
        ("Nx_kernel.Spec.Contract_view." ^ fn ^ ": the view has no init")

  let offset v o =
    check_init "offset" v o;
    v.offset.(operand_index o)

  let stride v o x =
    check_init "stride" v o;
    (match (o, x) with
    | A, Column | B, Row | (Init | Dst), Contracted ->
        invalid_arg
          "Nx_kernel.Spec.Contract_view.stride: an axis the operand does not \
           have"
    | _ -> ());
    v.stride.((4 * operand_index o) + axis_index x)
end
