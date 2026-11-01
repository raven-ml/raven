(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Quantity = Ymir.Quantity
module Unit = Ymir.Unit
module Time = Ymir.Time
module B = Rig.Buffer

let strf = Printf.sprintf

type 'd tensor = (float, Nx.float64_elt, 'd) Nx.t

(* NAIF's DAF and SPK Required Reading,
   https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/daf.html and
   .../req/spk.html, define the format: "The descriptor has six integer
   components" (target, centre, frame, type, first and last address), and
   "segments closer to the end of the file have higher priority". A type 2
   record holds MID, RADIUS and the X, Y, Z coefficients, of degree
   "( RSIZE - 2 ) / 3 - 1"; type 3 "contains six sets of coefficients
   instead of three". Its INIT and INTLEN locate the record of an epoch;
   the segment's last instant lies in its last record.

   DAF's layout: records of 1024 bytes, 128 double words, addressed from 1.
   The file record holds the ID word at byte 0, ND and NI at 8 and 12,
   FWARD at 76 and the binary format at 88. A summary record holds NEXT,
   PREV and NSUM, then NSUM summaries of ND + (NI + 1) / 2 double words: an
   SPK's two epochs, then its six integers, two to a double word. The
   record after each summary record holds their names, 8 characters to a
   double word of a summary. *)
let record = 1024
let word = 8
let spk_id = "DAF/SPK "
let nd = 2
let ni = 6
let summary_words = nd + ((ni + 1) / 2)

(* A summary record holds its three control words and whole summaries. *)
let max_summaries = ((record / word) - 3) / summary_words

(* SPK's frame 1 is J2000, which JPL's planetary ephemerides hold as
   ICRF. *)
let j2000 = 1
let day_s = 86400.

(* An epoch within 2^62 s of J2000 is a day within 2^53 days of MJD 0, which
   {!Ymir.Time} holds. *)
let epoch_bound = 0x1p62

type segment = {
  name : string;
  target : int;
  center : int;
  frame : int;
  kind : int;
  start : (Time.tdb, Nx.host) Time.t;
  stop : (Time.tdb, Nx.host) Time.t;
}

(* A Chebyshev segment of type 2 or 3: its directory and its [n] records of
   [rsize] words, [[n; rsize]], each MID, RADIUS, then [sets] sets of
   [coefficients]. *)
type 'd chebyshev = {
  init : float;
  intlen : float;
  records : int;
  sets : int;
  coefficients : int;
  table : 'd tensor;
}

(* A segment as read: its descriptor, its epochs in TDB seconds past J2000,
   and its data where ymir reads its type. *)
type 'd entry = {
  descriptor : segment;
  first : float;
  last : float;
  data : 'd chebyshev option;
}

type 'd t = { file : string; entries : 'd entry list }

let segments t = List.map (fun e -> e.descriptor) t.entries

(* [instant s] is the TDB instant [s] seconds past J2000, its day and
   seconds exact for whole seconds. *)
let instant s =
  let d = Float.floor ((s +. 43200.) /. day_s) in
  Time.v Time.tdb
    ~day:(Nx.scalar Nx.int64 (Int64.of_float (d +. 51544.)))
    ~second:
      (Quantity.v Unit.second
         (Nx.scalar Nx.float64 (s +. 43200. -. (d *. day_s))))

(* Reading *)

exception Fail of string

let fail fmt = Printf.ksprintf (fun s -> raise (Fail s)) fmt

(* [floats ~little src ~offset ~count] is the [count] float64 of [src] from
   byte [offset]: a host tensor over a host copy of those bytes alone, each
   element's bytes reversed where the file's order is not the host's. *)
let floats ~little src ~offset ~count =
  let n = count * word in
  let h = B.create Rig.host n in
  if n > 0 then B.copy ~src:(B.view src ~first:offset ~length:n) ~dst:h;
  let layout = Nx_array.Layout.contiguous (Iarray.of_array [| count; word |]) in
  let u = Nx.Repr.of_array Nx.Host.v (Nx_array.v Nx.uint8 layout h) in
  let u = if little = not Sys.big_endian then u else Nx.flip ~axes:[ 1 ] u in
  Nx.reshape [| count |] (Nx.bitcast Nx.float64 u)

let read_buffer ~file ~at src =
  let n = B.length src in
  let total = n / record in
  if n < record then fail "%s: %d bytes, shorter than a DAF file record" file n;
  (* [chunk r] is record [r]'s bytes, records counted from 1. *)
  let chunk r =
    if r < 1 || r > total then
      fail "%s: record %d lies outside the file's %d records" file r total;
    let s = Bytes.create record in
    B.blit_to_bytes src ((r - 1) * record) s 0 record;
    Bytes.unsafe_to_string s
  in
  let head = chunk 1 in
  if String.sub head 0 8 <> spk_id then
    fail "%s: the ID word is %S, not %S" file (String.sub head 0 8) spk_id;
  let little =
    match String.sub head 88 8 with
    | "LTL-IEEE" -> true
    | "BIG-IEEE" -> false
    | f -> fail "%s: the binary format is %S, not LTL-IEEE or BIG-IEEE" file f
  in
  let int32 s o =
    Int32.to_int
      (if little then String.get_int32_le s o else String.get_int32_be s o)
  in
  let double s o =
    Int64.float_of_bits
      (if little then String.get_int64_le s o else String.get_int64_be s o)
  in
  if int32 head 8 <> nd || int32 head 12 <> ni then
    fail "%s: ND = %d and NI = %d, not SPK's 2 and 6" file (int32 head 8)
      (int32 head 12);
  (* [words name a b] is the doubles at addresses [a] to [b], from 1. *)
  let words name a b =
    if a < 1 || b < a || b * word > n then
      fail "%s: segment %S's words %d-%d lie outside the file's %d bytes" file
        name a b n;
    floats ~little src ~offset:((a - 1) * word) ~count:(b - a + 1)
  in
  let chebyshev seg first last =
    let size = last - first + 1 in
    if size < 4 then fail "%s: segment %S has no directory" file seg.name;
    if last - 3 < 1 || last * word > n then
      fail "%s: segment %S's words %d-%d lie outside the file's %d bytes" file
        seg.name first last n;
    (* The directory's four words, read as bytes so that a read works where
       tensors are traced. *)
    let dir =
      let b = Bytes.create (4 * word) in
      B.blit_to_bytes src ((last - 4) * word) b 0 (4 * word);
      let b = Bytes.unsafe_to_string b in
      Array.init 4 (fun k -> double b (k * word))
    in
    let init = dir.(0) and intlen = dir.(1) in
    let rsize = dir.(2) and count = dir.(3) in
    let sets = if seg.kind = 2 then 3 else 6 in
    if not (Float.is_finite init && Float.is_finite intlen && intlen > 0.)
    then
      fail "%s: segment %S's INIT %g and INTLEN %g name no records" file
        seg.name init intlen;
    let whole x = Float.is_integer x && x >= 1. && x <= float_of_int size in
    if not (whole rsize && whole count) then
      fail "%s: segment %S's directory, RSIZE %g and N %g, does not fit its \
            %d words"
        file seg.name rsize count size;
    let rsize = int_of_float rsize and records = int_of_float count in
    if rsize < 2 + sets
       || (rsize - 2) mod sets <> 0
       || (records * rsize) + 4 <> size
    then
      fail "%s: segment %S's directory, RSIZE %d and N %d, does not fit its \
            %d words"
        file seg.name rsize records size;
    let data = words seg.name first (last - 4) in
    {
      init;
      intlen;
      records;
      sets;
      coefficients = (rsize - 2) / sets;
      table = Nx.place at (Nx.reshape [| records; rsize |] data);
    }
  in
  (* The summary records form a list from FWARD, each NEXT naming the next
     record or 0; a record met twice would make it a loop. *)
  let rec summaries r seen acc =
    if r = 0 then List.rev acc
    else begin
      if List.mem r seen then
        fail "%s: summary record %d comes twice in the list" file r;
      let s = chunk r and names = chunk (r + 1) in
      let next = double s 0 and count = double s 16 in
      if
        not
          (Float.is_integer next && next >= 0. && next <= float_of_int total)
      then fail "%s: summary record %d's NEXT, %g, names no record" file r next;
      if
        not
          (Float.is_integer count && count >= 0.
          && count <= float_of_int max_summaries)
      then
        fail "%s: summary record %d's NSUM, %g, is not 0 to %d" file r count
          max_summaries;
      let one acc k =
        let o = (3 + (k * summary_words)) * word in
        let i j = int32 s (o + (nd * word) + (4 * j)) in
        let name =
          String.trim
            (String.map
               (fun c -> if c = '\000' then ' ' else c)
               (String.sub names
                  (k * summary_words * word)
                  (summary_words * word)))
        in
        let first_s = double s o and last_s = double s (o + word) in
        let bounded x = Float.abs x < epoch_bound in
        if not (bounded first_s && bounded last_s) then
          fail "%s: segment %S's epochs, %g and %g s past J2000, are not \
                within 2^62 s of it"
            file name first_s last_s;
        let descriptor =
          {
            name;
            target = i 0;
            center = i 1;
            frame = i 2;
            kind = i 3;
            start = instant first_s;
            stop = instant last_s;
          }
        in
        let data =
          match descriptor.kind with
          | 2 | 3 -> Some (chebyshev descriptor (i 4) (i 5))
          | _ -> None
        in
        { descriptor; first = first_s; last = last_s; data } :: acc
      in
      summaries (int_of_float next) (r :: seen)
        (List.fold_left one acc (List.init (int_of_float count) Fun.id))
    end
  in
  let fward = int32 head 76 in
  if fward < 2 || fward >= total then
    fail "%s: FWARD, %d, names no summary record" file fward;
  { file; entries = summaries fward [] [] }

let catch f = match f () with t -> Ok t | exception Fail e -> Error e

let of_string ~at ~name s =
  let n = String.length s in
  let b = B.create Rig.host n in
  B.blit_from_string s 0 b 0 n;
  catch (fun () -> read_buffer ~file:name ~at b)

let read ~at path =
  match Rig_disk.of_file path with
  | Error e -> Error e
  | Ok b -> catch (fun () -> read_buffer ~file:path ~at b)

(* Evaluating *)

(* [seconds tdb] is [(d, s)]: TDB seconds past J2000 as [d], the whole
   seconds of the day's 0h, exact, and [s], the seconds since it. *)
let seconds tdb =
  let day = Time.day tdb in
  let s = Quantity.value Unit.second (Time.second tdb) in
  let d =
    Nx.sub_s (Nx.mul_s (Nx.cast Nx.float64 (Nx.sub_s day 51544L)) day_s) 43200.
  in
  let shape = Nx.broadcast_shapes [ Nx.shape d; Nx.shape s ] in
  (Nx.broadcast_to shape d, Nx.broadcast_to shape s)

let leading x n = List.init (Nx.ndim x - n) (fun _ -> Nx.A)
let column x c = Nx.slice (leading x 1 @ [ Nx.I c ]) x

(* [chebyshev c (d, s)] is [(p, v)], [c]'s position in km and velocity in
   km/s at [d + s]. Record [⌊(t − INIT) / INTLEN⌋], clamped to the last at
   the segment's end, maps [t] to [x = (t − MID) / RADIUS] in [[-1, 1]];
   the position is [Σ a_k T_k(x)]. A type 2 velocity is its derivative,
   [Σ a_k T'_k(x) / RADIUS], a type 3 velocity its own [Σ b_k T_k(x)].
   [d − MID] and [d − INIT] are exact for epochs on whole seconds. *)
let chebyshev c (d, s) =
  let i = Nx.floor (Nx.div_s (Nx.add (Nx.sub_s d c.init) s) c.intlen) in
  let i = Nx.minimum_s (Nx.maximum_s i 0.) (float_of_int (c.records - 1)) in
  let i = Nx.where (Nx.isnan i) (Nx.zeros_like i) i in
  let r = Nx.take ~axis:0 (Nx.cast Nx.int64 i) c.table in
  let mid = column r 0 and radius = column r 1 in
  let x = Nx.div (Nx.add (Nx.sub d mid) s) radius in
  let n = c.coefficients in
  (* T_0 .. T_{n-1} and their derivatives, by T_{k+1} = 2x T_k − T_{k−1}
     and T'_{k+1} = 2 T_k + 2x T'_k − T'_{k−1}. *)
  let rec go k ts dts =
    if k >= n then (List.rev ts, List.rev dts)
    else
      match (ts, dts) with
      | t1 :: t0 :: _, dt1 :: dt0 :: _ ->
          let t = Nx.sub (Nx.mul_s (Nx.mul x t1) 2.) t0 in
          let dt = Nx.sub (Nx.mul_s (Nx.add t1 (Nx.mul x dt1)) 2.) dt0 in
          go (k + 1) (t :: ts) (dt :: dts)
      | _ -> assert false
  in
  let ones = Nx.ones_like x and zeros = Nx.zeros_like x in
  let ts, dts =
    if n = 1 then ([ ones ], [ zeros ]) else go 2 [ x; ones ] [ ones; zeros ]
  in
  let basis l = Nx.unsqueeze ~axes:[ -2 ] (Nx.stack ~axis:(-1) l) in
  let set j =
    let shape = Array.append (Nx.shape x) [| 3; n |] in
    let first = 2 + (3 * j * n) in
    Nx.reshape shape
      (Nx.slice (leading r 1 @ [ Nx.R (first, first + (3 * n)) ]) r)
  in
  let sum a b = Nx.sum ~axes:[ -1 ] (Nx.mul a b) in
  let p = sum (set 0) (basis ts) in
  let v =
    if c.sets = 6 then sum (set 1) (basis ts)
    else Nx.div (sum (set 0) (basis dts)) (Nx.unsqueeze ~axes:[ -1 ] radius)
  in
  (p, v)

(* Bodies *)

let invalid fmt = Printf.ksprintf invalid_arg fmt

(* [usable fn t e] is [e]'s data, which the state of its target needs.
   Raises where ymir does not read it. *)
let usable fn t e =
  let s = e.descriptor in
  if s.frame <> j2000 then
    invalid "%s: %s's segment %S is on frame %d; ymir reads frame 1, J2000" fn
      t.file s.name s.frame;
  match e.data with
  | Some c -> c
  | None ->
      invalid
        "%s: %s's segment %S of body %d is of SPK type %d; ymir reads types 2 \
         and 3"
        fn t.file s.name s.target s.kind

(* A link of a chain: [body]'s state from [center] by its segments of that
   centre, in file order. *)
type 'd link = {
  body : int;
  center : int;
  links : ('d entry * 'd chebyshev) list;
}

let own t body = List.filter (fun e -> e.descriptor.target = body) t.entries

(* [chains fn t body seen] is every chain from [body] up its segments'
   centres to a body that is no segment's target, one per choice of centre
   at each body. *)
let rec chains fn t body seen =
  if List.mem body seen then
    invalid "%s: %s's segments of body %d chain back to it" fn t.file body;
  match own t body with
  | [] -> [ [] ]
  | segs ->
      let centres =
        List.sort_uniq compare (List.map (fun e -> e.descriptor.center) segs)
      in
      List.concat_map
        (fun center ->
          let links =
            List.filter_map
              (fun e ->
                if e.descriptor.center = center then Some (e, usable fn t e)
                else None)
              segs
          in
          List.map
            (fun up -> { body; center; links } :: up)
            (chains fn t center (body :: seen)))
        centres

let covers et e =
  Nx.logical_and (Nx.greater_equal_s et e.first) (Nx.less_equal_s et e.last)

let none et = Nx.less_s et Float.neg_infinity
let all et = Nx.logical_not (none et)

(* [serves t et l] is where [l] serves its body: where the latest of the
   body's segments that covers the instant has [l]'s centre. From the latest
   segment back, [covered] is where a later one covers. *)
let serves t et l =
  let _, wins =
    List.fold_left
      (fun (covered, wins) e ->
        let c = covers et e in
        let wins =
          if e.descriptor.center = l.center then
            Nx.logical_or wins (Nx.logical_and c (Nx.logical_not covered))
          else wins
        in
        (Nx.logical_or covered c, wins))
      (none et, none et)
      (List.rev (own t l.body))
  in
  wins

(* [link at l] is [l]'s state, each instant's from the latest of its
   segments that covers it. *)
let link at l =
  let d, s = at in
  let et = Nx.add d s in
  List.fold_left
    (fun acc (e, c) ->
      let p, v = chebyshev c at in
      match acc with
      | None -> Some (p, v)
      | Some (p0, v0) ->
          let m = Nx.unsqueeze ~axes:[ -1 ] (covers et e) in
          Some (Nx.where m p p0, Nx.where m v v0))
    None l.links
  |> Option.get

let at_index i =
  if Array.length i = 0 then ""
  else
    strf " at [%s]"
      (String.concat "; " (Array.to_list (Array.map string_of_int i)))

(* A state is a sum of links from each body up to a body both chains reach,
   the first the target's chain reaches. Each choice of a chain for the
   target and one for the centre is a candidate, and it gives the state
   where each of its links serves its body. *)
let state_named fn t ~target ~center tdb =
  let known b =
    List.exists
      (fun e -> e.descriptor.target = b || e.descriptor.center = b)
      t.entries
  in
  if not (known target) then
    invalid "%s: %s has no segment of body %d" fn t.file target;
  if not (known center) then
    invalid "%s: %s has no segment of body %d" fn t.file center;
  let ((d, s) as at) = seconds tdb in
  let et = Nx.add d s in
  let bodies chain b = b :: List.map (fun l -> l.center) chain in
  let until common chain =
    let rec go = function
      | l :: rest when l.body <> common -> l :: go rest
      | _ -> []
    in
    go chain
  in
  let candidates =
    List.concat_map
      (fun up ->
        List.filter_map
          (fun down ->
            let downs = bodies down center in
            List.find_opt (fun b -> List.mem b downs) (bodies up target)
            |> Option.map (fun common -> (until common up, until common down)))
          (chains fn t center []))
      (chains fn t target [])
  in
  if candidates = [] then
    invalid "%s: %s chains bodies %d and %d to no common body" fn t.file target
      center;
  (* A link serves several candidates: each is evaluated once. *)
  let memo f =
    let seen = Hashtbl.create 8 in
    fun l ->
      let key = (l.body, l.center) in
      match Hashtbl.find_opt seen key with
      | Some x -> x
      | None ->
          let x = f l in
          Hashtbl.add seen key x;
          x
  in
  let link = memo (link at) and serves = memo (serves t et) in
  let zeros () = Nx.zeros_like (Nx.stack ~axis:(-1) [ d; d; d ]) in
  let sum links =
    List.fold_left
      (fun (p0, v0) l ->
        let p, v = link l in
        (Nx.add p0 p, Nx.add v0 v))
      (zeros (), zeros ())
      links
  in
  let pieces =
    List.map
      (fun (up, down) ->
        let mask =
          List.fold_left
            (fun m l -> Nx.logical_and m (serves l))
            (all et) (up @ down)
        in
        let pu, vu = sum up and pd, vd = sum down in
        (mask, Nx.sub pu pd, Nx.sub vu vd))
      candidates
  in
  let served =
    List.fold_left (fun a (m, _, _) -> Nx.logical_or a m) (Nx.isnan et) pieces
  in
  Nx.check Nx.Ptree.tensor served et (fun i e ->
      Invalid_argument
        (strf
           "%s: TDB %.17g s past J2000%s is outside %s's segments that chain \
            body %d to body %d"
           fn (Nx.item [] e) (at_index i) t.file target center));
  let p, v =
    match pieces with
    | [] -> assert false
    | (_, p, v) :: rest ->
        List.fold_left
          (fun (p0, v0) (m, p, v) ->
            let m = Nx.unsqueeze ~axes:[ -1 ] m in
            (Nx.where m p p0, Nx.where m v v0))
          (p, v) rest
  in
  (Quantity.v Unit.(kilo metre) p, Quantity.v Unit.(kilo metre / second) v)

let state t = state_named "Spk.state" t

let position t ~target ~center tdb =
  fst (state_named "Spk.position" t ~target ~center tdb)
