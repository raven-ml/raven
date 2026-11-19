(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  HDF5, read-only, after the HDF Group's File Format Specification 3.0
  (FMT3): superblocks 0 and 1, version-1 object headers, groups stored as
  symbol tables, and datasets of version-3 layout.

  A load walks the groups from the root in path order, reads each object
  header it reaches once, and keeps each dataset's header messages, from a
  file or a partial file; it decodes no dataset's messages. [plan] decodes
  them when a dataset is read: its dataspace, datatype, layout, filters
  and fill value. [window] reads one box and [rows] a box for each window.
  Both find the chunks the boxes cover in the dataset's version-1 B-tree in
  one walk, name the ranges a partial file lacks of the B-tree's nodes and
  then of the chunks, then run each chunk's filters backwards once and copy
  its rows in each box into place. [values] is the window of the whole
  dataset, or a value over the file's pages.

  A walk over a partial file raises only while it has found no range
  lacking: past one, the whole file's walk could meet structures the
  partial file's skipped first, so a failure there returns the ranges
  instead.
  ---------------------------------------------------------------------------*)

open Bigarray
module A = Nx_array
module B = Rig.Buffer
module D = Nx_array.Dtype
module L = Nx_array.Layout

(* Where a dataset's bytes are: a file, or a partial file. *)
type source = File of B.t | Part of Partial.t

let length = function File b -> B.length b | Part p -> Partial.length p

(* The bytes of an address and of a length, as the superblock states. *)
type form = { offsets : int; lengths : int }

(* A header message: its type, its flags and its data. *)
type message = { kind : int; flags : int; data : string }


(* Numbers (FMT3, I.A: little-endian) *)

let ( let* ) = Option.bind
let ( let+ ) x f = Option.map f x

let byte s i = Char.code s.[i]
let u16 s i = String.get_uint16_le s i
let u32 s i = Int32.to_int (String.get_int32_le s i) land 0xffff_ffff

(* The [k]-byte unsigned integer at [i], refused past [max_int]. *)
let uint what s i k =
  let v = ref 0 in
  for j = k - 1 downto 0 do
    if !v lsr 54 <> 0 then Fail.failf "%s past any file" what;
    v := (!v lsl 8) lor byte s (i + j)
  done;
  !v

(* An address of the form's size, or [None] for the undefined address, all
   of whose bits are set (FMT3, V). *)
let address f s i =
  let rec undefined j =
    j = f.offsets || (s.[i + j] = '\xff' && undefined (j + 1))
  in
  if undefined 0 then None else Some (uint "an address" s i f.offsets)

let defined f s i what =
  match address f s i with
  | Some a -> a
  | None -> Fail.failf "%s has the undefined address" what

let size f s i = uint "a length" s i f.lengths

(* [need s i n what] checks that [s] holds [n] bytes from [i]. *)
let need s i n what =
  if i < 0 || n < 0 || i > String.length s - n then
    Fail.failf "%s ends inside its fields" what

(* Reading *)

(* A walk's reads: from [source], the ranges a partial file lacks in
   [lacks], and the bytes of structures read in [taken], which a load
   bounds by the file's length so that it reads linear in it. *)
type reader = {
  source : source;
  form : form;
  lacks : (int * int) list ref;
  taken : int ref;
}

(* The [n] bytes from [pos], or [None] where a partial file lacks some of
   them, whose ranges then join [r.lacks]. *)
let get r ~pos n =
  match r.source with
  | File b -> Some (Storage.string b ~pos n)
  | Part p -> (
      Storage.within ~length:(Partial.length p) ~pos n;
      match Partial.lacking p ~pos n with
      | [] -> Some (Partial.string p ~pos n)
      | rs ->
          r.lacks := rs @ !(r.lacks);
          None)

let take r n =
  let length = length r.source in
  r.taken := !(r.taken) + n;
  if !(r.taken) > length then
    Fail.failf "the file's structures take more than its %d bytes" length

(* [walk r f] is [Ok (f ())], or [Error] of the ranges [r] lacks, also
   when [f] raises [Failure] past a lacking range. *)
let walk r f =
  match f () with
  | x -> if !(r.lacks) = [] then Ok x else Error (Partial.ranges !(r.lacks))
  | exception Failure _ when !(r.lacks) <> [] ->
      Error (Partial.ranges !(r.lacks))

(* The superblock (FMT3, II.A) *)

let signature = "\137HDF\r\n\026\n"

(* The most bytes of a superblock of version 0 or 1: its fields, four
   addresses and the root group's symbol table entry, of addresses of 8
   bytes. *)
let superblock_bytes = 28 + (4 * 8) + 40

(* The superblock at byte 0, version 0 or 1: the sizes of addresses and
   lengths and the root group's object header. It is read in one piece, to
   save a partial file's caller a round. *)
let superblock r =
  let* s = get r ~pos:0 (Int.min superblock_bytes (length r.source)) in
  let what = "the superblock" in
  need s 0 9 what;
  if String.sub s 0 8 <> signature then
    failwith
      "no HDF5 signature at byte 0, where a file without a user block has it";
  let version = byte s 8 in
  if version > 1 then
    Fail.failf "superblock version %d; read are 0 and 1" version;
  need s 0 16 what;
  let offsets = byte s 13 and lengths = byte s 14 in
  let valid k = k = 2 || k = 4 || k = 8 in
  if not (valid offsets && valid lengths) then
    Fail.failf "addresses of %d bytes and lengths of %d" offsets lengths;
  let f = { offsets; lengths } in
  (* Version 1 adds the indexed storage K and two reserved bytes. *)
  let fields = 24 + (if version = 1 then 4 else 0) in
  let entry = (2 * offsets) + 24 in
  need s 0 (fields + (4 * offsets) + entry) what;
  let at k = fields + (k * offsets) in
  (match address f s (at 0) with
  | Some 0 -> ()
  | _ -> failwith "a base address other than 0");
  (match address f s (at 3) with
  | None -> ()
  | Some _ ->
      failwith "a driver information block, as a file split in parts has");
  (match address f s (at 2) with
  | Some eof when eof > length r.source ->
      Fail.failf "the file ends at byte %d, before its end of data at %d"
        (length r.source) eof
  | _ -> ());
  Some (f, defined f s (at 4 + offsets) "the root group's object header")

(* Checksums (FMT3, I.A: Jenkins' lookup3) *)

(* libhdf5's H5_checksum_lookup3 (H5checksum.c), Jenkins' hashlittle, of
   the first [n] bytes of [s] from the initial value [init]: the bytes in
   words of 12, little endian, mixed, and the last up to 12 mixed in
   finally. H5_checksum_metadata calls it with an initial value of 0. *)
let hash ~init s n =
  let m x = x land 0xffff_ffff in
  let rot x k = m ((x lsl k) lor (x lsr (32 - k))) in
  let a = ref (m (0xdeadbeef + n + init)) in
  let b = ref !a and c = ref !a in
  let mix () =
    a := m (!a - !c) lxor rot !c 4;
    c := m (!c + !b);
    b := m (!b - !a) lxor rot !a 6;
    a := m (!a + !c);
    c := m (!c - !b) lxor rot !b 8;
    b := m (!b + !a);
    a := m (!a - !c) lxor rot !c 16;
    c := m (!c + !b);
    b := m (!b - !a) lxor rot !a 19;
    a := m (!a + !c);
    c := m (!c - !b) lxor rot !b 4;
    b := m (!b + !a)
  in
  let final () =
    c := m ((!c lxor !b) - rot !b 14);
    a := m ((!a lxor !c) - rot !c 11);
    b := m ((!b lxor !a) - rot !a 25);
    c := m ((!c lxor !b) - rot !b 16);
    a := m ((!a lxor !c) - rot !c 4);
    b := m ((!b lxor !a) - rot !a 14);
    c := m ((!c lxor !b) - rot !b 24)
  in
  (* The bytes from [i] up to [k] of them, little endian, zeros past. *)
  let word i k =
    let v = ref 0 in
    for j = Int.min k 4 - 1 downto 0 do
      v := (!v lsl 8) lor byte s (i + j)
    done;
    !v
  in
  let i = ref 0 in
  while n - !i > 12 do
    a := m (!a + word !i 4);
    b := m (!b + word (!i + 4) 4);
    c := m (!c + word (!i + 8) 4);
    mix ();
    i := !i + 12
  done;
  let k = n - !i in
  if k > 0 then begin
    a := m (!a + word !i k);
    if k > 4 then b := m (!b + word (!i + 4) (k - 4));
    if k > 8 then c := m (!c + word (!i + 8) (k - 8));
    final ()
  end;
  !c

(* [checked s what] checks the checksum in [s]'s last 4 bytes. *)
let checked s what =
  let n = String.length s - 4 in
  if n < 0 || u32 s n <> hash ~init:0 s n then
    Fail.failf "%s fails its checksum" what

(* Object headers (FMT3, IV.A.1 and IV.A.2.q) *)

let continuation = 0x10

(* The messages of [s] from [i] to [last], and the continuation blocks
   they name, each an address and a length. A version-1 message's prefix
   is a type and a size of 2 bytes, flags and 3 reserved bytes; a
   version-2 one's a type of a byte, a size of 2, flags and, with [crt],
   a creation order of 2. A version-2 block ends in a gap shorter than a
   prefix. *)
let messages r ~v2 ~crt s i last what =
  let prefix = if not v2 then 8 else if crt then 6 else 4 in
  let rec go i acc conts =
    if i + prefix > last then (List.rev acc, List.rev conts)
    else begin
      let kind, len, flags =
        if v2 then (byte s i, u16 s (i + 1), byte s (i + 3))
        else (u16 s i, u16 s (i + 2), byte s (i + 4))
      in
      if i + prefix + len > last then
        Fail.failf "%s ends inside a message" what;
      let data = String.sub s (i + prefix) len in
      let conts =
        if kind <> continuation then conts
        else begin
          let f = r.form in
          need data 0 (f.offsets + f.lengths) "a continuation message";
          let pos = defined r.form data 0 "a continuation block" in
          let n = size r.form data r.form.offsets in
          take r n;
          (pos, n) :: conts
        end
      in
      go (i + prefix + len) ({ kind; flags; data } :: acc) conts
    end
  in
  go i [] []

(* The most bytes before a version-2 header's first message: its
   signature, version, flags, four times, two attribute counts and a size
   of 8 bytes. *)
let prefix_bytes = 4 + 2 + 16 + 4 + 8

(* The messages of the object header at [at], version 1 or 2, in order,
   its continuation blocks' included, or [None] where a partial file lacks
   some of them. *)
let header r at =
  let what = Printf.sprintf "the object header at byte %d" at in
  let length = length r.source in
  if at >= length then Fail.failf "%s lies past the file's end" what;
  let* s = get r ~pos:at (Int.min prefix_bytes (length - at)) in
  need s 0 16 what;
  let v2 = String.sub s 0 4 = "OHDR" in
  (* The first block, its messages from [first] to [last], and how each
     continuation block reads. *)
  let* chunk, first, last, crt =
    if not v2 then begin
      if byte s 0 <> 1 then Fail.failf "%s is of version %d" what (byte s 0);
      let n = u32 s 8 in
      take r (16 + n);
      let+ c = get r ~pos:(at + 16) n in
      (c, 0, n, false)
    end
    else begin
      if byte s 4 <> 2 then Fail.failf "%s is of version %d" what (byte s 4);
      let flags = byte s 5 in
      let p = 6 + (if flags land 0x20 <> 0 then 16 else 0)
              + if flags land 0x10 <> 0 then 4 else 0 in
      let w = 1 lsl (flags land 3) in
      need s p w what;
      let n = uint "an object header's size" s p w in
      let first = p + w in
      take r (first + n + 4);
      let+ c = get r ~pos:at (first + n + 4) in
      checked c what;
      (c, first, first + n, flags land 4 <> 0)
    end
  in
  let ms, conts = messages r ~v2 ~crt chunk first last what in
  let rec blocks todo acc =
    match todo with
    | [] -> Some (List.concat (List.rev acc))
    | (pos, n) :: rest ->
        let what = Printf.sprintf "the continuation block at byte %d" pos in
        let* c = get r ~pos n in
        let first, last =
          if not v2 then (0, n)
          else begin
            if n < 8 || String.sub c 0 4 <> "OCHK" then
              Fail.failf "no continuation block at byte %d" pos;
            checked c what;
            (4, n - 4)
          end
        in
        let ms, conts = messages r ~v2 ~crt c first last what in
        blocks (rest @ conts) (ms :: acc)
  in
  blocks conts [ ms ]

(* Groups (FMT3, III.A.1, III.B, III.C, III.D, IV.A.2.c, g and r) *)

let symbol_table = 0x11
let layout = 0x08
let link_info = 0x02
let link_kind = 0x06

(* Where a link leads: an object header, or a soft link's path. *)
type target = Hard of int | Soft of string

(* The local heap at [at]'s data segment. *)
let heap r at =
  let f = r.form in
  let* s = get r ~pos:at (8 + (2 * f.lengths) + f.offsets) in
  if String.sub s 0 4 <> "HEAP" then
    Fail.failf "no local heap at byte %d" at;
  let n = size f s 8 in
  let data = defined f s (8 + (2 * f.lengths)) "a local heap's data" in
  take r n;
  get r ~pos:data n

(* The name at [off] in the heap [h]. *)
let name h off =
  let ends =
    if off < String.length h then String.index_from_opt h off '\000' else None
  in
  match ends with
  | Some e -> String.sub h off (e - off)
  | None -> Fail.failf "a link name at %d past its heap's end" off

(* [node r ~kind ~key at] is the version-1 B-tree node of type [kind] at
   [at], whose keys take [key] bytes: its level, its keys and its
   children. *)
let node r ~kind ~key at =
  let f = r.form in
  let head = 8 + (2 * f.offsets) in
  let* s = get r ~pos:at head in
  if String.sub s 0 4 <> "TREE" then
    Fail.failf "no B-tree node at byte %d" at;
  if byte s 4 <> kind then
    Fail.failf "the B-tree node at byte %d is of type %d, not %d" at (byte s 4)
      kind;
  let level = byte s 5 and n = u16 s 6 in
  let len = ((n + 1) * key) + (n * f.offsets) in
  take r (head + len);
  let+ s = get r ~pos:(at + head) len in
  let entry i = i * (key + f.offsets) in
  let keys = Array.init (n + 1) (fun i -> String.sub s (entry i) key) in
  let children =
    Array.init n (fun i ->
        defined f s (entry i + key) "a B-tree node's child")
  in
  (level, keys, children)

let valid n =
  if n = "" || n = "." || String.contains n '/' then
    Fail.failf "a link named %S" n

(* The links of the group whose symbol table message is [m], or [None]
   where a partial file lacks some. A tree that comes back to a node, or
   reaches one twice, is refused. *)
let symbols r m =
  let f = r.form in
  need m.data 0 (2 * f.offsets) "a symbol table message";
  let root = defined f m.data 0 "a group's B-tree" in
  let* h = heap r (defined f m.data f.offsets "a group's local heap") in
  let seen = Hashtbl.create 8 in
  let visit at =
    if Hashtbl.mem seen at then
      Fail.failf "the B-tree node or symbol table node at byte %d is reached \
                  twice" at;
    Hashtbl.add seen at ()
  in
  let entry = (2 * f.offsets) + 24 in
  let snod at =
    visit at;
    let* s = get r ~pos:at 8 in
    if String.sub s 0 4 <> "SNOD" then
      Fail.failf "no symbol table node at byte %d" at;
    let n = u16 s 6 in
    take r (8 + (n * entry));
    let+ s = get r ~pos:(at + 8) (n * entry) in
    List.init n (fun i ->
        let e = i * entry in
        let n = name h (uint "a name's offset" s e f.offsets) in
        valid n;
        (* Cache type 2 is a soft link, whose path is in the heap at the
           offset the scratch pad's first 4 bytes hold. *)
        if u32 s (e + (2 * f.offsets)) = 2 then
          (n, Soft (name h (u32 s (e + (2 * f.offsets) + 8))))
        else (n, Hard (defined f s (e + f.offsets) "a link's object header")))
  in
  let rec tree at ~level =
    visit at;
    let* l, _, children = node r ~kind:0 ~key:f.lengths at in
    (match level with
    | Some v when v <> l ->
        Fail.failf "the B-tree node at byte %d is at level %d, not %d" at l v
    | _ -> ());
    let below c = if l = 0 then snod c else tree c ~level:(Some (l - 1)) in
    let parts = Array.to_list (Array.map below children) in
    if List.mem None parts then None
    else Some (List.concat_map Option.get parts)
  in
  tree root ~level:None

let child parent n = if parent = "/" then "/" ^ n else parent ^ "/" ^ n

(* The link message [s] of the group at [path] (FMT3, IV.A.2.g): its name
   and target. *)
let link f path s =
  need s 0 2 "a link message";
  if byte s 0 <> 1 then Fail.failf "link message version %d" (byte s 0);
  let flags = byte s 1 in
  let kind, i =
    if flags land 8 = 0 then (0, 2)
    else begin
      need s 2 1 "a link message";
      (byte s 2, 3)
    end
  in
  (* A creation order of 8 bytes, then a character set of 1. *)
  let i = i + (if flags land 4 <> 0 then 8 else 0) in
  let i = i + if flags land 0x10 <> 0 then 1 else 0 in
  let w = 1 lsl (flags land 3) in
  need s i w "a link message";
  let len = uint "a link name's length" s i w in
  need s (i + w) len "a link message";
  let n = String.sub s (i + w) len in
  let i = i + w + len in
  valid n;
  let at = child path n in
  match kind with
  | 0 ->
      need s i f.offsets "a link message";
      (n, Hard (defined f s i "a link's object header"))
  | 1 ->
      need s i 2 "a link message";
      let len = u16 s i in
      need s (i + 2) len "a link message";
      (n, Soft (String.sub s (i + 2) len))
  | 64 -> Fail.failf "%s: an external link, to an object of another file" at
  | k -> Fail.failf "%s: a link of type %d" at k

(* What an object is, from its header's messages: a group and its links, a
   dataset, or another object; [None] where a partial file lacks some. *)
type kind = Group of (string * target) list | Dataset | Other

let classify r path messages =
  let has k = List.exists (fun m -> m.kind = k) messages in
  let find k = List.find_opt (fun m -> m.kind = k) messages in
  match (find symbol_table, find link_info) with
  | Some _, Some _ -> failwith "a symbol table and a link info message at once"
  | Some m, None ->
      if has layout then failwith "a group and a dataset at once";
      let+ links = symbols r m in
      Group links
  | None, Some m ->
      if has layout then failwith "a group and a dataset at once";
      (* The link info message (FMT3, IV.A.2.c): a version, flags, a
         maximum creation index when flag 0 is set, then the fractal heap
         that holds a group's links densely. *)
      need m.data 0 2 "a link info message";
      let at = if byte m.data 1 land 1 <> 0 then 10 else 2 in
      need m.data at r.form.offsets "a link info message";
      if address r.form m.data at <> None then
        failwith "links stored densely, in a fractal heap";
      let links =
        List.filter_map
          (fun m ->
            if m.kind = link_kind then Some (link r.form path m.data) else None)
          messages
      in
      Some (Group links)
  | None, None ->
      if has link_kind then
        failwith "link messages without a link info message";
      Some (if has layout then Dataset else Other)

(* The walk *)

(* A path's names, which order paths name by name: a path comes before
   every path below it, and [/a/b] before [/a-]. *)
module Pending = Set.Make (struct
  type t = string list * target * int

  let compare = compare
end)

let path_of = function [] -> "/" | ns -> "/" ^ String.concat "/" ns
let names p =
  List.filter (fun n -> n <> "" && n <> ".") (String.split_on_char '/' p)

(* An object read by a walk: its path, as messages name it, and its
   header's messages. *)
type obj = {
  name : string;
  source : source;
  form : form;
  path : string;
  messages : message list;
}

type t = obj
type group = obj

let path (t : obj) = t.path

(* [objects name r] is the groups and the datasets [r] reaches from the
   root, each in path order and read once, at the first of its paths
   pending, which is the least: the path of a soft link is that of its
   target. A soft link whose target is missing, or another soft link of a
   cycle, leads nowhere and is not listed. *)
let objects name r =
  match superblock r with
  | None -> ([], [])
  | Some (form, root) ->
      let r = { r with form } in
      let memo = Hashtbl.create 64 in
      let read path at =
        match Hashtbl.find_opt memo at with
        | Some o -> Some o
        | None ->
            let* messages = header r at in
            let+ kind = classify r path messages in
            Hashtbl.replace memo at (messages, kind);
            (messages, kind)
      in
      (* The object [value] names from the group at [base], or [Some None]
         when it names none. [follows] holds the soft links followed, so a
         cycle of them names none. *)
      let rec resolve path ~base ~follows value =
        let rec down at ns =
          match ns with
          | [] -> Some (Some at)
          | n :: rest -> (
              match read path at with
              | None -> None
              | Some (_, (Dataset | Other)) -> Some None
              | Some (_, Group links) -> (
                  match List.assoc_opt n links with
                  | None -> Some None
                  | Some (Hard a) -> down a rest
                  | Some (Soft v) ->
                      if List.mem (at, n) follows then Some None
                      else
                        let* o =
                          resolve path ~base:at ~follows:((at, n) :: follows) v
                        in
                        match o with
                        | None -> Some None
                        | Some a -> down a rest))
        in
        let absolute = String.starts_with ~prefix:"/" value in
        down (if absolute then root else base) (names value)
      in
      let seen = Hashtbl.create 64 in
      let rec next pending groups datasets =
        match Pending.min_elt_opt pending with
        | None -> (List.rev groups, List.rev datasets)
        | Some ((ns, target, base) as p) -> (
            let pending = Pending.remove p pending in
            let path = path_of ns in
            let at =
              Fail.file path @@ fun () ->
              match target with
              | Hard a -> Some (Some a)
              | Soft v -> resolve path ~base ~follows:[] v
            in
            match at with
            | None | Some None -> next pending groups datasets
            | Some (Some at) when Hashtbl.mem seen at ->
                next pending groups datasets
            | Some (Some at) -> (
                Hashtbl.add seen at ();
                let source = r.source in
                match Fail.file path (fun () -> read path at) with
                | None | Some (_, Other) -> next pending groups datasets
                | Some (messages, Group links) ->
                    let add ps (n, t) = Pending.add (ns @ [ n ], t, at) ps in
                    let g = { name; source; form; path; messages } in
                    let pending = List.fold_left add pending links in
                    next pending (g :: groups) datasets
                | Some (messages, Dataset) ->
                    let d = { name; source; form; path; messages } in
                    next pending groups (d :: datasets)))
      in
      next (Pending.singleton ([], Hard root, root)) [] []

let reader source =
  let form = { offsets = 8; lengths = 8 } in
  { source; form; lacks = ref []; taken = ref 0 }

let read path file =
  let r = reader (File file) in
  objects path r

let decode p =
  let name = Partial.name p in
  Fail.file name @@ fun () ->
  let r = reader (Part p) in
  walk r (fun () -> objects name r)

(* Datatypes (FMT3, IV.A.2.d) *)

(* How an element lies in the file: as [trailing] elements of [dt], in
   C order, each of the file's byte order, swapped when [swap]. An element
   of a datatype nx lacks is its bytes: [dt] is uint8 and [trailing] its
   size. A fixed-length string, or an array of them, has the padding type
   its datatype states in [pad]. *)
type element = {
  dt : D.any;
  trailing : int list;
  swap : bool;
  bytes : int;
  pad : int option;
}

let refuse fmt = Printf.ksprintf (fun m -> failwith ("datatype: " ^ m)) fmt

let as_bytes n =
  { dt = D.Any D.Uint8; trailing = [ n ]; swap = false; bytes = n; pad = None }

(* The integers of [n] bytes, signed or not. *)
let integer ~signed n =
  match (n, signed) with
  | 1, true -> D.Any D.Int8
  | 1, false -> D.Any D.Uint8
  | 2, true -> D.Any D.Int16
  | 2, false -> D.Any D.Uint16
  | 4, true -> D.Any D.Int32
  | 4, false -> D.Any D.Uint32
  | 8, true -> D.Any D.Int64
  | _ -> D.Any D.Uint64

(* The IEEE layouts of the floats nx has: size, sign bit, exponent location
   and size, mantissa location and size, exponent bias. *)
let floats =
  [
    ((2, 15, 10, 5, 0, 10, 15), D.Any D.Float16);
    ((2, 15, 7, 8, 0, 7, 127), D.Any D.Bfloat16);
    ((4, 31, 23, 8, 0, 23, 127), D.Any D.Float32);
    ((8, 63, 52, 11, 0, 52, 1023), D.Any D.Float64);
  ]

(* The element the datatype message [s] from [i] describes. *)
let rec element s i =
  need s i 8 "a datatype message";
  let cls = byte s i land 0xf and version = byte s i lsr 4 in
  let bits = byte s (i + 1) lor (byte s (i + 2) lsl 8) in
  let n = u32 s (i + 4) in
  if n = 0 then refuse "a type of no byte";
  let p = i + 8 in
  (* Bit 0 is the byte order, set for big-endian. *)
  let swap = bits land 1 = 1 <> Sys.big_endian && n > 1 in
  let whole dt = { dt; trailing = []; swap; bytes = n; pad = None } in
  match cls with
  | 0 | 4 ->
      need s p 4 "a datatype message";
      let offset = u16 s p and precision = u16 s (p + 2) in
      if not (List.mem n [ 1; 2; 4; 8 ] && offset = 0 && precision = 8 * n)
      then
        refuse "%s of %d bits at bit %d in %d bytes"
          (if cls = 0 then "fixed-point" else "bitfield")
          precision offset n;
      let signed = cls = 0 && bits land 8 <> 0 in
      whole (integer ~signed n)
  | 1 -> (
      need s p 12 "a datatype message";
      if bits land 0x40 <> 0 then refuse "a VAX floating-point order";
      let sign = byte s (i + 2) in
      let key =
        (n, sign, byte s (p + 4), byte s (p + 5), byte s (p + 6),
         byte s (p + 7), u32 s (p + 8))
      in
      let normalised = (bits lsr 4) land 3 = 2 in
      let exact = u16 s p = 0 && u16 s (p + 2) = 8 * n in
      match List.assoc_opt key floats with
      | Some dt when normalised && exact -> whole dt
      | _ ->
          refuse "floating point of %d bytes other than IEEE binary16, \
                  bfloat16, binary32 and binary64" n)
  | 3 ->
      (* Bits 0 to 3 are the padding type, 4 to 7 the character set. *)
      { (as_bytes n) with pad = Some (bits land 0xf) }
  | 5 | 6 -> as_bytes n
  | 8 ->
      let e = element s p in
      if e.bytes <> n then
        refuse "an enumeration of %d bytes over %d" n e.bytes;
      e
  | 10 ->
      if version < 2 then refuse "an array datatype of version %d" version;
      need s p 1 "a datatype message";
      let rank = byte s p in
      let dims, base =
        if version = 2 then (p + 4, p + 4 + (8 * rank))
        else (p + 1, p + 1 + (4 * rank))
      in
      need s dims (4 * rank) "a datatype message";
      let dims = List.init rank (fun k -> u32 s (dims + (4 * k))) in
      let e = element s base in
      if List.fold_left ( * ) e.bytes dims <> n then
        refuse "an array of %d bytes over elements of %d" n e.bytes;
      { e with trailing = dims @ e.trailing; bytes = n }
  | 11 -> (
      let e = element s p in
      let rectangular = bits land 7 = 1 in
      match e.dt with
      | D.Any D.Float32 when rectangular && n = 8 ->
          { e with dt = D.Any D.Complex64; bytes = n }
      | D.Any D.Float64 when rectangular && n = 16 ->
          { e with dt = D.Any D.Complex128; bytes = n }
      | _ ->
          refuse
            "a complex type of %d bytes other than rectangular float32 or \
             float64 parts"
            n)
  | 2 -> refuse "the time class"
  | 7 -> refuse "a reference"
  | 9 ->
      refuse "variable-length, %s"
        (if bits land 0xf = 1 then "a string" else "a sequence")
  | c -> refuse "class %d" c

(* Dataspaces (FMT3, IV.A.2.b) *)

let dataspace f s =
  need s 0 8 "a dataspace message";
  let version = byte s 0 and rank = byte s 1 in
  let first =
    match version with
    | 1 -> 8
    | 2 ->
        if byte s 3 = 2 then failwith "dataspace: null, of no element";
        4
    | v -> Fail.failf "dataspace message version %d" v
  in
  need s first (rank * f.lengths) "a dataspace message";
  let shape = Array.init rank (fun k -> size f s (first + (k * f.lengths))) in
  Storage.check_shape shape;
  shape

(* Plans *)

type storage =
  | Compact of string
  | Contiguous of int option
  | Chunked of int option * int array

type filter = Deflate | Shuffle of int | Fletcher | Lzf

type plan = {
  shape : int array;
  e : element;
  storage : storage;
  filters : filter list;
  fill : string;
}

let dataspace_kind = 0x01
let datatype_kind = 0x03
let fill_kind = 0x05
let external_kind = 0x07
let pipeline_kind = 0x0B
let attribute_kind = 0x0C
let attribute_info = 0x15

(* The last message type FMT3 defines. *)
let known = 0x18

let message (t : t) kind =
  match List.find_opt (fun m -> m.kind = kind) t.messages with
  | None -> None
  | Some m ->
      if m.flags land 2 <> 0 then
        if kind = datatype_kind then
          failwith "a committed datatype, stored in another object"
        else Fail.failf "message %d is shared, stored in another object" kind;
      Some m.data

let required (t : t) kind what =
  match message t kind with
  | Some d -> d
  | None -> Fail.failf "no %s message" what

let shape (t : t) =
  Fail.file t.name @@ fun () ->
  Fail.file t.path @@ fun () ->
  dataspace t.form (required t dataspace_kind "dataspace")

(* The layout (FMT3, IV.A.2.i): version 3. *)
let storage (t : t) ~shape ~elt =
  let f = t.form in
  let rank = Array.length shape in
  let bytes = Array.fold_left ( * ) elt shape in
  let s = required t layout "layout" in
  need s 0 2 "a layout message";
  if byte s 0 <> 3 then
    Fail.failf "layout message version %d; read is 3" (byte s 0);
  match byte s 1 with
  | 0 ->
      need s 2 2 "a layout message";
      let n = u16 s 2 in
      need s 4 n "a layout message";
      if n <> bytes then
        Fail.failf "compact data of %d bytes for %d" n bytes;
      Compact (String.sub s 4 n)
  | 1 ->
      need s 2 (f.offsets + f.lengths) "a layout message";
      let n = size f s (2 + f.offsets) in
      let at = address f s 2 in
      if at <> None && n <> bytes then
        Fail.failf "contiguous data of %d bytes for %d" n bytes;
      Contiguous at
  | 2 ->
      need s 2 1 "a layout message";
      let k = byte s 2 in
      need s 3 (f.offsets + (4 * k)) "a layout message";
      if k <> rank + 1 then
        Fail.failf "chunks of rank %d for a dataspace of rank %d" (k - 1) rank;
      let dims = Array.init k (fun j -> u32 s (3 + f.offsets + (4 * j))) in
      if dims.(rank) <> elt then
        Fail.failf "chunks of elements of %d bytes for %d" dims.(rank) elt;
      let dims = Array.sub dims 0 rank in
      if Array.exists (( = ) 0) dims then failwith "a chunk of no element";
      Storage.check_shape dims;
      let count = Array.fold_left ( * ) 1 dims in
      if count > max_int / elt then
        Fail.failf "chunks of %s elements of %d bytes" (Fail.pp_shape dims) elt;
      Chunked (address f s 3, dims)
  | 3 -> failwith "layout class 3 (virtual)"
  | c -> Fail.failf "layout class %d" c

(* The filter pipeline (FMT3, IV.A.2.l), version 1, the one libhdf5 writes
   into files of superblock 0 or 1, in the order a write applies it. *)
let filters (t : t) =
  match message t pipeline_kind with
  | None -> []
  | Some s ->
      need s 0 8 "a filter pipeline message";
      if byte s 0 <> 1 then
        Fail.failf "filter pipeline message version %d; read is 1" (byte s 0);
      let n = byte s 1 in
      (* Each filter: its identifier, the length of its name, flags, the
         number of client values, the name padded to 8 bytes, the values
         of 4 bytes each, padded to 8. *)
      let rec read i k acc =
        if k = n then List.rev acc
        else begin
          need s i 8 "a filter";
          let id = u16 s i and name_n = u16 s (i + 2) in
          let values = u16 s (i + 6) in
          let i = i + 8 in
          need s i name_n "a filter";
          let name = String.sub s i name_n in
          let i = i + name_n in
          need s i (4 * values) "a filter";
          let cd = List.init values (fun j -> u32 s (i + (4 * j))) in
          let i = i + (4 * values) + if values mod 2 = 1 then 4 else 0 in
          let filter =
            match (id, cd) with
            | 1, _ -> Deflate
            | 2, [ e ] when e > 0 -> Shuffle e
            | 3, _ -> Fletcher
            | 32000, _ -> Lzf
            | 2, [ e ] ->
                Fail.failf "filter 2 (shuffle) of elements of %d bytes" e
            | 2, _ -> Fail.failf "filter 2 (shuffle) with %d values" values
            | id, _ ->
                let name =
                  match String.index_opt name '\000' with
                  | Some e -> String.sub name 0 e
                  | None -> name
                in
                Fail.failf "filter %d%s; read are 1 (deflate), 2 (shuffle), \
                            3 (Fletcher-32) and 32000 (LZF)"
                  id (if name = "" then "" else Printf.sprintf " (%s)" name)
          in
          read i (k + 1) (filter :: acc)
        end
      in
      read 8 0 []

(* The fill value (FMT3, IV.A.2.f): one element's bytes, zeros without
   one. libhdf5 writes the message's version 2 into files of superblock 0
   or 1, and beside it, for a fill value the program set, the old fill
   value message (IV.A.2.e), which holds the same value and is not read. *)
let fill (t : t) ~bytes =
  let s = required t fill_kind "fill value" in
  need s 0 4 "a fill value message";
  if byte s 0 <> 2 then
    Fail.failf "fill value message version %d; read is 2" (byte s 0);
  if byte s 3 <> 1 then String.make bytes '\000'
  else begin
    need s 4 4 "a fill value message";
    let n = u32 s 4 in
    need s 8 n "a fill value message";
    if n = 0 then String.make bytes '\000'
    else if n <> bytes then
      Fail.failf "a fill value of %d bytes for elements of %d" n bytes
    else String.sub s 8 n
  end

let plan (t : t) =
  let shape = dataspace t.form (required t dataspace_kind "dataspace") in
  let e = element (required t datatype_kind "datatype") 0 in
  if List.exists (fun m -> m.kind = external_kind) t.messages then
    failwith "storage in external files";
  List.iter
    (fun m ->
      if m.kind > known && m.flags land 0x80 <> 0 then
        Fail.failf "message type %d, which a reader must understand" m.kind)
    t.messages;
  let n = Array.fold_left ( * ) 1 shape in
  if e.bytes > 0 && n > max_int / e.bytes then
    Fail.failf "%d elements of %d bytes" n e.bytes;
  let storage = storage t ~shape ~elt:e.bytes in
  let filters = filters t in
  (match storage with
  | Chunked _ -> ()
  | _ -> if filters <> [] then failwith "filters on data that is not chunked");
  (* A compressor's output is the size its write's input had, which only a
     compressor before it leaves unknown. *)
  if List.length (List.filter (fun f -> f = Deflate || f = Lzf) filters) > 1
  then failwith "two compressing filters";
  { shape; e; storage; filters; fill = fill t ~bytes:e.bytes }

let chunk (t : t) =
  Fail.file t.name @@ fun () ->
  Fail.file t.path @@ fun () ->
  let shape = dataspace t.form (required t dataspace_kind "dataspace") in
  let e = element (required t datatype_kind "datatype") 0 in
  match storage t ~shape ~elt:e.bytes with
  | Chunked (_, dims) -> Some dims
  | Compact _ | Contiguous _ -> None

(* Filters, backwards *)

(* libhdf5's H5_checksum_fletcher32 (H5checksum.c): sums of the data's
   16-bit words, each its first byte high, an odd last byte as a word of
   its own, folded each 360 words. *)
let fletcher32 (b : Storage.bytes) n =
  let s1 = ref 0 and s2 = ref 0 in
  let fold () =
    s1 := (!s1 land 0xffff) + (!s1 lsr 16);
    s2 := (!s2 land 0xffff) + (!s2 lsr 16)
  in
  let words = n / 2 in
  let i = ref 0 in
  while !i < words do
    let batch = Int.min 360 (words - !i) in
    for k = !i to !i + batch - 1 do
      let w = (Array1.get b (2 * k) lsl 8) lor Array1.get b ((2 * k) + 1) in
      s1 := !s1 + w;
      s2 := !s2 + !s1
    done;
    i := !i + batch;
    fold ()
  done;
  if n mod 2 = 1 then begin
    s1 := !s1 + (Array1.get b (n - 1) lsl 8);
    s2 := !s2 + !s1;
    fold ()
  end;
  fold ();
  ((!s2 lsl 16) lor !s1) land 0xffff_ffff

(* The checksum's bytes swapped within each half, which libhdf5 before
   1.6.3 wrote on little-endian hosts and libhdf5 still accepts. *)
let reversed c =
  let swap h = ((h land 0xff) lsl 8) lor (h lsr 8) in
  (swap (c lsr 16) lsl 16) lor swap (c land 0xffff)

let create n = Array1.create int8_unsigned c_layout n

(* libhdf5's unshuffle (H5Zshuffle.c): byte [b] of element [j] is byte
   [b * m + j] of the input, of [m] elements of [e] bytes; the bytes past
   the last whole element stay in place. *)
let unshuffle e (src : Storage.bytes) n =
  let m = n / e in
  if e <= 1 || m <= 1 then src
  else begin
    let dst = create n in
    for b = 0 to e - 1 do
      for j = 0 to m - 1 do
        Array1.unsafe_set dst ((j * e) + b)
          (Array1.unsafe_get src ((b * m) + j))
      done
    done;
    let rest = n - (m * e) in
    Array1.blit (Array1.sub src (m * e) rest) (Array1.sub dst (m * e) rest);
    dst
  end

let filter_name = function
  | Deflate -> "1 (deflate)"
  | Shuffle _ -> "2 (shuffle)"
  | Fletcher -> "3 (Fletcher-32)"
  | Lzf -> "32000 (LZF)"

(* The most bytes a filter's one stored byte decodes to: deflate's 258-byte
   match from a length code and a distance code of a bit each (RFC 1951),
   and LZF's 264-byte match from 3 bytes. *)
let ratio = function Deflate -> 1032 | Lzf -> 88 | Shuffle _ | Fletcher -> 1

(* The filters a chunk's [mask] does not skip, with the size of their
   output on a write, the chunk's [bytes] grown by each checksum before
   it. *)
let applied filters ~mask ~bytes =
  let _, acc =
    List.fold_left
      (fun (k, acc) f ->
        let skipped = mask land (1 lsl k) <> 0 in
        let before = match acc with (_, n) :: _ -> n | [] -> bytes in
        if skipped then (k + 1, acc)
        else
          let after = match f with Fletcher -> before + 4 | _ -> before in
          (k + 1, (f, after) :: acc))
      (0, []) filters
  in
  acc

(* The chunk [src] of [stored] bytes through [filters] backwards, to its
   [bytes]. Each filter's output is the size the write gave its input. *)
let unfilter ~filters ~mask ~bytes src stored =
  let steps = applied filters ~mask ~bytes in
  let rec back steps src n =
    match steps with
    | [] ->
        if n <> bytes then Fail.failf "%d bytes for a chunk of %d" n bytes;
        src
    | (f, _) :: rest ->
        (* What this filter's write was given: the output of the one before
           it, or the chunk. *)
        let size = match rest with (_, m) :: _ -> m | [] -> bytes in
        let fail what = Fail.failf "filter %s: %s" (filter_name f) what in
        let src, n =
          match f with
          | Fletcher ->
              if n <= 4 then fail "no data before the checksum";
              let stored = ref 0 in
              for k = 3 downto 0 do
                stored := (!stored lsl 8) lor Array1.get src (n - 4 + k)
              done;
              let c = fletcher32 src (n - 4) in
              if !stored <> c && !stored <> reversed c then
                fail "the checksum does not match the data";
              (src, n - 4)
          | Shuffle e -> (unshuffle e src n, n)
          | Deflate | Lzf -> (
              let dst = create size in
              let decode =
                if f = Lzf then Lz.Lzf.decode else Lz.Zlib.decode
              in
              match decode (Array1.sub src 0 n) dst with
              | Ok { produced; _ } when produced = size -> (dst, size)
              | Ok { produced; _ } ->
                  fail (Printf.sprintf "%d bytes, not %d" produced size)
              | Error e -> fail (Format.asprintf "%a" Lz.pp_error e))
        in
        back rest src n
  in
  back steps src stored

(* Boxes *)

(* [each lo hi f] applies [f] to every index between [lo] and [hi], the
   last axis excluded, in C order. *)
let each lo hi f =
  let r = Array.length lo in
  if Array.for_all2 ( < ) lo hi then begin
    let i = Array.copy lo in
    let rec go a =
      if a >= r - 1 then f i
      else
        for v = lo.(a) to hi.(a) - 1 do
          i.(a) <- v;
          go (a + 1)
        done
    in
    go 0
  end

(* The byte strides of an array of [dims] elements of [e] bytes. *)
let strides dims e =
  let r = Array.length dims in
  let s = Array.make r e in
  for a = r - 2 downto 0 do
    s.(a) <- s.(a + 1) * dims.(a + 1)
  done;
  s

(* [copy ~src ~origin ~dims ~e ~lo ~hi dst ~start ~shape] copies the
   elements [lo] to [hi] (exclusive) of an array of [dims] elements of [e]
   bytes, whose first is the dataset's element [origin] and whose [n] bytes
   from [at] are [src at n], into [dst], the box of [shape] elements from
   [start]. *)
let copy ~src ~origin ~dims ~e ~lo ~hi dst ~start ~shape =
  let r = Array.length lo in
  if r = 0 then Array1.blit (src 0 e) (Array1.sub dst 0 e)
  else begin
    let cs = strides dims e and os = strides shape e in
    let span = (hi.(r - 1) - lo.(r - 1)) * e in
    each lo hi (fun i ->
        let at = ref 0 and out = ref 0 in
        for a = 0 to r - 1 do
          let v = if a = r - 1 then lo.(a) else i.(a) in
          at := !at + ((v - origin.(a)) * cs.(a));
          out := !out + ((v - start.(a)) * os.(a))
        done;
        Array1.blit (src !at span) (Array1.sub dst !out span))
  end

(* [with_bytes source ~first ~length f] is [f bytes], [bytes] the [length]
   bytes of [source] from byte [first], which a partial file must hold:
   their pages in the file, which the host borrows for the extent of [f], or
   a copy of them alone where the system cannot map the file, or the partial
   file's own. *)
let with_bytes source ~first ~length f =
  match source with
  | File b -> Storage.read (Storage.pages b ~first ~length) f
  | Part p -> f (Partial.bytes p ~pos:first length)

(* Chunk B-trees (FMT3, III.A.1, node type 1) *)

(* A chunk's record: its offset in elements, its address, its stored bytes
   and its filter mask. *)
type record = { at : int; stored : int; mask : int }

(* The records of the chunks at [targets], offsets sorted in C order, in
   the B-tree at [root], by offset; a target the tree lacks is a chunk
   never written. *)
let chunks r ~rank root targets =
  let key = 8 + (8 * (rank + 1)) in
  let offset k =
    Array.init rank (fun a -> uint "a chunk offset" k (8 + (8 * a)) 8)
  in
  let found = Hashtbl.create 64 in
  let rec visit at ~level targets =
    if targets <> [] then
      match node r ~kind:1 ~key at with
      | None -> ()
      | Some (l, keys, children) ->
          (match level with
          | Some v when v <> l ->
              Fail.failf "the B-tree node at byte %d is at level %d, not %d" at
                l v
          | _ -> ());
          let lows = Array.map offset keys in
          let n = Array.length children in
          (* Child [i] holds the chunks from key [i], up to key [i + 1]. *)
          let rec split i targets =
            if i < n then begin
              let mine, rest =
                if i = n - 1 then (targets, [])
                else
                  List.partition (fun t -> compare t lows.(i + 1) < 0) targets
              in
              let mine = List.filter (fun t -> compare t lows.(i) >= 0) mine in
              if l = 0 then
                List.iter
                  (fun t ->
                    if t = lows.(i) then
                      let k = keys.(i) in
                      Hashtbl.replace found t
                        { at = children.(i); stored = u32 k 0; mask = u32 k 4 })
                  mine
              else visit children.(i) ~level:(Some (l - 1)) mine;
              split (i + 1) rest
            end
          in
          split 0 targets
  in
  visit root ~level:None targets;
  found

(* Windows *)

let output (e : element) shape buf =
  let (D.Any dt) = e.dt in
  let dims = Array.append shape (Array.of_list e.trailing) in
  let layout = L.contiguous (Iarray.of_array dims) in
  let x = Nx.Repr.unsafe_of_array Nx.Host.v (A.v dt layout buf) in
  Nx.P (if e.swap then Storage.byteswap x else x)

(* A fill row of [n] bytes, the fill element repeated. *)
let fill_row fill n =
  let e = String.length fill in
  let b = create n in
  for i = 0 to n - 1 do
    Array1.unsafe_set b i (Char.code fill.[i mod e])
  done;
  b

let box_error (t : t) ~start ~shape dims =
  let r = Array.length dims in
  if Array.length start <> r || Array.length shape <> r then
    Fail.invalid_argf
      "Nx_io.Hdf5.window: a box of rank %d from rank %d for %s of rank %d"
      (Array.length shape) (Array.length start) t.path r;
  for a = 0 to r - 1 do
    if start.(a) < 0 || shape.(a) < 0 || start.(a) > dims.(a) - shape.(a) then
      Fail.invalid_argf
        "Nx_io.Hdf5.window: %d elements from %d lie outside axis %d of %s, \
         of %d"
        shape.(a) start.(a) a t.path dims.(a)
  done

(* A box of the dataset, [shape] elements from [start], and the byte [out]
   of the result from which its elements lie, in C order. A window is one
   box; rows are a box for each window, one after the other. *)
type box = { start : int array; shape : int array; out : int }

let box_bytes e b = Array.fold_left ( * ) e b.shape
let box_end b = Array.map2 ( + ) b.start b.shape

(* The result of [dims] elements, the boxes' elements, from bytes the
   source holds. [plan] has checked every byte it reads lies in the file.
   [records] are the chunks the boxes cover, each read once. *)
let assemble (t : t) p ~dims ~boxes ~records =
  let e = p.e.bytes and r = Array.length p.shape in
  let total = Array.fold_left ( * ) e dims in
  let buf = B.create Rig.host total in
  let boxes = List.filter (fun b -> box_bytes e b > 0) boxes in
  let span b = if r = 0 then e else b.shape.(r - 1) * e in
  let row = List.fold_left (fun n b -> Int.max n (span b)) 0 boxes in
  let filled = lazy (fill_row p.fill row) in
  (* [into dst b ~src ~origin ~dims ~lo ~hi] copies the elements [lo] to
     [hi] of [b], from [src] of [dims] elements from [origin]. *)
  let into dst b ~src ~origin ~dims ~lo ~hi =
    let dst = Array1.sub dst b.out (box_bytes e b) in
    copy ~src ~origin ~dims ~e ~lo ~hi dst ~start:b.start ~shape:b.shape
  in
  let fill dst b ~origin ~dims ~lo ~hi =
    let f = Lazy.force filled in
    into dst b ~src:(fun _ n -> Array1.sub f 0 n) ~origin ~dims ~lo ~hi
  in
  let zero = Array.make r 0 in
  let whole dst b ~src =
    into dst b ~src ~origin:zero ~dims:p.shape ~lo:b.start ~hi:(box_end b)
  in
  if total > 0 then
    B.write buf int8_unsigned (fun dst ->
        match p.storage with
        | Compact s ->
            let src = create (String.length s) in
            String.iteri (fun i c -> Array1.unsafe_set src i (Char.code c)) s;
            List.iter (whole dst ~src:(Array1.sub src)) boxes
        | Contiguous None ->
            List.iter
              (fun b ->
                fill dst b ~origin:zero ~dims:p.shape ~lo:b.start
                  ~hi:(box_end b))
              boxes
        | Contiguous (Some at) -> (
            match t.source with
            | Part q ->
                let src i n = Partial.bytes q ~pos:(at + i) n in
                List.iter (whole dst ~src) boxes
            | File _ ->
                (* Each box's bytes in the file, from its first element's
                   to past its last's. *)
                let cs = strides p.shape e in
                let offset idx =
                  Array.fold_left ( + ) 0
                    (Array.mapi (fun a v -> v * cs.(a)) idx)
                in
                List.iter
                  (fun b ->
                    let lo = offset b.start in
                    let last = Array.map (fun h -> h - 1) (box_end b) in
                    let length = offset last + e - lo in
                    with_bytes t.source ~first:(at + lo) ~length (fun bytes ->
                        let src i n = Array1.sub bytes (i - lo) n in
                        whole dst b ~src))
                  boxes)
        | Chunked (_, dims) ->
            let bytes_of = Array.fold_left ( * ) e dims in
            (* The chunks come in C order, so their first rows never
               decrease. A box opens once a chunk reaches its first row and
               closes once a chunk starts past its last, so a chunk meets
               the open boxes alone: those whose rows it shares, and those
               about to close. *)
            (* A box's or a chunk's first row, and the row past its last;
               a scalar's one element is row 0. *)
            let row a = if r = 0 then 0 else a.(0) in
            let past a = if r = 0 then 1 else a.(0) in
            let by_row a b = Int.compare (row a.start) (row b.start) in
            let sorted = Array.of_list (List.stable_sort by_row boxes) in
            let next = ref 0 and opened = ref [] in
            List.iter
              (fun (origin, record) ->
                let top = Array.map2 ( + ) origin dims in
                while
                  !next < Array.length sorted
                  && row sorted.(!next).start < past top
                do
                  opened := sorted.(!next) :: !opened;
                  incr next
                done;
                opened :=
                  List.filter (fun b -> past (box_end b) > row origin) !opened;
                (* The part of each open box the chunk holds. *)
                let parts =
                  List.filter_map
                    (fun b ->
                      let lo = Array.map2 Int.max origin b.start in
                      let hi = Array.map2 Int.min top (box_end b) in
                      if Array.for_all2 ( < ) lo hi then Some (b, lo, hi)
                      else None)
                    !opened
                in
                match record with
                | None ->
                    List.iter
                      (fun (b, lo, hi) -> fill dst b ~origin ~dims ~lo ~hi)
                      parts
                | Some c ->
                    with_bytes t.source ~first:c.at ~length:c.stored
                    @@ fun stored ->
                    let chunk =
                      try
                        unfilter ~filters:p.filters ~mask:c.mask
                          ~bytes:bytes_of stored c.stored
                      with Failure m ->
                        Fail.failf "chunk at %s, byte %d: %s"
                          (Fail.pp_shape origin) c.at m
                    in
                    List.iter
                      (fun (b, lo, hi) ->
                        into dst b ~src:(Array1.sub chunk) ~origin ~dims ~lo
                          ~hi)
                      parts)
              records);
  output p.e dims buf

(* The offsets of the chunks of [dims] a box covers, in C order. *)
let covered dims b =
  let r = Array.length dims in
  if Array.exists (( = ) 0) b.shape then []
  else begin
    let lo = Array.init r (fun a -> b.start.(a) / dims.(a)) in
    let hi =
      Array.init r (fun a -> ((b.start.(a) + b.shape.(a) - 1) / dims.(a)) + 1)
    in
    let acc = ref [] in
    let i = Array.copy lo in
    let rec go a =
      if a = r then acc := Array.mapi (fun k v -> v * dims.(k)) i :: !acc
      else
        for v = lo.(a) to hi.(a) - 1 do
          i.(a) <- v;
          go (a + 1)
        done
    in
    go 0;
    List.rev !acc
  end

(* The ranges a partial file lacks of what the boxes read, and the chunks
   they read, each once with its record, in C order, after checking every
   byte read lies in the file. *)
let reads (t : t) p r boxes =
  let e = p.e.bytes in
  let length = length t.source in
  match p.storage with
  | Compact _ | Contiguous None -> []
  | Contiguous (Some at) ->
      let total = Array.fold_left ( * ) e p.shape in
      Storage.within ~length ~pos:at total;
      (match t.source with
      | File _ -> ()
      | Part q ->
          let rank = Array.length p.shape in
          let cs = strides p.shape e in
          let lacks b =
            if box_bytes e b > 0 then begin
              let span = if rank = 0 then e else b.shape.(rank - 1) * e in
              let row i =
                let off = ref at in
                for a = 0 to rank - 1 do
                  let v = if a = rank - 1 then b.start.(a) else i.(a) in
                  off := !off + (v * cs.(a))
                done;
                r.lacks := Partial.lacking q ~pos:!off span @ !(r.lacks)
              in
              if rank = 0 then row [||] else each b.start (box_end b) row
            end
          in
          List.iter lacks boxes);
      []
  | Chunked (root, dims) ->
      let targets =
        List.sort_uniq compare (List.concat_map (covered dims) boxes)
      in
      let found =
        match root with
        | None -> Hashtbl.create 0
        | Some root -> chunks r ~rank:(Array.length dims) root targets
      in
      let bytes = Array.fold_left ( * ) e dims in
      List.map
        (fun origin ->
          match Hashtbl.find_opt found origin with
          | None -> (origin, None)
          | Some c ->
              Storage.within ~length ~pos:c.at c.stored;
              let bound =
                List.fold_left
                  (fun n f ->
                    if n > max_int / ratio f then max_int else n * ratio f)
                  c.stored p.filters
              in
              if bound < bytes then
                Fail.failf "the chunk at %s holds %d bytes, too few for its %d"
                  (Fail.pp_shape origin) c.stored bytes;
              (match t.source with
              | File _ -> ()
              | Part q ->
                  let rs = Partial.lacking q ~pos:c.at c.stored in
                  r.lacks := rs @ !(r.lacks));
              (origin, Some c))
        targets

let reader_of (t : t) = { (reader t.source) with form = t.form }

(* The result of [dims] elements that holds the boxes, or the ranges a
   partial file lacks of them. *)
let read_boxes (t : t) p ~dims boxes =
  let r = reader_of t in
  let reads () = Fail.file t.path (fun () -> reads t p r boxes) in
  match walk r reads with
  | Error rs -> Error rs
  | Ok records ->
      Ok (Fail.file t.path (fun () -> assemble t p ~dims ~boxes ~records))

let window (t : t) ~start ~shape =
  Fail.file t.name @@ fun () ->
  let p = Fail.file t.path (fun () -> plan t) in
  box_error t ~start ~shape p.shape;
  read_boxes t p ~dims:shape [ { start; shape; out = 0 } ]

let rows (t : t) ~row ~rows =
  let by = "Nx_io.Hdf5.rows" in
  if Array.length row <> Array.length rows then
    Fail.invalid_argf "%s: %d windows' rows and %d counts" by (Array.length row)
      (Array.length rows);
  Fail.file t.name @@ fun () ->
  let p = Fail.file t.path (fun () -> plan t) in
  let rank = Array.length p.shape in
  if rank = 0 then Fail.invalid_argf "%s: %s is scalar, of no row" by t.path;
  let n = p.shape.(0) and cell = Array.sub p.shape 1 (rank - 1) in
  let numel =
    List.fold_left ( * ) (Array.fold_left ( * ) 1 cell) p.e.trailing
  in
  let limit = L.max_numel / Int.max 1 numel in
  let row_bytes = Array.fold_left ( * ) p.e.bytes cell in
  let k = ref 0 and boxes = ref [] in
  Array.iteri
    (fun j r ->
      let m = rows.(j) in
      if r < 0 || m < 0 then
        Fail.invalid_argf "%s: window %d, row %d, rows %d" by j r m;
      if r > n - m then
        Fail.invalid_argf
          "%s: window %d, %d rows from row %d, ends past %d rows" by j m r n;
      if m > limit - !k then
        Fail.invalid_argf "%s: the windows hold more than 2^53 %s" by
          (if numel = 0 then "rows" else "elements");
      let start = Array.make rank 0 and shape = Array.copy p.shape in
      start.(0) <- r;
      shape.(0) <- m;
      boxes := { start; shape; out = !k * row_bytes } :: !boxes;
      k := !k + m)
    row;
  read_boxes t p ~dims:(Array.append [| !k |] cell) (List.rev !boxes)

let values (t : t) =
  let p = Fail.file t.name (fun () -> Fail.file t.path (fun () -> plan t)) in
  match (t.source, p.storage) with
  | File file, Contiguous (Some at) ->
      Fail.file t.name @@ fun () ->
      let (D.Any dt) = p.e.dt in
      let total = Array.fold_left ( * ) p.e.bytes p.shape in
      Fail.file t.path (fun () ->
          Storage.within ~length:(B.length file) ~pos:at total);
      let dims = Array.append p.shape (Array.of_list p.e.trailing) in
      let layout = L.contiguous (Iarray.of_array dims) in
      let x = Storage.value (B.view file ~first:at ~length:total) dt layout in
      Nx.P (if p.e.swap then Storage.byteswap x else x)
  | _ -> (
      let start = Array.make (Array.length p.shape) 0 in
      match window t ~start ~shape:p.shape with
      | Ok x -> x
      | Error rs ->
          let at, n = List.hd rs in
          Fail.failf "%s: the partial file lacks bytes %d to %d" t.name at
            (at + n - 1))

(* Strings (FMT3, IV.A.2.d, class 3): the padding types. *)

let null_terminated = 0
let null_padded = 1
let space_padded = 2

(* [trim c s] is [s] without the [c]s that end it. *)
let trim c s =
  let n = ref (String.length s) in
  while !n > 0 && s.[!n - 1] = c do
    decr n
  done;
  String.sub s 0 !n

let strings (t : t) (Nx.P x) =
  let by = "Nx_io.Hdf5.strings" in
  let e, pad =
    Fail.file t.name @@ fun () ->
    Fail.file t.path @@ fun () ->
    let e = element (required t datatype_kind "datatype") 0 in
    match e.pad with
    | Some p when p > space_padded ->
        Fail.failf "string padding %d, which the format reserves" p
    | pad -> (e, pad)
  in
  let pad, size =
    match (pad, List.rev e.trailing) with
    | Some pad, size :: _ -> (pad, size)
    | _ -> Fail.invalid_argf "%s: %s holds no fixed-length strings" by t.path
  in
  let dims = Nx.shape x in
  let r = Array.length dims in
  let x : (int, int8_unsigned_elt, Nx.host) Nx.t =
    match D.equal_witness (Nx.dtype x) D.Uint8 with
    | Some Type.Equal when r > 0 && dims.(r - 1) = size -> x
    | _ ->
        Fail.invalid_argf
          "%s: %s: %s of shape %s, uint8 with a last extent of %d expected" by
          t.path
          (D.name (Nx.dtype x))
          (Fail.pp_shape dims) size
  in
  let all = Nx.to_array x in
  let cut s =
    if pad = null_terminated then
      match String.index_opt s '\000' with
      | Some i -> String.sub s 0 i
      | None -> s
    else if pad = null_padded then trim '\000' s
    else trim ' ' s
  in
  Array.init (Array.length all / size) (fun i ->
      cut (String.init size (fun k -> Char.chr all.((i * size) + k))))

(* Attributes (FMT3, IV.A.2.m), versions 1 to 3. libhdf5 writes version 3
   for a name whose encoding is UTF-8, which differs from version 2 by
   that encoding's byte. *)

let pad8 n = (n + 7) land lnot 7

(* An attribute message: its name [key], whether its datatype or
   dataspace is shared, its datatype, its dataspace, and its data from
   [at]. *)
type attr = {
  key : string;
  shared : bool;
  dt : string;
  ds : string;
  at : int;
}

let attr s =
  need s 0 8 "an attribute message";
  let version = byte s 0 in
  let name_n = u16 s 2 and dt_n = u16 s 4 and ds_n = u16 s 6 in
  let first, pad =
    match version with
    | 1 -> (8, pad8)
    | 2 -> (8, Fun.id)
    | 3 -> (9, Fun.id)
    | v -> Fail.failf "attribute message version %d; read are 1 to 3" v
  in
  let dt_at = first + pad name_n in
  let ds_at = dt_at + pad dt_n in
  let at = ds_at + pad ds_n in
  need s first name_n "an attribute message";
  need s dt_at dt_n "an attribute message";
  need s ds_at ds_n "an attribute message";
  need s at 0 "an attribute message";
  let key = String.sub s first name_n in
  let key =
    match String.index_opt key '\000' with
    | Some e -> String.sub key 0 e
    | None -> key
  in
  let shared = version > 1 && byte s 1 land 3 <> 0 in
  let dt = String.sub s dt_at dt_n and ds = String.sub s ds_at ds_n in
  { key; shared; dt; ds; at }

(* The attributes of [t]'s header, each with its message's data. *)
let attrs (t : t) =
  (match message t attribute_info with
  | Some s ->
      (* The attribute info message (FMT3, IV.A.2.v): a version, flags,
         a maximum creation index when flag 0 is set, then the fractal
         heap's address. *)
      need s 0 2 "an attribute info message";
      let at = if byte s 1 land 1 <> 0 then 4 else 2 in
      need s at t.form.offsets "an attribute info message";
      if address t.form s at <> None then
        failwith "attributes stored in a fractal heap"
  | None -> ());
  List.filter_map
    (fun m ->
      if m.kind <> attribute_kind then None else Some (attr m.data, m.data))
    t.messages

let attributes (t : t) =
  Fail.file t.name @@ fun () ->
  Fail.file t.path @@ fun () -> List.map (fun (a, _) -> a.key) (attrs t)

let attribute (t : t) name =
  Fail.file t.name @@ fun () ->
  Fail.file t.path @@ fun () ->
  match List.find_opt (fun (a, _) -> a.key = name) (attrs t) with
  | None -> Fail.failf "no attribute named %S" name
  | Some (a, data) ->
      Fail.file name @@ fun () ->
      if a.shared then failwith "a shared datatype or dataspace";
      let e = element a.dt 0 in
      let shape = dataspace t.form a.ds in
      let n = Array.fold_left ( * ) e.bytes shape in
      need data a.at n "an attribute's data";
      let buf = B.create Rig.host n in
      if n > 0 then
        B.write buf int8_unsigned (fun dst ->
            for i = 0 to n - 1 do
              Array1.unsafe_set dst i (Char.code data.[a.at + i])
            done);
      output e shape buf

module Group = struct
  type t = group

  let path (g : t) = g.path
  let attributes = attributes
  let attribute = attribute
end

let lookup3 ?(init = 0) s = hash ~init s (String.length s)
