(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Rig.Buffer
module Sub = Rig.Submission

let strf = Printf.sprintf

(* Descriptions *)

type leaf =
  | Address of { memory : int; on : int }
  | Handle of int
  | Entry of { image : int; name : string }
  | Ready of int

(* Before holes, whose [at] an unannotated field then means. *)
type use = Read | Write | Unordered
type ref = { slot : int; at : int; use : use }
type width = W32 | W64
type 'a hole = {
  at : int;
  width : width;
  terms : ('a * int) iarray;
  add : int;
  shift : int;
}
type 'a data = { bytes : string; holes : 'a hole iarray }
type value = Fixed of int | Int of int | Leaf of leaf
type copies = One | Two
type area = Outbound | Inbound | Counts

type memory =
  | Alloc of {
      device : int;
      kind : B.kind;
      bytes : int;
      init : leaf data;
      copies : copies;
    }
  | Rail of { rail : int; area : area }

type image = { device : int; binary : leaf data }
type view = { memory : int; offset : int; length : int }
type slot = Memory of view | Input of int | Ints

type part =
  | Words of view
  | Fill of { fill : leaf; params : value data; refs : ref iarray }
  | Copy of {
      src : int;
      src_offset : (value * int) iarray;
      dst : int;
      dst_offset : (value * int) iarray;
      length : int;
    }
  | Launch of {
      image : int;
      name : string;
      params : value data;
      refs : ref iarray;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type submit = {
  device : int;
  parts : part iarray;
  slots : slot iarray;
  fixed : (view * Rig.access) iarray;
}

type step =
  | Submit of submit
  | Move of { src : slot; dst : slot }
  | Loop of {
      trips : value;
      trip : int option;
      flag : (slot * int) option;
      body : step iarray;
    }

type input = { device : int; bytes : int }

type t = {
  devices : string iarray;
  memory : memory iarray;
  images : image iarray;
  inputs : input iarray;
  ints : int;
  steps : step iarray;
}

(* A refusal of the description, which [load] answers as [Error]. *)
exception Refused of string

let refuse fmt = Printf.ksprintf (fun why -> raise (Refused why)) fmt

(* An offset's sum past an OCaml [int]. *)
exception Overflow

let add_exn a b =
  let s = a + b in
  if a >= 0 = (b >= 0) && s >= 0 <> (a >= 0) then raise Overflow else s

let mul_exn a b =
  if a = 0 then 0
  else
    let p = a * b in
    if (a = -1 && b = min_int) || p / a <> b then raise Overflow else p

(* The run's ints are memory [Iarray.length t.memory] of two copies, which
   [load] makes on the host: a slot [Ints] is its view. *)
let ints_view t =
  { memory = Iarray.length t.memory; offset = 0; length = 8 * t.ints }

(* Checks *)

(* What [load] refuses before it makes anything: every index, size and hole, and
   where a leaf of [Two] memory may stand. *)

let index what n i =
  if i < 0 || i >= n then refuse "%s %d: no such %s" what i what

let two t m =
  match Iarray.get t.memory m with
  | Alloc { copies; _ } -> copies = Two
  | Rail _ -> false

let check_leaf t ~images = function
  | Address { memory; on } ->
      index "memory" (Iarray.length t.memory) memory;
      index "device" (Iarray.length t.devices) on
  | Handle m -> index "memory" (Iarray.length t.memory) m
  | Entry { image; _ } -> index "image" images image
  | Ready _ -> ()

let leaf_two t = function
  | Address { memory; _ } | Handle memory -> two t memory
  | Entry _ | Ready _ -> false

let width h = match h.width with W32 -> 4 | W64 -> 8

(* Bounds are compared by subtraction: a description's integers may be any, and
   a sum of two may wrap. *)
let check_holes what len check holes =
  Iarray.iter
    (fun h ->
      if h.at < 0 || h.at > len - width h then
        refuse "%s: a hole at %d outside its %d bytes" what h.at len;
      if h.shift < 0 || h.shift > 62 then
        refuse "%s: a hole's shift %d outside 0 to 62" what h.shift;
      Iarray.iter (fun (x, _) -> check x) h.terms)
    holes;
  let sorted =
    List.sort (fun a b -> compare a.at b.at) (Iarray.to_list holes)
  in
  ignore
    (List.fold_left
       (fun prev h ->
         (match prev with
         | Some p when p.at + width p > h.at ->
             refuse "%s: holes at %d and %d share a byte" what p.at h.at
         | _ -> ());
         Some h)
       None sorted)

let check_view t sizes (v : view) =
  index "memory" (Iarray.length t.memory) v.memory;
  let n = Option.value sizes.(v.memory) ~default:max_int in
  if v.offset < 0 || v.length < 0 || v.offset > n - v.length then
    refuse "memory %d: a view of %d bytes at %d outside its %d bytes" v.memory
      v.length v.offset n

let check_slot t sizes = function
  | Memory v -> check_view t sizes v
  | Input i -> index "input" (Iarray.length t.inputs) i
  | Ints -> ()

let check_value t : value -> unit = function
  | Fixed _ -> ()
  | Int i -> index "int" t.ints i
  | Leaf l -> check_leaf t ~images:(Iarray.length t.images) l

let refs_of = function
  | Fill { refs; _ } | Launch { refs; _ } -> refs
  | Words _ | Copy _ -> Iarray.of_list []

let check_part t sizes = function
  | Words v -> check_view t sizes v
  | Fill { fill; params; _ } ->
      check_leaf t ~images:(Iarray.length t.images) fill;
      let n = String.length params.bytes in
      if n mod 4 <> 0 then refuse "fill: invalid parameter length %d" n;
      check_holes "fill parameters" n (check_value t) params.holes
  | Copy { src_offset; dst_offset; length; _ } ->
      if length < 0 then refuse "a copy of %d bytes" length;
      let term = function
        | Leaf _, _ -> refuse "a copy's offset names a leaf"
        | x, _ -> check_value t x
      in
      Iarray.iter term src_offset;
      Iarray.iter term dst_offset
  | Launch
      { image; params; groups = gx, gy, gz; threads = tx, ty, tz; shared; _ } ->
      index "image" (Iarray.length t.images) image;
      let n = String.length params.bytes in
      if n mod 4 <> 0 then
        refuse
          "image %d: %d bytes of parameters, not a whole number of 32-bit words"
          image n;
      check_holes
        (strf "image %d's parameters" image)
        n (check_value t) params.holes;
      List.iter (check_value t) [ gx; gy; gz; tx; ty; tz; shared ]

(* An offset's value where it holds only constants, which [load] knows. *)
let fixed_offset terms =
  try
    Iarray.fold_left
      (fun sum (x, factor) ->
        match (sum, x) with
        | Some s, Fixed n -> Some (add_exn s (mul_exn n factor))
        | _ -> None)
      (Some 0) terms
  with Overflow -> refuse "a copy's offset past an int"

(* Refuses a copy whose end lies outside its slot, or whose two ends share a
   byte of one buffer, where load knows the offsets. A view's and the ints'
   lengths are known; an input's is only at least its [bytes]. *)
let check_copy t slots ~src ~src_offset ~dst ~dst_offset ~length =
  let check_end i offset =
    match fixed_offset offset with
    | None -> ()
    | Some o -> (
        if o < 0 then refuse "a copy from byte %d of slot %d" o i;
        let n =
          match Iarray.get slots i with
          | Memory v -> Some v.length
          | Ints -> Some (8 * t.ints)
          | Input _ -> None
        in
        match n with
        | Some n when o > n - length ->
            refuse "a copy of %d bytes from byte %d of slot %d's %d" length o i
              n
        | _ -> ())
  in
  check_end src src_offset;
  check_end dst dst_offset;
  let one =
    match (Iarray.get slots src, Iarray.get slots dst) with
    | Ints, Ints -> true
    | Input i, Input j -> i = j
    | _ -> false
  in
  match (fixed_offset src_offset, fixed_offset dst_offset) with
  | Some a, Some b when one && length > 0 && a - b < length && b - a < length
    ->
      refuse "a copy between overlapping bytes %d and %d of one buffer" a b
  | _ -> ()

let rec check_step t sizes ndev = function
  | Submit s ->
      index "device" ndev s.device;
      Iarray.iter (check_part t sizes) s.parts;
      Iarray.iter (check_slot t sizes) s.slots;
      (* Per slot, whether an unordered ref names it, and whether an ordered
         ref or a copy does. *)
      let n = Iarray.length s.slots in
      let unordered = Array.make n false and ordered = Array.make n false in
      let name k =
        index "slot" n k;
        ordered.(k) <- true
      in
      Iarray.iter
        (function
          | Copy { src; src_offset; dst; dst_offset; length } ->
              name src;
              name dst;
              check_copy t s.slots ~src ~src_offset ~dst ~dst_offset ~length
          | p ->
              Iarray.iter
                (fun r ->
                  match r.use with
                  | Unordered ->
                      index "slot" n r.slot;
                      unordered.(r.slot) <- true
                  | Read | Write -> name r.slot)
                (refs_of p))
        s.parts;
      for i = 0 to n - 1 do
        match (unordered.(i), ordered.(i)) with
        | false, false -> refuse "slot %d: no ref or copy names it" i
        | true, true ->
            refuse
              "slot %d: named by an unordered ref and an ordered one or a copy"
              i
        | _ -> ()
      done;
      Iarray.iter (fun (v, _) -> check_view t sizes v) s.fixed
  | Move { src; dst } ->
      check_slot t sizes src;
      check_slot t sizes dst
  | Loop { trips; trip; flag; body } ->
      check_value t trips;
      Option.iter (index "int" t.ints) trip;
      Option.iter
        (fun (slot, at) ->
          check_slot t sizes slot;
          let bytes =
            match slot with
            | Memory v -> Some v.length
            | Ints -> Some (8 * t.ints)
            | Input _ -> None
          in
          match bytes with
          | _ when at < 0 -> refuse "a loop's flag at byte %d" at
          | Some n when at >= n ->
              refuse "a loop's flag at byte %d of a slot of %d bytes" at n
          | _ -> ())
        flag;
      Iarray.iter (check_step t sizes ndev) body

(* A memory leaf denotes its whole allocation. A step reading an initialized
   container must also name every pointee, including those reached through other
   initialized containers. Indices and views have already been checked. *)
let check_memory t sizes =
  let whole views m =
    List.exists
      (fun (v : view) ->
        v.memory = m && v.offset = 0
        && Option.fold ~none:true ~some:(( = ) v.length) sizes.(m))
      views
  in
  let views slots =
    List.filter_map
      (function Memory v -> Some v | Input _ | Ints -> None)
      slots
  in
  let named s =
    let parts =
      Iarray.to_list s.parts
      |> List.concat_map (function
        | Words v -> [ v ]
        | Copy _ | Fill _ | Launch _ -> [])
    in
    List.map fst (Iarray.to_list s.fixed)
    @ views (Iarray.to_list s.slots)
    @ parts
  in
  let rec fixing m on steps =
    Iarray.exists
      (function
        | Submit s ->
            s.device = on && whole (List.map fst (Iarray.to_list s.fixed)) m
        | Loop { body; _ } -> fixing m on body
        | Move _ -> false)
      steps
  in
  let terms holes f =
    Iarray.iter (fun h -> Iarray.iter (fun (x, _) -> f x) h.terms) holes
  in
  let closure views =
    (* Each pointee must itself occur whole in views, so checking every named
       container also checks every transitive pointee, including cycles. *)
    List.iter
      (fun (v : view) ->
        match Iarray.get t.memory v.memory with
        | Rail _ -> ()
        | Alloc { init; _ } ->
            terms init.holes (function
              | Address { memory; _ } | Handle memory ->
                  if not (whole views memory) then
                    refuse
                      "memory %d: its init names memory %d, which its step \
                       does not name whole"
                      v.memory memory
              | Entry _ | Ready _ -> ()))
      views
  in
  let value ~on ~fixed views = function
    | Leaf (Address { memory; on = device }) ->
        if on <> Some device || not (whole fixed memory) then
          refuse
            "memory %d: an Address needs its step to fix it whole on device %d"
            memory device
    | Leaf (Handle memory) ->
        if not (whole views memory) then
          refuse "memory %d: a Handle needs its step to name it whole" memory
    | Fixed _ | Int _ | Leaf (Entry _ | Ready _) -> ()
  in
  Iarray.iteri
    (fun i (image : image) ->
      terms image.binary.holes (function
        | Address _ | Handle _ -> refuse "image %d: a hole names memory" i
        | Entry _ | Ready _ -> ()))
    t.images;
  Iarray.iteri
    (fun m -> function
      | Rail _ -> ()
      | Alloc { init; _ } ->
          terms init.holes (function
            | Address { memory; on } when not (fixing memory on t.steps) ->
                refuse
                  "memory %d: its init takes memory %d's address on device %d, \
                   where no step fixes it whole"
                  m memory on
            | Address _ | Handle _ | Entry _ | Ready _ -> ()))
    t.memory;
  let rec step = function
    | Submit s ->
        let views = named s in
        closure views;
        let check =
          value ~on:(Some s.device)
            ~fixed:(List.map fst (Iarray.to_list s.fixed))
            views
        in
        Iarray.iter
          (function
            | Fill { fill; params; _ } ->
                (match fill with
                | Address _ | Handle _ ->
                    refuse "fill: memory is not a callable function"
                | Entry _ | Ready _ -> ());
                terms params.holes check
            | Launch
                { params; groups = gx, gy, gz; threads = tx, ty, tz; shared; _ }
              ->
                terms params.holes check;
                List.iter check [ gx; gy; gz; tx; ty; tz; shared ]
            | Words _ | Copy _ -> ())
          s.parts
    | Loop { trips; flag; body; _ } ->
        value ~on:None ~fixed:[]
          (views (Option.to_list (Option.map fst flag)))
          trips;
        Iarray.iter step body
    | Move _ -> ()
  in
  Iarray.iter step t.steps

(* [sizes] gives known extents. Only the agent knows a remote rail's extent;
   it checks the bounds and whole-memory claims again before loading. *)
let check t sizes devices =
  if Array.length devices <> Iarray.length t.devices then
    refuse "%d devices for a program of %d" (Array.length devices)
      (Iarray.length t.devices);
  Iarray.iteri
    (fun i arch ->
      let a = Rig.arch devices.(i) in
      if a <> arch then
        refuse "device %d: %s is %S, not %S" i (Rig.name devices.(i)) a arch)
    t.devices;
  if t.ints < 0 || t.ints > max_int / 8 then refuse "%d ints" t.ints;
  let ndev = Iarray.length t.devices in
  Iarray.iteri
    (fun i -> function
      | Alloc { device; bytes; init; copies; _ } ->
          index "device" ndev device;
          if bytes < 0 then refuse "memory %d: %d bytes" i bytes;
          let n = String.length init.bytes in
          if n > bytes then
            refuse "memory %d: %d bytes of init in %d bytes" i n bytes;
          check_holes (strf "memory %d" i) n
            (fun l ->
              check_leaf t ~images:(Iarray.length t.images) l;
              if copies = One && leaf_two t l then
                refuse "memory %d: of one copy, a hole names memory of two" i)
            init.holes
      | Rail _ -> ())
    t.memory;
  Iarray.iteri
    (fun i (im : image) ->
      index "device" ndev im.device;
      check_holes (strf "image %d" i)
        (String.length im.binary.bytes)
        (fun l ->
          check_leaf t ~images:i l;
          if leaf_two t l then refuse "image %d: a hole names memory of two" i)
        im.binary.holes)
    t.images;
  Iarray.iteri
    (fun i (input : input) ->
      index "device" ndev input.device;
      if input.bytes < 0 then refuse "input %d: %d bytes" i input.bytes)
    t.inputs;
  Iarray.iter (check_step t sizes ndev) t.steps;
  check_memory t sizes

(* Bytes *)

(* A description as bytes: the magic, the format's version, then the
   description. An integer is 64 bits, little-endian; a string or an array its
   count, then its elements; a variant its tag, a byte, then its arguments. *)

let magic = "rig.program\n"
let version = 13

module W = struct
  let int b n = Buffer.add_int64_le b (Int64.of_int n)
  let tag b n = Buffer.add_uint8 b n

  let string b s =
    int b (String.length s);
    Buffer.add_string b s

  let array f b a =
    int b (Iarray.length a);
    Iarray.iter (f b) a

  let option f b = function
    | None -> tag b 0
    | Some x ->
        tag b 1;
        f b x

  let leaf b = function
    | Address { memory; on } ->
        tag b 0;
        int b memory;
        int b on
    | Handle m ->
        tag b 1;
        int b m
    | Entry { image; name } ->
        tag b 2;
        int b image;
        string b name
    | Ready r ->
        tag b 3;
        int b r

  let value b = function
    | Fixed n ->
        tag b 0;
        int b n
    | Int i ->
        tag b 1;
        int b i
    | Leaf l ->
        tag b 2;
        leaf b l

  let terms f b ts =
    array
      (fun b (x, factor) ->
        f b x;
        int b factor)
      b ts

  let data f b (d : _ data) =
    string b d.bytes;
    array
      (fun b h ->
        int b h.at;
        tag b (match h.width with W32 -> 0 | W64 -> 1);
        terms f b h.terms;
        int b h.add;
        int b h.shift)
      b d.holes

  let memory_kind b (k : B.kind) =
    tag b (match k with Device -> 0 | Pinned -> 1 | Mapped -> 2)

  let access b (a : Rig.access) =
    tag b (match a with Read -> 0 | Read_write -> 1)

  let ref b (r : ref) =
    int b r.slot;
    int b r.at;
    tag b (match r.use with Read -> 0 | Write -> 1 | Unordered -> 2)

  let memory b = function
    | Alloc { device; kind; bytes; init; copies } ->
        tag b 0;
        int b device;
        memory_kind b kind;
        int b bytes;
        data leaf b init;
        tag b (match copies with One -> 0 | Two -> 1)
    | Rail { rail; area } ->
        tag b 1;
        int b rail;
        tag b (match area with Outbound -> 0 | Inbound -> 1 | Counts -> 2)

  let view b (v : view) =
    int b v.memory;
    int b v.offset;
    int b v.length

  let slot b = function
    | Memory v ->
        tag b 0;
        view b v
    | Input i ->
        tag b 1;
        int b i
    | Ints -> tag b 2

  let triple b (x, y, z) =
    value b x;
    value b y;
    value b z

  let part b = function
    | Words v ->
        tag b 0;
        view b v
    | Fill { fill; params; refs } ->
        tag b 1;
        leaf b fill;
        data value b params;
        array ref b refs
    | Copy { src; src_offset; dst; dst_offset; length } ->
        tag b 2;
        int b src;
        terms value b src_offset;
        int b dst;
        terms value b dst_offset;
        int b length
    | Launch { image; name; params; refs; groups; threads; shared } ->
        tag b 3;
        int b image;
        string b name;
        data value b params;
        array ref b refs;
        triple b groups;
        triple b threads;
        value b shared

  let rec step b = function
    | Submit s ->
        tag b 0;
        int b s.device;
        array part b s.parts;
        array slot b s.slots;
        array
          (fun b (v, a) ->
            view b v;
            access b a)
          b s.fixed
    | Move { src; dst } ->
        tag b 1;
        slot b src;
        slot b dst
    | Loop { trips; trip; flag; body } ->
        tag b 2;
        value b trips;
        option int b trip;
        option
          (fun b (s, at) ->
            slot b s;
            int b at)
          b flag;
        array step b body

  let t b t =
    array string b t.devices;
    array memory b t.memory;
    array
      (fun b (i : image) ->
        int b i.device;
        data leaf b i.binary)
      b t.images;
    array
      (fun b (i : input) ->
        int b i.device;
        int b i.bytes)
      b t.inputs;
    int b t.ints;
    array step b t.steps
end

module R = struct
  exception Malformed of int * string

  type r = { s : string; mutable at : int }

  let fail r fmt =
    Printf.ksprintf (fun why -> raise (Malformed (r.at, why))) fmt

  let left r = String.length r.s - r.at

  let int r =
    if left r < 8 then fail r "an integer past the end";
    let x = String.get_int64_le r.s r.at in
    let n = Int64.to_int x in
    if Int64.of_int n <> x then fail r "an integer of more than 63 bits";
    r.at <- r.at + 8;
    n

  let tag r =
    if left r < 1 then fail r "a tag past the end";
    let n = String.get_uint8 r.s r.at in
    r.at <- r.at + 1;
    n

  (* A count of elements, each of at least one byte. *)
  let count r =
    let n = int r in
    if n < 0 || n > left r then fail r "a count of %d" n;
    n

  let string r =
    let n = count r in
    let s = String.sub r.s r.at n in
    r.at <- r.at + n;
    s

  let array f r : _ iarray =
    let n = count r in
    if n = 0 then [||]
    else begin
      let a = Array.make n (f r) in
      for i = 1 to n - 1 do
        a.(i) <- f r
      done;
      Iarray.of_array a
    end

  let option f r =
    match tag r with
    | 0 -> None
    | 1 -> Some (f r)
    | n -> fail r "an option of tag %d" n

  let bad r what n = fail r "%s of tag %d" what n

  let leaf r =
    match tag r with
    | 0 ->
        let memory = int r in
        let on = int r in
        Address { memory; on }
    | 1 -> Handle (int r)
    | 2 ->
        let image = int r in
        let name = string r in
        Entry { image; name }
    | 3 -> Ready (int r)
    | n -> bad r "a leaf" n

  let value r =
    match tag r with
    | 0 -> Fixed (int r)
    | 1 -> Int (int r)
    | 2 -> Leaf (leaf r)
    | n -> bad r "a value" n

  let width r = match tag r with 0 -> W32 | 1 -> W64 | n -> bad r "a width" n

  let terms f r =
    array
      (fun r ->
        let x = f r in
        let factor = int r in
        (x, factor))
      r

  let data f r =
    let bytes = string r in
    let holes =
      array
        (fun r ->
          let at = int r in
          let width = width r in
          let terms = terms f r in
          let add = int r in
          let shift = int r in
          { at; width; terms; add; shift })
        r
    in
    { bytes; holes }

  let memory_kind r : B.kind =
    match tag r with
    | 0 -> Device
    | 1 -> Pinned
    | 2 -> Mapped
    | n -> bad r "a memory kind" n

  let access r : Rig.access =
    match tag r with 0 -> Read | 1 -> Read_write | n -> bad r "an access" n

  let ref r =
    let slot = int r in
    let at = int r in
    let use =
      match tag r with
      | 0 -> Read
      | 1 -> Write
      | 2 -> Unordered
      | n -> bad r "a use" n
    in
    { slot; at; use }

  let copies r = match tag r with 0 -> One | 1 -> Two | n -> bad r "copies" n

  let memory r =
    match tag r with
    | 0 ->
        let device = int r in
        let kind = memory_kind r in
        let bytes = int r in
        let init = data leaf r in
        let copies = copies r in
        Alloc { device; kind; bytes; init; copies }
    | 1 ->
        let rail = int r in
        let area =
          match tag r with
          | 0 -> Outbound
          | 1 -> Inbound
          | 2 -> Counts
          | n -> bad r "an area" n
        in
        Rail { rail; area }
    | n -> bad r "a memory" n

  let view r =
    let memory = int r in
    let offset = int r in
    let length = int r in
    { memory; offset; length }

  let slot r =
    match tag r with
    | 0 -> Memory (view r)
    | 1 -> Input (int r)
    | 2 -> Ints
    | n -> bad r "a slot" n

  let triple r =
    let x = value r in
    let y = value r in
    let z = value r in
    (x, y, z)

  let part r =
    match tag r with
    | 0 -> Words (view r)
    | 1 ->
        let fill = leaf r in
        let params = data value r in
        let refs = array ref r in
        Fill { fill; params; refs }
    | 2 ->
        let src = int r in
        let src_offset = terms value r in
        let dst = int r in
        let dst_offset = terms value r in
        let length = int r in
        Copy { src; src_offset; dst; dst_offset; length }
    | 3 ->
        let image = int r in
        let name = string r in
        let params = data value r in
        let refs = array ref r in
        let groups = triple r in
        let threads = triple r in
        let shared = value r in
        Launch { image; name; params; refs; groups; threads; shared }
    | n -> bad r "a part" n

  let rec step r =
    match tag r with
    | 0 ->
        let device = int r in
        let parts = array part r in
        let slots = array slot r in
        let fixed =
          array
            (fun r ->
              let v = view r in
              let a = access r in
              (v, a))
            r
        in
        Submit { device; parts; slots; fixed }
    | 1 ->
        let src = slot r in
        let dst = slot r in
        Move { src; dst }
    | 2 ->
        let trips = value r in
        let trip = option int r in
        let flag =
          option
            (fun r ->
              let s = slot r in
              let at = int r in
              (s, at))
            r
        in
        let body = array step r in
        Loop { trips; trip; flag; body }
    | n -> bad r "a step" n

  let t r =
    let devices = array string r in
    let memory = array memory r in
    let images =
      array
        (fun r ->
          let device = int r in
          let binary = data leaf r in
          { device; binary })
        r
    in
    let inputs =
      array
        (fun r ->
          let device = int r in
          let bytes = int r in
          { device; bytes })
        r
    in
    let ints = int r in
    let steps = array step r in
    { devices; memory; images; inputs; ints; steps }
end

let write_description b t =
  Buffer.add_string b magic;
  W.int b version;
  W.t b t

let to_string t =
  let b = Buffer.create 4096 in
  write_description b t;
  Buffer.contents b

let of_string s =
  let r = { R.s; at = 0 } in
  let m = String.length magic in
  if String.length s < m || String.sub s 0 m <> magic then
    Error "not a program description"
  else begin
    r.R.at <- m;
    let malformed at why =
      Error (strf "a malformed program description at byte %d: %s" at why)
    in
    match R.int r with
    | exception R.Malformed (at, why) -> malformed at why
    | v when v <> version ->
        Error
          (strf "a program description of format version %d, not %d" v version)
    | _ -> (
        match R.t r with
        | t when R.left r = 0 -> Ok t
        | _ -> Error (strf "%d bytes after a program description" (R.left r))
        | exception R.Malformed (at, why) -> malformed at why)
  end

(* Loading *)

(* A value once loaded, for one copy: a leaf is resolved at load, so a run meets
   no refusal of it. *)
type known = Known of int | Word of int (* Of the run's ints. *)

(* A block's words that a run writes into its block before each submit: its
   parameters' holes over the run's values, and its geometry. *)
type block_run = {
  block : Sub.block;
  params : string;
  holes : known hole array;
  geometry : ((known * known * known) * (known * known * known) * known) option;
      (* A launch's geometry where a run changes it. *)
}

(* A copy's end whose buffer each run passes: [count] bytes of the slot
   [from], at the offset [offset]'s terms sum to. *)
type moving = {
  index : int; (* Of [sub]'s slots. *)
  from : slot;
  held : B.t; (* [from]'s buffer, or [unset] for an input. *)
  offset : (known * int) array;
  count : int;
}

(* A step's submission for one copy, its run, and the buffers it is passed:
   memory set at load, inputs and moving copy ends at each run. *)
type prepared = {
  sub : Sub.t;
  srun : Sub.Run.t;
  buffers : B.t array;
  input_slots : (int * int) array;
      (* Each slot of [sub] that holds an input, with the input. *)
  moving : moving array;
  blocks : block_run array;
}

type lstep =
  | Lsubmit of { spec : submit; copies : prepared array }
  | Lmove of { src : slot; dst : slot }
  | Lloop of {
      trips : known array; (* By copy. *)
      trip : int option;
      flag : (slot * int) option;
      body : lstep array;
    }

type frame = { inputs : B.t array; ints : int array }

(* Whether each of [s]'s slots is written: a [Write] or an [Unordered] ref
   names it, or it is a copy's [dst]. [s]'s indices are checked. *)
let written (s : submit) =
  let w = Array.make (Iarray.length s.slots) false in
  Iarray.iter
    (function
      | Copy { dst; _ } -> w.(dst) <- true
      | p ->
          Iarray.iter
            (fun r -> if r.use <> Read then w.(r.slot) <- true)
            (refs_of p))
    s.parts;
  w

(* Each input's access, as its steps use it: [Read_write] where a step writes
   it, a [Submit] slot written or a [Move]'s destination; [Read] otherwise.
   [t]'s indices are checked. *)
let input_access (t : t) =
  let a = Array.make (Iarray.length t.inputs) Rig.Read in
  let write = function
    | Input i -> a.(i) <- Read_write
    | Memory _ | Ints -> ()
  in
  let rec step = function
    | Submit s ->
        let w = written s in
        Iarray.iteri (fun i slot -> if w.(i) then write slot) s.slots
    | Move { dst; _ } -> write dst
    | Loop { body; _ } -> Iarray.iter step body
  in
  Iarray.iter step t.steps;
  a

(* An input as a run binds it on device [on]. *)
type bind = { input : int; on : int }

(* What a run binds: each input on its own device, first, then on each other
   device that names it through a Submit slot. [t]'s indices are checked. *)
let binds (t : t) =
  let nd = Iarray.length t.devices and n = Iarray.length t.inputs in
  let named = Array.make (n * nd) false in
  let slot f = function Input i -> f i | Memory _ | Ints -> () in
  let rec step = function
    | Submit s ->
        Iarray.iter
          (slot (fun i -> named.((i * nd) + s.device) <- true))
          s.slots
    | Move _ -> ()
    | Loop { body; _ } -> Iarray.iter step body
  in
  Iarray.iter step t.steps;
  let bind i on = { input = i; on }
  and own i = (Iarray.get t.inputs i).device in
  let others = ref [] in
  for i = n - 1 downto 0 do
    for d = nd - 1 downto 0 do
      if d <> own i && named.((i * nd) + d) then
        others := bind i d :: !others
    done
  done;
  Array.append
    (Array.init n (fun i -> bind i (own i)))
    (Array.of_list !others)

(* Each launch of [t], in the order of its steps, a loop's body in place: its
   image, its function and its threads. *)
let launches (t : t) =
  let acc = ref [] in
  let part = function
    | Launch { image; name; threads; _ } ->
        acc := (image, name, threads) :: !acc
    | Words _ | Fill _ | Copy _ -> ()
  in
  let rec step = function
    | Submit s -> Iarray.iter part s.parts
    | Loop { body; _ } -> Iarray.iter step body
    | Move _ -> ()
  in
  Iarray.iter step t.steps;
  Array.of_list (List.rev !acc)

(* The words of the run's ints that [t]'s steps may write: [None] for any, as a
   [Submit] slot written or a [Move]'s destination may; else the loops' trip
   words. *)
let ints_written (t : t) =
  let any = ref false and trips = ref [] in
  let ints = function Ints -> any := true | Memory _ | Input _ -> () in
  let rec step = function
    | Submit s ->
        let w = written s in
        Iarray.iteri (fun i slot -> if w.(i) then ints slot) s.slots
    | Move { dst; _ } -> ints dst
    | Loop { trip; body; _ } ->
        Option.iter (fun w -> trips := w :: !trips) trip;
        Iarray.iter step body
  in
  Iarray.iter step t.steps;
  if !any then None else Some !trips

(* A launch whose threads a run reads from the frame's ints, which no step
   writes, with the most threads a group of its function has: a run on another
   machine checks it at the call. *)
type watched = { kernel : string; sizes : value * value * value; most : int }

(* The launches of [t] to watch, [limits] giving each launch's most threads. *)
let watched (t : t) limits =
  let ls = launches t in
  if Array.length limits <> Array.length ls then
    refuse "the agent states the threads of %d launches, the program has %d"
      (Array.length limits) (Array.length ls);
  match ints_written t with
  | None -> [||]
  | Some trips ->
      let known = function
        | Fixed _ -> true
        | Int i -> not (List.mem i trips)
        | Leaf _ -> false
      in
      let read = function Int _ -> true | Fixed _ | Leaf _ -> false in
      let watch k (_, kernel, ((x, y, z) as sizes)) =
        match limits.(k) with
        | Some most
          when List.for_all known [ x; y; z ] && List.exists read [ x; y; z ]
          ->
            Some { kernel; sizes; most }
        | Some _ | None -> None
      in
      Array.of_list (List.filter_map Fun.id (List.mapi watch (Array.to_list ls)))

(* A program loaded on this machine. *)
type here = {
  t : t;
  devices : Rig.t array;
  mem : B.t array array;
      (* Each memory's copies, on its device, then the run's ints. *)
  ints : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t array;
      (* Each copy of the run's ints, as words. *)
  given : int array;
      (* The frame's ints, its first [ngiven] words, as the run read them
         before it waits for run [n - 2]. *)
  mutable ngiven : int;
  borrows : (int * int * int, B.t) Hashtbl.t;
      (* By memory, copy and device: the borrows made at load. *)
  fixes : (int * int * int, Sub.t) Hashtbl.t;
      (* A submission fixing each whole memory, copy and device. *)
  mutable images : Rig.Image.t array; (* Those loaded so far, while loading. *)
  rails : int -> Rig_remote_abi.end_ option;
  kept : Rig.Image.t array Stdlib.ref * Rig_remote_abi.end_ list Stdlib.ref;
      (* What every submission keeps until its work is done: the images, and
         the rail ends [rails] gave, whose functions and addresses its work
         may name on any device. *)
  mutable steps : lstep array;
  twos : B.t array array; (* Copy [k] of each memory of two copies. *)
  flag : B.t; (* The byte a loop's flag is read into. *)
  ran : bool array; (* By device: whether the run submitted there. *)
  mutable last : Rig.Point.t array;
      (* By device: the run's last point, where [ran]; empty before the first
         submit. *)
  access : Rig.access array; (* By input, as its steps use it. *)
  binds : bind array;
  bound : B.t array;
      (* By input [i] and device [d], at [i * devices + d]: a run's buffer of
         each of its binds, and [unset] between runs. *)
  lock : Mutex.t;
  mutable runs : int;
}

(* This machine's end of the rail [id]. *)
let rail_end rails id =
  match rails id with
  | Some e -> e
  | None -> refuse "no rail %d on this machine" id

let rail_area (e : Rig_remote_abi.end_) = function
  | Outbound -> e.outbound
  | Inbound -> e.inbound
  | Counts -> e.counts

(* Memory [m]'s copy [k] as device [d] names it. *)
let on p m k d =
  let b = p.mem.(m).(k) and dev = p.devices.(d) in
  if Rig.equal (B.device b) dev then b
  else
    match Hashtbl.find_opt p.borrows (m, k, d) with
    | Some b -> b
    | None -> (
        match B.borrow dev b with
        | Some b' ->
            Hashtbl.replace p.borrows (m, k, d) b';
            b'
        | None -> refuse "memory %d: %s cannot borrow it" m (Rig.name dev))

let copy_of p m k = if Array.length p.mem.(m) = 2 then k else 0

let view p k d (v : view) =
  B.view
    (on p v.memory (copy_of p v.memory k) d)
    ~first:v.offset ~length:v.length

(* The value of [l] for copy [k]. *)
let leaf_value p k = function
  | Address { memory; on = d } -> (
      let b = on p memory (copy_of p memory k) d in
      try
        Sub.address (Hashtbl.find p.fixes (memory, copy_of p memory k, d)) b
      with
      | Not_found ->
          refuse "memory %d: no step fixes it whole on device %d" memory d
      | Invalid_argument _ ->
          refuse "memory %d: named by its handle only, it has no address" memory
      )
  | Handle m -> Nativeint.to_int (B.handle p.mem.(m).(copy_of p m k))
  | Entry { image; name } -> (
      let i = p.images.(image) in
      match Rig.Image.entry i name with
      | Some e -> e
      | None ->
          refuse "%s: image %d has no function %S"
            (Rig.name (Rig.Image.device i))
            image name
      | exception Invalid_argument why -> refuse "%s" why)
  | Ready r -> Nativeint.to_int (rail_end p.rails r).ready_fn

(* A hole whose value meets set bits of its bytes, at its byte: the value
   would clobber a constant the description holds there. *)
exception Clobbers of int

(* The sum hole [h] writes before its shift, [value] giving its terms'
   values. *)
let hole_sum h value =
  Iarray.fold_left (fun v (x, factor) -> v + (value x * factor)) h.add h.terms

(* The word hole [h] makes of the sum [v] over [word], the word of its bytes
   there: the sum's 64-bit word, shifted, its low bits ORed into it. *)
let hole_word h v word =
  let w = Int64.shift_right_logical (Int64.of_int v) h.shift in
  let w = match h.width with W32 -> Int64.logand w 0xffff_ffffL | W64 -> w in
  if Int64.logand word w <> 0L then raise (Clobbers h.at);
  Int64.logor word w

let get_word b h =
  match h.width with
  | W32 ->
      Int64.logand (Int64.of_int32 (String.get_int32_le b h.at)) 0xffff_ffffL
  | W64 -> String.get_int64_le b h.at

(* [d]'s bytes with its holes filled for copy [k]. *)
let filled p k (d : leaf data) =
  let b = Bytes.of_string d.bytes in
  let fill h =
    let v = hole_sum h (leaf_value p k) in
    let w = hole_word h v (get_word d.bytes h) in
    match h.width with
    | W32 -> Bytes.set_int32_le b h.at (Int64.to_int32 w)
    | W64 -> Bytes.set_int64_le b h.at w
  in
  (try Iarray.iter fill d.holes
   with Clobbers at ->
     refuse "a hole at byte %d: its value meets set bits of the bytes" at);
  Bytes.unsafe_to_string b

(* Whether [s] names memory of [Two] copies, so that it is made once per
   copy. *)
let per_copy t (s : submit) =
  let view (v : view) = two t v.memory in
  let slot = function Memory v -> view v | Ints -> true | Input _ -> false in
  let value : value -> bool = function
    | Leaf l -> leaf_two t l
    | Fixed _ | Int _ -> false
  in
  let holes =
    Iarray.exists (fun h -> Iarray.exists (fun (x, _) -> value x) h.terms)
  in
  let part = function
    | Words v -> view v
    | Fill { fill; params; _ } -> leaf_two t fill || holes params.holes
    | Copy _ -> false
    | Launch { params; groups = gx, gy, gz; threads = tx, ty, tz; shared; _ } ->
        holes params.holes
        || List.exists value [ gx; gy; gz; tx; ty; tz; shared ]
  in
  Iarray.exists part s.parts || Iarray.exists slot s.slots
  || Iarray.exists (fun (v, _) -> view v) s.fixed

let per_run = function Word _ -> true | Known _ -> false
let hole_per_run h = Iarray.exists (fun (x, _) -> per_run x) h.terms

(* Stores hole [h] of a launch's parameters [params] over its sum [v] into its
   block [b]. A 64-bit word goes as two 32-bit halves: an [int] setter would
   drop its bit 63. *)
let store_hole r b params h v =
  let w = hole_word h v (get_word params h) in
  Sub.Run.int32 r b h.at (Int64.to_int (Int64.logand w 0xffff_ffffL));
  if h.width = W64 then
    Sub.Run.int32 r b (h.at + 4) (Int64.to_int (Int64.shift_right_logical w 32))

let store_geometry r b (gx, gy, gz) (tx, ty, tz) shared value =
  Sub.Run.groups r b (value gx) (value gy) (value gz);
  Sub.Run.threads r b (value tx) (value ty) (value tz);
  Sub.Run.shared r b (value shared)

(* The buffer an input's slot holds until a run passes the input. *)
let unset = B.of_string ""

(* [v] for copy [k], its leaf resolved. *)
let resolve p k : value -> known = function
  | Fixed n -> Known n
  | Leaf l -> Known (leaf_value p k l)
  | Int i -> Word i

(* The slot of [s]'s submission for each of [s]'s slots that a ref names, in
   order, or [-1]; and the count of those slots. A copy's end takes a slot of
   its own after them unless it is a buffer set at make. *)
let sub_slots (s : submit) =
  let n = Iarray.length s.slots in
  let reffed = Array.make n false in
  Iarray.iter
    (fun p -> Iarray.iter (fun r -> reffed.(r.slot) <- true) (refs_of p))
    s.parts;
  let index = Array.make n (-1) and next = ref 0 in
  for i = 0 to n - 1 do
    if reffed.(i) then begin
      index.(i) <- !next;
      incr next
    end
  done;
  (index, !next)

(* The submission of [s] for copy [k], and its run holding the words of its
   blocks that no run changes. *)
let prepared p k (s : submit) =
  let d = s.device in
  let known = resolve p k in
  let index, reffed = sub_slots s in
  let next = ref reffed and moving = ref [] in
  let sub_ref r =
    let slot = index.(r.slot) in
    match r.use with
    | Read -> Sub.read slot ~at:r.at
    | Write -> Sub.write slot ~at:r.at
    | Unordered -> Sub.unordered slot ~at:r.at
  in
  let refs rs = Array.map sub_ref (Iarray.to_array rs) in
  let part = function
    | Words v -> Sub.words (view p k d v)
    | Fill { fill; params; refs = rs } ->
        let fill = Nativeint.of_int (leaf_value p k fill) in
        Sub.fill ~refs:(refs rs) fill ~params:(String.length params.bytes)
    | Copy { src; src_offset; dst; dst_offset; length } ->
        (* An end of memory at an offset load knows is that memory's bytes,
           set at make; any other end is a slot each run passes. *)
        let side i offset =
          let terms =
            Iarray.to_array (Iarray.map (fun (x, f) -> (known x, f)) offset)
          in
          match (Iarray.get s.slots i, fixed_offset offset) with
          | Memory v, Some o ->
              Sub.Buffer (B.view (view p k d v) ~first:o ~length)
          | slot, _ ->
              let held =
                match slot with
                | Memory v -> view p k d v
                | Ints -> view p k d (ints_view p.t)
                | Input _ -> unset
              in
              let j = !next in
              incr next;
              moving :=
                { index = j; from = slot; held; offset = terms; count = length }
                :: !moving;
              Sub.Slot j
        in
        let src = side src src_offset in
        Sub.copy ~src ~dst:(side dst dst_offset)
    | Launch { image; name; params; refs = rs; _ } ->
        Sub.launch ~refs:(refs rs) p.images.(image) name
          ~params:(String.length params.bytes)
  in
  let parts = Iarray.to_array (Iarray.map part s.parts) in
  let fixed =
    Iarray.to_list (Iarray.map (fun (v, a) -> (view p k d v, a)) s.fixed)
  in
  let sub = Sub.make ~keep:p.kept ~fixed p.devices.(d) parts in
  Iarray.iter
    (fun ((v : view), _) ->
      if
        v.offset = 0
        && v.length = B.length p.mem.(v.memory).(copy_of p v.memory k)
      then Hashtbl.replace p.fixes (v.memory, copy_of p v.memory k, d) sub)
    s.fixed;
  let buffers = Array.make !next unset and inputs = ref [] in
  Iarray.iteri
    (fun i slot ->
      let j = index.(i) in
      if j >= 0 then
        match slot with
        | Input input -> inputs := (j, input) :: !inputs
        | Memory v -> buffers.(j) <- view p k d v
        | Ints -> buffers.(j) <- view p k d (ints_view p.t))
    s.slots;
  let srun = Sub.Run.make () in
  let value = function Known n -> n | Word _ -> 0 in
  let block i = function
    | Words _ | Copy _ -> None
    | (Launch { params; _ } | Fill { params; _ }) as work ->
        let b = Sub.block sub i in
        for w = 0 to (String.length params.bytes / 4) - 1 do
          Sub.Run.int32 srun b (4 * w)
            (Int32.to_int (String.get_int32_le params.bytes (4 * w)))
        done;
        let holes =
          Iarray.map
            (fun h ->
              let terms = Iarray.map (fun (x, f) -> (known x, f)) h.terms in
              { h with terms })
            params.holes
        in
        Iarray.iter
          (fun h ->
            if not (hole_per_run h) then
              store_hole srun b params.bytes h (hole_sum h value))
          holes;
        let geometry =
          match work with
          | Launch { groups = gx, gy, gz; threads = tx, ty, tz; shared; _ } ->
              let groups = (known gx, known gy, known gz)
              and threads = (known tx, known ty, known tz)
              and shared = known shared in
              store_geometry srun b groups threads shared value;
              let gx, gy, gz = groups and tx, ty, tz = threads in
              if List.exists per_run [ gx; gy; gz; tx; ty; tz; shared ] then
                Some (groups, threads, shared)
              else None
          | Fill _ -> None
          | Words _ | Copy _ -> assert false
        in
        Some
          {
            block = b;
            params = params.bytes;
            holes =
              Array.of_list
                (List.filter hole_per_run (Iarray.to_list holes));
            geometry;
          }
  in
  {
    sub;
    srun;
    buffers;
    input_slots = Array.of_list !inputs;
    moving = Array.of_list (List.rev !moving);
    blocks =
      Array.of_list
        (List.filter_map Fun.id (Iarray.to_list (Iarray.mapi block s.parts)));
  }

(* A submission [Rig.Submission.make] or a setter refuses is a defect of the
   description. *)
let prepare p i k s =
  try prepared p k s with
  | Invalid_argument why -> refuse "step %d: %s" i why
  | Clobbers at ->
      refuse "step %d: a parameter hole at byte %d meets set bits of its bytes"
        i at

let rec lstep p i = function
  | Submit s ->
      let n = if per_copy p.t s then 2 else 1 in
      Lsubmit { spec = s; copies = Array.init n (fun k -> prepare p i k s) }
  | Move { src; dst } -> Lmove { src; dst }
  | Loop { trips; trip; flag; body } ->
      let trips = Array.init 2 (fun k -> resolve p k trips) in
      Lloop
        {
          trips;
          trip;
          flag;
          body = Iarray.to_array (Iarray.map (lstep p i) body);
        }

let make_memory devices rails = function
  | Alloc { device; kind; bytes; copies; _ } ->
      let n = match copies with One -> 1 | Two -> 2 in
      Array.init n (fun _ -> B.create ~memory:kind devices.(device) bytes)
  | Rail { rail; area } ->
      [| B.of_bigarray (rail_area (rail_end rails rail) area) |]

let load_image p (im : image) =
  match Rig.Image.load p.devices.(im.device) (filled p 0 im.binary) with
  | Ok i -> i
  | Error why -> raise (Refused why)

let write_init p m k =
  match Iarray.get p.t.memory m with
  | Rail _ -> ()
  | Alloc { init; _ } ->
      let n = String.length init.bytes in
      if n > 0 then
        B.copy
          ~src:(B.of_string (filled p k init))
          ~dst:(B.view p.mem.(m).(k) ~first:0 ~length:n)

let words b =
  if B.length b = 0 then Bigarray.(Array1.create int64 c_layout 0)
  else B.export b Bigarray.int64

(* A device borrows host memory that starts on a page, as a host buffer of 64
   KiB or more does (rig.mli, [Buffer.create]). *)
let page_bytes = 65536

let rec names_ints = function
  | Submit s ->
      let ints = function Ints -> true | Memory _ | Input _ -> false in
      Iarray.exists ints s.slots
  | Move _ -> false
  | Loop { body; _ } -> Iarray.exists names_ints body

(* One copy of the run's ints: a view of a buffer a device can borrow where a
   step names them. *)
let make_ints (t : t) =
  let n = 8 * t.ints in
  if Iarray.exists names_ints t.steps then
    B.view (B.create Rig.host (max n page_bytes)) ~first:0 ~length:n
  else B.create Rig.host n

let load_here ~rails t devices =
  (* Every end [rails] gives is kept with the images: a rail ends once its end
     is unreachable. *)
  let ends = ref [] in
  let rails id =
    let e = rails id in
    Option.iter (fun e -> ends := e :: !ends) e;
    e
  in
  let size = function
    | Alloc { bytes; _ } -> Some bytes
    | Rail { rail; area } ->
        Some (Bigarray.Array1.dim (rail_area (rail_end rails rail) area))
  in
  match check t (Iarray.to_array (Iarray.map size t.memory)) devices with
  | exception Refused why -> Error why
  | () -> (
      let ints = Array.init 2 (fun _ -> make_ints t) in
      let mem =
        Array.append
          (Iarray.to_array (Iarray.map (make_memory devices rails) t.memory))
          [| ints |]
      in
      let twos k =
        Array.to_list mem
        |> List.filter (fun c -> Array.length c = 2)
        |> List.map (fun c -> c.(k))
        |> Array.of_list
      in
      let pairs = Iarray.length t.inputs * Array.length devices in
      let images = ref [||] in
      let p =
        {
          t;
          devices;
          mem;
          ints = Array.map words ints;
          given = Array.make t.ints 0;
          ngiven = 0;
          borrows = Hashtbl.create 8;
          fixes = Hashtbl.create 8;
          images = [||];
          rails;
          kept = (images, ends);
          steps = [||];
          twos = [| twos 0; twos 1 |];
          flag = B.create Rig.host 1;
          ran = Array.make (Array.length devices) false;
          last = [||];
          access = input_access t;
          binds = binds t;
          bound = Array.make pairs unset;
          lock = Mutex.create ();
          runs = 0;
        }
      in
      match
        (* An image's holes name earlier images' entries: each loads once those
           before it did. *)
        Iarray.iter
          (fun im -> p.images <- Array.append p.images [| load_image p im |])
          t.images;
        images := p.images;
        p.steps <- Iarray.to_array (Iarray.mapi (lstep p) t.steps);
        Iarray.iteri
          (fun m _ -> Array.iteri (fun k _ -> write_init p m k) mem.(m))
          t.memory
      with
      | () -> Ok p
      | exception Refused why -> Error why)

(* Running *)

let invalid fmt = Printf.ksprintf invalid_arg ("Rig_program.run: " ^^ fmt)

(* The frame's counts: as many inputs as [t]'s, at most [t]'s ints. *)
let check_counts (t : t) (f : frame) =
  let n = Iarray.length t.inputs in
  if Array.length f.inputs <> n then
    invalid "%d inputs for a program of %d" (Array.length f.inputs) n;
  if Array.length f.ints > t.ints then
    invalid "%d ints for a program of %d" (Array.length f.ints) t.ints

(* [b], input [i] of [spec], on [spec]'s device [d] and holding its bytes. *)
let check_input i (spec : input) d b =
  if not (Rig.equal (B.device b) d) then
    invalid "input %d is on %s, not %s" i (Rig.name (B.device b)) (Rig.name d);
  if B.length b < spec.bytes then
    invalid "input %d holds %d bytes, fewer than %d" i (B.length b) spec.bytes

(* The run's buffer of input [i] as device [d] names it. *)
let bound (p : here) i d = p.bound.((i * Array.length p.devices) + d)

(* Binds [f] for the run: copies its ints, and reads each input once, checks
   it, and stores it with its borrow on each other device that names it. A
   frame the run refuses raises here, before its first step, and a change to
   [f] during the run changes nothing. *)
let bind (p : here) (f : frame) =
  check_counts p.t f;
  let n = Array.length f.ints in
  Array.blit f.ints 0 p.given 0 n;
  p.ngiven <- n;
  let nd = Array.length p.devices in
  for j = 0 to Array.length p.binds - 1 do
    let { input = i; on } = p.binds.(j) in
    let spec = Iarray.get p.t.inputs i in
    let b =
      if on = spec.device then begin
        let b = f.inputs.(i) in
        check_input i spec p.devices.(on) b;
        if p.access.(i) = Rig.Read_write && B.access b = Rig.Read then
          invalid "input %d is read-only memory, which the program writes" i;
        b
      end
      else
        let dev = p.devices.(on) in
        match B.borrow dev (bound p i spec.device) with
        | Some b -> b
        | None -> invalid "input %d: %s cannot borrow it" i (Rig.name dev)
    in
    p.bound.((i * nd) + on) <- b
  done

(* Drops the run's inputs, so a loaded program keeps no caller's buffer past its
   run. *)
let unbind (p : here) =
  let nd = Array.length p.devices in
  for j = 0 to Array.length p.binds - 1 do
    let { input; on; _ } = p.binds.(j) in
    p.bound.((input * nd) + on) <- unset
  done

(* The host reads and writes the ints as it does any rig memory: after the work
   its access must follow, which a step that names them may still run. *)
let ints_buffer (p : here) k = p.mem.(Iarray.length p.t.memory).(k)

let int_at p k i =
  B.wait (ints_buffer p k) Rig.Read;
  Int64.to_int (Bigarray.Array1.unsafe_get p.ints.(k) i)

let set_trip p k w i =
  B.wait (ints_buffer p k) Rig.Read_write;
  Bigarray.Array1.unsafe_set p.ints.(k) w (Int64.of_int i)

let run_value (p : here) k = function Known n -> n | Word i -> int_at p k i

let pass p q d =
  for x = 0 to Array.length q.input_slots - 1 do
    let i, j = q.input_slots.(x) in
    q.buffers.(i) <- bound p j d
  done

(* Drops the inputs and copy ends [q]'s buffers held, so a loaded program
   keeps no caller's buffer past its run. *)
let unpass q =
  for x = 0 to Array.length q.input_slots - 1 do
    q.buffers.(fst q.input_slots.(x)) <- unset
  done;
  for x = 0 to Array.length q.moving - 1 do
    q.buffers.(q.moving.(x).index) <- unset
  done

let mem_buffer p k (v : view) =
  B.view
    p.mem.(v.memory).(copy_of p v.memory k)
    ~first:v.offset ~length:v.length

let slot_buffer p k = function
  | Input j -> bound p j (Iarray.get p.t.inputs j).device
  | Memory v -> mem_buffer p k v
  | Ints -> mem_buffer p k (ints_view p.t)

let submit_step p k waits (spec : submit) copies =
  let q = copies.(if Array.length copies = 2 then k else 0) in
  let d = spec.device in
  (try
     for i = 0 to Array.length q.blocks - 1 do
       let l = q.blocks.(i) in
       for j = 0 to Array.length l.holes - 1 do
         let h = l.holes.(j) in
         let v = ref h.add in
         for t = 0 to Iarray.length h.terms - 1 do
           let x, factor = Iarray.get h.terms t in
           v := !v + (run_value p k x * factor)
         done;
         store_hole q.srun l.block l.params h !v
       done;
       match l.geometry with
       | None -> ()
       | Some (groups, threads, shared) ->
           store_geometry q.srun l.block groups threads shared (run_value p k)
     done
   with Clobbers at ->
     invalid
       "a parameter hole at byte %d: its value meets set bits of its bytes" at);
  pass p q d;
  for x = 0 to Array.length q.moving - 1 do
    let e = q.moving.(x) in
    let held = match e.from with Input j -> bound p j d | _ -> e.held in
    let o = ref 0 in
    (try
       for t = 0 to Array.length e.offset - 1 do
         let v, factor = e.offset.(t) in
         o := add_exn !o (mul_exn (run_value p k v) factor)
       done
     with Overflow ->
       unpass q;
       invalid "a copy's offset past an int");
    let o = !o and n = B.length held in
    if o < 0 || o > n - e.count then begin
      unpass q;
      invalid "a copy of %d bytes from byte %d of a slot of %d bytes" e.count o
        n
    end;
    q.buffers.(e.index) <-
      (if o = 0 && e.count = n then held
       else B.view held ~first:o ~length:e.count)
  done;
  let waits = if p.ran.(d) then None else waits in
  match Rig.submit ?waits q.sub ~run:q.srun ~buffers:q.buffers with
  | point ->
      unpass q;
      if Array.length p.last = 0 then
        p.last <- Array.make (Array.length p.devices) point;
      p.last.(d) <- point;
      p.ran.(d) <- true
  | exception e ->
      unpass q;
      raise e

(* Whether byte [at] of [slot]'s buffer is not [0], once the work that wrote
   it is done. *)
let flag_holds p k (slot, at) =
  let b, first =
    match slot with
    | Memory v -> (p.mem.(v.memory).(copy_of p v.memory k), v.offset + at)
    | Ints -> (ints_buffer p k, at)
    | Input j -> (bound p j (Iarray.get p.t.inputs j).device, at)
  in
  if first >= B.length b then
    invalid "a loop's flag at byte %d of an input of %d bytes" at
      (B.length b);
  B.copy ~src:(B.view b ~first ~length:1) ~dst:p.flag;
  let c = Bytes.create 1 in
  B.blit_to_bytes p.flag 0 c 0 1;
  Bytes.get c 0 <> '\000'

let rec exec (p : here) k waits = function
  | Lmove { src; dst } ->
      B.copy ~src:(slot_buffer p k src) ~dst:(slot_buffer p k dst)
  | Lsubmit { spec; copies } -> submit_step p k waits spec copies
  | Lloop { trips; trip; flag; body } ->
      let n = run_value p k trips.(k) in
      let rec go i =
        if i < n && Option.fold ~none:true ~some:(flag_holds p k) flag then begin
          Option.iter (fun w -> set_trip p k w i) trip;
          for s = 0 to Array.length body - 1 do
            exec p k waits body.(s)
          done;
          go (i + 1)
        end
      in
      go 0

(* The run's last point on each device it submitted to, in the devices'
   order. *)
let points (p : here) =
  let n = ref 0 in
  Array.iter (fun r -> if r then incr n) p.ran;
  if !n = 0 then [||]
  else begin
    let a = Array.make !n p.last.(0) and j = ref 0 in
    for d = 0 to Array.length p.ran - 1 do
      if p.ran.(d) then begin
        a.(!j) <- p.last.(d);
        incr j
      end
    done;
    a
  end

let run_steps ~waits (p : here) (f : frame) =
  bind p f;
  let k = p.runs land 1 in
  p.runs <- p.runs + 1;
  let twos = p.twos.(k) in
  for i = 0 to Array.length twos - 1 do
    B.wait twos.(i) Rig.Read_write
  done;
  let ints = p.ints.(k) in
  for i = 0 to p.ngiven - 1 do
    Bigarray.Array1.set ints i (Int64.of_int p.given.(i))
  done;
  Array.fill p.ran 0 (Array.length p.ran) false;
  for s = 0 to Array.length p.steps - 1 do
    exec p k waits p.steps.(s)
  done;
  points p

(* Re-entry

   A call of rig's from a signal handler, finaliser or Gc.Memprof callback
   inside a lock of its thread's raises as it starts. A load or a run checks
   first, since its later calls of rig would raise after it changed its
   state. A run holds its program's lock across its calls of rig, so that
   lock does not count among the thread's: a callback between them runs as
   any code does. *)

(* The locks the calling thread holds ([rig_holds] in [rig_edge.h]). *)
external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

let check_held fn =
  if holds () <> 0 then
    invalid_arg
      ("Rig_program." ^ fn
     ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
        inside a lock of the calling thread's")

(* Takes a program's lock, after [check_held]. A take by the thread that
   holds it, a run from a callback inside a run of the same program, raises
   having taken nothing. A signal handler that runs inside a contended take
   may raise a [Sys_error] of its own. *)
let lock m =
  check_held "run";
  match Mutex.lock m with
  | () -> ()
  | exception Sys_error w when String.starts_with ~prefix:"Mutex.lock:" w ->
      invalid_arg "Rig_program.run: called on the thread that runs it"

(* The lock without [Mutex.protect]'s closure: a run allocates only its
   answer. *)
let run_here ~waits (p : here) (f : frame) =
  lock p.lock;
  match run_steps ~waits p f with
  | points ->
      unbind p;
      Mutex.unlock p.lock;
      points
  | exception e ->
      unbind p;
      Mutex.unlock p.lock;
      raise e

(* Programs of another machine *)

(* The binary [load] sends another machine's host: the magic, then each device's
   id on that machine, then the description's bytes. A run's words: the
   program's entry, each input's memory id, offset and length, the number of the
   frame's ints, then the description's [ints] words, the frame's first. *)

let share_magic = "rig.share\n"

(* A program loaded on another machine, which a submission on its host runs. *)
type there = {
  t : t;
  devices : Rig.t array;
  access : Rig.access array;
  binds : bind array;
  limits : int option array; (* What the agent stated at load ([threads]). *)
  watch : watched array;
  entry : int;
  sub : Sub.t;
  srun : Sub.Run.t;
  words :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  lock : Mutex.t;
}

type loaded = Here of here | There of there

let words_bytes (t : t) = 8 + (24 * Iarray.length t.inputs) + 8 + (8 * t.ints)

(* [d]'s id on its machine, as its agent names it. *)
let proxy_id d =
  match Rig.capability d Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host _) -> 0
  | Some (Rig_remote_abi.Device d) -> d.Rig_remote_abi.id
  | None -> refuse "%s is no device of an agent" (Rig.name d)

let load_there t devices =
  match
    (* The areas of rails are the agent's to know: their views are checked
       there. *)
    let size = function Alloc { bytes; _ } -> Some bytes | Rail _ -> None in
    check t (Iarray.to_array (Iarray.map size t.memory)) devices;
    let inputs = Iarray.length t.inputs in
    if
      inputs > (max_int - 16) / 24
      || t.ints > (max_int - 16 - (24 * inputs)) / 8
    then
      refuse "%d inputs and %d ints: more than a run's words hold" inputs t.ints;
    let host = Rig.host_of devices.(0) in
    match Rig.capability host Rig_remote_abi.key with
    | Some (Rig_remote_abi.Host { threads; _ }) ->
        (Array.map proxy_id devices, threads)
    | Some (Rig_remote_abi.Device _) | None ->
        refuse "%s is no host of an agent" (Rig.name host)
  with
  | exception Refused why -> Error why
  | ids, threads -> (
      let b = Buffer.create 4096 in
      Buffer.add_string b share_magic;
      W.array W.int b (Iarray.of_array ids);
      write_description b t;
      let host = Rig.host_of devices.(0) in
      match Rig.Image.load host (Buffer.contents b) with
      | Error why -> Error why
      | Ok image -> (
          match Rig.Image.entry image "run" with
          | None -> Error (strf "%s: the program has no run" (Rig.name host))
          | Some entry -> (
              let limits = threads entry in
              match watched t limits with
              | exception Refused why -> Error why
              | watch ->
              let buffer = B.create Rig.host (words_bytes t) in
              let part = Sub.words buffer in
              (* The image stays loaded until the runs' work is done. *)
              let sub = Sub.make ~keep:image host [| part |] in
              Ok
                (There
                   {
                     t;
                     devices;
                     access = input_access t;
                     binds = binds t;
                     limits;
                     watch;
                     entry;
                     sub;
                     srun = Sub.Run.make ();
                     words = B.export buffer Bigarray.char;
                     lock = Mutex.create ();
                   }))))

let no_rails _ = None

let load ?rails t devices =
  check_held "load";
  let devices = Iarray.to_array devices in
  let here () =
    let rails = Option.value rails ~default:no_rails in
    Result.map (fun p -> Here p) (load_here ~rails t devices)
  in
  if Array.length devices = 0 then here ()
  else begin
    let h = Rig.host_of devices.(0) in
    Array.iter
      (fun d ->
        if not (Rig.equal (Rig.host_of d) h) then
          invalid_arg "Rig_program.load: devices of several machines")
      devices;
    if Rig.equal h Rig.host then here ()
    else if Option.is_some rails then
      invalid_arg "Rig_program.load: rails for another machine's devices"
    else load_there t devices
  end

(* Writes [v] at [at] as a little-endian int64, sign-extended as [R.int] reads
   it back: -1 is eight 0xff bytes. *)
let set_word words at v =
  Bigarray.Array1.(
    for i = 0 to 7 do
      unsafe_set words (at + i) (Char.unsafe_chr ((v asr (8 * i)) land 0xff))
    done)

let threads = function
  | Here p ->
      Array.map
        (fun (image, kernel, _) -> Rig.Image.threads p.images.(image) kernel)
        (launches p.t)
  | There p -> p.limits

(* A watched size as [f] sets it, or -1 for one the agent knows. *)
let watched_size (f : frame) = function
  | Fixed n -> n
  | Int i when i < Array.length f.ints -> f.ints.(i)
  | Int _ | Leaf _ -> -1

(* Whether [n] is a size of at least 1 that a header word holds. *)
let header_size n = n > 0 && n <= 0xffff_ffff

(* Refuses a watched launch whose threads per group, as the frame's ints set
   them, exceed its function's most. A size outside a header word, or a word
   the frame does not write, is left to the agent. The product is compared a
   factor at a time, so it never overflows. A loop, so each run allocates
   nothing here. *)
let check_threads watch (f : frame) =
  for i = 0 to Array.length watch - 1 do
    let { kernel; sizes = x, y, z; most } = watch.(i) in
    let x = watched_size f x
    and y = watched_size f y
    and z = watched_size f z in
    if
      header_size x && header_size y && header_size z
      && (x > most || y > most / x || z > most / (x * y))
    then
      invalid "%s runs at most %d threads per group; the frame sets %dx%dx%d"
        kernel most x y z
  done

(* The words are placed on the host's queue when [submit] hands them over, so
   the next run writes them again once it returned. Each input is read once,
   checked and written: the agent checks again what it decodes. A borrow here
   asks the agent's own borrow, once per allocation and device. *)
let run_words ~waits p (f : frame) =
  check_counts p.t f;
  check_threads p.watch f;
  set_word p.words 0 p.entry;
  for i = 0 to Array.length f.inputs - 1 do
    let b = f.inputs.(i) and spec = Iarray.get p.t.inputs i in
    check_input i spec p.devices.(spec.device) b;
    if p.access.(i) = Rig.Read_write && B.access b = Rig.Read then
      invalid "input %d is read-only memory, which the program writes" i;
    for j = 0 to Array.length p.binds - 1 do
      let { input; on } = p.binds.(j) in
      let dev = p.devices.(on) in
      if input <> i || on = spec.device then ()
      else if Option.is_none (B.borrow dev b) then
        invalid "input %d: %s cannot borrow it" i (Rig.name dev)
    done;
    let at = 8 + (24 * i) in
    set_word p.words at (Nativeint.to_int (B.handle b));
    set_word p.words (at + 8) (B.offset b);
    set_word p.words (at + 16) (B.length b)
  done;
  let at = 8 + (24 * Array.length f.inputs) in
  set_word p.words at (Array.length f.ints);
  for i = 0 to Array.length f.ints - 1 do
    set_word p.words (at + 8 + (8 * i)) f.ints.(i)
  done;
  [| Rig.submit ?waits p.sub ~run:p.srun ~buffers:[||] |]

(* The lock as [run_here] takes it. *)
let run_there ~waits p (f : frame) =
  lock p.lock;
  match run_words ~waits p f with
  | points ->
      Mutex.unlock p.lock;
      points
  | exception e ->
      Mutex.unlock p.lock;
      raise e

let run ?waits p f =
  match p with Here p -> run_here ~waits p f | There p -> run_there ~waits p f

let load_received ?rails b ~device =
  let r = { R.s = b; at = 0 } in
  let m = String.length share_magic in
  if String.length b < m || String.sub b 0 m <> share_magic then
    Error "not a program another process loads"
  else begin
    r.R.at <- m;
    match R.array R.int r with
    | exception R.Malformed (at, why) ->
        Error (strf "a malformed program at byte %d: %s" at why)
    | ids -> (
        match of_string (String.sub b r.R.at (R.left r)) with
        | Error _ as e -> e
        | Ok t -> (
            match
              Iarray.map
                (fun id ->
                  match device id with
                  | Some d -> d
                  | None -> refuse "no device %d on this machine" id)
                ids
            with
            | exception Refused why -> Error why
            | devices -> load ?rails t devices))
  end

let run_received ?waits w ~program ~region =
  let r = { R.s = w; at = 0 } in
  match
    let p =
      match program (R.int r) with
      | Some (Here p) -> p
      | Some (There _) | None -> R.fail r "no program of this entry"
    in
    let input _ =
      let id = R.int r in
      let first = R.int r in
      let length = R.int r in
      match region id with
      | None -> R.fail r "no memory %d" id
      | Some b -> (
          try B.view b ~first ~length
          with Invalid_argument _ ->
            R.fail r "%d bytes at %d past memory %d" length first id)
    in
    let inputs = Array.init (Iarray.length p.t.inputs) input in
    let n = R.int r in
    if n < 0 || n > p.t.ints then
      R.fail r "%d ints for a program of %d" n p.t.ints;
    let ints = Array.init p.t.ints (fun _ -> R.int r) in
    (p, { inputs; ints = Array.sub ints 0 n })
  with
  | exception R.Malformed (at, why) ->
      Error (strf "a malformed run at byte %d: %s" at why)
  | p, f -> Ok (run_here ~waits p f)
