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
  | Code of int
  | Ready of int
  | Ready_arg of int

type width = W32 | W64
type 'a hole = { at : int; width : width; leaf : 'a; add : int; shift : int }
type 'a data = { bytes : string; holes : 'a hole array }

type value =
  | Fixed of int
  | Int of int
  | Input of { input : int; on : int }
  | Leaf of leaf

type copies = One | Two
type area = Outbound | Inbound | Counts

type memory =
  | Alloc of {
      device : int;
      kind : B.memory;
      bytes : int;
      init : leaf data;
      copies : copies;
    }
  | Rail of { rail : int; area : area }

type image = { device : int; binary : leaf data }
type view = { memory : int; offset : int; length : int }
type slot = Memory of view | Input of int | Ints

type work =
  | Words of view
  | Fill of { fill : leaf; arg : view; ring_units : int; segment_bytes : int }
  | Copy of { src : view; dst : view }
  | Launch of {
      image : int;
      kernel : string;
      params : value data;
      refs : Sub.ref array;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type part = { queue : string; after : int array; work : work }

type submit = {
  device : int;
  parts : part array;
  reads : slot array;
  writes : slot array;
  fixed : (view * B.access) array;
}

type code = { obj : string; entry : string }
type split = { extent : value; blocks : int; lo : int; hi : int }

type step =
  | Submit of submit
  | Move of { src : slot; dst : slot }
  | Host of {
      code : int;
      buffers : (slot * B.access) array;
      values : value array;
      split : split option;
    }
  | Loop of {
      trips : value;
      trip : int option;
      flag : view option;
      body : step array;
    }

type input = { device : int; bytes : int; access : B.access }

type t = {
  devices : string array;
  memory : memory array;
  images : image array;
  code : code array;
  inputs : input array;
  ints : int;
  steps : step array;
}

(* A refusal of the description, which [load] answers as [Error]. *)
exception Refused of string

let refuse fmt = Printf.ksprintf (fun why -> raise (Refused why)) fmt

(* The run's ints are memory [Array.length t.memory] of two copies, which [load]
   makes on the host: a slot [Ints] is its view. *)
let ints_view t =
  { memory = Array.length t.memory; offset = 0; length = 8 * t.ints }

(* Checks *)

(* What [load] refuses before it makes anything: every index, size and hole, and
   where a leaf of [Two] memory may stand. *)

let index what n i =
  if i < 0 || i >= n then refuse "%s %d: no such %s" what i what

let two t m =
  match t.memory.(m) with
  | Alloc { copies; _ } -> copies = Two
  | Rail _ -> false

let check_leaf t ~images = function
  | Address { memory; on } ->
      index "memory" (Array.length t.memory) memory;
      index "device" (Array.length t.devices) on
  | Handle m -> index "memory" (Array.length t.memory) m
  | Entry { image; _ } -> index "image" images image
  | Code i -> index "code" (Array.length t.code) i
  | Ready _ | Ready_arg _ -> ()

let leaf_two t = function
  | Address { memory; _ } | Handle memory -> two t memory
  | Entry _ | Code _ | Ready _ | Ready_arg _ -> false

let width h = match h.width with W32 -> 4 | W64 -> 8

let check_holes what len check holes =
  let sorted = List.sort (fun a b -> compare a.at b.at) (Array.to_list holes) in
  ignore
    (List.fold_left
       (fun prev h ->
         (match prev with
         | Some p when p.at + width p > h.at ->
             refuse "%s: holes at %d and %d share a byte" what p.at h.at
         | _ -> ());
         Some h)
       None sorted);
  Array.iter
    (fun h ->
      if h.at < 0 || h.at + width h > len then
        refuse "%s: a hole at %d outside its %d bytes" what h.at len;
      if h.shift < 0 || h.shift > 62 then
        refuse "%s: a hole's shift %d outside 0 to 62" what h.shift;
      check h.leaf)
    holes

let check_view t sizes (v : view) =
  index "memory" (Array.length t.memory) v.memory;
  let n = sizes.(v.memory) in
  if v.offset < 0 || v.length < 0 || v.offset + v.length > n then
    refuse "memory %d: a view of %d bytes at %d outside its %d bytes" v.memory
      v.length v.offset n

let check_slot t sizes = function
  | Memory v -> check_view t sizes v
  | Input i -> index "input" (Array.length t.inputs) i
  | Ints -> ()

let check_value t : value -> unit = function
  | Fixed _ -> ()
  | Int i -> index "int" t.ints i
  | Input { input; on } ->
      index "input" (Array.length t.inputs) input;
      index "device" (Array.length t.devices) on
  | Leaf l -> check_leaf t ~images:(Array.length t.images) l

let check_work t sizes = function
  | Words v -> check_view t sizes v
  | Fill { fill; arg; _ } ->
      check_leaf t ~images:(Array.length t.images) fill;
      check_view t sizes arg
  | Copy { src; dst } ->
      check_view t sizes src;
      check_view t sizes dst
  | Launch
      { image; params; groups = gx, gy, gz; threads = tx, ty, tz; shared; _ } ->
      index "image" (Array.length t.images) image;
      let n = String.length params.bytes in
      if n mod 4 <> 0 || n > 4096 then
        refuse
          "image %d: %d bytes of parameters, not a whole number of 32-bit \
           words up to 4096"
          image n;
      check_holes
        (strf "image %d's parameters" image)
        n (check_value t) params.holes;
      List.iter (check_value t) [ gx; gy; gz; tx; ty; tz; shared ]

let rec check_step t sizes ndev = function
  | Submit s ->
      index "device" ndev s.device;
      Array.iter (fun p -> check_work t sizes p.work) s.parts;
      Array.iter (check_slot t sizes) s.reads;
      Array.iter (check_slot t sizes) s.writes;
      Array.iter (fun (v, _) -> check_view t sizes v) s.fixed
  | Move { src; dst } ->
      check_slot t sizes src;
      check_slot t sizes dst
  | Host { code; buffers; values; split } ->
      index "code" (Array.length t.code) code;
      Array.iter (fun (s, _) -> check_slot t sizes s) buffers;
      Array.iter (check_value t) values;
      Option.iter
        (fun s ->
          check_value t s.extent;
          let n = Array.length values in
          if
            s.blocks < 1 || s.lo < 0 || s.lo >= n || s.hi < 0 || s.hi >= n
            || s.lo = s.hi
          then
            refuse "code %d: a split of %d blocks over values %d and %d of %d"
              code s.blocks s.lo s.hi n)
        split
  | Loop { trips; trip; flag; body } ->
      check_value t trips;
      Option.iter (index "int" t.ints) trip;
      Option.iter
        (fun (v : view) ->
          check_view t sizes v;
          if v.length < 1 then refuse "memory %d: a flag of no byte" v.memory)
        flag;
      Array.iter (check_step t sizes ndev) body

(* [sizes] is each memory's bytes; a rail's area's where its end is known. *)
let check t sizes devices =
  if Array.length devices <> Array.length t.devices then
    refuse "%d devices for a program of %d" (Array.length devices)
      (Array.length t.devices);
  Array.iteri
    (fun i arch ->
      let a = Rig.arch devices.(i) in
      if a <> arch then
        refuse "device %d: %s is %S, not %S" i (Rig.name devices.(i)) a arch)
    t.devices;
  if t.ints < 0 then refuse "%d ints" t.ints;
  let ndev = Array.length t.devices in
  Array.iteri
    (fun i -> function
      | Alloc { device; bytes; init; copies; _ } ->
          index "device" ndev device;
          if bytes < 0 then refuse "memory %d: %d bytes" i bytes;
          let n = String.length init.bytes in
          if n > bytes then
            refuse "memory %d: %d bytes of init in %d bytes" i n bytes;
          check_holes (strf "memory %d" i) n
            (fun l ->
              check_leaf t ~images:(Array.length t.images) l;
              if copies = One && leaf_two t l then
                refuse "memory %d: of one copy, a hole names memory of two" i)
            init.holes
      | Rail _ -> ())
    t.memory;
  Array.iteri
    (fun i (im : image) ->
      index "device" ndev im.device;
      check_holes (strf "image %d" i)
        (String.length im.binary.bytes)
        (fun l ->
          check_leaf t ~images:i l;
          if leaf_two t l then refuse "image %d: a hole names memory of two" i)
        im.binary.holes)
    t.images;
  Array.iteri
    (fun i (input : input) ->
      index "device" ndev input.device;
      if input.bytes < 0 then refuse "input %d: %d bytes" i input.bytes)
    t.inputs;
  Array.iter (check_step t sizes ndev) t.steps

(* Bytes *)

(* A description as bytes: the magic, the format's version, then the
   description. An integer is 64 bits, little-endian; a string or an array its
   count, then its elements; a variant its tag, a byte, then its arguments. *)

let magic = "rig.program\n"
let version = 1

module W = struct
  let int b n = Buffer.add_int64_le b (Int64.of_int n)
  let tag b n = Buffer.add_uint8 b n

  let string b s =
    int b (String.length s);
    Buffer.add_string b s

  let array f b a =
    int b (Array.length a);
    Array.iter (f b) a

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
    | Code i ->
        tag b 3;
        int b i
    | Ready r ->
        tag b 4;
        int b r
    | Ready_arg r ->
        tag b 5;
        int b r

  let value b = function
    | Fixed n ->
        tag b 0;
        int b n
    | Int i ->
        tag b 1;
        int b i
    | Input { input; on } ->
        tag b 2;
        int b input;
        int b on
    | Leaf l ->
        tag b 3;
        leaf b l

  let data f b (d : _ data) =
    string b d.bytes;
    array
      (fun b h ->
        int b h.at;
        tag b (match h.width with W32 -> 0 | W64 -> 1);
        f b h.leaf;
        int b h.add;
        int b h.shift)
      b d.holes

  let memory_kind b (k : B.memory) =
    tag b (match k with Device -> 0 | Pinned -> 1 | Mapped -> 2)

  let access b (a : B.access) =
    tag b (match a with Read -> 0 | Read_write -> 1)

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

  let work b = function
    | Words v ->
        tag b 0;
        view b v
    | Fill { fill; arg; ring_units; segment_bytes } ->
        tag b 1;
        leaf b fill;
        view b arg;
        int b ring_units;
        int b segment_bytes
    | Copy { src; dst } ->
        tag b 2;
        view b src;
        view b dst
    | Launch { image; kernel; params; refs; groups; threads; shared } ->
        tag b 3;
        int b image;
        string b kernel;
        data value b params;
        array
          (fun b (r : Sub.ref) ->
            int b r.Sub.at;
            int b r.Sub.slot)
          b refs;
        triple b groups;
        triple b threads;
        value b shared

  let part b (p : part) =
    string b p.queue;
    array int b p.after;
    work b p.work

  let rec step b = function
    | Submit s ->
        tag b 0;
        int b s.device;
        array part b s.parts;
        array slot b s.reads;
        array slot b s.writes;
        array
          (fun b (v, a) ->
            view b v;
            access b a)
          b s.fixed
    | Move { src; dst } ->
        tag b 1;
        slot b src;
        slot b dst
    | Host { code; buffers; values; split } ->
        tag b 2;
        int b code;
        array
          (fun b (s, a) ->
            slot b s;
            access b a)
          b buffers;
        array value b values;
        option
          (fun b (s : split) ->
            value b s.extent;
            int b s.blocks;
            int b s.lo;
            int b s.hi)
          b split
    | Loop { trips; trip; flag; body } ->
        tag b 3;
        value b trips;
        option int b trip;
        option view b flag;
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
      (fun b (c : code) ->
        string b c.obj;
        string b c.entry)
      b t.code;
    array
      (fun b (i : input) ->
        int b i.device;
        int b i.bytes;
        access b i.access)
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

  let array f r =
    let n = count r in
    if n = 0 then [||]
    else begin
      let a = Array.make n (f r) in
      for i = 1 to n - 1 do
        a.(i) <- f r
      done;
      a
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
    | 3 -> Code (int r)
    | 4 -> Ready (int r)
    | 5 -> Ready_arg (int r)
    | n -> bad r "a leaf" n

  let value r =
    match tag r with
    | 0 -> Fixed (int r)
    | 1 -> Int (int r)
    | 2 ->
        let input = int r in
        let on = int r in
        Input { input; on }
    | 3 -> Leaf (leaf r)
    | n -> bad r "a value" n

  let width r = match tag r with 0 -> W32 | 1 -> W64 | n -> bad r "a width" n

  let data f r =
    let bytes = string r in
    let holes =
      array
        (fun r ->
          let at = int r in
          let width = width r in
          let leaf = f r in
          let add = int r in
          let shift = int r in
          { at; width; leaf; add; shift })
        r
    in
    { bytes; holes }

  let memory_kind r : B.memory =
    match tag r with
    | 0 -> Device
    | 1 -> Pinned
    | 2 -> Mapped
    | n -> bad r "a memory kind" n

  let access r : B.access =
    match tag r with 0 -> Read | 1 -> Read_write | n -> bad r "an access" n

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

  let work r =
    match tag r with
    | 0 -> Words (view r)
    | 1 ->
        let fill = leaf r in
        let arg = view r in
        let ring_units = int r in
        let segment_bytes = int r in
        Fill { fill; arg; ring_units; segment_bytes }
    | 2 ->
        let src = view r in
        let dst = view r in
        Copy { src; dst }
    | 3 ->
        let image = int r in
        let kernel = string r in
        let params = data value r in
        let refs =
          array
            (fun r ->
              let at = int r in
              let slot = int r in
              { Sub.at; slot })
            r
        in
        let groups = triple r in
        let threads = triple r in
        let shared = value r in
        Launch { image; kernel; params; refs; groups; threads; shared }
    | n -> bad r "a work" n

  let part r =
    let queue = string r in
    let after = array int r in
    let work = work r in
    { queue; after; work }

  let rec step r =
    match tag r with
    | 0 ->
        let device = int r in
        let parts = array part r in
        let reads = array slot r in
        let writes = array slot r in
        let fixed =
          array
            (fun r ->
              let v = view r in
              let a = access r in
              (v, a))
            r
        in
        Submit { device; parts; reads; writes; fixed }
    | 1 ->
        let src = slot r in
        let dst = slot r in
        Move { src; dst }
    | 2 ->
        let code = int r in
        let buffers =
          array
            (fun r ->
              let s = slot r in
              let a = access r in
              (s, a))
            r
        in
        let values = array value r in
        let split =
          option
            (fun r ->
              let extent = value r in
              let blocks = int r in
              let lo = int r in
              let hi = int r in
              { extent; blocks; lo; hi })
            r
        in
        Host { code; buffers; values; split }
    | 3 ->
        let trips = value r in
        let trip = option int r in
        let flag = option view r in
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
    let code =
      array
        (fun r ->
          let obj = string r in
          let entry = string r in
          { obj; entry })
        r
    in
    let inputs =
      array
        (fun r ->
          let device = int r in
          let bytes = int r in
          let access = access r in
          { device; bytes; access })
        r
    in
    let ints = int r in
    let steps = array step r in
    { devices; memory; images; code; inputs; ints; steps }
end

let to_string t =
  let b = Buffer.create 4096 in
  Buffer.add_string b magic;
  W.int b version;
  W.t b t;
  Buffer.contents b

let of_string s =
  let r = { R.s; at = 0 } in
  let m = String.length magic in
  if String.length s < m || String.sub s 0 m <> magic then
    Error "not a program description"
  else begin
    r.R.at <- m;
    match R.int r with
    | exception R.Malformed (_, why) -> Error why
    | v when v <> version ->
        Error
          (strf "a program description of format version %d, not %d" v version)
    | _ -> (
        match R.t r with
        | t when R.left r = 0 -> Ok t
        | _ -> Error (strf "%d bytes after a program description" (R.left r))
        | exception R.Malformed (at, why) ->
            Error (strf "a malformed program description at byte %d: %s" at why)
        )
  end

(* Loading *)

external host_address : B.t -> int = "caml_rig_program_host"

(* A launch's words that a run writes into its block before each submit: its
   parameters' holes over the run's values, and its geometry. *)
type launch_run = {
  block : Sub.block;
  params : string;
  holes : value hole array;
  groups : value * value * value;
  threads : value * value * value;
  shared : value;
  geometry_per_run : bool; (* Whether a value of its geometry is per run. *)
}

(* A step's submission for one copy, its run, and the buffers it is passed:
   memory set at load, inputs at each run. *)
type prepared = {
  sub : Sub.t;
  srun : Sub.Run.t;
  reads : B.t array;
  writes : B.t array;
  launches : launch_run array;
}

type lstep =
  | Lsubmit of { spec : submit; copies : prepared array }
  | Lmove of { src : slot; dst : slot }
  | Lhost of {
      code : Rig_host.t;
      buffers : (slot * B.access) array;
      values : value array;
      split : split option;
      addresses : int array;
      words : int array;
    }
  | Lloop of {
      trips : value;
      trip : int option;
      flag : view option;
      body : lstep array;
    }

(* A program loaded on this machine. *)
type here = {
  t : t;
  devices : Rig.t array;
  mem : B.t array array;
      (* Each memory's copies, on its device, then the run's ints. *)
  ints : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t array;
      (* Each copy of the run's ints, as words. *)
  borrows : (int * int * int, B.t) Hashtbl.t;
      (* By memory, copy and device: the borrows made at load. *)
  mutable images : Rig.Image.t array; (* Those loaded so far, while loading. *)
  code : Rig_host.t array;
  rails : int -> Rig_remote_abi.end_ option;
  hold : Rig.Hold.t;
      (* What every submission keeps until its work is done: the code, which an
         address in its work may name. *)
  mutable steps : lstep array;
  twos : B.t array array; (* Copy [k] of each memory of two copies. *)
  flag : B.t; (* The byte a loop's flag is read into. *)
  ran : bool array; (* By device: whether the run submitted there. *)
  mutable last : Rig.Point.t array;
      (* By device: the run's last point, where [ran]; empty before the first
         submit. *)
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

(* The memory of [s], [None] for an input. *)
let slot_view p = function
  | Memory v -> Some v
  | Ints -> Some (ints_view p.t)
  | Input _ -> None

(* The value of [l] for copy [k]. *)
let leaf_value p k = function
  | Address { memory; on = d } -> (
      let b = on p memory (copy_of p memory k) d in
      try B.address b
      with Invalid_argument _ ->
        refuse "memory %d: named by its handle only, it has no address" memory)
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
  | Code i -> Rig_host.address p.code.(i)
  | Ready r -> Nativeint.to_int (rail_end p.rails r).ready_fn
  | Ready_arg r -> Nativeint.to_int (rail_end p.rails r).ready_arg

(* The word hole [h] makes of the value [v] over [word], the word of its bytes
   there: the value's low bits ORed into it. *)
let hole_word h v word =
  let w = Int64.of_int ((v + h.add) lsr h.shift) in
  match h.width with
  | W32 -> Int64.logand (Int64.logor word w) 0xffff_ffffL
  | W64 -> Int64.logor word w

let get_word b h =
  match h.width with
  | W32 ->
      Int64.logand (Int64.of_int32 (String.get_int32_le b h.at)) 0xffff_ffffL
  | W64 -> String.get_int64_le b h.at

(* [d]'s bytes with its holes filled for copy [k]. *)
let filled p k (d : leaf data) =
  let b = Bytes.of_string d.bytes in
  Array.iter
    (fun h ->
      let w = hole_word h (leaf_value p k h.leaf) (get_word d.bytes h) in
      match h.width with
      | W32 -> Bytes.set_int32_le b h.at (Int64.to_int32 w)
      | W64 -> Bytes.set_int64_le b h.at w)
    d.holes;
  Bytes.unsafe_to_string b

(* Whether [s] names memory of [Two] copies, so that it is made once per
   copy. *)
let per_copy t (s : submit) =
  let view (v : view) = two t v.memory in
  let slot = function Memory v -> view v | Ints -> true | Input _ -> false in
  let value : value -> bool = function
    | Leaf l -> leaf_two t l
    | Fixed _ | Int _ | Input _ -> false
  in
  let work = function
    | Words v -> view v
    | Fill { fill; arg; _ } -> leaf_two t fill || view arg
    | Copy { src; dst } -> view src || view dst
    | Launch { params; groups = gx, gy, gz; threads = tx, ty, tz; shared; _ } ->
        Array.exists (fun h -> value h.leaf) params.holes
        || List.exists value [ gx; gy; gz; tx; ty; tz; shared ]
  in
  Array.exists (fun (q : part) -> work q.work) s.parts
  || Array.exists slot s.reads || Array.exists slot s.writes
  || Array.exists (fun (v, _) -> view v) s.fixed

let per_run : value -> bool = function
  | Int _ | Input _ -> true
  | Fixed _ | Leaf _ -> false

(* Stores hole [h] of a launch's parameters [params] over [v] into its block
   [b]. A 64-bit word goes as two 32-bit halves: an [int] setter would drop its
   bit 63. *)
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

(* The submission of [s] for copy [k], and its run holding the words of its
   blocks that no run changes. *)
let prepared p k (s : submit) =
  let d = s.device in
  let work = function
    | Words v -> Sub.Words (view p k d v)
    | Fill { fill; arg; ring_units; segment_bytes } ->
        let fill = Nativeint.of_int (leaf_value p k fill) in
        Sub.Fill { fill; arg = view p k d arg; ring_units; segment_bytes }
    | Copy { src; dst } ->
        Sub.Copy { src = view p k d src; dst = view p k d dst }
    | Launch { image; kernel; params; refs; _ } ->
        let params = String.length params.bytes in
        Sub.Launch { image = p.images.(image); kernel; params; refs }
  in
  let parts =
    Array.map
      (fun (q : part) ->
        { Sub.queue = q.queue; after = q.after; work = work q.work })
      s.parts
  in
  let fixed =
    Array.to_list (Array.map (fun (v, a) -> (view p k d v, a)) s.fixed)
  in
  let reads = Array.length s.reads and writes = Array.length s.writes in
  let sub = Sub.make ~hold:p.hold ~fixed ~reads ~writes p.devices.(d) parts in
  let slot s =
    match slot_view p s with Some v -> view p k d v | None -> unset
  in
  let srun = Sub.Run.make () in
  let value : value -> int = function
    | Fixed n -> n
    | Leaf l -> leaf_value p k l
    | Int _ | Input _ -> 0
  in
  let launch i (q : part) =
    match q.work with
    | Words _ | Fill _ | Copy _ -> None
    | Launch { params; groups; threads; shared; _ } ->
        let b = Sub.block sub i in
        for w = 0 to (String.length params.bytes / 4) - 1 do
          Sub.Run.int32 srun b (4 * w)
            (Int32.to_int (String.get_int32_le params.bytes (4 * w)))
        done;
        Array.iter
          (fun h ->
            if not (per_run h.leaf) then
              store_hole srun b params.bytes h (value h.leaf))
          params.holes;
        store_geometry srun b groups threads shared value;
        let gx, gy, gz = groups and tx, ty, tz = threads in
        Some
          {
            block = b;
            params = params.bytes;
            holes =
              Array.of_list
                (List.filter
                   (fun h -> per_run h.leaf)
                   (Array.to_list params.holes));
            groups;
            threads;
            shared;
            geometry_per_run =
              List.exists per_run [ gx; gy; gz; tx; ty; tz; shared ];
          }
  in
  {
    sub;
    srun;
    reads = Array.map slot s.reads;
    writes = Array.map slot s.writes;
    launches =
      Array.of_list
        (List.filter_map Fun.id (Array.to_list (Array.mapi launch s.parts)));
  }

(* A submission [Rig.Submission.make] or a setter refuses is a defect of the
   description. *)
let prepare p i k s =
  try prepared p k s with Invalid_argument why -> refuse "step %d: %s" i why

(* A [Host] step's memory is the host's to address, in each copy. *)
let check_host p i buffers =
  Array.iter
    (fun (s, _) ->
      match slot_view p s with
      | Some v ->
          Array.iter
            (fun b ->
              if host_address b = 0 then
                refuse "step %d: the host does not address memory %d" i v.memory)
            p.mem.(v.memory)
      | None -> ())
    buffers

let rec lstep p i = function
  | Submit s ->
      let n = if per_copy p.t s then 2 else 1 in
      Lsubmit { spec = s; copies = Array.init n (fun k -> prepare p i k s) }
  | Move { src; dst } -> Lmove { src; dst }
  | Host { code; buffers; values; split } ->
      check_host p i buffers;
      Lhost
        {
          code = p.code.(code);
          buffers;
          values;
          split;
          addresses = Array.make (Array.length buffers) 0;
          words = Array.make (Array.length values) 0;
        }
  | Loop { trips; trip; flag; body } ->
      Lloop { trips; trip; flag; body = Array.map (lstep p i) body }

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

let link (c : code) =
  match Rig_host.link ~entry:c.entry c.obj with
  | Ok c -> c
  | Error why -> raise (Refused why)

let write_init p m k =
  match p.t.memory.(m) with
  | Rail _ -> ()
  | Alloc { init; _ } ->
      let n = String.length init.bytes in
      if n > 0 then
        B.copy
          ~src:(B.of_string (filled p k init))
          ~dst:(B.view p.mem.(m).(k) ~first:0 ~length:n)

let words b =
  if B.length b = 0 then Bigarray.(Array1.create int64 c_layout 0)
  else B.bigarray Bigarray.int64 b

let load_here ~rails t devices =
  let size = function
    | Alloc { bytes; _ } -> bytes
    | Rail { rail; area } ->
        Bigarray.Array1.dim (rail_area (rail_end rails rail) area)
  in
  match
    check t (Array.map size t.memory) devices;
    Array.map link t.code
  with
  | exception Refused why -> Error why
  | code -> (
      let ints = Array.init 2 (fun _ -> B.create Rig.host (8 * t.ints)) in
      let mem =
        Array.append (Array.map (make_memory devices rails) t.memory) [| ints |]
      in
      let twos k =
        Array.to_list mem
        |> List.filter (fun c -> Array.length c = 2)
        |> List.map (fun c -> c.(k))
        |> Array.of_list
      in
      let p =
        {
          t;
          devices;
          mem;
          ints = Array.map words ints;
          borrows = Hashtbl.create 8;
          images = [||];
          code;
          rails;
          hold = Rig.Hold.make (fun () -> ignore (Sys.opaque_identity code));
          steps = [||];
          twos = [| twos 0; twos 1 |];
          flag = B.create Rig.host 1;
          ran = Array.make (Array.length devices) false;
          last = [||];
          lock = Mutex.create ();
          runs = 0;
        }
      in
      match
        (* An image's holes name earlier images' entries: each loads once those
           before it did. *)
        Array.iter
          (fun im -> p.images <- Array.append p.images [| load_image p im |])
          t.images;
        Array.iteri
          (fun m _ -> Array.iteri (fun k _ -> write_init p m k) mem.(m))
          t.memory;
        p.steps <- Array.mapi (lstep p) t.steps
      with
      | () -> Ok p
      | exception Refused why -> Error why)

(* Running *)

type frame = { inputs : B.t array; ints : int array }

let invalid fmt = Printf.ksprintf invalid_arg ("Rig_program.run: " ^^ fmt)

let check_frame (t : t) devices (f : frame) =
  let n = Array.length t.inputs in
  if Array.length f.inputs <> n then
    invalid "%d inputs for a program of %d" (Array.length f.inputs) n;
  for i = 0 to n - 1 do
    let spec = t.inputs.(i) in
    let b = f.inputs.(i) and d = devices.(spec.device) in
    if not (Rig.equal (B.device b) d) then
      invalid "input %d is on %s, not %s" i (Rig.name (B.device b)) (Rig.name d);
    if B.length b < spec.bytes then
      invalid "input %d holds %d bytes, fewer than %d" i (B.length b) spec.bytes;
    if spec.access = B.Read_write && B.access b = B.Read then
      invalid "input %d is read-only memory, which the program writes" i
  done;
  if Array.length f.ints > t.ints then
    invalid "%d ints for a program of %d" (Array.length f.ints) t.ints

(* Input [i] of [f] as device [d] names it. *)
let input_on p (f : frame) i d =
  let b = f.inputs.(i) and dev = p.devices.(d) in
  if Rig.equal (B.device b) dev then b
  else
    match B.borrow dev b with
    | Some b -> b
    | None -> invalid "input %d: %s cannot borrow it" i (Rig.name dev)

let run_value (p : here) f k : value -> int = function
  | Fixed n -> n
  | Int i -> Int64.to_int (Bigarray.Array1.unsafe_get p.ints.(k) i)
  | Input { input; on } -> B.address (input_on p f input on)
  | Leaf l -> leaf_value p k l

let pass p f (s : slot array) bufs d =
  for i = 0 to Array.length s - 1 do
    match s.(i) with
    | Input j -> bufs.(i) <- input_on p f j d
    | Memory _ | Ints -> ()
  done

(* Drops the inputs [bufs] held, so a loaded program keeps no caller's buffer
   past its run. *)
let unpass (s : slot array) bufs =
  for i = 0 to Array.length s - 1 do
    match s.(i) with Input _ -> bufs.(i) <- unset | Memory _ | Ints -> ()
  done

let mem_buffer p k (v : view) =
  B.view
    p.mem.(v.memory).(copy_of p v.memory k)
    ~first:v.offset ~length:v.length

let slot_buffer p (f : frame) k = function
  | Input j -> f.inputs.(j)
  | Memory v -> mem_buffer p k v
  | Ints -> mem_buffer p k (ints_view p.t)

let submit_step p f k after (spec : submit) copies =
  let q = copies.(if Array.length copies = 2 then k else 0) in
  let d = spec.device in
  pass p f spec.reads q.reads d;
  pass p f spec.writes q.writes d;
  for i = 0 to Array.length q.launches - 1 do
    let l = q.launches.(i) in
    for j = 0 to Array.length l.holes - 1 do
      let h = l.holes.(j) in
      store_hole q.srun l.block l.params h (run_value p f k h.leaf)
    done;
    if l.geometry_per_run then
      store_geometry q.srun l.block l.groups l.threads l.shared
        (run_value p f k)
  done;
  let waits = if p.ran.(d) then [||] else after in
  match Rig.submit q.sub ~run:q.srun ~reads:q.reads ~writes:q.writes ~waits with
  | point ->
      unpass spec.reads q.reads;
      unpass spec.writes q.writes;
      if Array.length p.last = 0 then
        p.last <- Array.make (Array.length p.devices) point;
      p.last.(d) <- point;
      p.ran.(d) <- true
  | exception e ->
      unpass spec.reads q.reads;
      unpass spec.writes q.writes;
      raise e

let host p f k code buffers values split addresses words =
  Array.iteri
    (fun i (s, access) ->
      let b = slot_buffer p f k s in
      B.wait b access;
      let a = host_address b in
      (* [load] checked the memory: only an input can fail. *)
      if a = 0 then
        invalid "a host step's buffer %d: the host does not address it" i;
      addresses.(i) <- a)
    buffers;
  Array.iteri (fun i v -> words.(i) <- run_value p f k v) values;
  let split =
    Option.map
      (fun (s : split) ->
        {
          Rig_host.extent = run_value p f k s.extent;
          blocks = s.blocks;
          lo = s.lo;
          hi = s.hi;
        })
      split
  in
  Rig_host.call ?split code addresses words

(* Whether [flag]'s first byte is not [0], once the work that wrote it is
   done. *)
let flag_holds p k (v : view) =
  B.copy ~src:(mem_buffer p k { v with length = 1 }) ~dst:p.flag;
  let c = Bytes.create 1 in
  B.blit_to_bytes p.flag 0 c 0 1;
  Bytes.get c 0 <> '\000'

let rec exec (p : here) f k after = function
  | Lmove { src; dst } ->
      B.copy ~src:(slot_buffer p f k src) ~dst:(slot_buffer p f k dst)
  | Lsubmit { spec; copies } -> submit_step p f k after spec copies
  | Lhost { code; buffers; values; split; addresses; words } ->
      host p f k code buffers values split addresses words
  | Lloop { trips; trip; flag; body } ->
      let n = run_value p f k trips in
      let rec go i =
        if i < n && Option.fold ~none:true ~some:(flag_holds p k) flag then begin
          Option.iter
            (fun w -> Bigarray.Array1.set p.ints.(k) w (Int64.of_int i))
            trip;
          for s = 0 to Array.length body - 1 do
            exec p f k after body.(s)
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

let run_steps ~after (p : here) (f : frame) =
  check_frame p.t p.devices f;
  let k = p.runs land 1 in
  p.runs <- p.runs + 1;
  let twos = p.twos.(k) in
  for i = 0 to Array.length twos - 1 do
    B.wait twos.(i) B.Read_write
  done;
  let ints = p.ints.(k) in
  for i = 0 to Array.length f.ints - 1 do
    Bigarray.Array1.set ints i (Int64.of_int f.ints.(i))
  done;
  Array.fill p.ran 0 (Array.length p.ran) false;
  for s = 0 to Array.length p.steps - 1 do
    exec p f k after p.steps.(s)
  done;
  points p

(* The lock without [Mutex.protect]'s closure: a run allocates only its
   answer. *)
let run_here ~after (p : here) (f : frame) =
  Mutex.lock p.lock;
  match run_steps ~after p f with
  | points ->
      Mutex.unlock p.lock;
      points
  | exception e ->
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
  entry : int;
  sub : Sub.t;
  srun : Sub.Run.t;
  words :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  lock : Mutex.t;
}

type loaded = Here of here | There of there

let words_bytes (t : t) = 8 + (24 * Array.length t.inputs) + 8 + (8 * t.ints)

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
    let size = function Alloc { bytes; _ } -> bytes | Rail _ -> max_int in
    check t (Array.map size t.memory) devices;
    Array.map proxy_id devices
  with
  | exception Refused why -> Error why
  | ids -> (
      let b = Buffer.create 4096 in
      Buffer.add_string b share_magic;
      W.array W.int b ids;
      Buffer.add_string b (to_string t);
      let host = Rig.host_of devices.(0) in
      match Rig.Image.load host (Buffer.contents b) with
      | Error why -> Error why
      | Ok image -> (
          match Rig.Image.entry image "run" with
          | None -> Error (strf "%s: the program has no run" (Rig.name host))
          | Some entry ->
              let buffer = B.create Rig.host (words_bytes t) in
              let part =
                {
                  Sub.queue = "COMPUTE:0";
                  after = [||];
                  work = Sub.Words buffer;
                }
              in
              (* The image stays loaded until the runs' work is done. *)
              let hold =
                Rig.Hold.make (fun () -> ignore (Sys.opaque_identity image))
              in
              let sub = Sub.make ~hold ~reads:0 ~writes:0 host [| part |] in
              Ok
                (There
                   {
                     t;
                     devices;
                     entry;
                     sub;
                     srun = Sub.Run.make ();
                     words = B.bigarray Bigarray.char buffer;
                     lock = Mutex.create ();
                   })))

let no_rails _ = None

let load ?rails t devices =
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

let set_word words at v =
  Bigarray.Array1.(
    for i = 0 to 7 do
      unsafe_set words (at + i) (Char.unsafe_chr ((v lsr (8 * i)) land 0xff))
    done)

(* The words are placed on the host's queue when [submit] hands them over, so
   the next run writes them again once it returned. *)
let run_there ~after p (f : frame) =
  Mutex.protect p.lock @@ fun () ->
  check_frame p.t p.devices f;
  set_word p.words 0 p.entry;
  Array.iteri
    (fun i b ->
      let at = 8 + (24 * i) in
      set_word p.words at (Nativeint.to_int (B.handle b));
      set_word p.words (at + 8) (B.offset b);
      set_word p.words (at + 16) (B.length b))
    f.inputs;
  let at = 8 + (24 * Array.length f.inputs) in
  set_word p.words at (Array.length f.ints);
  Array.iteri (fun i v -> set_word p.words (at + 8 + (8 * i)) v) f.ints;
  [| Rig.submit p.sub ~run:p.srun ~reads:[||] ~writes:[||] ~waits:after |]

let run ?(after = [||]) p f =
  match p with Here p -> run_here ~after p f | There p -> run_there ~after p f

let load_share ?rails b ~device =
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
              Array.map
                (fun id ->
                  match device id with
                  | Some d -> d
                  | None -> refuse "no device %d on this machine" id)
                ids
            with
            | exception Refused why -> Error why
            | devices -> load ?rails t devices))
  end

let run_share ?(after = [||]) w ~program ~region =
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
    let inputs = Array.init (Array.length p.t.inputs) input in
    let n = R.int r in
    if n < 0 || n > p.t.ints then
      R.fail r "%d ints for a program of %d" n p.t.ints;
    let ints = Array.init p.t.ints (fun _ -> R.int r) in
    (p, { inputs; ints = Array.sub ints 0 n })
  with
  | exception R.Malformed (at, why) ->
      Error (strf "a malformed run at byte %d: %s" at why)
  | p, f -> Ok (run_here ~after p f)
