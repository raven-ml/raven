(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module B = Rig.Buffer
module Sub = Rig.Submission

let strf = Printf.sprintf

(* Descriptions *)

(* Before scalars, whose [at] an unannotated field then means. *)
type use = Read | Write | Unordered
type ref = { slot : int; at : int; use : use }
type width = W32 | W64
type value = { add : int; terms : (int * int) iarray }
type scalar = { at : int; width : width; value : value }
type area = Outbound | Inbound | Counts

type memory =
  | Alloc of { device : int; kind : B.kind; bytes : int; init : string }
  | Rail of { rail : int; area : area }

type image = { device : int; binary : string }
type view = { memory : int; offset : int; length : int }
type slot = Memory of view | Input of int | Ints

type part =
  | Words of view
  | Copy of {
      src : int;
      src_offset : value;
      dst : int;
      dst_offset : value;
      length : int;
    }
  | Launch of {
      image : int;
      name : string;
      label : string;
      params : string;
      scalars : scalar iarray;
      refs : ref iarray;
      groups : value * value * value;
      threads : value * value * value;
      shared : value;
    }

type submit = { device : int; parts : part iarray; slots : slot iarray }

type step =
  | Submit of submit
  | Move of {
      src : slot;
      src_offset : value;
      dst : slot;
      dst_offset : value;
      length : int;
    }
  | Loop of {
      trips : value;
      trip : int option;
      flag : (slot * int) option;
      body : step iarray;
    }
  | Send of { rail : int; count : value }

type input = { device : int; bytes : int }

type t = {
  devices : string iarray;
  memory : memory iarray;
  images : image iarray;
  inputs : input iarray;
  ints : int;
  steps : step iarray;
}

(* What a device refuses as the description loads, which [load] answers as
   [Error]. *)
exception Refused of string

let refuse fmt = Printf.ksprintf (fun why -> raise (Refused why)) fmt

(* A rule of the description it breaks, which [load] raises as
   [Invalid_argument] and [load_received], whose bytes another process sent,
   answers as [Error]. *)
exception Broken of string

let broken fmt = Printf.ksprintf (fun why -> raise (Broken why)) fmt

(* A value's sum past an OCaml [int]. *)
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

(* The rules [load] checks before it makes anything: every index, size and
   scalar, and each constant value. *)

let index what n i =
  if i < 0 || i >= n then broken "%s %d: no such %s" what i what

let is_constant (v : value) = Iarray.length v.terms = 0

(* A constant value's [add], which [load] knows. *)
let known (v : value) = if is_constant v then Some v.add else None

let check_value t (v : value) =
  Iarray.iter (fun (w, _) -> index "int" t.ints w) v.terms

let width_bytes = function W32 -> 4 | W64 -> 8

(* Whether a scalar of [width] holds [v]: 32 bits hold it as a signed or an
   unsigned integer, and 64 bits hold every [int]. *)
let fits width v =
  match width with W32 -> v >= -0x8000_0000 && v <= 0xffff_ffff | W64 -> true

(* Bounds are compared by subtraction: a description's integers may be any, and
   a sum of two may wrap. *)
let check_scalars t what len scalars =
  Iarray.iter
    (fun s ->
      if s.at < 0 || s.at > len - width_bytes s.width then
        broken "%s: a scalar at %d outside its %d bytes" what s.at len;
      check_value t s.value;
      if is_constant s.value && not (fits s.width s.value.add) then
        broken
          "%s: the scalar at %d holds %d, which 32 bits hold neither signed \
           nor unsigned"
          what s.at s.value.add)
    scalars;
  let sorted =
    List.sort (fun a b -> compare a.at b.at) (Iarray.to_list scalars)
  in
  ignore
    (List.fold_left
       (fun prev s ->
         (match prev with
         | Some p when p.at + width_bytes p.width > s.at ->
             broken "%s: scalars at %d and %d share a byte" what p.at s.at
         | _ -> ());
         Some s)
       None sorted)

let check_view t sizes (v : view) =
  index "memory" (Iarray.length t.memory) v.memory;
  let n = Option.value sizes.(v.memory) ~default:max_int in
  if v.offset < 0 || v.length < 0 || v.offset > n - v.length then
    broken "memory %d: a view of %d bytes at %d outside its %d bytes" v.memory
      v.length v.offset n

let check_slot t sizes = function
  | Memory v -> check_view t sizes v
  | Input i -> index "input" (Iarray.length t.inputs) i
  | Ints -> ()

let refs_of = function
  | Launch { refs; _ } -> refs
  | Words _ | Copy _ -> Iarray.of_list []

let check_part t sizes = function
  | Words v -> check_view t sizes v
  | Copy _ -> ()
  | Launch
      {
        image;
        params;
        scalars;
        groups = gx, gy, gz;
        threads = tx, ty, tz;
        shared;
        _;
      } ->
      index "image" (Iarray.length t.images) image;
      let n = String.length params in
      if n mod 4 <> 0 then
        broken
          "image %d: %d bytes of parameters, not a whole number of 32-bit words"
          image n;
      check_scalars t (strf "image %d's parameters" image) n scalars;
      List.iter (check_value t) [ gx; gy; gz; tx; ty; tz; shared ]

(* Refuses a copy part or a move of a negative length, or, where load knows
   the offsets, whose end lies outside its slot or whose two ends share a byte
   of one buffer. A view's and the ints' lengths are known; an input's is only
   at least its [bytes]. *)
let check_copy t ~src ~src_offset ~dst ~dst_offset ~length =
  if length < 0 then broken "a copy of %d bytes" length;
  check_value t src_offset;
  check_value t dst_offset;
  let check_end slot offset =
    match known offset with
    | None -> ()
    | Some o -> (
        if o < 0 then broken "a copy from byte %d of its slot" o;
        let n =
          match slot with
          | Memory v -> Some v.length
          | Ints -> Some (8 * t.ints)
          | Input _ -> None
        in
        match n with
        | Some n when o > n - length ->
            broken "a copy of %d bytes from byte %d of a slot of %d" length o n
        | _ -> ())
  in
  check_end src src_offset;
  check_end dst dst_offset;
  (* Where both ends are of one buffer, the byte each slot starts at in it: a
     view's offset into its memory, which [check_view] bounds, and [0] in the
     ints or an input. *)
  let shared =
    match (src, dst) with
    | Ints, Ints -> Some (0, 0)
    | Input i, Input j when i = j -> Some (0, 0)
    | Memory a, Memory b when a.memory = b.memory -> Some (a.offset, b.offset)
    | _ -> None
  in
  match (shared, known src_offset, known dst_offset) with
  | Some (s, d), Some a, Some b when length > 0 ->
      let a = s + a and b = d + b in
      if a - b < length && b - a < length then
        broken "a copy between overlapping bytes %d and %d of one buffer" a b
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
              check_copy t ~src:(Iarray.get s.slots src) ~src_offset
                ~dst:(Iarray.get s.slots dst) ~dst_offset ~length
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
        | false, false -> broken "slot %d: no ref or copy names it" i
        | true, true ->
            broken
              "slot %d: named by an unordered ref and an ordered one or a copy"
              i
        | _ -> ()
      done
  | Move { src; src_offset; dst; dst_offset; length } ->
      check_slot t sizes src;
      check_slot t sizes dst;
      check_copy t ~src ~src_offset ~dst ~dst_offset ~length
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
          | _ when at < 0 -> broken "a loop's flag at byte %d" at
          | Some n when at >= n ->
              broken "a loop's flag at byte %d of a slot of %d bytes" at n
          | _ -> ())
        flag;
      Iarray.iter (check_step t sizes ndev) body
  | Send { count; _ } -> (
      check_value t count;
      match known count with
      | Some c when c < 1 -> broken "a send of count %d, below 1" c
      | _ -> ())

(* The features an arm64 arch names after "arm64", each after a "+";
   [None] for another arch. *)
let arm64_features arch =
  match String.split_on_char '+' arch with
  | "arm64" :: features -> Some features
  | _ -> None

(* Whether a device of arch [device] runs code compiled for [arch]: its own;
   on x86-64 that of a level at most its own, since "the levels are
   cumulative" (x86-64 psABI, §3.1.1); and on arm64 that of an arm64 arch
   whose features are all its own, since code runs where its "dependent
   features are all available" (ACLE, "Function Multi Versioning",
   "Selection"). Features compare by name, so a feature added later needs no
   table here. *)
let runs ~device arch =
  let level a = List.find_index (String.equal a) Rig.x86_levels in
  match (level device, level arch) with
  | Some d, Some a -> a <= d
  | _ -> (
      match (arm64_features device, arm64_features arch) with
      | Some has, Some needs -> List.for_all (fun f -> List.mem f has) needs
      | _ -> device = arch)

(* [sizes] gives known extents. Only the agent knows a remote rail's extent;
   it checks the bounds again before loading. *)
let check t sizes devices =
  if Array.length devices <> Iarray.length t.devices then
    broken "%d devices for a program of %d" (Array.length devices)
      (Iarray.length t.devices);
  Iarray.iteri
    (fun i arch ->
      let a = Rig.arch devices.(i) in
      if not (runs ~device:a arch) then
        broken "device %d: %s is %S, which does not run %S" i
          (Rig.name devices.(i)) a arch)
    t.devices;
  if t.ints < 0 || t.ints > max_int / 8 then broken "%d ints" t.ints;
  let ndev = Iarray.length t.devices in
  Iarray.iteri
    (fun i -> function
      | Alloc { device; bytes; init; _ } ->
          index "device" ndev device;
          if bytes < 0 then broken "memory %d: %d bytes" i bytes;
          let n = String.length init in
          if n > bytes then
            broken "memory %d: %d bytes of init in %d bytes" i n bytes
      | Rail _ -> ())
    t.memory;
  Iarray.iter (fun (im : image) -> index "device" ndev im.device) t.images;
  Iarray.iteri
    (fun i (input : input) ->
      index "device" ndev input.device;
      if input.bytes < 0 then broken "input %d: %d bytes" i input.bytes)
    t.inputs;
  Iarray.iter (check_step t sizes ndev) t.steps

(* Bytes *)

(* A description as bytes: the magic, the format's version, then the
   description. An integer is 64 bits, little-endian; a string or an array its
   count, then its elements; a record its fields in order; a variant its tag, a
   byte, then its arguments. *)

let magic = "rig.program\n"
let version = 16

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
end

(* Codecs: each type's bytes described once, as its writer and its reader. A
   record is its fields in order ([record], [field], [finish]), a variant its
   cases, tagged by their place in the list ([variant], [case]). *)
module C = struct
  type 'a t = { w : Buffer.t -> 'a -> unit; r : R.r -> 'a }

  let int =
    { w = (fun b n -> Buffer.add_int64_le b (Int64.of_int n)); r = R.int }

  let string =
    let w b s =
      int.w b (String.length s);
      Buffer.add_string b s
    in
    let r (r : R.r) =
      let n = R.count r in
      let s = String.sub r.s r.at n in
      r.at <- r.at + n;
      s
    in
    { w; r }

  let array c =
    let w b a =
      int.w b (Iarray.length a);
      Iarray.iter (c.w b) a
    in
    let r r : _ iarray =
      let n = R.count r in
      if n = 0 then [||]
      else begin
        let a = Array.make n (c.r r) in
        for i = 1 to n - 1 do
          a.(i) <- c.r r
        done;
        Iarray.of_array a
      end
    in
    { w; r }

  (* Records *)

  type ('o, 'f) fields = { fw : Buffer.t -> 'o -> unit; fr : R.r -> 'f }

  let record make = { fw = (fun _ _ -> ()); fr = (fun _ -> make) }

  let field c get f =
    let fw b o =
      f.fw b o;
      c.w b (get o)
    in
    let fr r =
      let k = f.fr r in
      k (c.r r)
    in
    { fw; fr }

  let finish f = { w = f.fw; r = f.fr }

  (* Tuples *)

  let pair a b =
    record (fun x y -> (x, y)) |> field a fst |> field b snd |> finish

  let triple a b c =
    record (fun x y z -> (x, y, z))
    |> field a (fun (x, _, _) -> x)
    |> field b (fun (_, y, _) -> y)
    |> field c (fun (_, _, z) -> z)
    |> finish

  (* Variants

     A variant is its cases, each tagged by its place in the variant's list,
     and one writer that matches on the value, so the compiler checks that it
     writes every constructor. [caseN a1 .. aN inj] is a case whose arguments
     [a1] to [aN] describe and [inj] makes the value of: its [read] reads them
     and makes the value, and its [put b x1 .. xN] writes its tag and them,
     allocating nothing. *)

  type ('a, 'put) case = {
    mutable tag : int; (* its place in its variant's list, once listed *)
    read : R.r -> 'a;
    put : 'put;
  }

  type 'a arm = Arm : ('a, _) case -> 'a arm

  let tag b c =
    if c.tag < 0 then invalid_arg "Rig_program: a case of no variant";
    Buffer.add_uint8 b c.tag

  let case0 v =
    let rec c = { tag = -1; read = (fun _ -> v); put = (fun b -> tag b c) } in
    c

  let case1 a inj =
    let rec c =
      {
        tag = -1;
        read = (fun r -> inj (a.r r));
        put =
          (fun b x ->
            tag b c;
            a.w b x);
      }
    in
    c

  let case2 a b inj =
    let read r =
      let x = a.r r in
      let y = b.r r in
      inj x y
    in
    let rec c =
      {
        tag = -1;
        read;
        put =
          (fun buf x y ->
            tag buf c;
            a.w buf x;
            b.w buf y);
      }
    in
    c

  let case4 a b c d inj =
    let read r =
      let x = a.r r in
      let y = b.r r in
      let z = c.r r in
      let u = d.r r in
      inj x y z u
    in
    let rec k =
      {
        tag = -1;
        read;
        put =
          (fun buf x y z u ->
            tag buf k;
            a.w buf x;
            b.w buf y;
            c.w buf z;
            d.w buf u);
      }
    in
    k

  let case5 a b c d e inj =
    let read r =
      let x = a.r r in
      let y = b.r r in
      let z = c.r r in
      let u = d.r r in
      let v = e.r r in
      inj x y z u v
    in
    let rec k =
      {
        tag = -1;
        read;
        put =
          (fun buf x y z u v ->
            tag buf k;
            a.w buf x;
            b.w buf y;
            c.w buf z;
            d.w buf u;
            e.w buf v);
      }
    in
    k

  let case9 a b c d e f g h i inj =
    let read r =
      let x = a.r r in
      let y = b.r r in
      let z = c.r r in
      let u = d.r r in
      let v = e.r r in
      let s = f.r r in
      let t = g.r r in
      let w = h.r r in
      let q = i.r r in
      inj x y z u v s t w q
    in
    let rec k =
      {
        tag = -1;
        read;
        put =
          (fun buf x y z u v s t w q ->
            tag buf k;
            a.w buf x;
            b.w buf y;
            c.w buf z;
            d.w buf u;
            e.w buf v;
            f.w buf s;
            g.w buf t;
            h.w buf w;
            i.w buf q);
      }
    in
    k

  (* [variant what arms w] is the variant of the cases [arms], in tag order,
     written by [w], which puts each value through its case. *)
  let variant what arms w =
    let reads =
      Array.of_list
        (List.mapi
           (fun i (Arm c) ->
             if c.tag >= 0 then invalid_arg "Rig_program: a case listed twice";
             c.tag <- i;
             c.read)
           arms)
    in
    let r r =
      let n = R.tag r in
      if n >= Array.length reads then R.fail r "%s of tag %d" what n;
      reads.(n) r
    in
    { w; r }

  (* A variant whose cases take no arguments, [values] in tag order and [tag]
     each value's place among them, which its construction checks. *)
  let enum what values tag =
    List.iteri
      (fun i v ->
        if tag v <> i then invalid_arg ("Rig_program: the tags of " ^ what))
      values;
    let values = Array.of_list values in
    let r r =
      let n = R.tag r in
      if n >= Array.length values then R.fail r "%s of tag %d" what n;
      values.(n)
    in
    { w = (fun b v -> Buffer.add_uint8 b (tag v)); r }

  let option c =
    let none = case0 None and some = case1 c Option.some in
    variant "an option" [ Arm none; Arm some ] (fun b -> function
      | None -> none.put b
      | Some x -> some.put b x)

  (* The codec [f] makes of itself, for a type that holds itself. [f] only
     keeps the codec it is given, so the codec it makes is in place before
     anything writes or reads. *)
  let fix f =
    let self = ref None in
    let get () = Option.get !self in
    let c = { w = (fun b x -> (get ()).w b x); r = (fun r -> (get ()).r r) } in
    self := Some (f c);
    c
end

(* The description's codecs. *)
module D = struct
  open C

  let value =
    record (fun add terms -> { add; terms })
    |> field int (fun (v : value) -> v.add)
    |> field (array (pair int int)) (fun v -> v.terms)
    |> finish

  let width = enum "a width" [ W32; W64 ] (function W32 -> 0 | W64 -> 1)

  let scalar =
    record (fun at width value -> { at; width; value })
    |> field int (fun (s : scalar) -> s.at)
    |> field width (fun s -> s.width)
    |> field value (fun s -> s.value)
    |> finish

  let kind =
    enum "a memory kind"
      B.[ Device; Pinned; Mapped ]
      (function B.Device -> 0 | Pinned -> 1 | Mapped -> 2)

  let use =
    enum "a use" [ Read; Write; Unordered ] (function
      | Read -> 0
      | Write -> 1
      | Unordered -> 2)

  let ref =
    record (fun slot at use -> { slot; at; use })
    |> field int (fun (r : ref) -> r.slot)
    |> field int (fun (r : ref) -> r.at)
    |> field use (fun (r : ref) -> r.use)
    |> finish

  let area =
    enum "an area" [ Outbound; Inbound; Counts ] (function
      | Outbound -> 0
      | Inbound -> 1
      | Counts -> 2)

  let memory =
    let alloc =
      case4 int kind int string (fun device kind bytes init ->
          Alloc { device; kind; bytes; init })
    and rail = case2 int area (fun rail area -> Rail { rail; area }) in
    variant "a memory" [ Arm alloc; Arm rail ] (fun b -> function
      | Alloc { device; kind; bytes; init } -> alloc.put b device kind bytes init
      | Rail { rail = r; area } -> rail.put b r area)

  let view =
    record (fun memory offset length -> { memory; offset; length })
    |> field int (fun (v : view) -> v.memory)
    |> field int (fun (v : view) -> v.offset)
    |> field int (fun (v : view) -> v.length)
    |> finish

  let slot =
    let memory = case1 view (fun v -> Memory v)
    and input = case1 int (fun i -> Input i)
    and ints = case0 Ints in
    variant "a slot" [ Arm memory; Arm input; Arm ints ] (fun b -> function
      | Memory v -> memory.put b v
      | Input i -> input.put b i
      | Ints -> ints.put b)

  let sizes = triple value value value

  let part =
    let words = case1 view (fun v -> Words v)
    and copy =
      case5 int value int value int
        (fun src src_offset dst dst_offset length ->
          Copy { src; src_offset; dst; dst_offset; length })
    and launch =
      case9 int string string string (array scalar) (array ref) sizes sizes
        value
        (fun image name label params scalars refs groups threads shared ->
          Launch
            {
              image;
              name;
              label;
              params;
              scalars;
              refs;
              groups;
              threads;
              shared;
            })
    in
    variant "a part" [ Arm words; Arm copy; Arm launch ] (fun b -> function
      | Words v -> words.put b v
      | Copy { src; src_offset; dst; dst_offset; length } ->
          copy.put b src src_offset dst dst_offset length
      | Launch
          {
            image;
            name;
            label;
            params;
            scalars;
            refs;
            groups;
            threads;
            shared;
          } ->
          launch.put b image name label params scalars refs groups threads
            shared)

  let submit =
    record (fun device parts slots -> { device; parts; slots })
    |> field int (fun (s : submit) -> s.device)
    |> field (array part) (fun s -> s.parts)
    |> field (array slot) (fun s -> s.slots)
    |> finish

  let step =
    fix @@ fun step ->
    let submit = case1 submit (fun s -> Submit s)
    and move =
      case5 slot value slot value int
        (fun src src_offset dst dst_offset length ->
          Move { src; src_offset; dst; dst_offset; length })
    and loop =
      case4 value (option int)
        (option (pair slot int))
        (array step)
        (fun trips trip flag body -> Loop { trips; trip; flag; body })
    and send = case2 int value (fun rail count -> Send { rail; count }) in
    variant "a step" [ Arm submit; Arm move; Arm loop; Arm send ]
      (fun b -> function
      | Submit s -> submit.put b s
      | Move { src; src_offset; dst; dst_offset; length } ->
          move.put b src src_offset dst dst_offset length
      | Loop { trips; trip; flag; body } -> loop.put b trips trip flag body
      | Send { rail; count } -> send.put b rail count)

  let image =
    record (fun device binary -> { device; binary })
    |> field int (fun (i : image) -> i.device)
    |> field string (fun i -> i.binary)
    |> finish

  let input =
    record (fun device bytes -> { device; bytes })
    |> field int (fun (i : input) -> i.device)
    |> field int (fun (i : input) -> i.bytes)
    |> finish

  let description =
    record (fun devices memory images inputs ints steps ->
        { devices; memory; images; inputs; ints; steps })
    |> field (array string) (fun t -> t.devices)
    |> field (array memory) (fun t -> t.memory)
    |> field (array image) (fun t -> t.images)
    |> field (array input) (fun t -> t.inputs)
    |> field int (fun t -> t.ints)
    |> field (array step) (fun t -> t.steps)
    |> finish

  (* Each device's id on the machine that loads a program another sent. *)
  let ids = array int
end

let write_description b t =
  Buffer.add_string b magic;
  C.int.w b version;
  D.description.w b t

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
        match D.description.r r with
        | t when R.left r = 0 -> Ok t
        | _ -> Error (strf "%d bytes after a program description" (R.left r))
        | exception R.Malformed (at, why) -> malformed at why)
  end

(* Loading *)

(* A launch whose values a run changes before each submit: its index among
   the submit's values, the scalars that read the ints, and its geometry
   where a run changes it. *)
type block_run = {
  index : int;
  scalars : scalar array;
  geometry : ((value * value * value) * (value * value * value) * value) option;
}

(* A copy's end whose buffer each run passes: [count] bytes of the slot
   [from], at the byte [offset] sums to. *)
type moving = {
  index : int; (* Of [sub]'s slots. *)
  from : slot;
  held : B.t; (* [from]'s buffer, or [unset] for an input. *)
  offset : value;
  count : int;
}

(* A step's submission for one copy of the ints, the buffers it is passed,
   memory set at load, inputs and moving copy ends at each run, and the
   values of its launches, constant bytes set at load and the rest at each
   run. Runs of a program take turns, and a submit copies the values, so
   each run writes them in place. *)
type prepared = {
  sub : Sub.t;
  buffers : B.t array;
  input_slots : (int * int) array;
      (* Each slot of the submission that holds an input, with the input. *)
  moving : moving array;
  args : Sub.args array;
  blocks : block_run array;
}

type lstep =
  | Lsubmit of { spec : submit; copies : prepared array; step : int }
    (* [step] is the index among the description's steps of the step, or of
       the loop that holds it. *)
  | Lmove of {
      src : slot;
      src_offset : value;
      dst : slot;
      dst_offset : value;
      boxes : B.box array; (* One row of the move's length. *)
    }
  | Lloop of {
      trips : value;
      trip : int option;
      flag : (slot * int) option;
      body : lstep array;
    }
  | Lsend of {
      end_ : Rig_remote_abi.end_;
      outbound : B.t array; (* The program's memory of its [Outbound]. *)
      count : value;
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
    | Send _ -> ()
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
    | Move _ | Send _ -> ()
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
    | Words _ | Copy _ -> ()
  in
  let rec step = function
    | Submit s -> Iarray.iter part s.parts
    | Loop { body; _ } -> Iarray.iter step body
    | Move _ | Send _ -> ()
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
    | Send _ -> ()
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
      let unchanged (v : value) =
        Iarray.for_all (fun (w, _) -> not (List.mem w trips)) v.terms
      in
      let reads v = not (is_constant v) in
      let watch k (_, kernel, ((x, y, z) as sizes)) =
        match limits.(k) with
        | Some most
          when List.for_all unchanged [ x; y; z ] && List.exists reads [ x; y; z ]
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
      (* Each memory, on its device, then the two copies of the run's ints. *)
  ints : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t array;
      (* Each copy of the run's ints, as words. *)
  given : int array;
      (* The frame's ints, its first [ngiven] words, as the run read them
         before it waits for run [n - 2]. *)
  mutable ngiven : int;
  borrows : (int * int * int, B.t) Hashtbl.t;
      (* By memory, copy and device: the borrows made at load, and those of
         host memory a move made. *)
  unborrowed : (int * int * int, unit) Hashtbl.t;
      (* By memory, copy and device: host memory a move's device did not
         borrow, which its moves stage. *)
  mutable images : Rig.Image.t array; (* Empty until they load. *)
  funcs : (int * string, Rig.Image.func) Hashtbl.t;
      (* By image and name: the functions resolved so far. *)
  rails : int -> Rig_remote_abi.end_ option;
  kept :
    (int * string, Rig.Image.func) Hashtbl.t
    * Rig_remote_abi.end_ list Stdlib.ref;
      (* What every submission keeps until its work is done: the functions,
         and the rail ends [rails] gave, whose memory its work may use on any
         device. *)
  mutable steps : lstep array;
  flag : bytes; (* The byte a loop's flag is read into. *)
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
  | None -> broken "no rail %d on this machine" id

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

(* The copy of memory [m] that run [n] uses, for [k = n mod 2]: the ints have
   two copies, and the program's memory one. *)
let copy_of p m k = if Array.length p.mem.(m) = 2 then k else 0

let view p k d (v : view) =
  B.view
    (on p v.memory (copy_of p v.memory k) d)
    ~first:v.offset ~length:v.length

(* Image [image]'s function [name], resolved at its first need. *)
let func p image name =
  match Hashtbl.find_opt p.funcs (image, name) with
  | Some fn -> fn
  | None -> (
      let i = p.images.(image) in
      match Rig.Image.resolve i name with
      | Some fn ->
          Hashtbl.replace p.funcs (image, name) fn;
          fn
      | None ->
          refuse "%s: image %d has no function %S"
            (Rig.name (Rig.Image.device i))
            image name
      | exception Invalid_argument why -> refuse "%s" why)

let is_ints = function Ints -> true | Memory _ | Input _ -> false

(* Whether [s] names the run's ints, so that it is made once per copy of
   them. *)
let per_copy (s : submit) = Iarray.exists is_ints s.slots

(* Writes the scalar [s] of value [v], which [s.width] holds, into the
   parameters [b], little-endian, as rig reads them. *)
let store_scalar b s v =
  match s.width with
  | W32 -> Bytes.set_int32_le b s.at (Int32.of_int v)
  | W64 -> Bytes.set_int64_le b s.at (Int64.of_int v)

(* What load sets of [v]: its sum where it is constant, and [0] for a sum
   each run sets. *)
let at_load (v : value) = if is_constant v then v.add else 0

(* Whether threads per group of [x] by [y] by [z], each in [0] to
   [2^32 - 1], exceed [most], as every submit refuses them
   ({!Rig.section-valid}): [x * y] past [most] whatever [z] is, or the
   product of the three where [z] is known, [-1] where it is not. A [most]
   of [0] bounds nothing, as for a function whose driver states none. The
   products are compared a factor at a time, so none overflows: [x * y] is
   at most [most] where it is taken. *)
let past_most ~most x y z =
  most > 0 && x > 0 && y > 0
  && (y > most / x || (z >= 0 && z > most / (x * y)))

(* Threads [x] by [y] by [z], [z] [-1] where it is not known. *)
let threads_of x y z =
  if z >= 0 then strf "%dx%dx%d" x y z else strf "%dx%d in x and y" x y

(* Refuses at load the constant sizes of a launch of [fn] that every submit
   would refuse ({!Rig.section-valid}): one outside [0] to [2^32 - 1], or
   threads per group past [fn]'s most, where its driver knows it and [x]
   and [y] are constant. Each submit checks the sizes a run sets. *)
let check_sizes fn label (gx, gy, gz) (tx, ty, tz) shared =
  let word (v : value) =
    if is_constant v && (v.add < 0 || v.add > 0xffff_ffff) then
      invalid_arg
        (strf "%s has a size of %d, negative or above 2^32 - 1" label v.add)
  in
  List.iter word [ gx; gy; gz; tx; ty; tz; shared ];
  let z = Option.value (known tz) ~default:(-1) in
  match (Rig.Image.threads fn, known tx, known ty) with
  | Some most, Some x, Some y when past_most ~most x y z ->
      invalid_arg
        (strf "%s runs at most %d threads per group; the launch sets %s" label
           most (threads_of x y z))
  | _ -> ()

(* The buffer an input's slot holds until a run passes the input. *)
let unset = B.of_string ""

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

(* The submission of [s] for copy [k] of the ints, with its buffers and its
   launches' values as load knows them. *)
let prepared p k (s : submit) =
  let d = s.device in
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
    | Copy { src; src_offset; dst; dst_offset; length } ->
        (* An end of memory at an offset load knows is that memory's bytes,
           set at make; any other end is a slot each run passes. *)
        let side i offset =
          match (Iarray.get s.slots i, known offset) with
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
              moving := { index = j; from = slot; held; offset; count = length }
                        :: !moving;
              Sub.Slot j
        in
        let src = side src src_offset in
        Sub.copy ~src ~dst:(side dst dst_offset)
    | Launch { image; name; label; params; refs = rs; _ } ->
        Sub.launch ~refs:(refs rs) ~label (func p image name)
          ~params:(String.length params)
  in
  let parts = Iarray.to_array (Iarray.map part s.parts) in
  let sub = Sub.make ~keep:p.kept p.devices.(d) parts in
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
  (* Each launch's values, in the order of [s]'s parts, with the constant
     bytes, scalars and sizes set, and the launches whose values a run
     changes. *)
  let values = ref [] and blocks = ref [] and n = ref 0 in
  let launch = function
    | Words _ | Copy _ -> ()
    | Launch
        {
          image;
          name;
          label;
          params;
          scalars;
          groups = (gx, gy, gz) as groups;
          threads = (tx, ty, tz) as threads;
          shared;
          _;
        } ->
        check_sizes (func p image name) label groups threads shared;
        let b = Bytes.of_string params in
        Iarray.iter
          (fun sc ->
            if is_constant sc.value then store_scalar b sc sc.value.add)
          scalars;
        values :=
          Sub.
            {
              groups = (at_load gx, at_load gy, at_load gz);
              threads = (at_load tx, at_load ty, at_load tz);
              shared = at_load shared;
              params = b;
            }
          :: !values;
        let per_run sc = not (is_constant sc.value) in
        let scalars = List.filter per_run (Iarray.to_list scalars) in
        let geometry =
          if List.for_all is_constant [ gx; gy; gz; tx; ty; tz; shared ] then
            None
          else Some (groups, threads, shared)
        in
        if not (List.is_empty scalars && Option.is_none geometry) then
          blocks :=
            { index = !n; scalars = Array.of_list scalars; geometry }
            :: !blocks;
        incr n
  in
  Iarray.iter launch s.parts;
  {
    sub;
    buffers;
    input_slots = Array.of_list !inputs;
    moving = Array.of_list (List.rev !moving);
    args = Array.of_list (List.rev !values);
    blocks = Array.of_list (List.rev !blocks);
  }

(* A submission [Rig.Submission.make] refuses, or a launch's constant size
   every submit would refuse, is a defect of the description. *)
let submission p i k s =
  try prepared p k s with Invalid_argument why -> refuse "step %d: %s" i why

(* The program's memory of the [Outbound] area of the rail [rail]. *)
let outbound p rail =
  let views = ref [] in
  Iarray.iteri
    (fun m -> function
      | Rail { rail = r; area = Outbound } when r = rail ->
          views := p.mem.(m).(0) :: !views
      | Rail _ | Alloc _ -> ())
    p.t.memory;
  Array.of_list !views

(* Step [i]'s loaded form. *)
let rec lstep p i = function
  | Submit s ->
      let n = if per_copy s then 2 else 1 in
      let copies = Array.init n (fun k -> submission p i k s) in
      Lsubmit { spec = s; copies; step = i }
  | Move { src; src_offset; dst; dst_offset; length } ->
      Lmove { src; src_offset; dst; dst_offset; boxes = [| B.row length |] }
  | Loop { trips; trip; flag; body } ->
      let body = Array.map (lstep p i) (Iarray.to_array body) in
      Lloop { trips; trip; flag; body }
  | Send { rail; count } ->
      Lsend { end_ = rail_end p.rails rail; outbound = outbound p rail; count }

(* A device borrows host memory that starts on a page, as a host buffer of 64
   KiB or more does (rig.mli, [Buffer.create]). *)
let page_bytes = 65536

let rec names_ints = function
  | Submit s -> Iarray.exists is_ints s.slots
  | Move { src; dst; _ } -> is_ints src || is_ints dst
  | Loop { body; _ } -> Iarray.exists names_ints body
  | Send _ -> false

(* Whether a move of [step] names memory [m]. *)
let rec names_memory m = function
  | Submit _ | Send _ -> false
  | Move { src; dst; _ } ->
      let is = function Memory v -> v.memory = m | Ints | Input _ -> false in
      is src || is dst
  | Loop { body; _ } -> Iarray.exists (names_memory m) body

(* The bytes of the buffer made for memory [m] of [t] on its device [d]: on
   the host and named by a move, a page's worth at least, so that a device can
   borrow it, as the ints are. *)
let memory_bytes (t : t) d m bytes =
  if Rig.equal d Rig.host && Iarray.exists (names_memory m) t.steps then
    max bytes page_bytes
  else bytes

(* Memory [m] of [t]: a view of its {!memory_bytes}. *)
let make_memory (t : t) devices rails m = function
  | Alloc { device; kind; bytes; _ } ->
      let d = devices.(device) in
      let n = memory_bytes t d m bytes in
      if n = bytes then [| B.create ~memory:kind d bytes |]
      else [| B.view (B.create ~memory:kind d n) ~first:0 ~length:bytes |]
  | Rail { rail; area } ->
      [| B.of_bigarray (rail_area (rail_end rails rail) area) |]

let load_image p (im : image) =
  match Rig.Image.load p.devices.(im.device) im.binary with
  | Ok i -> i
  | Error why -> raise (Refused why)

let write_init p m =
  match Iarray.get p.t.memory m with
  | Rail _ -> ()
  | Alloc { init; _ } ->
      let n = String.length init in
      if n > 0 then
        B.copy ~src:(B.of_string init)
          ~dst:(B.view p.mem.(m).(0) ~first:0 ~length:n)

let words b =
  if B.length b = 0 then Bigarray.(Array1.create int64 c_layout 0)
  else B.export b Bigarray.int64

(* The bytes of the host buffer made for each of the two copies of the run's
   ints: a page's worth at least where a step names them, so that a device
   can borrow it. *)
let ints_bytes (t : t) =
  let n = 8 * t.ints in
  if Iarray.exists names_ints t.steps then max n page_bytes else n

(* One copy of the run's ints: a view of its {!ints_bytes}. *)
let make_ints (t : t) =
  let n = 8 * t.ints and m = ints_bytes t in
  if m = n then B.create Rig.host n
  else B.view (B.create Rig.host m) ~first:0 ~length:n

let footprint t devices =
  let devices = Iarray.to_list devices in
  let held = List.map (fun d -> (d, ref 0)) devices in
  let held =
    if List.exists (fun d -> Rig.equal d Rig.host) devices then held
    else held @ [ (Rig.host, ref 0) ]
  in
  let add d n =
    let total = snd (List.find (fun (d', _) -> Rig.equal d d') held) in
    total := !total + n
  in
  Iarray.iteri
    (fun m -> function
      | Alloc { device; kind; bytes; _ } ->
          let d = List.nth devices device in
          let n = memory_bytes t d m bytes in
          add (Rig.charged ~memory:kind d) (Rig.allocation ~memory:kind d n)
      | Rail _ -> ())
    t.memory;
  add Rig.host (2 * Rig.allocation Rig.host (ints_bytes t));
  List.map (fun (d, n) -> (d, !n)) held

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
  check t (Iarray.to_array (Iarray.map size t.memory)) devices;
  (* A send's rail is checked before load makes anything, as Rail memory's. *)
  let rec sends steps =
    Iarray.iter
      (function
        | Send { rail; _ } -> ignore (rail_end rails rail)
        | Loop { body; _ } -> sends body
        | Submit _ | Move _ -> ())
      steps
  in
  sends t.steps;
  let ints = Array.init 2 (fun _ -> make_ints t) in
  let mem =
    Array.append
      (Iarray.to_array (Iarray.mapi (make_memory t devices rails) t.memory))
      [| ints |]
  in
  let pairs = Iarray.length t.inputs * Array.length devices in
  let funcs = Hashtbl.create 8 in
  let p =
    {
      t;
      devices;
      mem;
      ints = Array.map words ints;
      given = Array.make t.ints 0;
      ngiven = 0;
      borrows = Hashtbl.create 8;
      unborrowed = Hashtbl.create 8;
      images = [||];
      funcs;
      rails;
      kept = (funcs, ends);
      steps = [||];
      flag = Bytes.create 1;
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
    p.images <- Iarray.to_array (Iarray.map (load_image p) t.images);
    (* Every launched function starts being made before the steps resolve
       the first, so that a device makes them concurrently. *)
    Array.iter
      (fun (image, name, _) ->
        if not (Hashtbl.mem funcs (image, name)) then
          Rig.Image.prepare p.images.(image) name)
      (launches t);
    p.steps <- Array.mapi (lstep p) (Iarray.to_array t.steps);
    Iarray.iteri (fun m _ -> write_init p m) t.memory
  with
  | () -> Ok p
  | exception Refused why -> Error why

(* Running *)

let invalid fmt = Printf.ksprintf invalid_arg ("Rig_program.run: " ^^ fmt)

(* The frame's counts: as many inputs as [t]'s, and at most [t]'s ints. *)
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

(* [v]'s sum over copy [k] of the ints, or [Overflow] past an [int]. *)
let value_at p k (v : value) =
  let s = ref v.add in
  for i = 0 to Iarray.length v.terms - 1 do
    let w, factor = Iarray.get v.terms i in
    s := add_exn !s (mul_exn (int_at p k w) factor)
  done;
  !s

(* [value_at], refused past an [int] as the sum of [what]. *)
let sum_at p k what v =
  try value_at p k v with Overflow -> invalid "%s past an int" what

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

(* Writes [q]'s scalars for copy [k] of the ints into its launches'
   parameters, gives a launch whose geometry moves new values, and passes
   the submit's buffers. A run's sizes are checked by its submit. *)
let prepare_step p k (q : prepared) d =
  for i = 0 to Array.length q.blocks - 1 do
    let l = q.blocks.(i) in
    let a = q.args.(l.index) in
    for j = 0 to Array.length l.scalars - 1 do
      let s = l.scalars.(j) in
      let v = sum_at p k "a parameter's value" s.value in
      if not (fits s.width v) then
        invalid
          "the parameter at byte %d takes %d, which 32 bits hold neither \
           signed nor unsigned"
          s.at v;
      store_scalar a.params s v
    done;
    match l.geometry with
    | None -> ()
    | Some ((gx, gy, gz), (tx, ty, tz), shared) ->
        let g = "a launch's geometry" in
        let groups = (sum_at p k g gx, sum_at p k g gy, sum_at p k g gz)
        and threads = (sum_at p k g tx, sum_at p k g ty, sum_at p k g tz)
        and shared = sum_at p k g shared in
        q.args.(l.index) <- { a with groups; threads; shared }
  done;
  pass p q d;
  for x = 0 to Array.length q.moving - 1 do
    let e = q.moving.(x) in
    let held = match e.from with Input j -> bound p j d | _ -> e.held in
    let o =
      try value_at p k e.offset
      with Overflow ->
        unpass q;
        invalid "a copy's offset past an int"
    in
    let n = B.length held in
    if o < 0 || o > n - e.count then begin
      unpass q;
      invalid "a copy of %d bytes from byte %d of a slot of %d bytes" e.count o
        n
    end;
    q.buffers.(e.index) <-
      (if o = 0 && e.count = n then held
       else B.view held ~first:o ~length:e.count)
  done

(* Records [point] as the run's last on its device [d]. *)
let set_last (p : here) d point =
  if Array.length p.last = 0 then
    p.last <- Array.make (Array.length p.devices) point;
  p.last.(d) <- point;
  p.ran.(d) <- true

(* The program's lock admits one run at a time, so the step's values are
   this run's to write. With a [timer], a run of [time]'s, the submit is
   timed as its [step]. *)
let submit_step p timer k waits (spec : submit) copies step =
  let q = copies.(if Array.length copies = 2 then k else 0) in
  let d = spec.device in
  prepare_step p k q d;
  let waits = if p.ran.(d) then None else waits in
  match
    match timer with
    | None -> Rig.submit ?waits q.sub ~buffers:q.buffers ~args:q.args
    | Some t ->
        Rig.Timer.submit t step ?waits q.sub ~buffers:q.buffers ~args:q.args
  with
  | point ->
      unpass q;
      set_last p d point
  | exception e ->
      unpass q;
      raise e

(* The [length] bytes of [slot]'s buffer from the byte [offset] sums to for
   copy [k] of the ints, as a move's end. *)
let move_end p k slot offset length =
  let b, first, n =
    match slot with
    | Memory v -> (p.mem.(v.memory).(copy_of p v.memory k), v.offset, v.length)
    | Ints ->
        let b = ints_buffer p k in
        (b, 0, B.length b)
    | Input j ->
        let b = bound p j (Iarray.get p.t.inputs j).device in
        (b, 0, B.length b)
  in
  let o = sum_at p k "a move's offset" offset in
  if o < 0 || o > n - length then
    invalid "a move of %d bytes from byte %d of a slot of %d bytes" length o n;
  if first + o = 0 && length = B.length b then b
  else B.view b ~first:(first + o) ~length

(* The index of [point]'s device among [p]'s devices, or [-1]. *)
let device_index (p : here) point =
  match Rig.Point.device point with
  | None -> -1
  | Some d ->
      let i = ref (-1) and j = ref 0 in
      while !i < 0 && !j < Array.length p.devices do
        if Rig.equal d p.devices.(!j) then i := !j;
        incr j
      done;
      !i

(* The index among [p]'s devices of [d], or [-1]. *)
let index_of (p : here) d =
  let i = ref (-1) and j = ref 0 in
  while !i < 0 && !j < Array.length p.devices do
    if Rig.equal d p.devices.(!j) then i := !j;
    incr j
  done;
  !i

(* The index among [p.mem] of the program's memory [slot] names, or [-1] for
   an input. *)
let slot_memory p = function
  | Memory v -> v.memory
  | Ints -> Iarray.length p.t.memory
  | Input _ -> -1

(* A move's end [b] in the program's host memory, of [slot], as the device of
   [other], the move's other end, names it: a view of the memory's borrow on
   that device where it is one of [p]'s and can borrow it, asked once, kept
   with [p]'s borrows or its refusals. A copy from the host memory itself
   would pass through staging memory, and one into it would make the host
   wait for its bytes. Elsewhere [b] as it is. *)
let named p k slot b other =
  let m = slot_memory p slot in
  if m < 0 || not (Rig.equal (B.device b) Rig.host) then b
  else
    let d = index_of p (B.device other) in
    if d < 0 || Rig.equal p.devices.(d) Rig.host then b
    else
      let c = copy_of p m k in
      let whole = p.mem.(m).(c) and key = (m, c, d) in
      let borrowed =
        match Hashtbl.find_opt p.borrows key with
        | Some _ as found -> found
        | None when Hashtbl.mem p.unborrowed key -> None
        | None -> (
            (* A device that cannot borrow the memory, or has no room to
               map it, stages it: the move costs a wait, never the run. *)
            match B.borrow p.devices.(d) whole with
            | Some b ->
                Hashtbl.replace p.borrows key b;
                Some b
            | None | (exception Rig.Out_of_memory _) ->
                Hashtbl.replace p.unborrowed key ();
                None)
      in
      match borrowed with
      | None -> b
      | Some w ->
          B.view w ~first:(B.offset b - B.offset whole) ~length:(B.length b)

(* A move follows [waits], since it may be the first work of whichever device
   runs it. Its point becomes the run's last on its device where that device
   is one of [p]'s. Otherwise the step waits for it, as where a device outside
   the program copies from a borrow of its memory, so that the run's ends
   cover every move. A move the host makes is done when [submit_copy]
   returns. *)
let move_step p k waits ~src ~src_offset ~dst ~dst_offset boxes =
  let length = boxes.(0).B.width in
  let s = move_end p k src src_offset length
  and t = move_end p k dst dst_offset length in
  let src = named p k src s t and dst = named p k dst t s in
  let point = B.submit_copy ?waits ~src ~dst boxes in
  match device_index p point with
  | -1 -> Rig.Point.wait point
  | d -> set_last p d point

(* Whether byte [at] of [slot]'s buffer is not [0], once the work that wrote
   it is done. *)
let flag_holds p k (slot, at) =
  let b =
    match slot with
    | Memory v -> p.mem.(v.memory).(copy_of p v.memory k)
    | Ints -> ints_buffer p k
    | Input j -> bound p j (Iarray.get p.t.inputs j).device
  in
  let first = match slot with Memory v -> v.offset + at | _ -> at in
  if first >= B.length b then
    invalid "a loop's flag at byte %d of an input of %d bytes" at
      (B.length b);
  B.blit_to_bytes b first p.flag 0 1;
  Bytes.get p.flag 0 <> '\000'

(* Tells the rail's end [e] that the transfer of count [count] lies in its
   [Outbound], once the points of [waits] are reached and the work that writes
   the program's memory of that area, [outbound], is done. [waits] may hold
   work of another program that writes the area through memory of its own,
   which no stamp of [outbound] records. *)
let send_step p k waits (e : Rig_remote_abi.end_) outbound count =
  Option.iter (Array.iter Rig.Point.wait) waits;
  for i = 0 to Array.length outbound - 1 do
    B.wait outbound.(i) Rig.Read
  done;
  let c = sum_at p k "a send's count" count in
  if c < 1 then invalid "a send of count %d, below 1" c;
  e.ready c

let rec exec (p : here) timer k waits = function
  | Lmove { src; src_offset; dst; dst_offset; boxes } ->
      move_step p k waits ~src ~src_offset ~dst ~dst_offset boxes
  | Lsubmit { spec; copies; step } ->
      submit_step p timer k waits spec copies step
  | Lsend { end_; outbound; count } -> send_step p k waits end_ outbound count
  | Lloop { trips; trip; flag; body } ->
      (* A loop without closures: a run allocates only its answer. *)
      let n = sum_at p k "a loop's trips" trips in
      let i = ref 0 in
      while
        !i < n && match flag with None -> true | Some f -> flag_holds p k f
      do
        (match trip with None -> () | Some w -> set_trip p k w !i);
        for s = 0 to Array.length body - 1 do
          exec p timer k waits body.(s)
        done;
        incr i
      done

(* The run's last point on each device that ran its submissions or moves, in
   the devices' order. *)
let ends (p : here) =
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

let run_steps ~waits ~timer (p : here) (f : frame) =
  bind p f;
  let k = p.runs land 1 in
  p.runs <- p.runs + 1;
  B.wait (ints_buffer p k) Rig.Read_write;
  let ints = p.ints.(k) in
  for i = 0 to p.ngiven - 1 do
    Bigarray.Array1.set ints i (Int64.of_int p.given.(i))
  done;
  Array.fill p.ran 0 (Array.length p.ran) false;
  for s = 0 to Array.length p.steps - 1 do
    exec p timer k waits p.steps.(s)
  done;
  ends p

(* Re-entry

   A call of rig's from a signal handler, finaliser or Gc.Memprof callback
   inside a lock of its thread's raises as it starts. A load or a run checks
   first, since its later calls of rig would raise after it changed its
   state. A run holds its program's lock across its calls of rig, so that
   lock does not count among the thread's: a callback between them runs as
   any code does. *)

let check_held fn =
  if Rig_edge_primitive.holds () <> 0 then
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
  match run_steps ~waits ~timer:None p f with
  | points ->
      unbind p;
      Mutex.unlock p.lock;
      points
  | exception e ->
      unbind p;
      Mutex.unlock p.lock;
      raise e

(* Timing *)

let is_loop = function Loop _ -> true | Submit _ | Move _ | Send _ -> false

(* Refuses a program with a loop, a step of which a run may submit once per
   trip, before any work. *)
let check_unlooped (t : t) =
  if Iarray.exists is_loop t.steps then
    invalid_arg "Rig_program.time: a program with a loop"

(* [run_here] with each step's submit timed, once the run's work is done: its
   ends and its steps' times. Without a loop each step submits at most once.
   The lock covers the steps alone, so another run of [p] may start while
   this one's work is waited for. *)
let time_here ~waits (p : here) (f : frame) =
  let timer = Rig.Timer.make (Iarray.length p.t.steps) in
  lock p.lock;
  match run_steps ~waits ~timer:(Some timer) p f with
  | exception e ->
      unbind p;
      Mutex.unlock p.lock;
      Rig.Timer.drop timer;
      raise e
  | ends -> (
      unbind p;
      Mutex.unlock p.lock;
      match
        Array.iter Rig.Point.wait ends;
        Rig.Timer.read timer
      with
      | times -> (ends, times)
      | exception e ->
          Rig.Timer.drop timer;
          raise e)

(* Programs of another machine *)

(* The binary [load] sends another machine's host: the magic, then each device's
   id on that machine, then the description's bytes. A run's words: the
   program's entry, the memory id of the region a time's agent writes the
   steps' times into or [no_times] for a run, each input's memory id, offset
   and length, the number of the frame's ints, then the description's [ints]
   words, the frame's first. *)

let share_magic = "rig.share\n"
let no_times = -1

(* What a time of a program of another machine uses: the region on that
   machine's host that its agent writes the steps' times into, as
   little-endian 64-bit integers, and a submission of the run's words and a
   copy of the region into its slot 0, the time's own memory of this
   process. *)
type times_there = { region : B.t; timed : Sub.t }

(* A program loaded on another machine, which a submission on its host runs. *)
type there = {
  t : t;
  devices : Rig.t array;
  access : Rig.access array;
  binds : bind array;
  limits : int option array; (* What the agent stated at load ([threads]). *)
  watch : watched array;
  entry : int;
  sub : Sub.t; (* Of the run's words alone. *)
  times : times_there option;
      (* [None] for a program that has a loop, which [time] refuses, or no
         [Submit] step, whose times are all [0]. *)
  words :
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
  lock : Mutex.t;
}

type loaded = Here of here | There of there

let words_bytes (t : t) =
  8 + 8 + (24 * Iarray.length t.inputs) + 8 + (8 * t.ints)

(* Where a run's words hold each input's memory id, offset and length. *)
let input_word i = 16 + (24 * i)

let is_submit = function Submit _ -> true | Loop _ | Move _ | Send _ -> false

(* [t]'s times on another machine, whose host is [host], with the words of
   [words] run as [run]. *)
let times_there (t : t) host words run =
  if Iarray.exists is_loop t.steps || not (Iarray.exists is_submit t.steps)
  then None
  else
    let region = B.create host (8 * Iarray.length t.steps) in
    let copy = Sub.copy ~src:(Sub.Buffer region) ~dst:(Sub.Slot 0) in
    Some { region; timed = Sub.make ~keep:run host [| Sub.words words; copy |] }

(* [d]'s id on its machine, as its agent names it. *)
let proxy_id d =
  match Rig.capability d Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host _) -> 0
  | Some (Rig_remote_abi.Device d) -> d.Rig_remote_abi.id
  | None -> broken "%s is no device of an agent" (Rig.name d)

let load_there t devices =
  (* The areas of rails are the agent's to know: their views are checked
     there. *)
  let size = function Alloc { bytes; _ } -> Some bytes | Rail _ -> None in
  check t (Iarray.to_array (Iarray.map size t.memory)) devices;
  let inputs = Iarray.length t.inputs in
  if
    inputs > (max_int - 24) / 24 || t.ints > (max_int - 24 - (24 * inputs)) / 8
  then
    broken "%d inputs and %d ints: more than a run's words hold" inputs t.ints;
  let host = Rig.host_of devices.(0) in
  let ids, threads =
    match Rig.capability host Rig_remote_abi.key with
    | Some (Rig_remote_abi.Host { threads; _ }) ->
        (Array.map proxy_id devices, threads)
    | Some (Rig_remote_abi.Device _) | None ->
        broken "%s is no host of an agent" (Rig.name host)
  in
  let b = Buffer.create 4096 in
  Buffer.add_string b share_magic;
  D.ids.w b (Iarray.of_array ids);
  write_description b t;
  match Rig.Image.load host (Buffer.contents b) with
  | Error why -> Error why
  | Ok image -> (
      match Rig.Image.resolve image "run" with
      | exception Invalid_argument why -> Error why
      | None -> Error (strf "%s: the program has no run" (Rig.name host))
      | Some run -> (
          let entry = Rig.Image.entry run in
          let limits = threads entry in
          match watched t limits with
          | exception Refused why -> Error why
          | watch ->
          let buffer = B.create Rig.host (words_bytes t) in
          let part = Sub.words buffer in
          (* The function stays until the runs' work is done. *)
          let sub = Sub.make ~keep:run host [| part |] in
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
                 times = times_there t host buffer run;
                 words = B.export buffer Bigarray.char;
                 lock = Mutex.create ();
               })))

let no_rails _ = None

(* [load], raising [Broken] for a rule of [t] it breaks. *)
let load_checked ?rails t devices =
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

let load ?rails t devices =
  try load_checked ?rails t devices
  with Broken why -> invalid_arg ("Rig_program.load: " ^ why)

(* Writes [v] at [at] as a little-endian int64, sign-extended as [R.int] reads
   it back: -1 is eight 0xff bytes. *)
let set_word words at v =
  Bigarray.Array1.(
    for i = 0 to 7 do
      unsafe_set words (at + i) (Char.unsafe_chr ((v asr (8 * i)) land 0xff))
    done)

(* The little-endian int64 at [at] of [words], which [set_word] wrote. *)
let get_word words at =
  let v = ref 0 in
  for i = 7 downto 0 do
    v := (!v lsl 8) lor Char.code (Bigarray.Array1.unsafe_get words (at + i))
  done;
  !v

let threads = function
  | Here p ->
      Array.map
        (fun (image, kernel, _) ->
          Rig.Image.threads (Hashtbl.find p.funcs (image, kernel)))
        (launches p.t)
  | There p -> p.limits

(* A watched size as [f] sets it, or -1 for one the agent knows: one that
   reads a word past the frame's, or whose sum is past an [int]. *)
let watched_size (f : frame) (v : value) =
  let n = Array.length f.ints in
  let s = ref v.add and known = ref true in
  for i = 0 to Iarray.length v.terms - 1 do
    let w, factor = Iarray.get v.terms i in
    if w >= n then known := false
    else if !known then
      match add_exn !s (mul_exn f.ints.(w) factor) with
      | x -> s := x
      | exception Overflow -> known := false
  done;
  if !known then !s else -1

(* Whether [n] is a size that a header word holds. *)
let header_size n = n >= 0 && n <= 0xffff_ffff

(* Refuses a watched launch whose threads per group, as the frame's ints set
   them, exceed its function's most, as load and every submit refuse them
   ([past_most]). An [x] or a [y] outside a header word, or that reads a
   word the frame does not write, is left to the agent, and so is the
   product with such a [z]. A loop, so each run allocates nothing here. *)
let check_threads watch (f : frame) =
  for i = 0 to Array.length watch - 1 do
    let { kernel; sizes = x, y, z; most } = watch.(i) in
    let x = watched_size f x
    and y = watched_size f y
    and z = watched_size f z in
    let z = if header_size z then z else -1 in
    if header_size x && header_size y && past_most ~most x y z then
      invalid "%s runs at most %d threads per group; the frame sets %s" kernel
        most (threads_of x y z)
  done

(* Submits [sub] over [buffers], which runs [p]'s words, naming the region
   [times] for the steps' times, and is its point. The words are placed on
   the host's queue when [submit] hands them over, so the next run writes
   them again once it returned. Each input is read once, checked and written:
   the agent checks again what it decodes. A borrow here asks the agent's own
   borrow, once per allocation and device. *)
let run_words ~waits p sub ~times ~buffers (f : frame) =
  check_counts p.t f;
  check_threads p.watch f;
  set_word p.words 0 p.entry;
  set_word p.words 8 times;
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
    let at = input_word i in
    set_word p.words at (Nativeint.to_int (B.handle b));
    set_word p.words (at + 8) (B.offset b);
    set_word p.words (at + 16) (B.length b)
  done;
  let at = input_word (Array.length f.inputs) in
  set_word p.words at (Array.length f.ints);
  for i = 0 to Array.length f.ints - 1 do
    set_word p.words (at + 8 + (8 * i)) f.ints.(i)
  done;
  Rig.submit ?waits sub ~buffers ~args:[||]

(* [run_words] under the lock, as [run_here] takes it. *)
let submit_there ~waits p sub ~times ~buffers (f : frame) =
  lock p.lock;
  match run_words ~waits p sub ~times ~buffers f with
  | point ->
      Mutex.unlock p.lock;
      point
  | exception e ->
      Mutex.unlock p.lock;
      raise e

let run_there ~waits p (f : frame) =
  [| submit_there ~waits p p.sub ~times:no_times ~buffers:[||] f |]

(* A run of [p] whose agent writes the steps' times into [x]'s region, which
   a copy brings back into memory of this time's own. The region needs no
   lock: the agent runs a later run's words once this run's copy is done. *)
let time_times ~waits p x (f : frame) =
  let n = Iarray.length p.t.steps in
  let region = Nativeint.to_int (B.handle x.region) in
  let local = B.create Rig.host (8 * n) in
  let point =
    submit_there ~waits p x.timed ~times:region ~buffers:[| local |] f
  in
  Rig.Point.wait point;
  B.read local Bigarray.char (fun a ->
      Array.init n (fun i -> get_word a (8 * i)))

let time_there ~waits p (f : frame) =
  match p.times with
  | Some x -> time_times ~waits p x f
  | None ->
      Array.iter Rig.Point.wait (run_there ~waits p f);
      Array.make (Iarray.length p.t.steps) 0

let run ?waits p f =
  match p with Here p -> run_here ~waits p f | There p -> run_there ~waits p f

let time ?waits p f =
  match p with
  | Here p ->
      check_unlooped p.t;
      snd (time_here ~waits p f)
  | There p ->
      check_unlooped p.t;
      time_there ~waits p f

let load_received ?rails b ~device =
  let r = { R.s = b; at = 0 } in
  let m = String.length share_magic in
  if String.length b < m || String.sub b 0 m <> share_magic then
    Error "not a program another process loads"
  else begin
    r.R.at <- m;
    match D.ids.r r with
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
            | devices -> (
                try load_checked ?rails t devices
                with Broken why -> Error why)))
  end

(* A run's words that name a region for its times make it [time_here], whose
   times the agent writes there for the controller's copy to bring back. *)
let run_received ?waits w ~program ~region =
  let r = { R.s = w; at = 0 } in
  match
    let p =
      match program (R.int r) with
      | Some (Here p) -> p
      | Some (There _) | None -> R.fail r "no program of this entry"
    in
    let times =
      match R.int r with
      | id when id = no_times -> None
      | id -> (
          let bytes = 8 * Iarray.length p.t.steps in
          match region id with
          | None -> R.fail r "no memory %d" id
          | Some b when B.length b <> bytes ->
              R.fail r "memory %d holds %d bytes, not the %d of the times" id
                (B.length b) bytes
          | Some b -> Some b)
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
    (p, times, { inputs; ints = Array.sub ints 0 n })
  with
  | exception R.Malformed (at, why) ->
      Error (strf "a malformed run at byte %d: %s" at why)
  | p, None, f -> Ok (run_here ~waits p f)
  | p, Some b, f ->
      check_unlooped p.t;
      let ends, times = time_here ~waits p f in
      B.write b Bigarray.char (fun a ->
          Array.iteri (fun i v -> set_word a (8 * i) v) times);
      Ok ends
