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

type width = W32 | W64
type 'a hole = { at : int; width : width; leaf : 'a; add : int; shift : int }
type 'a data = { bytes : string; holes : 'a hole array }
type value = Fixed of int | Input of { input : int; on : int } | Leaf of leaf
type copies = One | Two

type memory =
  | Alloc of {
      device : int;
      kind : B.memory;
      bytes : int;
      init : leaf data;
      copies : copies;
    }

type image = { device : int; binary : leaf data }
type view = { memory : int; offset : int; length : int }
type slot = Memory of view | Input of int

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

type step = Submit of submit | Move of { src : slot; dst : slot }
type input = { device : int; bytes : int; access : B.access }

type t = {
  devices : string array;
  memory : memory array;
  images : image array;
  inputs : input array;
  steps : step array;
}

(* A refusal of the description, which [load] answers as [Error]. *)
exception Refused of string

let refuse fmt = Printf.ksprintf (fun why -> raise (Refused why)) fmt

(* Checks *)

(* What [load] refuses before it makes anything: every index, size and hole, and
   where a leaf of [Two] memory may stand. *)

let index what n i =
  if i < 0 || i >= n then refuse "%s %d: no such %s" what i what

let two t m = match t.memory.(m) with Alloc { copies; _ } -> copies = Two
let mem_bytes t m = match t.memory.(m) with Alloc { bytes; _ } -> bytes

let check_leaf t ~images = function
  | Address { memory; on } ->
      index "memory" (Array.length t.memory) memory;
      index "device" (Array.length t.devices) on
  | Handle m -> index "memory" (Array.length t.memory) m
  | Entry { image; _ } -> index "image" images image

let leaf_two t = function
  | Address { memory; _ } | Handle memory -> two t memory
  | Entry _ -> false

let check_holes what len check holes =
  let width h = match h.width with W32 -> 4 | W64 -> 8 in
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
      let w = match h.width with W32 -> 4 | W64 -> 8 in
      if h.at < 0 || h.at + w > len then
        refuse "%s: a hole at %d outside its %d bytes" what h.at len;
      if h.shift < 0 || h.shift > 62 then
        refuse "%s: a hole's shift %d outside 0 to 62" what h.shift;
      check h.leaf)
    holes

let check_view t (v : view) =
  index "memory" (Array.length t.memory) v.memory;
  let n = mem_bytes t v.memory in
  if v.offset < 0 || v.length < 0 || v.offset + v.length > n then
    refuse "memory %d: a view of %d bytes at %d outside its %d bytes" v.memory
      v.length v.offset n

let check_slot t = function
  | Memory v -> check_view t v
  | Input i -> index "input" (Array.length t.inputs) i

let check_value t : value -> unit = function
  | Fixed _ -> ()
  | Input { input; on } ->
      index "input" (Array.length t.inputs) input;
      index "device" (Array.length t.devices) on
  | Leaf l -> check_leaf t ~images:(Array.length t.images) l

let check_work t = function
  | Words v -> check_view t v
  | Fill { fill; arg; _ } ->
      check_leaf t ~images:(Array.length t.images) fill;
      check_view t arg
  | Copy { src; dst } ->
      check_view t src;
      check_view t dst
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

let check t devices =
  if Array.length devices <> Array.length t.devices then
    refuse "%d devices for a program of %d" (Array.length devices)
      (Array.length t.devices);
  Array.iteri
    (fun i arch ->
      let a = Rig.arch devices.(i) in
      if a <> arch then
        refuse "device %d: %s is %S, not %S" i (Rig.name devices.(i)) a arch)
    t.devices;
  let ndev = Array.length t.devices in
  Array.iteri
    (fun i (Alloc { device; bytes; init; copies; _ }) ->
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
        init.holes)
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
  Array.iter
    (function
      | Submit s ->
          index "device" ndev s.device;
          Array.iter (fun p -> check_work t p.work) s.parts;
          Array.iter (check_slot t) s.reads;
          Array.iter (check_slot t) s.writes;
          Array.iter (fun (v, _) -> check_view t v) s.fixed
      | Move { src; dst } ->
          check_slot t src;
          check_slot t dst)
    t.steps

(* Loading *)

(* A launch's words that a run writes into its block before each submit: its
   parameters' holes over the run's inputs, and its geometry. *)
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

type loaded = {
  t : t;
  devices : Rig.t array;
  mem : B.t array array; (* Each memory's copies, on its device. *)
  borrows : (int * int * int, B.t) Hashtbl.t;
      (* By memory, copy and device: the borrows made at load. *)
  mutable images : Rig.Image.t array; (* Those loaded so far, while loading. *)
  mutable steps : lstep array;
  twos : B.t array array; (* Copy [k] of each memory of [Two] copies. *)
  last : Rig.Point.t option array; (* By device: the run's last point. *)
  lock : Mutex.t;
  mutable runs : int;
}

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

let copy_of p m k = if two p.t m then k else 0

let view p k d (v : view) =
  B.view
    (on p v.memory (copy_of p v.memory k) d)
    ~first:v.offset ~length:v.length

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
  let slot = function Memory v -> view v | Input _ -> false in
  let value : value -> bool = function
    | Leaf l -> leaf_two t l
    | Fixed _ | Input _ -> false
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
  | Input _ -> true
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
  let sub = Sub.make ~fixed ~reads ~writes p.devices.(d) parts in
  let slot = function Memory v -> view p k d v | Input _ -> unset in
  let srun = Sub.Run.make () in
  let value : value -> int = function
    | Fixed n -> n
    | Leaf l -> leaf_value p k l
    | Input _ -> 0
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
        let holes =
          List.filter (fun h -> per_run h.leaf) (Array.to_list params.holes)
        in
        let gx, gy, gz = groups and tx, ty, tz = threads in
        let geometry_per_run =
          List.exists per_run [ gx; gy; gz; tx; ty; tz; shared ]
        in
        Some
          {
            block = b;
            params = params.bytes;
            holes = Array.of_list holes;
            groups;
            threads;
            shared;
            geometry_per_run;
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

let make_memory devices = function
  | Alloc { device; kind; bytes; copies; _ } ->
      let n = match copies with One -> 1 | Two -> 2 in
      Array.init n (fun _ -> B.create ~memory:kind devices.(device) bytes)

let load_image p (im : image) =
  match Rig.Image.load p.devices.(im.device) (filled p 0 im.binary) with
  | Ok i -> i
  | Error why -> raise (Refused why)

let write_init p m k =
  let (Alloc { init; _ }) = p.t.memory.(m) in
  let n = String.length init.bytes in
  if n > 0 then
    B.copy
      ~src:(B.of_string (filled p k init))
      ~dst:(B.view p.mem.(m).(k) ~first:0 ~length:n)

let one_machine devices =
  if Array.length devices > 0 then begin
    let h = Rig.host_of devices.(0) in
    Array.iter
      (fun d ->
        if not (Rig.equal (Rig.host_of d) h) then
          invalid_arg "Rig_program.load: devices of several machines")
      devices
  end

let load t devices =
  one_machine devices;
  match check t devices with
  | exception Refused why -> Error why
  | () -> (
      let mem = Array.map (make_memory devices) t.memory in
      let twos k =
        List.filteri (fun m _ -> two t m) (Array.to_list mem)
        |> List.map (fun c -> c.(k))
        |> Array.of_list
      in
      let p =
        {
          t;
          devices;
          mem;
          borrows = Hashtbl.create 8;
          images = [||];
          steps = [||];
          twos = [| twos 0; twos 1 |];
          last = Array.make (Array.length devices) None;
          lock = Mutex.create ();
          runs = 0;
        }
      in
      let step i = function
        | Submit s ->
            let n = if per_copy t s then 2 else 1 in
            Lsubmit
              { spec = s; copies = Array.init n (fun k -> prepare p i k s) }
        | Move { src; dst } -> Lmove { src; dst }
      in
      match
        (* An image's holes name earlier images' entries: each loads once those
           before it did. *)
        Array.iter
          (fun im -> p.images <- Array.append p.images [| load_image p im |])
          t.images;
        Array.iteri (fun m c -> Array.iteri (fun k _ -> write_init p m k) c) mem;
        p.steps <- Array.mapi step t.steps
      with
      | () -> Ok p
      | exception Refused why -> Error why)

(* Running *)

type frame = { inputs : B.t array }

let invalid fmt = Printf.ksprintf invalid_arg ("Rig_program.run: " ^^ fmt)

let check_frame p (f : frame) =
  let n = Array.length p.t.inputs in
  if Array.length f.inputs <> n then
    invalid "%d inputs for a program of %d" (Array.length f.inputs) n;
  Array.iteri
    (fun i (spec : input) ->
      let b = f.inputs.(i) and d = p.devices.(spec.device) in
      if not (Rig.equal (B.device b) d) then
        invalid "input %d is on %s, not %s" i
          (Rig.name (B.device b))
          (Rig.name d);
      if B.length b < spec.bytes then
        invalid "input %d holds %d bytes, fewer than %d" i (B.length b)
          spec.bytes;
      if spec.access = B.Read_write && B.access b = B.Read then
        invalid "input %d is read-only memory, which the program writes" i)
    p.t.inputs

(* Input [i] of [f] as device [d] names it. *)
let input_on p (f : frame) i d =
  let b = f.inputs.(i) and dev = p.devices.(d) in
  if Rig.equal (B.device b) dev then b
  else
    match B.borrow dev b with
    | Some b -> b
    | None -> invalid "input %d: %s cannot borrow it" i (Rig.name dev)

let run_value p f k : value -> int = function
  | Fixed n -> n
  | Input { input; on } -> B.address (input_on p f input on)
  | Leaf l -> leaf_value p k l

let pass p f (s : slot array) bufs d =
  Array.iteri
    (fun i -> function
      | Input j -> bufs.(i) <- input_on p f j d | Memory _ -> ())
    s

(* Drops the inputs [bufs] held, so a loaded program keeps no caller's buffer
   past its run. *)
let unpass (s : slot array) bufs =
  Array.iteri
    (fun i -> function Input _ -> bufs.(i) <- unset | Memory _ -> ())
    s

let slot_buffer p (f : frame) k = function
  | Memory v ->
      B.view
        p.mem.(v.memory).(copy_of p v.memory k)
        ~first:v.offset ~length:v.length
  | Input j -> f.inputs.(j)

let exec p f k after = function
  | Lmove { src; dst } ->
      B.copy ~src:(slot_buffer p f k src) ~dst:(slot_buffer p f k dst)
  | Lsubmit { spec; copies } ->
      let q = copies.(if Array.length copies = 2 then k else 0) in
      let d = spec.device in
      pass p f spec.reads q.reads d;
      pass p f spec.writes q.writes d;
      Array.iter
        (fun l ->
          Array.iter
            (fun h ->
              store_hole q.srun l.block l.params h (run_value p f k h.leaf))
            l.holes;
          if l.geometry_per_run then
            store_geometry q.srun l.block l.groups l.threads l.shared
              (run_value p f k))
        q.launches;
      let waits = match p.last.(d) with None -> after | Some _ -> [||] in
      let point =
        Fun.protect
          ~finally:(fun () ->
            unpass spec.reads q.reads;
            unpass spec.writes q.writes)
          (fun () ->
            Rig.submit q.sub ~run:q.srun ~reads:q.reads ~writes:q.writes ~waits)
      in
      p.last.(d) <- Some point

let run ?(after = [||]) p f =
  Mutex.protect p.lock @@ fun () ->
  check_frame p f;
  let k = p.runs land 1 in
  p.runs <- p.runs + 1;
  Array.iter (fun b -> B.wait b B.Read_write) p.twos.(k);
  Array.fill p.last 0 (Array.length p.last) None;
  Array.iter (exec p f k after) p.steps;
  Array.of_list (List.filter_map Fun.id (Array.to_list p.last))
