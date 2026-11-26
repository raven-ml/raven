(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module G = Rig_program
open Rig_program

let strf = Printf.sprintf

(* Fresh Polled devices: a name opens once. *)
let opened = Atomic.make 0
let fresh name = strf "program:%s-%d" name (Atomic.fetch_and_add opened 1)

let polled ?peers ?copies ?host_visible name =
  P.open_ ?peers ?copies ?host_visible (fresh name)

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let read b =
  let n = B.length b in
  let h = B.create Rig.host n in
  B.copy ~src:b ~dst:h;
  let s = Bytes.create n in
  B.blit_to_bytes h 0 s 0 n;
  Bytes.to_string s

let of_bytes d s =
  let b = B.create d (String.length s) in
  B.copy ~src:(B.of_string s) ~dst:b;
  b

let functions = "functions"

(* The value [n], and the value of word [w] of the run's ints. *)
let fixed n = { G.add = n; terms = [||] }
let int_at w = { G.add = 0; terms = [| (w, 1) |] }
let one = (fixed 1, fixed 1, fixed 1)
let scalar ?(width = G.W64) at value = { G.at; width; value }

(* A launch of Polled's [fill]: for [groups] groups, the 64-bit word [i] of its
   slot [0] is [base + i]. *)
let fill ?(scalars : _ iarray = [||]) ?(use = G.Write) ~image ~groups base =
  Launch
    {
      image;
      name = "fill";
      label = "fill";
      params = String.make 8 '\000' ^ le64 base;
      scalars;
      refs = [| { G.slot = 0; at = 0; use } |];
      groups = (fixed groups, fixed 1, fixed 1);
      threads = one;
      shared = fixed 0;
    }

(* A launch of Polled's [fill] over the groups the run's int [w] holds. *)
let fill_at w base =
  match fill ~image:0 ~groups:1 base with
  | G.Launch l -> G.Launch { l with groups = (int_at w, fixed 1, fixed 1) }
  | part -> part

(* A launch of Polled's [copy]: [bytes] bytes from its slot [src] (defaults to
   [0]) to its slot [dst] (defaults to [1]). *)
let kcopy ?(src = 0) ?(dst = 1) ~image bytes =
  Launch
    {
      image;
      name = "copy";
      label = "copy";
      params = String.make 16 '\000' ^ le64 bytes;
      scalars = [||];
      refs =
        [|
          { G.slot = src; at = 0; use = Read };
          { G.slot = dst; at = 8; use = Write };
        |];
      groups = one;
      threads = one;
      shared = fixed 0;
    }

(* A submit on [device] of [parts] over [slots]. *)
let submit ?(slots : _ iarray = [||]) device parts =
  G.Submit { device; parts; slots }

let alloc ?(init = "") device bytes =
  G.Alloc { device; kind = B.Device; bytes; init }

let all memory = G.Memory { memory; offset = 0; length = 64 }

(* A copy part of [length] bytes from slot [src] (defaults to [0]) at
   [src_offset] (defaults to [0]) to slot [dst] (defaults to [1]) at
   [dst_offset] (defaults to [0]). *)
let pcopy ?(src = 0) ?(src_offset = fixed 0) ?(dst = 1) ?(dst_offset = fixed 0)
    length =
  G.Copy { src; src_offset; dst; dst_offset; length }

(* A move of [length] bytes from [src] at [src_offset] (defaults to [0]) to
   [dst] at [dst_offset] (defaults to [0]). *)
let move ?(src_offset = fixed 0) ?(dst_offset = fixed 0) src dst length =
  G.Move { src; src_offset; dst; dst_offset; length }

let load_ok t ds =
  require_ok ~pp:Format.pp_print_string (G.load t (Iarray.of_list ds))

(* The host code of [f]'s object in [dir], an image on device [device]. *)
let host_image ?(dir = "../fixtures") device f =
  { G.device; binary = Rig_host_support.fixture ~dir f }

(* Programs *)

let word s i = Int64.to_int (String.get_int64_le s (8 * i))

let one_device_fill d =
  {
    G.devices = [| Rig.arch d |];
    ints = 0;
    memory = [| alloc 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    steps = [| submit 0 ~slots:[| all 0 |] [| fill ~image:0 ~groups:1 7 |] |];
  }

(* A fill of the run's ints on device 0. *)
let ints_fill d =
  {
    G.devices = [| Rig.arch d |];
    ints = 1;
    memory = [||];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    steps = [| submit 0 ~slots:[| Ints |] [| fill ~image:0 ~groups:1 7 |] |];
  }

let mib = 1 lsl 20
let empty_frame = { G.inputs = [||]; ints = [||] }

let out_of_memory d n = function
  | Rig.Out_of_memory (d', n') -> Rig.equal d d' && n = n'
  | _ -> false

let lost d why = function
  | Rig.Lost (d', why') -> Rig.equal d d' && why = why'
  | _ -> false

(* [one_device_fill d] with memory of [bytes] bytes. *)
let sized bytes d = { (one_device_fill d) with memory = [| alloc 0 bytes |] }

let base d =
  {
    G.devices = [| Rig.arch d |];
    ints = 0;
    memory = [| alloc 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 64 } |];
    steps = [| submit 0 ~slots:[| Input 0 |] [| fill ~image:0 ~groups:1 0 |] |];
  }

(* [t] whose one step is a submit on device 0, input 0 its slot, of [part]. *)
let with_part part (t : G.t) =
  { t with steps = [| submit 0 ~slots:[| Input 0 |] [| part |] |] }

let with_params params t =
  match fill ~image:0 ~groups:1 0 with
  | Launch l -> with_part (Launch { l with params; refs = [||] }) t
  | part -> with_part part t

(* [base d] whose fill writes its parameters' scalars [scalars]. *)
let with_scalars ?(ints = 0) scalars d =
  { (with_part (fill ~scalars ~image:0 ~groups:1 0) (base d)) with ints }

(* [base d] whose step copies [length] bytes of its slot [0] into its slot [1],
   of [slots], and fills its slot [0] with a ref of [use]. *)
let copying ?(ints = 0) ?use ?src_offset ?dst_offset ?(length = 64) slots d =
  {
    (base d) with
    ints;
    steps =
      [|
        submit 0 ~slots
          [|
            fill ?use ~image:0 ~groups:1 0; pcopy ?src_offset ?dst_offset length;
          |];
      |];
  }

(* A launch on device [host] of test/fixtures' [affine], image [image], which
   stores [a * in[i] + c] at [out[i]] for [i] below [n]: [x.(0) <- a * x.(0) +
   c]. The launch names the function [name]. *)
let step_affine ?(host = 1) ?(image = 1) ?(name = "affine") x ~a ~c =
  submit host ~slots:[| x |]
    [|
      Launch
        {
          image;
          name;
          label = name;
          params = String.make 16 '\000' ^ le64 1 ^ String.make 16 '\000';
          scalars = [| scalar 24 a; scalar 32 c |];
          refs =
            [|
              { G.slot = 0; at = 0; use = Write };
              { G.slot = 0; at = 8; use = Read };
            |];
          groups = one;
          threads = one;
          shared = fixed 0;
        };
    |]

(* A program on Polled's [d] (device 0) and the host (device 1), with Polled's
   functions as image 0 and [affine] as image 1. *)
let with_host d (t : G.t) =
  {
    t with
    G.devices = [| Rig.arch d; Rig.arch Rig.host |];
    images = [| { device = 0; binary = functions }; host_image 1 "affine" |];
  }

let one_word d = of_bytes d (le64 0)
let word_of b = word (read b) 0
let devices = lazy (fst (polled "model-a"), fst (polled "model-b"))

let pp_points points =
  List.map (Format.asprintf "%a" Rig.Point.pp) (Array.to_list points)

(* Order *)

(* Three memories and two inputs of 64 bytes, each on one of two devices, and
   steps over them. *)

let size = 64

type loc = Mem of int | In of int

(* [length] bytes of [src] from [src_at] into [dst] from [dst_at], at offsets
   the run reads from its ints where [per_run] holds. *)
type copy = {
  src : loc;
  dst : loc;
  length : int;
  src_at : int;
  dst_at : int;
  per_run : bool;
}

type op =
  | Fill of { dev : int; dst : loc; groups : int; base : int }
  | Kcopy of { dev : int; src : loc; dst : loc; bytes : int }
  | Pcopy of { dev : int; copy : copy }
  | Move of copy
  | Empty of { dev : int }
(* A submission of no parts. *)

(* With [loop], the steps run in a loop of one trip. *)
type case = {
  mem_dev : int array;
  init : string array;
  in_dev : int array;
  inputs : string array;
  ops : op list;
  loop : bool;
}

let pp_loc ppf = function
  | Mem i -> Format.fprintf ppf "m%d" i
  | In i -> Format.fprintf ppf "x%d" i

let pp_copy ppf c =
  Format.fprintf ppf "%a+%d->%a+%d %d bytes%s" pp_loc c.src c.src_at pp_loc
    c.dst c.dst_at c.length
    (if c.per_run then " per run" else "")

let pp_op ppf = function
  | Fill { dev; dst; groups; base } ->
      Format.fprintf ppf "fill@%d %a %d groups from %d" dev pp_loc dst groups
        base
  | Kcopy { dev; src; dst; bytes } ->
      Format.fprintf ppf "copy@%d %a->%a %d bytes" dev pp_loc src pp_loc dst
        bytes
  | Pcopy { dev; copy } -> Format.fprintf ppf "copy part@%d %a" dev pp_copy copy
  | Move copy -> Format.fprintf ppf "move %a" pp_copy copy
  | Empty { dev } -> Format.fprintf ppf "no part@%d" dev

let pp_case ppf c =
  Format.fprintf ppf "memory on [%s]; inputs on [%s]; %s[%a]"
    (String.concat ";" (Array.to_list (Array.map string_of_int c.mem_dev)))
    (String.concat ";" (Array.to_list (Array.map string_of_int c.in_dev)))
    (if c.loop then "a loop of " else "")
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp_op)
    c.ops

(* Cases whose field [loop] the generator [loop] draws. *)
let gen_case loop =
  let open Gen in
  let dev = int_range 0 1 in
  let loc =
    one_of
      [
        map (fun i -> Mem i) (int_range 0 2);
        map (fun i -> In i) (int_range 0 1);
      ]
  in
  let two_locs =
    let* src = loc in
    let+ dst = such_that (fun d -> d <> src) loc in
    (src, dst)
  in
  (* A copy between two slots, or within one slot between its two halves. *)
  let copy =
    frequency
      [
        ( 2,
          let+ src, dst = two_locs
          and+ length = one_of [ constant size; int_range 0 size ]
          and+ src_at = int_range 0 size
          and+ dst_at = int_range 0 size
          and+ per_run = bool in
          let fit at = at mod (size - length + 1) in
          {
            src;
            dst;
            length;
            src_at = fit src_at;
            dst_at = fit dst_at;
            per_run;
          } );
        ( 1,
          let+ at = loc
          and+ length = int_range 0 (size / 2)
          and+ a = int_range 0 size
          and+ b = int_range 0 size
          and+ down = bool
          and+ per_run = bool in
          let half = size / 2 in
          let low = a mod (half - length + 1)
          and high = half + (b mod (half - length + 1)) in
          let src_at, dst_at = if down then (high, low) else (low, high) in
          { src = at; dst = at; length; src_at; dst_at; per_run } );
      ]
  in
  let op =
    frequency
      [
        ( 3,
          let+ dev = dev
          and+ dst = loc
          and+ groups = int_range 1 (size / 8)
          and+ base = int_range 0 1_000_000 in
          Fill { dev; dst; groups; base } );
        ( 2,
          let+ dev = dev
          and+ src, dst = two_locs
          and+ words = int_range 0 (size / 8) in
          Kcopy { dev; src; dst; bytes = 8 * words } );
        ( 3,
          let+ dev = dev and+ copy = copy in
          Pcopy { dev; copy } );
        (2, map (fun c -> Move c) copy);
        (1, map (fun dev -> Empty { dev }) dev);
      ]
  in
  let bytes = string_of ~size:(constant size) char in
  let+ mem_dev = array ~size:(constant 3) dev
  and+ init = array ~size:(constant 3) bytes
  and+ in_dev = array ~size:(constant 2) dev
  and+ inputs = array ~size:(constant 2) bytes
  and+ ops = list ~size:(int_range 1 6) op
  and+ loop in
  { mem_dev; init; in_dev; inputs; ops; loop }

let gen_case loop = Gen.with_pp pp_case (gen_case loop)

let some_loops =
  Gen.frequency [ (3, Gen.constant false); (1, Gen.constant true) ]

(* Devices whose memory the host does not address, so that a device's copy
   engine runs each move. *)
let hidden =
  lazy
    ( fst (polled ~host_visible:false "model-hidden-a"),
      fst (polled ~host_visible:false "model-hidden-b") )

(* The offset [at] of a copy's end: a constant, or [2 * w + at mod 2] where word
   [w] of the run's ints holds [at / 2]. *)
let copy_offset ~per_run w at =
  if per_run then { G.add = at land 1; terms = [| (w, 2) |] } else fixed at

(* The run's ints: words [2 * i] and [2 * i + 1] hold half the offsets of op
   [i], a copy part or a move. *)
let copy_ints c =
  let ops = Array.of_list c.ops in
  Array.init
    (2 * Array.length ops)
    (fun w ->
      match ops.(w / 2) with
      | Pcopy { copy = { src_at; dst_at; _ }; _ } | Move { src_at; dst_at; _ }
        ->
          (if w land 1 = 0 then src_at else dst_at) / 2
      | Fill _ | Kcopy _ | Empty _ -> 0)

(* The device that holds [l]: a move runs on its source's, whose copy engine
   reaches the other device's memory. *)
let loc_dev c = function Mem m -> c.mem_dev.(m) | In i -> c.in_dev.(i)

let describe c =
  let slot = function Mem m -> all m | In i -> G.Input i in
  let image d = d in
  let step i = function
    | Fill { dev; dst; groups; base } ->
        submit dev
          ~slots:[| slot dst |]
          [| fill ~image:(image dev) ~groups base |]
    | Kcopy { dev; src; dst; bytes } ->
        submit dev
          ~slots:[| slot src; slot dst |]
          [| kcopy ~image:(image dev) bytes |]
    | Pcopy { dev; copy = { src; dst; length; src_at; dst_at; per_run } } ->
        submit dev
          ~slots:[| slot src; slot dst |]
          [|
            Copy
              {
                src = 0;
                src_offset = copy_offset ~per_run (2 * i) src_at;
                dst = 1;
                dst_offset = copy_offset ~per_run ((2 * i) + 1) dst_at;
                length;
              };
          |]
    | Move { src; dst; length; src_at; dst_at; per_run } ->
        G.Move
          {
            src = slot src;
            src_offset = copy_offset ~per_run (2 * i) src_at;
            dst = slot dst;
            dst_offset = copy_offset ~per_run ((2 * i) + 1) dst_at;
            length;
          }
    | Empty { dev } -> submit dev [||]
  in
  let steps = Iarray.of_list (List.mapi step c.ops) in
  let a, b = Lazy.force hidden in
  {
    G.devices = [| Rig.arch a; Rig.arch b |];
    ints = 2 * List.length c.ops;
    memory = Iarray.init 3 (fun m -> alloc ~init:c.init.(m) c.mem_dev.(m) size);
    images =
      [|
        { device = 0; binary = functions }; { device = 1; binary = functions };
      |];
    inputs =
      Iarray.map
        (fun device -> { G.device; bytes = size })
        (Iarray.of_array c.in_dev);
    steps =
      (if c.loop then
         [| Loop { trips = fixed 1; trip = None; flag = None; body = steps } |]
       else steps);
  }

(* The model: memory keeps what each run leaves, and each run's inputs start
   from the case's. *)
let model c =
  let mem = Array.map Bytes.of_string c.init in
  fun () ->
    let inputs = Array.map Bytes.of_string c.inputs in
    let cell = function Mem m -> mem.(m) | In i -> inputs.(i) in
    let copy { src; dst; length; src_at; dst_at; _ } =
      Bytes.blit (cell src) src_at (cell dst) dst_at length
    in
    List.iter
      (function
        | Fill { dst; groups; base; _ } ->
            for i = 0 to groups - 1 do
              Bytes.set_int64_le (cell dst) (8 * i) (Int64.of_int (base + i))
            done
        | Kcopy { src; dst; bytes; _ } ->
            Bytes.blit (cell src) 0 (cell dst) 0 bytes
        | Pcopy { copy = c; _ } | Move c -> copy c
        | Empty _ -> ())
      c.ops;
    Array.map Bytes.to_string inputs

(* The device each op runs on, none for a move of no bytes, which the host
   makes. *)
let op_dev c = function
  | Fill { dev; _ } | Kcopy { dev; _ } | Pcopy { dev; _ } | Empty { dev } ->
      Some dev
  | Move { length = 0; _ } -> None
  | Move { src; _ } -> Some (loc_dev c src)

(* The device whose submission of parts an op is, which [time] times. *)
let timed_dev = function
  | Fill { dev; _ } | Kcopy { dev; _ } | Pcopy { dev; _ } -> Some dev
  | Move _ | Empty _ -> None

(* Case [c]'s program loaded on the two hidden devices, and a fresh frame of its
   inputs for each run. *)
let loaded c =
  let a, b = Lazy.force hidden in
  let ds = [| a; b |] in
  let p = load_ok (describe c) [ a; b ] in
  let frame () =
    let xs = Array.map2 (fun d s -> of_bytes ds.(d) s) c.in_dev c.inputs in
    { inputs = xs; ints = copy_ints c }
  in
  (ds, p, frame)

(* A step that names memory another device's step wrote before. *)
let rec crossed c last = function
  | [] -> false
  | op :: ops ->
      let reads, writes =
        match op with
        | Fill { dst; _ } -> ([], [ dst ])
        | Kcopy { src; dst; _ }
        | Pcopy { copy = { src; dst; _ }; _ }
        | Move { src; dst; _ } ->
            ([ src ], [ dst ])
        | Empty _ -> ([], [])
      in
      let dev = op_dev c op in
      let seen l =
        match (List.assoc_opt l last, dev) with
        | Some (Some d'), Some d -> d' <> d
        | _ -> false
      in
      List.exists seen (reads @ writes)
      || crossed c (List.map (fun l -> (l, dev)) writes @ last) ops

(* Memory keeps what each run leaves, so three calls over fresh inputs each
   leave what the model's next run does: calls 0 and 2 are times unless [c] has
   a loop, which time refuses. The rarest covers, the moves', are drawn in about
   0.16 of cases, so 200 cases miss one with a probability below 10^-13. *)
let bytes_law c =
  let _, p, frame = loaded c in
  let expect = model c in
  for n = 0 to 2 do
    let f = frame () in
    if n <> 1 && not c.loop then ignore (G.time p f)
    else Array.iter Rig.Point.wait (G.run p f);
    equal
      ~msg:(strf "the inputs after call %d" n)
      (array string) (expect ()) (Array.map read f.inputs)
  done;
  let copies f =
    List.exists
      (function Pcopy { copy; _ } | Move copy -> f copy | _ -> false)
      c.ops
  in
  let parts f =
    List.exists (function Pcopy { copy; _ } -> f copy | _ -> false) c.ops
  in
  let moves f = List.exists (function Move m -> f m | _ -> false) c.ops in
  let input = function In _ -> true | Mem _ -> false in
  cover "a step follows another device's step" (crossed c [] c.ops);
  cover "steps in a loop" c.loop;
  cover "an input another device borrows"
    (List.exists
       (function
         | Fill { dev; dst = In i; _ }
         | Kcopy { dev; dst = In i; _ }
         | Pcopy { dev; copy = { dst = In i; _ } } ->
             dev <> c.in_dev.(i)
         | _ -> false)
       c.ops);
  cover "a move into an input" (moves (fun m -> input m.dst));
  cover "a move out of an input" (moves (fun m -> input m.src));
  cover "a move between two devices"
    (moves (fun m -> loc_dev c m.src <> loc_dev c m.dst));
  cover "a move within one slot" (moves (fun m -> m.src = m.dst));
  cover "a move at offsets the run reads"
    (moves (fun m -> m.per_run && m.src_at + m.dst_at > 0));
  cover "a copy part into an input" (parts (fun x -> input x.dst));
  cover "a copy part out of an input" (parts (fun x -> input x.src));
  cover "a copy part between memories"
    (parts (fun x -> (not (input x.src)) && not (input x.dst)));
  cover "a copy of part of its slots" (copies (fun x -> x.length < size));
  cover "a copy part within one slot" (parts (fun x -> x.src = x.dst));
  cover "a copy part at offsets the run reads"
    (parts (fun x -> x.per_run && x.src_at + x.dst_at > 0))

(* A run's ends are the last point of each device that ran an op, in the order
   of the program's devices: the device's submitted value once the run returns,
   since nothing else submits to it. Three calls use both copies of the ints:
   the first a time where [c] has no loop, the others runs. *)
let ends_law c =
  let ds, p, frame = loaded c in
  let ran =
    List.filter
      (fun d -> List.exists (fun op -> op_dev c op = Some d) c.ops)
      [ 0; 1 ]
  in
  for n = 0 to 2 do
    if n = 0 && not c.loop then ignore (G.time p (frame ()))
    else begin
      let ends = G.run p (frame ()) in
      equal
        ~msg:(strf "the ends of call %d" n)
        (list string)
        (List.map
           (fun d -> strf "%s:%d" (Rig.name ds.(d)) (Rig.submitted ds.(d)))
           ran)
        (pp_points ends);
      Array.iter Rig.Point.wait ends
    end
  done;
  cover "one device ran" (List.length ran = 1);
  cover "both devices ran" (List.length ran = 2);
  cover "steps in a loop" c.loop

(* A time has a duration per step: [0] for a move and a submission of no parts,
   and for the others, which Polled times on the host's clock, at least [0] and
   on each device at most the host's time around the call together. *)
let times_law c =
  let _, p, frame = loaded c in
  let f = frame () in
  let t0 = Rig.Profile.now () in
  let times = G.time p f in
  let window = Rig.Profile.now () - t0 in
  let ops = Array.of_list c.ops in
  let steps = List.init (Array.length ops) Fun.id in
  equal ~msg:"one per step" int (Array.length ops) (Array.length times);
  let untimed = List.filter (fun i -> timed_dev ops.(i) = None) steps in
  equal ~msg:"a move's and a submission's of no parts" (list int)
    (List.map (fun _ -> 0) untimed)
    (List.map (fun i -> times.(i)) untimed);
  for d = 0 to 1 do
    let mine = List.filter (fun i -> timed_dev ops.(i) = Some d) steps in
    List.iter
      (fun i -> at_least ~msg:(strf "step %d's time" i) int ~than:0 times.(i))
      mine;
    at_most
      ~msg:(strf "device %d's times together" d)
      int ~than:window
      (List.fold_left (fun s i -> s + times.(i)) 0 mine)
  done;
  let has f = List.exists f c.ops in
  cover "a submission of no parts"
    (has (function Empty _ -> true | _ -> false));
  cover "a move" (has (function Move _ -> true | _ -> false));
  cover "both devices' steps"
    (List.for_all (fun d -> has (fun op -> timed_dev op = Some d)) [ 0; 1 ])

(* A time of a program with a loop raises before any step: no device was given
   work, and the inputs keep their bytes. It leaves the program to the next run,
   which leaves the model's first run's bytes and its ends. *)
let loop_refused_law c =
  let ds, p, frame = loaded c in
  let f = frame () in
  let before = Array.map Rig.submitted ds in
  raises_match (Exn.invalid_arg ~sub:"Rig_program.time") (fun () -> G.time p f);
  equal ~msg:"the devices' submitted values" (array int) before
    (Array.map Rig.submitted ds);
  equal ~msg:"the inputs" (array string) c.inputs (Array.map read f.inputs);
  let ends = G.run p f in
  let ran =
    List.filter
      (fun d -> List.exists (fun op -> op_dev c op = Some d) c.ops)
      [ 0; 1 ]
  in
  equal ~msg:"the next run's ends" (list string)
    (List.map
       (fun d -> strf "%s:%d" (Rig.name ds.(d)) (Rig.submitted ds.(d)))
       ran)
    (pp_points ends);
  equal ~msg:"the inputs after the next run" (array string) (model c ())
    (Array.map read f.inputs)

(* A host step's launches are done when run returns: affine doubles and adds one
   to the word of a host buffer, read through its bigarray, which waits for
   nothing. *)
let test_host_step_done () =
  let t =
    {
      G.devices = [| Rig.arch Rig.host |];
      ints = 0;
      memory = [||];
      images = [| host_image 0 "affine" |];
      inputs = [| { G.device = 0; bytes = 8 } |];
      steps =
        [| step_affine ~host:0 ~image:0 (Input 0) ~a:(fixed 2) ~c:(fixed 1) |];
    }
  in
  let p = load_ok t [ Rig.host ] in
  let a =
    Bigarray.Array1.init Bigarray.int64 Bigarray.c_layout 1 (fun _ -> 5L)
  in
  ignore (G.run p { inputs = [| B.of_bigarray a |]; ints = [||] });
  equal ~msg:"the word when run returned" int64 11L a.{0}

(* Rails *)

(* [n] bytes of 0 that start on a page, which Polled maps: a host buffer of
   [Rig.Buffer.paged_from] bytes starts on one. *)
let area n =
  let a = B.export (B.create Rig.host B.paged_from) Bigarray.char in
  let a = Bigarray.Array1.sub a 0 n in
  Bigarray.Array1.fill a '\000';
  a

(* An end of rail 7 of this process whose ready function records each count with
   word 0 of the outbound area at the call. *)
let recording_end () =
  let outbound = area 512 and seen = ref [] in
  let ready c =
    let w = String.init 8 (Bigarray.Array1.get outbound) in
    seen := (c, word w 0) :: !seen
  in
  let e =
    Rig_remote_abi.end_ ~outbound ~inbound:(area 512) ~counts:(area 384) ~ready
      ~ready_fn:Rig_support.ready
  in
  ((fun id -> if id = 7 then Some e else None), seen)

(* On Polled's [d]: a fill of [42] into word 0 of rail 7's outbound area, then a
   send of [count]. *)
let send_program ?(count = int_at 0) d =
  {
    G.devices = [| Rig.arch d |];
    memory = [| Rail { rail = 7; area = Outbound } |];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    ints = 1;
    steps =
      [|
        submit 0
          ~slots:[| Memory { memory = 0; offset = 0; length = 8 } |]
          [| fill ~image:0 ~groups:1 42 |];
        Send { rail = 7; count };
      |];
  }

(* [Rail] memory is the area of the rail's end, which every run uses: a copy
   part of Polled's takes word 0 of the inbound area into the outbound area and
   word 1 into the counts, and each run reads the inbound words it finds
   there. *)
let test_rail_areas () =
  let d, _ = polled "rail-areas" in
  let outbound = area 512 and inbound = area 512 and counts = area 384 in
  let e =
    Rig_remote_abi.end_ ~outbound ~inbound ~counts ~ready:ignore
      ~ready_fn:Rig_support.ready
  in
  let rail area = G.Rail { rail = 7; area } in
  let word memory offset = G.Memory { memory; offset; length = 8 } in
  let t =
    {
      G.devices = [| Rig.arch d |];
      memory = [| rail Outbound; rail Inbound; rail Counts |];
      images = [||];
      inputs = [||];
      ints = 0;
      steps =
        [|
          submit 0
            ~slots:[| word 1 0; word 0 0; word 1 8; word 2 0 |]
            [| pcopy ~src:0 ~dst:1 8; pcopy ~src:2 ~dst:3 8 |];
        |];
    }
  in
  let rails id = if id = 7 then Some e else None in
  let p = require_ok ~pp:Format.pp_print_string (G.load ~rails t [| d |]) in
  let first8 a = String.init 8 (Bigarray.Array1.get a) in
  List.iter
    (fun (w0, w1) ->
      String.iteri (Bigarray.Array1.set inbound) (le64 w0 ^ le64 w1);
      Array.iter Rig.Point.wait (G.run p empty_frame);
      equal ~msg:"outbound's word 0" string (le64 w0) (first8 outbound);
      equal ~msg:"the counts' word 0" string (le64 w1) (first8 counts))
    [ (42, 7); (5, 9) ]

(* A send calls the end's ready function with its count once the work that
   writes the outbound area is done: Polled runs its work at a wait. *)
let test_send () =
  let d, _ = polled "send" in
  let rails, seen = recording_end () in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load ~rails (send_program d) [| d |])
  in
  ignore (G.run p { inputs = [||]; ints = [| 3 |] });
  equal ~msg:"the count and the outbound word at the call"
    (list (pair int int))
    [ (3, 42) ]
    !seen

(* A count below 1 is refused at load where it is constant, and at the run where
   it is not, before the end's ready function. *)
let test_send_count () =
  let d, _ = polled "send-count" in
  let rails, seen = recording_end () in
  raises_match ~msg:"a count of 0" (Exn.invalid_arg ~sub:"below 1") (fun () ->
      G.load ~rails (send_program ~count:(fixed 0) d) [| d |]);
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load ~rails (send_program d) [| d |])
  in
  raises_match ~msg:"an int of 0" (Exn.invalid_arg ~sub:"below 1") (fun () ->
      G.run p { inputs = [||]; ints = [| 0 |] });
  equal ~msg:"no call" (list (pair int int)) [] !seen

(* A send follows the run's [waits]: program [b] sends what program [a] wrote
   into the outbound area through [Rail] memory of its own, which [b]'s memory
   does not order. Polled runs [a]'s work at a wait. *)
let test_send_waits () =
  let d, _ = polled "send-waits" in
  let rails, seen = recording_end () in
  let described = send_program d in
  let a = { described with steps = [| Iarray.get described.steps 0 |] }
  and b =
    {
      described with
      images = [||];
      steps = [| Send { rail = 7; count = fixed 1 } |];
    }
  in
  let load t =
    require_ok ~pp:Format.pp_print_string (G.load ~rails t [| d |])
  in
  let a = load a and b = load b in
  let frame = { G.inputs = [||]; ints = [| 0 |] } in
  let waits = G.run a frame in
  ignore (G.run ~waits b frame);
  equal ~msg:"the count and the outbound word at the call"
    (list (pair int int))
    [ (1, 42) ]
    !seen

let test_no_rail () =
  let d, _ = polled "no-rail" in
  raises_match ~msg:"names the rail" (Exn.invalid_arg ~sub:"rail 7") (fun () ->
      G.load (send_program d) [| d |]);
  let send = { (send_program d) with memory = [| alloc 0 8 |] } in
  raises_match ~msg:"a send's rail" (Exn.invalid_arg ~sub:"rail 7") (fun () ->
      G.load send [| d |])

(* Domains *)

(* Runs of one loaded program, from two domains, each read the word the run
   before it wrote: a run copies its memory to its input, then writes its int
   there. *)
let swap_program d =
  let word = { G.memory = 0; offset = 0; length = 8 } in
  {
    G.devices = [| Rig.arch d |];
    ints = 1;
    memory = [| alloc ~init:(le64 0) 0 8 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 8 } |];
    steps =
      [|
        submit 0 ~slots:[| Memory word; Input 0 |] [| kcopy ~image:0 8 |];
        submit 0 ~slots:[| Memory word |]
          [| fill ~scalars:[| scalar 8 (int_at 0) |] ~image:0 ~groups:1 0 |];
      |];
  }

let swap_device = lazy (fst (polled "swap"))
let swapping = abstract "p"

let swap_commands =
  [
    command "load"
      (Gen.unit @-> makes swapping)
      (fun () -> ref 0)
      (fun () ->
        let d = Lazy.force swap_device in
        load_ok (swap_program d) [ d ]);
    command "run"
      (swapping ^-> Gen.int_range 1 1000 @-> returns int)
      (fun last v ->
        let w = !last in
        last := v;
        w)
      (fun p v ->
        let out = one_word (Lazy.force swap_device) in
        Array.iter Rig.Point.wait
          (G.run p { inputs = [| out |]; ints = [| v |] });
        word_of out);
  ]

(* Value *)

(* A fill of one group into input 0 whose base, at byte 8 of its parameters, is
   [s] over bytes 0. *)
let scalar_fill d s =
  {
    G.devices = [| Rig.arch d |];
    ints = 2;
    memory = [||];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 8 } |];
    steps =
      [|
        submit 0 ~slots:[| Input 0 |]
          [| fill ~scalars:[| s |] ~image:0 ~groups:1 0 |];
      |];
  }

(* A scalar of [width] whose value is [add] plus each term's int times its
   factor, and the run's two ints, one pair per run. *)
type sum = {
  width : G.width;
  add : int;
  terms : (int * int) list;
  frames : (int * int) list;
}

let pp_sum ppf c =
  let pairs f l = String.concat "; " (List.map f l) in
  Format.fprintf ppf "%s, add %d, terms [%s], frames [%s]"
    (match c.width with W32 -> "W32" | W64 -> "W64")
    c.add
    (pairs (fun (w, f) -> strf "int %d * %d" w f) c.terms)
    (pairs (fun (a, b) -> strf "%d,%d" a b) c.frames)

let gen_sum =
  let open Gen in
  let big = of_list [ 1 lsl 31; 1 lsl 32; max_int; min_int ] in
  let int = frequency [ (3, int_range (-(1 lsl 33)) (1 lsl 33)); (1, big) ] in
  let factor = frequency [ (3, int_range (-4) 4); (1, big) ] in
  let+ width = of_list [ G.W32; G.W64 ]
  and+ add = int
  and+ terms = list ~size:(int_range 0 3) (pair (int_range 0 1) factor)
  and+ frames = list ~size:(int_range 1 3) (pair int int) in
  { width; add; terms; frames }

let gen_sum = Gen.with_pp pp_sum gen_sum

(* The .mli's example: [1 + max_int] is past an [int], though the sum is
   [max_int]. *)
let sum_examples =
  [
    {
      width = W64;
      add = 1;
      terms = [ (0, max_int); (1, -1) ];
      frames = [ (1, 1) ];
    };
  ]

(* [a + b] and [a * b], or [None] past an [int]. *)
let add_int a b =
  let s = a + b in
  if a >= 0 = (b >= 0) && s >= 0 <> (a >= 0) then None else Some s

let mul_int a b =
  if a = 0 || b = 0 then Some 0
  else
    let p = a * b in
    if p / b <> a || (a = min_int && b = -1) then None else Some p

(* [c]'s sum over the ints [i0] and [i1], term by term from [add], or [None]
   where a product or a partial sum is past an [int]. *)
let exact c (i0, i1) =
  List.fold_left
    (fun s (w, f) ->
      Option.bind s (fun s ->
          Option.bind (mul_int (if w = 0 then i0 else i1) f) (add_int s)))
    (Some c.add) c.terms

let holds width v =
  match width with W64 -> true | W32 -> v >= -(1 lsl 31) && v < 1 lsl 32

(* A scalar writes, on each run, its sum over the run's ints, the low 32 bits of
   it for W32. The run refuses a sum past an [int] or past what the width holds;
   load refuses a value of no terms past it. *)
let sum_law c =
  let d = fst (Lazy.force devices) in
  let value = { G.add = c.add; terms = Iarray.of_list c.terms } in
  let t = scalar_fill d (scalar ~width:c.width 8 value) in
  let sums = List.map (exact c) c.frames in
  if c.terms = [] && not (holds c.width c.add) then
    raises_match ~msg:"load refuses the constant" Exn.invalid_arg (fun () ->
        G.load t [| d |])
  else begin
    let p = load_ok t [ d ] in
    List.iter2
      (fun (i0, i1) sum ->
        let x = B.create d 8 in
        let f = { G.inputs = [| x |]; ints = [| i0; i1 |] } in
        match sum with
        | Some v when holds c.width v ->
            Array.iter Rig.Point.wait (G.run p f);
            let low =
              match c.width with W64 -> v | W32 -> v land 0xffff_ffff
            in
            equal ~msg:"the scalar" int low (word (read x) 0)
        | _ ->
            raises_match ~msg:"the run refuses the sum" Exn.invalid_arg
              (fun () -> G.run p f))
      c.frames sums
  end;
  let past f = List.exists f sums in
  cover "two terms or more" (List.length c.terms >= 2);
  cover "no term" (c.terms = []);
  cover "a W32 scalar" (c.width = W32);
  cover "a sum past an int" (past Option.is_none);
  cover "a W32 sum past 32 bits"
    (c.width = W32
    && past (function Some v -> not (holds W32 v) | None -> false));
  cover "a negative sum written"
    (past (function Some v -> v < 0 && holds c.width v | None -> false))

(* Scalar *)

(* A 32-bit parameter's value and the low 32 bits it writes, or [None] where it
   fits 32 bits neither as a signed nor as an unsigned integer. *)
let w32_values =
  [
    (-(1 lsl 31) - 1, None);
    (-(1 lsl 31), Some 0x8000_0000);
    (-1, Some 0xffff_ffff);
    (0, Some 0);
    ((1 lsl 31) - 1, Some 0x7fff_ffff);
    ((1 lsl 32) - 1, Some 0xffff_ffff);
    (1 lsl 32, None);
    ((1 lsl 32) + 5, None);
  ]

(* A W32 scalar of value [v], a constant or the run's int 0, writes [v]'s low 32
   bits where [v] fits them; else load refuses the constant and the run the
   int. *)
let scalar_range (v, low) =
  let d = fst (Lazy.force devices) in
  let constant = scalar_fill d (scalar ~width:W32 8 (fixed v))
  and per_run = scalar_fill d (scalar ~width:W32 8 (int_at 0)) in
  let x = B.create d 8 in
  let frame = { G.inputs = [| x |]; ints = [| v; 0 |] } in
  match low with
  | Some low ->
      List.iter
        (fun (msg, t) ->
          Array.iter Rig.Point.wait (G.run (load_ok t [ d ]) frame);
          equal ~msg int low (word (read x) 0))
        [ ("a constant", constant); ("an int", per_run) ]
  | None ->
      raises_match ~msg:"load refuses the constant" Exn.invalid_arg (fun () ->
          G.load constant [| d |]);
      let p = load_ok per_run [ d ] in
      raises_match ~msg:"the run refuses the int" Exn.invalid_arg (fun () ->
          G.run p frame)

(* A launch's 16 constant parameter bytes, and a scalar of [width] at byte [at]
   whose value is [value], the run's int where [per_run], else constant. *)
type beside = {
  bytes : string;
  at : int;
  width : G.width;
  value : int;
  per_run : bool;
}

let gen_beside =
  let open Gen in
  let+ bytes = string_of ~size:(int_range 16 16) char
  and+ wide = bool
  and+ at = int_range 0 12
  and+ v64 = int
  and+ v32 = int_range (-(1 lsl 31)) ((1 lsl 32) - 1)
  and+ per_run = bool in
  if wide then { bytes; at = min at 8; width = W64; value = v64; per_run }
  else { bytes; at; width = W32; value = v32; per_run }

let gen_beside =
  Gen.with_pp
    (fun ppf b ->
      Format.fprintf ppf "bytes %S, %s at %d of %d, %s" b.bytes
        (match b.width with W32 -> "W32" | W64 -> "W64")
        b.at b.value
        (if b.per_run then "per run" else "constant"))
    gen_beside

(* A launch of [main] whose parameters are [b.bytes] with a scalar over
   [b.value]: the launch reads the bytes with the scalar's little-endian word in
   place of its bytes, every other byte kept. *)
let beside_law b =
  let d, pd = polled "beside" in
  let value = if b.per_run then int_at 0 else fixed b.value in
  let t =
    {
      G.devices = [| Rig.arch d |];
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [||];
      ints = 1;
      steps =
        [|
          submit 0
            [|
              Launch
                {
                  image = 0;
                  name = "main";
                  label = "main";
                  params = b.bytes;
                  scalars = [| scalar ~width:b.width b.at value |];
                  refs = [||];
                  groups = one;
                  threads = one;
                  shared = fixed 0;
                };
            |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  ignore (P.launches pd);
  Rig.Point.wait (G.run p { G.inputs = [||]; ints = [| b.value |] }).(0);
  let want = Bytes.of_string b.bytes in
  (match b.width with
  | W64 -> Bytes.set_int64_le want b.at (Int64.of_int b.value)
  | W32 -> Bytes.set_int32_le want b.at (Int32.of_int b.value));
  let l = List.hd (P.launches pd) in
  equal ~msg:"the bytes with the scalar in place" string (Bytes.to_string want)
    l.params;
  let size = match b.width with W32 -> 4 | W64 -> 8 in
  cover "a W32 scalar" (b.width = W32);
  cover "a value read per run" b.per_run;
  cover "a scalar off its width's alignment" (b.at mod size <> 0)

(* Ref *)

(* A launch's refs bind the input's address anew for every run. *)
let test_input_ref () =
  let d, _ = polled "input-ref" in
  let f = fill ~image:0 ~groups:1 0 in
  let f =
    match f with
    | Launch l ->
        G.Launch
          {
            l with
            refs =
              [|
                { G.slot = 0; at = 0; use = Write };
                { G.slot = 0; at = 8; use = Write };
              |];
          }
    | _ -> assert false
  in
  let t =
    { (base d) with steps = [| submit 0 ~slots:[| Input 0 |] [| f |] |] }
  in
  let p = load_ok t [ d ] in
  List.iter
    (fun x ->
      let keeper = Sub.make ~fixed:[ (x, Rig.Read) ] d [||] in
      let expected = Sub.address keeper x in
      ignore (G.run p { inputs = [| x |]; ints = [||] });
      equal ~msg:"the stored address" int expected (word_of x))
    [ B.create d 64; B.create d 64 ]

(* Part *)

(* A Copy part copies its source slot into its destination slot on its step's
   device, whether each holds the program's memory or a run's input: memory into
   an input, that input into another, the other into memory, and memory into
   memory, with each run's inputs. A move takes the last out. *)
let test_copy_part () =
  let d, _ = polled "copy-part" in
  let input = { G.device = 0; bytes = 8 } in
  let view memory offset = G.Memory { memory; offset; length = 8 } in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [| alloc ~init:(le64 42 ^ le64 7) 0 16; alloc 0 16 |];
      images = [||];
      inputs = [| input; input; input |];
      steps =
        [|
          submit 0
            ~slots:[| view 0 8; Input 0; Input 1; view 1 0; view 1 8 |]
            [|
              pcopy ~src:0 ~dst:1 8;
              pcopy ~src:1 ~dst:2 8;
              pcopy ~src:2 ~dst:3 8;
              pcopy ~src:3 ~dst:4 8;
            |];
          move (Memory { memory = 1; offset = 8; length = 8 }) (Input 2) 8;
        |];
    }
  in
  let p = load_ok t [ d ] in
  for run = 1 to 2 do
    let a = one_word d and b = one_word d and c = one_word d in
    Array.iter Rig.Point.wait (G.run p { inputs = [| a; b; c |]; ints = [||] });
    List.iteri
      (fun i x ->
        equal ~msg:(Printf.sprintf "run %d's input %d" run i) int 7 (word_of x))
      [ a; b; c ]
  done

(* A copy part in a loop moves with its trips: trip [i] copies row [i] of memory
   into row [3 - i] of the input, so the input holds the rows reversed. *)
let test_copy_loop () =
  let d, _ = polled "copy-loop" in
  let rows = [ 10; 11; 12; 13 ] in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [| alloc ~init:(String.concat "" (List.map le64 rows)) 0 32 |];
      images = [||];
      inputs = [| { device = 0; bytes = 32 } |];
      steps =
        [|
          Loop
            {
              trips = fixed 4;
              trip = Some 0;
              flag = None;
              body =
                [|
                  submit 0
                    ~slots:
                      [|
                        Memory { memory = 0; offset = 0; length = 32 }; Input 0;
                      |]
                    [|
                      pcopy
                        ~src_offset:{ G.add = 0; terms = [| (0, 8) |] }
                        ~dst_offset:{ G.add = 24; terms = [| (0, -8) |] }
                        8;
                    |];
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 32 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the rows reversed" string
    (String.concat "" (List.map le64 (List.rev rows)))
    (read x)

(* A copy within the ints, both ends at offsets the run reads from them: word 1
   into word 3. *)
let test_copy_ints () =
  let d, _ = polled "copy-ints" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 4;
      memory = [||];
      images = [||];
      inputs = [| { G.device = 0; bytes = 32 } |];
      steps =
        [|
          submit 0 ~slots:[| Ints; Ints |]
            [| pcopy ~src_offset:(int_at 0) ~dst_offset:(int_at 1) 8 |];
          move Ints (Input 0) 32;
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 32 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [| 8; 24; 42 |] });
  equal ~msg:"the ints" string (le64 8 ^ le64 24 ^ le64 42 ^ le64 24) (read x)

(* A slot that only copies name leaves the refs' slots after it in place: memory
   0, named by the copy alone, reaches memory 1, which the copy writes and a
   launch reads into input 0, named by the launch's write ref between them. *)
let test_renumbered () =
  let d, _ = polled "renumbered" in
  let bytes = String.init 64 (fun i -> Char.chr (i + 1)) in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [| alloc ~init:bytes 0 64; alloc 0 64 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 64 } |];
      steps =
        [|
          submit 0
            ~slots:[| all 0; Input 0; all 1 |]
            [| pcopy ~dst:2 64; kcopy ~src:2 ~dst:1 ~image:0 64 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = B.create d 64 in
  Array.iter Rig.Point.wait (G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"input 0" string bytes (read x)

(* A profile names a submission's work after its launches' labels, in the order
   of its parts, joined by ", " (rig.mli, [Profile.Work]): two launches of one
   function, labelled apart. *)
let test_label () =
  let d, _ = polled "label" in
  let labelled label =
    match fill ~image:0 ~groups:1 0 with
    | Launch l -> G.Launch { l with label }
    | part -> part
  in
  let t =
    {
      (base d) with
      steps =
        [|
          submit 0 ~slots:[| Input 0 |]
            [| labelled "first"; labelled "second" |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let (), events =
    Rig.Profile.take (fun () ->
        Array.iter Rig.Point.wait
          (G.run p { inputs = [| B.create d 64 |]; ints = [||] }))
  in
  equal ~msg:"the names of the device's work" (list string) [ "first, second" ]
    (List.filter_map
       (function
         | Rig.Profile.Work w when Rig.equal w.device d -> Some w.name
         | _ -> None)
       events)

(* Step *)

(* A move a device's copy engine runs is queued: the run returns without
   sleeping, its end on that device is the move's point, and the bytes are there
   once it is reached. *)
let test_move_queued () =
  let d, pd = polled ~host_visible:false "move-queued" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [| alloc ~init:(String.make 64 'm') 0 64 |];
      images = [||];
      inputs = [| { G.device = 0; bytes = 64 } |];
      steps = [| move (all 0) (Input 0) 64 |];
    }
  in
  let p = load_ok t [ d ] in
  let x = B.create d 64 in
  let before = P.sleeps pd in
  let ends = G.run p { inputs = [| x |]; ints = [||] } in
  equal ~msg:"sleeps" int before (P.sleeps pd);
  equal ~msg:"the ends" (list string)
    [ strf "%s:%d" (Rig.name d) (Rig.submitted d) ]
    (pp_points ends);
  Array.iter Rig.Point.wait ends;
  equal ~msg:"the moved bytes" string (String.make 64 'm') (read x)

(* Moves through the ints that a device runs are queued: input 0 into word 0 of
   the ints, and word 0 out into input 1. *)
let test_move_ints_queued () =
  let d, pd = polled ~host_visible:false "move-ints-queued" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [||];
      images = [||];
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
      steps = [| move (Input 0) Ints 8; move Ints (Input 1) 8 |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (le64 42) and y = one_word d in
  let before = P.sleeps pd in
  let ends = G.run p { inputs = [| x; y |]; ints = [||] } in
  equal ~msg:"sleeps" int before (P.sleeps pd);
  equal ~msg:"the ends" (list string)
    [ strf "%s:%d" (Rig.name d) (Rig.submitted d) ]
    (pp_points ends);
  Array.iter Rig.Point.wait ends;
  equal ~msg:"the word through the ints" int 42 (word_of y)

(* Moves between the program's host memory and a device of the program are
   queued, as through the ints: input 0 into memory 0, of 64 bytes on the host,
   and memory 0 out into input 1. *)
let test_move_host_queued () =
  let d, pd = polled ~host_visible:false "move-host-queued" in
  let t =
    {
      G.devices = [| Rig.arch d; Rig.arch Rig.host |];
      ints = 0;
      memory = [| alloc 1 64 |];
      images = [||];
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 64 });
      steps = [| move (Input 0) (all 0) 64; move (all 0) (Input 1) 64 |];
    }
  in
  let p = load_ok t [ d; Rig.host ] in
  let x = of_bytes d (String.make 64 'h')
  and y = of_bytes d (String.make 64 '0') in
  let before = P.sleeps pd in
  let ends = G.run p { inputs = [| x; y |]; ints = [||] } in
  equal ~msg:"sleeps" int before (P.sleeps pd);
  Array.iter Rig.Point.wait ends;
  equal ~msg:"the bytes through host memory" string (String.make 64 'h')
    (read y)

(* An int that a queued move wrote is read once the move is done: word 0 of the
   ints, which a move sets from input 0, picks the row of memory 0 that a second
   move takes into input 1. *)
let test_move_writes_int () =
  let d, _ = polled ~host_visible:false "move-writes-int" in
  let rows = [ 10; 11; 12; 13 ] in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [| alloc ~init:(String.concat "" (List.map le64 rows)) 0 32 |];
      images = [||];
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          move (Input 0) Ints 8;
          move
            ~src_offset:{ G.add = 0; terms = [| (0, 8) |] }
            (Memory { memory = 0; offset = 0; length = 32 })
            (Input 1) 8;
        |];
    }
  in
  let p = load_ok t [ d ] in
  List.iter
    (fun r ->
      let x = of_bytes d (le64 r) and y = one_word d in
      ignore (G.run p { inputs = [| x; y |]; ints = [||] });
      equal ~msg:(strf "row %d" r) int (List.nth rows r) (word_of y))
    [ 2; 0; 3 ]

(* A move in a loop moves with its trips: trip [i] moves row [i] of memory into
   row [3 - i] of the input, so the input holds the rows reversed. *)
let test_move_loop () =
  let d, _ = polled ~host_visible:false "move-loop" in
  let rows = [ 10; 11; 12; 13 ] in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [| alloc ~init:(String.concat "" (List.map le64 rows)) 0 32 |];
      images = [||];
      inputs = [| { G.device = 0; bytes = 32 } |];
      steps =
        [|
          Loop
            {
              trips = fixed 4;
              trip = Some 0;
              flag = None;
              body =
                [|
                  move
                    ~src_offset:{ G.add = 0; terms = [| (0, 8) |] }
                    ~dst_offset:{ G.add = 24; terms = [| (0, -8) |] }
                    (Memory { memory = 0; offset = 0; length = 32 })
                    (Input 0) 8;
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 32 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the rows reversed" string
    (String.concat "" (List.map le64 (List.rev rows)))
    (read x)

(* A move that a device outside the program's runs is done when the run returns:
   input 0 is a borrow, on a device that runs no copy, of another device's
   memory, whose copy engine runs the move. *)
let test_move_outside () =
  let a, _ = polled ~copies:false "move-outside-a"
  and b, _ = polled ~host_visible:false "move-outside-b" in
  let t =
    {
      G.devices = [| Rig.arch a |];
      ints = 0;
      memory = [||];
      images = [||];
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 64 });
      steps = [| move (Input 0) (Input 1) 64 |];
    }
  in
  let p = load_ok t [ a ] in
  let src = Option.get (B.borrow a (of_bytes b (String.make 64 'o'))) in
  let dst = B.create a 64 in
  let before = Rig.submitted b in
  let ends = G.run p { inputs = [| src; dst |]; ints = [||] } in
  equal ~msg:"the ends" (list string) [] (pp_points ends);
  equal ~msg:"b's submitted and signaled values when run returned"
    (pair int int)
    (before + 1, before + 1)
    (Rig.submitted b, Rig.signaled b);
  equal ~msg:"the moved bytes" string (String.make 64 'o') (read dst)

(* A loop of [k] trips: host code counts the trips in input 0, and a launch
   stores each trip's index in input 1. *)
let loop_law k =
  let d = fst (Lazy.force devices) in
  let t =
    with_host d
      {
        G.devices = [||];
        ints = 2;
        memory = [||];
        images = [| { device = 0; binary = functions } |];
        inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
        steps =
          [|
            Loop
              {
                trips = int_at 0;
                trip = Some 1;
                flag = None;
                body =
                  [|
                    step_affine (Input 0) ~a:(fixed 1) ~c:(fixed 1);
                    submit 0 ~slots:[| Input 1 |]
                      [|
                        fill
                          ~scalars:[| scalar 8 (int_at 1) |]
                          ~image:0 ~groups:1 0;
                      |];
                  |];
              };
          |];
      }
  in
  let p = load_ok t [ d; Rig.host ] in
  let counter = one_word d and last = one_word d in
  ignore (G.run p { inputs = [| counter; last |]; ints = [| k |] });
  equal ~msg:"trips" int (max 0 k) (word_of counter);
  equal ~msg:"the last trip's index" int (max 0 (k - 1)) (word_of last)

(* Host code that stores [0] into word [w] of the run's ints. *)
let clear_int w =
  submit 1 ~slots:[| G.Ints |]
    [|
      Launch
        {
          image = 1;
          name = "affine";
          label = "affine";
          params = le64 (8 * w) ^ le64 (8 * w) ^ le64 1 ^ le64 0 ^ le64 0;
          scalars = [||];
          refs =
            [|
              { G.slot = 0; at = 0; use = Write };
              { G.slot = 0; at = 8; use = Read };
            |];
          groups = one;
          threads = one;
          shared = fixed 0;
        };
    |]

(* A loop whose flag a step clears in its first trip runs once, in memory the
   program makes, in an input and in the ints, the flag's byte inside its slot:
   the flag is read after that work. *)
let test_flag () =
  let d, _ = polled "flag" in
  let mem = G.Memory { memory = 0; offset = 0; length = 8 } in
  let clear slot = submit 0 ~slots:[| slot |] [| fill ~image:0 ~groups:1 0 |] in
  let trips flag clear =
    let t =
      with_host d
        {
          G.devices = [||];
          ints = 3;
          memory = [| alloc ~init:(le64 1) 0 8 |];
          images = [||];
          inputs = [| { device = 0; bytes = 8 }; { device = 0; bytes = 8 } |];
          steps =
            [|
              Loop
                {
                  trips = fixed 10;
                  trip = None;
                  flag = Some flag;
                  body =
                    [| clear; step_affine (Input 0) ~a:(fixed 1) ~c:(fixed 1) |];
                };
            |];
        }
    in
    let p = load_ok t [ d; Rig.host ] in
    let counter = one_word d in
    let frame =
      { G.inputs = [| counter; of_bytes d (le64 1) |]; ints = [| 1; 1; 1 |] }
    in
    ignore (G.run p frame);
    word_of counter
  in
  equal ~msg:"memory" int 1 (trips (mem, 0) (clear mem));
  equal ~msg:"an input" int 1 (trips (Input 1, 0) (clear (Input 1)));
  equal ~msg:"the ints" int 1 (trips (Ints, 0) (clear_int 0));
  equal ~msg:"word 2 of the ints" int 1 (trips (Ints, 16) (clear_int 2))

(* Bytes *)

(* Descriptions of any shape, well formed or not: the format holds every value
   of the type. *)
let gen_t =
  let open Gen in
  let n = int_range (-2) 5000 in
  let few g = map Iarray.of_array (array ~size:(int_range 0 3) g) in
  let str = string_of ~size:(int_range 0 12) char in
  let value =
    let+ add = int and+ terms = few (pair n int) in
    ({ add; terms } : G.value)
  in
  let view =
    let+ memory = n and+ offset = n and+ length = n in
    { memory; offset; length }
  in
  let slot =
    one_of
      [
        map (fun v -> Memory v) view;
        map (fun i -> (Input i : slot)) n;
        constant Ints;
      ]
  in
  let ref =
    let+ at = n and+ slot = n and+ use = of_list [ G.Read; Write; Unordered ] in
    { G.slot; at; use }
  in
  let triple = triple value value value in
  let part =
    one_of
      [
        map (fun v -> Words v) view;
        (let+ src = n
         and+ src_offset = value
         and+ dst = n
         and+ dst_offset = value
         and+ length = n in
         (Copy { src; src_offset; dst; dst_offset; length } : G.part));
        (let+ image = n
         and+ name = str
         and+ label = str
         and+ params = str
         and+ scalars =
           few
             (let+ at = n
              and+ width = of_list [ W32; W64 ]
              and+ value = value in
              ({ at; width; value } : G.scalar))
         and+ refs = few ref
         and+ groups = triple
         and+ threads = triple
         and+ shared = value in
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
           });
      ]
  in
  let rec step depth =
    let leaves =
      [
        (let+ device = n and+ parts = few part and+ slots = few slot in
         Submit { device; parts; slots });
        (let+ src = slot
         and+ src_offset = value
         and+ dst = slot
         and+ dst_offset = value
         and+ length = n in
         G.Move { src; src_offset; dst; dst_offset; length });
        (let+ rail = n and+ count = value in
         Send { rail; count });
      ]
    in
    if depth = 0 then one_of leaves
    else
      one_of
        ((let+ trips = value
          and+ trip = option n
          and+ flag = option (pair slot n)
          and+ body = few (step (depth - 1)) in
          Loop { trips; trip; flag; body })
        :: leaves)
  in
  let+ devices = few str
  and+ memory =
    few
      (one_of
         [
           (let+ device = n
            and+ kind = of_list [ B.Device; B.Pinned; B.Mapped ]
            and+ bytes = n
            and+ init = str in
            Alloc { device; kind; bytes; init });
           (let+ rail = n and+ area = of_list [ Outbound; Inbound; Counts ] in
            Rail { rail; area });
         ])
  and+ images =
    few
      (let+ device = n and+ binary = str in
       { device; binary })
  and+ inputs =
    few
      (let+ device = n and+ bytes = n in
       { device; bytes })
  and+ ints = n
  and+ steps = few (step 2) in
  { devices; memory; images; inputs; ints; steps }

let pp_t ppf t =
  Format.fprintf ppf "a description of %d bytes, %d steps"
    (String.length (G.to_string t))
    (Iarray.length t.steps)

let gen_t = Gen.with_pp pp_t gen_t
let description = Testable.make ~pp:pp_t ~equal:( = )

let round_trip t =
  equal (result description string) (Ok t) (G.of_string (G.to_string t))

(* Every strict prefix of a description's bytes is malformed. *)
let cut_law (t, cut) =
  let s = G.to_string t in
  let cut = cut mod String.length s in
  is_error ~pp:pp_t (G.of_string (String.sub s 0 cut));
  cover "a cut past the first byte" (cut > 0)

(* A changed byte of a description's bytes decodes or answers [Error]. *)
let changed_law (t, at, byte) =
  let s = G.to_string t in
  let at = at mod String.length s in
  ignore
    (G.of_string
       (String.mapi (fun i c -> if i = at then Char.chr byte else c) s)
      : (G.t, string) result)

let gen_cut = Gen.(pair gen_t nat)
let gen_changed = Gen.(triple gen_t nat (int_range 0 255))

(* The format's version is the 64-bit word at byte 12 of a description, the word
   after its magic, as to_string wrote it. *)
let version_at = 12

let test_version () =
  let s = Bytes.of_string (G.to_string (base (fst (polled "version")))) in
  Bytes.set_int64_le s version_at 7L;
  expect (require_error (G.of_string (Bytes.to_string s)))
  @@ __POS_OF__ {|a program description of format version 7, not 16|}

let test_cut_version () =
  let s = G.to_string (base (fst (polled "cut-version"))) in
  expect (require_error (G.of_string (String.sub s 0 (version_at + 3))))
  @@ __POS_OF__
       {|a malformed program description at byte 12: an integer past the end|}

(* Footprint *)

(* Collects and drains [ds] for a fixed number of rounds, then is each one's
   allocated bytes: a collected buffer's memory returns a round late. *)
let settled ds =
  for _ = 1 to 4 do
    Gc.full_major ();
    List.iter
      (fun r ->
        ignore (B.create r 0);
        Rig.free_cache r)
      ds
  done;
  List.map Rig.allocated ds

let software () = Result.get_ok (Rig.software (fresh "footprint"))
let hidden_device () = fst (P.open_ ~host_visible:false (fresh "footprint"))
let kind_alloc kind bytes = G.Alloc { device = 0; kind; bytes; init = "" }

(* Programs on a device (device 0) and the host (device 1), by name: the device,
   the ints, the memory and the steps. *)
let footprints :
    (string * (unit -> Rig.t) * int * G.memory iarray * G.step iarray) list =
  [
    ( "memory of each device, and the ints",
      software,
      3,
      [| alloc 0 200; alloc 0 64; alloc 1 300 |],
      [||] );
    ( "host memory a move names, and the ints a move names",
      software,
      2,
      [| alloc 1 64 |],
      [| move (Input 0) (all 0) 64; move (all 0) Ints 8 |] );
    ( "Mapped memory of a software device",
      software,
      0,
      [| kind_alloc Mapped 100 |],
      [||] );
    ( "Pinned memory of a device the host does not address, in the host's budget",
      hidden_device,
      0,
      [| kind_alloc Pinned 100 |],
      [||] );
  ]

(* A load makes, in each device's budget, what footprint counts there. *)
let test_footprint (_, device, ints, memory, steps) =
  let d = device () in
  let ds = [ d; Rig.host ] in
  let t =
    {
      G.devices = [| Rig.arch d; Rig.arch Rig.host |];
      ints;
      memory;
      images = [||];
      inputs = [| { G.device = 0; bytes = 64 } |];
      steps;
    }
  in
  let want = G.footprint t (Iarray.of_list ds) in
  let before = settled ds in
  let p = load_ok t ds in
  let after = settled ds in
  ignore (Sys.opaque_identity p);
  equal
    (list (pair string int))
    (List.map (fun (r, n) -> (Rig.name r, n)) want)
    (List.map2
       (fun r (a, b) -> (Rig.name r, b - a))
       ds
       (List.combine before after))

(* Loaded *)

(* A loaded program that is unreachable returns its memory and unloads its
   images once the work of its last run is done. *)
let test_collected () =
  let d, pd = polled "collected" in
  let count call = List.length (List.filter (( = ) call) (P.log pd)) in
  (* Collects, then drains [d] as an allocation does, for a fixed number of
     rounds: the program's collection frees its images a round later. *)
  let drain () =
    for _ = 1 to 3 do
      Gc.full_major ();
      ignore (B.create d 0)
    done
  in
  let before = P.allocated pd Device in
  let[@inline never] run_dropped () =
    let p = load_ok (one_device_fill d) [ d ] in
    ignore (G.run p { inputs = [||]; ints = [||] })
  in
  run_dropped ();
  drain ();
  Rig.free_cache d;
  equal ~msg:"images while the run's work is unrun" int 0 (count "unload");
  equal ~msg:"memory while the run's work is unrun" int (before + 64)
    (P.allocated pd Device);
  ignore (P.run pd);
  drain ();
  Rig.free_cache d;
  equal ~msg:"images once the work ran" int 1 (count "unload");
  equal ~msg:"memory once the work ran" int before (P.allocated pd Device)

(* Load *)

(* Descriptions that break a rule their types or the order section state: load
   raises for each. *)
let breaks =
  let one d = [ d ] in
  let head = G.Memory { memory = 0; offset = 0; length = 8 } in
  (* Bytes [offset] to [offset + length] of memory 0: views [0, 16] and [8, 16]
     at offsets 8 and 0 lie apart within their views and share bytes 8 to 16 of
     the memory. *)
  let view offset length = G.Memory { memory = 0; offset; length } in
  [
    ("as many devices as the program's", (fun d -> base d), fun _ -> []);
    ( "a device's arch",
      (fun d -> { (base d) with devices = [| "no such arch" |] }),
      one );
    ( "a memory's device",
      (fun d -> { (base d) with memory = [| alloc 1 64 |] }),
      one );
    ( "a memory's size",
      (fun d -> { (base d) with memory = [| alloc 0 (-1) |] }),
      one );
    (* Polled's device offers no Pinned memory: the host addresses its
       memory. *)
    ( "a memory's kind its device does not offer, even of no bytes",
      (fun d -> { (base d) with memory = [| alloc 0 64; kind_alloc Pinned 0 |] }),
      one );
    ( "memory on an io device",
      (fun d ->
        let io = Rig_support.io "program:broken-io" in
        {
          (base d) with
          devices = [| Rig.arch d; Rig.arch io |];
          memory = [| alloc 0 64; alloc 1 64 |];
        }),
      fun d -> [ d; Rig_support.io "program:broken-io" ] );
    ( "an init longer than its memory",
      (fun d ->
        { (base d) with memory = [| alloc ~init:(String.make 65 'x') 0 64 |] }),
      one );
    ( "a view past its memory",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~slots:[| Memory { memory = 0; offset = 60; length = 8 } |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "an input's index",
      (fun d ->
        {
          (base d) with
          steps =
            [| submit 0 ~slots:[| Input 3 |] [| fill ~image:0 ~groups:1 0 |] |];
        }),
      one );
    ( "an image's index",
      (fun d -> with_part (fill ~image:5 ~groups:1 0) (base d)),
      one );
    ( "a view's memory index",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~slots:[| Memory { memory = 3; offset = 0; length = 8 } |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "an input of a negative size",
      (fun d -> { (base d) with inputs = [| { device = 0; bytes = -1 } |] }),
      one );
    ( "a copy naming no slot",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~slots:[| all 0 |]
                [| fill ~image:0 ~groups:1 0; pcopy ~src:0 ~dst:5 8 |];
            |];
        }),
      one );
    ( "parameters of no whole word",
      (fun d -> with_params (String.make 6 '\000') (base d)),
      one );
    ("a scalar past its parameters", with_scalars [| scalar 12 (fixed 0) |], one);
    ( "a scalar at a negative byte",
      with_scalars [| scalar ~width:W32 (-4) (fixed 0) |],
      one );
    ( "scalars that share a byte",
      with_scalars [| scalar 8 (fixed 0); scalar ~width:W32 12 (fixed 0) |],
      one );
    ( "a constant scalar past 32 bits",
      with_scalars [| scalar ~width:W32 8 (fixed (1 lsl 32)) |],
      one );
    ( "a scalar of a word past the ints",
      with_scalars ~ints:1 [| scalar 8 (int_at 1) |],
      one );
    ( "a geometry of a word past the ints",
      (fun d ->
        let part =
          match fill ~image:0 ~groups:1 0 with
          | Launch l ->
              G.Launch { l with groups = (int_at 0, fixed 1, fixed 1) }
          | part -> part
        in
        with_part part (base d)),
      one );
    ( "a slot no ref names",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0 ~slots:[| Input 0; Input 0 |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "memory named by an unordered ref and a copy",
      copying ~use:Unordered [| all 0; Input 0 |],
      one );
    ( "an input named by an unordered ref and a copy",
      copying ~use:Unordered [| Input 0; all 0 |],
      one );
    ( "memory named by an unordered and a read ref",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~slots:[| all 0; Input 0 |]
                [| fill ~use:Unordered ~image:0 ~groups:1 0; kcopy ~image:0 8 |];
            |];
        }),
      one );
    ( "a copy past the end of its source",
      copying ~ints:2 ~length:16 [| head; Ints |],
      one );
    ( "a copy past the end of its destination",
      copying ~ints:1 ~length:16 [| all 0; Ints |],
      one );
    ( "a copy at a negative offset",
      copying ~src_offset:(fixed (-8)) ~length:8 [| all 0; Input 0 |],
      one );
    ( "a copy of a negative length",
      copying ~length:(-1) [| all 0; Input 0 |],
      one );
    ( "a copy between overlapping bytes of the ints",
      copying ~ints:2 ~dst_offset:(fixed 4) ~length:8 [| Ints; Ints |],
      one );
    ( "a copy between overlapping bytes of an input",
      copying ~dst_offset:(fixed 4) ~length:8 [| Input 0; Input 0 |],
      one );
    ( "a copy offset of a word past the ints",
      copying ~ints:1 ~src_offset:(int_at 1) ~length:8 [| all 0; Input 0 |],
      one );
    ( "a move of a negative length",
      (fun d -> { (base d) with steps = [| move (all 0) (Input 0) (-1) |] }),
      one );
    ( "a move past the end of its source",
      (fun d -> { (base d) with steps = [| move head (Input 0) 16 |] }),
      one );
    ( "a move past the end of its destination",
      (fun d ->
        {
          (base d) with
          steps = [| move ~dst_offset:(fixed 60) (Input 0) (all 0) 8 |];
        }),
      one );
    ( "a move at a negative offset",
      (fun d ->
        {
          (base d) with
          steps = [| move ~src_offset:(fixed (-8)) (all 0) (Input 0) 8 |];
        }),
      one );
    ( "a move offset of a word past the ints",
      (fun d ->
        {
          (base d) with
          steps = [| move ~src_offset:(int_at 0) (all 0) (Input 0) 8 |];
        }),
      one );
    ( "a move between overlapping bytes of one memory",
      (fun d ->
        {
          (base d) with
          steps = [| move ~src_offset:(fixed 8) (view 0 16) (view 8 16) 8 |];
        }),
      one );
    ( "a move between overlapping bytes of an input",
      (fun d ->
        {
          (base d) with
          steps = [| move ~dst_offset:(fixed 4) (Input 0) (Input 0) 8 |];
        }),
      one );
    ( "a copy between overlapping bytes of one memory",
      copying ~src_offset:(fixed 8) ~length:8 [| view 0 16; view 8 16 |],
      one );
    ( "a loop of trips of a word past the ints",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              Loop
                {
                  trips = int_at 0;
                  trip = None;
                  flag = None;
                  body = (base d).steps;
                };
            |];
        }),
      one );
    ( "a ref past its slots",
      (fun d ->
        {
          (base d) with
          steps = [| submit 0 ~slots:[| Input 0 |] [| kcopy ~image:0 8 |] |];
        }),
      one );
  ]

(* A launch of Polled's [fill] of [threads] per group, which runs at most
   1024. *)
let fill_threads threads =
  match fill ~image:0 ~groups:1 0 with
  | Launch l -> G.Launch { l with threads }
  | part -> part

(* What a refusal's [why] names, as load states: the name of the device that
   refused first, Polled's or the host's, or the step. *)
type why = Device_first | Host_first | Step of int

(* Descriptions a device refuses: load answers Error for each. *)
let refusals =
  let one d = [ d ] in
  [
    ( "a grid of more than 2^32 - 1 groups",
      (fun d -> with_part (fill ~image:0 ~groups:(1 lsl 33) 0) (base d)),
      one,
      Step 0 );
    ( "a negative count of threads",
      (fun d ->
        with_part (fill_threads (fixed 1, fixed (-1), fixed 1)) (base d)),
      one,
      Step 0 );
    ( "threads past the most of the function",
      (fun d -> with_part (fill_threads (fixed 33, fixed 32, fixed 1)) (base d)),
      one,
      Step 0 );
    ( "threads past the most of the function in x and y, and none in z",
      (fun d -> with_part (fill_threads (fixed 33, fixed 32, fixed 0)) (base d)),
      one,
      Step 0 );
    ( "threads past the most of the function in x and y, z set by each run",
      (fun d ->
        let t = with_part (fill_threads (fixed 33, fixed 32, int_at 0)) in
        { (t (base d)) with ints = 1 }),
      one,
      Step 0 );
    ( "an image its device refuses",
      (fun d ->
        { (base d) with images = [| { device = 0; binary = "nonsense" } |] }),
      one,
      Device_first );
    ( "a function its image lacks",
      (fun d ->
        let part =
          match fill ~image:0 ~groups:1 0 with
          | Launch l -> G.Launch { l with name = "nope" }
          | part -> part
        in
        with_part part (base d)),
      one,
      Device_first );
  ]

let test_refusal (_, t, ds, names) =
  let d = fst (polled "refused") in
  let why = require_error (G.load (t d) (Iarray.of_list (ds d))) in
  match names with
  | Device_first -> starts_with ~prefix:(Rig.name d) why
  | Host_first -> starts_with ~prefix:(Rig.name Rig.host) why
  | Step i -> contains ~sub:(strf "step %d" i) why

let broken = Exn.invalid_arg ~sub:"Rig_program.load: "

(* load raises before it makes anything: the device's driver was asked for
   nothing. *)
let test_break (_, t, ds) =
  let d, pd = polled "broken" in
  let t = t d in
  let before = P.log pd in
  raises_match broken (fun () -> G.load t (Iarray.of_list (ds d)));
  equal ~msg:"the driver's calls" (list string) before (P.log pd)

let host_breaks =
  let two d = [ d; Rig.host ] in
  let with_steps ?(ints = 0) d steps =
    with_host d { (base d) with ints; steps }
  in
  let loop ?trip flag = Loop { trips = fixed 1; trip; flag; body = [||] } in
  [
    ( "an int past the ints",
      (fun d ->
        with_steps ~ints:1 d [| step_affine Ints ~a:(int_at 1) ~c:(fixed 0) |]),
      two );
    ( "a trip past the ints",
      (fun d -> with_steps ~ints:1 d [| loop ~trip:1 None |]),
      two );
    ( "a flag of no byte",
      (fun d ->
        with_steps d
          [| loop (Some (Memory { memory = 0; offset = 0; length = 0 }, 0)) |]),
      two );
    ( "a flag past its view",
      (fun d ->
        with_steps d
          [| loop (Some (Memory { memory = 0; offset = 0; length = 8 }, 8)) |]),
      two );
    ( "a flag at a negative byte",
      (fun d -> with_steps d [| loop (Some (Input 0, -1)) |]),
      two );
    ( "a flag past the ints",
      (fun d -> with_steps ~ints:1 d [| loop (Some (Ints, 8)) |]),
      two );
  ]

let host_refusals =
  let two d = [ d; Rig.host ] in
  let with_steps d steps = with_host d { (base d) with steps } in
  [
    ( "host code that does not link",
      (fun d ->
        {
          (with_steps d [||]) with
          images = [| { device = 1; binary = "nonsense" } |];
        }),
      two,
      Host_first );
    ( "a host function its image lacks",
      (fun d ->
        with_steps d
          [| step_affine ~name:"nope" (Input 0) ~a:(fixed 0) ~c:(fixed 0) |]),
      two,
      Host_first );
  ]

(* Memory a host launch names that the host does not address. *)
let test_host_unaddressed () =
  let d, _ = P.open_ ~host_visible:false (fresh "hidden") in
  let t =
    with_host d
      {
        (base d) with
        steps = [| step_affine (all 0) ~a:(fixed 1) ~c:(fixed 0) |];
      }
  in
  is_error
    ~pp:(fun ppf _ -> Format.pp_print_string ppf "a loaded program")
    (G.load t [| d; Rig.host |])

(* A description's integers may come from another process's bytes: each field
   whose sum with another bounds an access is checked where a sum wraps. *)
type extreme =
  | Ints of int
  | Scalar_at of int
  | Word of int
  | View_at of int
  | View_length of int

let pp_extreme ppf = function
  | Ints n -> Format.fprintf ppf "ints %d" n
  | Scalar_at n -> Format.fprintf ppf "a scalar at %d" n
  | Word n -> Format.fprintf ppf "a value of word %d" n
  | View_at n -> Format.fprintf ppf "a view at %d" n
  | View_length n -> Format.fprintf ppf "a view of %d bytes" n

let extremes =
  List.concat_map
    (fun n ->
      let fields = [ Scalar_at n; Word n; View_at n; View_length n ] in
      if n < 0 || n > max_int / 8 then Ints n :: fields else fields)
    [
      max_int;
      max_int - 1;
      max_int - 3;
      max_int - 7;
      min_int;
      min_int + 1;
      -1;
      1 lsl 60;
      (max_int / 8) + 1;
      (max_int / 16) + 1;
    ]

(* load raises for each, a rule the description breaks. *)
let extreme_law e =
  let d = fst (Lazy.force devices) in
  let t = base d in
  let view offset length =
    submit 0
      ~slots:[| Memory { memory = 0; offset; length } |]
      [| fill ~image:0 ~groups:1 0 |]
  in
  let t =
    match e with
    | Ints n -> { t with ints = n }
    | Scalar_at n -> with_scalars [| scalar n (fixed 0) |] d
    | Word n -> with_scalars ~ints:1 [| scalar 8 (int_at n) |] d
    | View_at n -> { t with steps = [| view n 8 |] }
    | View_length n -> { t with steps = [| view 8 n |] }
  in
  raises_match ~msg:"load raises" broken (fun () -> G.load t [| d |])

(* A device runs code of its own arch, and a device of an x86-64 level the code
   of each level below it: "the levels are cumulative" (x86-64 psABI, §3.1.1).
   An arm64 arch, "arm64" with features after a "+", as in "arm64+fp16", runs
   arm64 code whose features are all among its own: "dependent features are all
   available" (ACLE, "Function Multi Versioning", "Selection"). Code of another
   arch, of a level above, of a feature the device lacks, or of another family
   is refused. Another arch that holds a "+", as AMD's target IDs do
   ("gfx90a:xnack+"), runs only its own code. *)
let x86_levels = [ "x86-64"; "x86-64-v2"; "x86-64-v3"; "x86-64-v4" ]

let arm64s =
  [
    ("arm64", []);
    ("arm64+fp16", [ "fp16" ]);
    ("arm64+bf16+fp16", [ "bf16"; "fp16" ]);
  ]

let archs =
  x86_levels @ List.map fst arm64s
  @ [ "gfx1201"; "gfx90a:xnack"; "gfx90a:xnack+"; "polled" ]

let runs ~device code =
  let level a = List.find_index (String.equal a) x86_levels in
  match (level device, level code) with
  | Some d, Some c -> c <= d
  | _ -> (
      match (List.assoc_opt device arm64s, List.assoc_opt code arm64s) with
      | Some has, Some needs -> List.for_all (fun f -> List.mem f has) needs
      | _ -> String.equal device code)

let of_arch =
  let opened = Hashtbl.create 8 in
  fun arch ->
    match Hashtbl.find_opt opened arch with
    | Some d -> d
    | None ->
        let d = fst (P.open_ ~arch (strf "program:arch-%s" arch)) in
        Hashtbl.replace opened arch d;
        d

(* Whether load takes code of arch [code] on a device of arch [device] iff
   [runs]. *)
let loads ~runs code device =
  let d = of_arch device in
  let t = { (base d) with devices = [| code |] } in
  if runs then match G.load t [| d |] with Ok _ -> () | Error why -> fail why
  else raises_match ~msg:"load raises" broken (fun () -> G.load t [| d |])

let test_arch_order (code, device) = loads ~runs:(runs ~device code) code device

(* arm64 archs of drawn features, written in lexicographic order as ACLE writes
   them. *)
let arm64_features = [ "bf16"; "dotprod"; "fp16"; "i8mm" ]

let arm64_arch mask =
  List.filteri (fun i _ -> (mask lsr i) land 1 = 1) arm64_features
  |> List.fold_left (fun a f -> a ^ "+" ^ f) "arm64"

let arm64_pairs =
  let n = 1 lsl List.length arm64_features in
  Gen.(pair (int_range 0 (n - 1)) (int_range 0 (n - 1)))

let inclusion (code, device) =
  let subset = code land device = code in
  cover ~share:0.05 "the code's features are the device's" (code = device);
  cover ~share:0.1 "fewer features than the device's" (subset && code <> device);
  cover ~share:0.3 "a feature the device lacks" (not subset);
  loads ~runs:subset (arm64_arch code) (arm64_arch device)

let test_unborrowable () =
  let a, _ = polled ~peers:false "lone-a"
  and b, _ = polled ~peers:false "lone-b" in
  let t =
    {
      G.devices = [| Rig.arch a; Rig.arch b |];
      ints = 0;
      memory = [| alloc 1 64 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [||];
      steps = [| submit 0 ~slots:[| all 0 |] [| fill ~image:0 ~groups:1 0 |] |];
    }
  in
  let why = require_error (G.load t [| a; b |]) in
  contains ~msg:"names the device" ~sub:(Rig.name a) why

let test_machines () =
  let d = fst (polled "machines") in
  let far =
    Rig_support.machine (strf "program-%d" (Atomic.fetch_and_add opened 1))
  in
  let t = { (base d) with devices = [| Rig.arch d; Rig.arch far |] } in
  raises_match Exn.invalid_arg (fun () -> G.load t [| d; far |])

(* Ends in two views of one memory that share no byte are accepted, though their
   offsets within the views are equal: a move and a copy part from bytes [0, 32]
   of memory 0 into bytes [32, 64], each then moved into input 0. *)
let test_disjoint_views () =
  let d, _ = polled ~host_visible:false "disjoint-views" in
  let low = G.Memory { memory = 0; offset = 0; length = 32 }
  and high = G.Memory { memory = 0; offset = 32; length = 32 } in
  let program first =
    {
      (base d) with
      memory = [| alloc ~init:(String.make 32 'l' ^ String.make 32 'h') 0 64 |];
      steps = [| first; move (all 0) (Input 0) 64 |];
    }
  in
  let copy_part = submit 0 ~slots:[| low; high |] [| pcopy 32 |] in
  List.iter
    (fun (what, first) ->
      let p = load_ok (program first) [ d ] in
      let x = of_bytes d (String.make 64 '\000') in
      ignore (G.run p { inputs = [| x |]; ints = [||] });
      equal ~msg:what string (String.make 64 'l') (read x))
    [ ("a move", move low high 32); ("a copy part", copy_part) ]

(* An io device makes empty buffers of every kind (rig.mli, Buffer.create),
   though it offers none. *)
let test_io_empty () =
  let io = Rig_support.io "program:io-empty" in
  let t =
    {
      G.devices = [| Rig.arch io |];
      ints = 0;
      memory = [| kind_alloc Device 0; kind_alloc Pinned 0; kind_alloc Mapped 0 |];
      images = [||];
      inputs = [||];
      steps = [||];
    }
  in
  ignore (load_ok t [ io ])

(* Loading on another machine's devices goes through its agent, which takes no
   rails from this process. *)
let test_far_rails () =
  let far =
    Rig_support.machine
      (strf "program-rails-%d" (Atomic.fetch_and_add opened 1))
  in
  let t = { (base far) with devices = [| Rig.arch far |] } in
  raises_match Exn.invalid_arg (fun () ->
      G.load ~rails:(fun _ -> None) t [| far |]);
  raises_match ~msg:"without rails: no agent's device"
    (Exn.invalid_arg ~sub:"of an agent") (fun () -> G.load t [| far |])

(* A device whose budget is 64 KiB, or whose driver refuses every allocation,
   which rig answers after its rounds of reclamation. *)
let refusers =
  [
    ("its budget", fun () -> fst (P.open_ ~budget:65536 (fresh "budget")));
    ( "its driver",
      fun () ->
        let d, pd = P.open_ (fresh "refused") in
        P.fail_at pd 1 (`Refuse max_int);
        d );
  ]

(* A load whose memory the device refuses raises Out_of_memory for the memory's
   bytes on the device, which is not lost. *)
let test_load_refused (_, device) =
  let d = device () in
  raises_match (out_of_memory d mib) (fun () ->
      G.load (sized mib d) (Iarray.of_list [ d ]));
  equal ~msg:"the device after" (option string) None (Rig.lost d)

(* A load on a device that faults raises Lost with the fault, as does a load on
   the device it lost. *)
let test_load_lost () =
  let d, pd = polled "load-lost" in
  P.fault pd "the engine hung";
  let load () = G.load (one_device_fill d) (Iarray.of_list [ d ]) in
  raises_match ~msg:"the load that finds the fault" (lost d "the engine hung")
    load;
  raises_match ~msg:"a load on the lost device" (lost d "the engine hung") load

let reentered = "Rig_program.run: called on the thread that runs it"

let inside_lock fn =
  "Rig_program." ^ fn
  ^ ": called from a signal handler, finaliser or Gc.Memprof callback inside a \
     lock of the calling thread's"

(* What a callback's load and run met: whether the thread held a lock of rig's,
   "loaded" or why the load refused, "ran" or why the run raised, and the bytes
   the device allocated meanwhile. *)
type landing = { held : bool; loaded : string; ran : string; allocated : int }

(* A load and a run of a program from a [Gc.Memprof] callback at the [n]-th
   allocation of a run of it and the wait for that run, on a fresh device, whose
   first sleep logs itself in a section of its lock that allocates. [None] where
   the run made fewer allocations. *)
let reenter n =
  let d, _ = polled "reenter" in
  let t = one_device_fill d in
  let p = load_ok t [ d ] in
  let landing = ref None in
  let call () =
    let held = Rig_edge_primitive.holds () > 0 in
    let before = Rig.allocated d in
    let loaded =
      match G.load t (Iarray.of_list [ d ]) with
      | Ok _ -> "loaded"
      | Error why -> why
      | exception Invalid_argument why -> why
    in
    let ran =
      match G.run p empty_frame with
      | _ -> "ran"
      | exception Invalid_argument why -> why
    in
    landing := Some { held; loaded; ran; allocated = Rig.allocated d - before }
  in
  let reached =
    Rig_support.interrupted ~landing:(Call call) n (fun () ->
        Array.iter Rig.Point.wait (G.run p empty_frame))
  in
  Array.iter Rig.Point.wait (G.run p empty_frame);
  if reached then !landing else None

(* The landings of every allocation of a run, in order. *)
let landings () =
  let rec from n acc =
    match reenter n with
    | None -> List.rev acc
    | Some l -> from (n + 1) (l :: acc)
  in
  from 1 []

(* Inside a lock, a load raises having made nothing; outside every lock it
   loads. *)
let test_load_reentry () =
  let ls = landings () in
  List.iter
    (fun l ->
      if l.held then begin
        equal ~msg:"a load inside a lock" string (inside_lock "load") l.loaded;
        equal ~msg:"bytes allocated inside a lock" int 0 l.allocated
      end
      else equal ~msg:"a load outside every lock" string "loaded" l.loaded)
    ls;
  greater ~msg:"landings inside a lock" int ~than:0
    (List.length (List.filter (fun l -> l.held) ls))

(* Inside a lock a run raises; outside every lock it runs, or raises where the
   callback runs on the thread that runs the program. *)
let test_run_reentry () =
  let ls = landings () in
  List.iter
    (fun l ->
      if l.held then
        equal ~msg:"a run inside a lock" string (inside_lock "run") l.ran
      else if l.ran <> "ran" then
        equal ~msg:"a run outside every lock" string reentered l.ran)
    ls;
  greater ~msg:"landings inside a lock" int ~than:0
    (List.length (List.filter (fun l -> l.held) ls));
  greater ~msg:"runs on the thread that runs the program" int ~than:0
    (List.length (List.filter (fun l -> l.ran = reentered) ls))

(* Run *)

let test_after () =
  let a, _ = polled "after-a" and b, _ = polled "after-b" in
  let pa = load_ok (one_device_fill a) [ a ] in
  let pb = load_ok (one_device_fill b) [ b ] in
  let pt = (G.run pa empty_frame).(0) in
  less ~msg:"a's signaled value, before the run waits" int
    ~than:(Rig.Point.value pt) (Rig.signaled a);
  let last = G.run ~waits:[| pt |] pb empty_frame in
  greater ~msg:"b's run followed it" int
    ~than:(Rig.Point.value pt - 1)
    (Rig.signaled a);
  Array.iter Rig.Point.wait last

(* A run whose first step on a device is a move follows [waits] there: the move
   takes them, and the submit after it on that device follows them through the
   move. *)
let test_after_move () =
  let a, _ = polled "after-move-a"
  and b, _ = polled ~host_visible:false "after-move-b" in
  let pa = load_ok (one_device_fill a) [ a ] in
  let t =
    {
      (one_device_fill b) with
      memory = [| alloc 0 64; alloc 0 64 |];
      steps =
        [|
          move (all 1) (all 0) 64;
          submit 0 ~slots:[| all 0 |] [| fill ~image:0 ~groups:1 7 |];
        |];
    }
  in
  let pb = load_ok t [ b ] in
  let pt = (G.run pa empty_frame).(0) in
  less ~msg:"a's signaled value, before the run waits" int
    ~than:(Rig.Point.value pt) (Rig.signaled a);
  let last = G.run ~waits:[| pt |] pb empty_frame in
  greater ~msg:"b's move followed it" int
    ~than:(Rig.Point.value pt - 1)
    (Rig.signaled a);
  Array.iter Rig.Point.wait last

(* Polled runs its work at a wait: run 2 waits for run 0's fill of its copy of
   the ints. *)
let test_ints_wait () =
  let d, pd = polled "two" in
  let p = load_ok (ints_fill d) [ d ] in
  ignore (P.launches pd);
  ignore (G.run p empty_frame);
  ignore (G.run p empty_frame);
  equal ~msg:"runs 0 and 1 handed over, neither run" int 0
    (List.length (P.launches pd));
  let last = G.run p empty_frame in
  greater ~msg:"run 2 waited for run 0's work on its copy" int ~than:0
    (List.length (P.launches pd));
  Array.iter Rig.Point.wait last

(* Polled runs its work only at a wait on the host: three runs of a program
   without ints ran none of their launches, so none waited for an earlier run's
   work. *)
let test_memory_waits_not () =
  let d, pd = polled "one" in
  let p = load_ok (one_device_fill d) [ d ] in
  ignore (P.launches pd);
  let last = List.init 3 (fun _ -> G.run p empty_frame) in
  equal ~msg:"launches run once the three runs returned" int 0
    (List.length (P.launches pd));
  List.iter (Array.iter Rig.Point.wait) last

(* A loop of two trips stores its trip in int 0, and each trip's launch copies
   that word's low byte, read through an Ints slot, to byte [trip] of the input:
   the launch's offset is a scalar of the trip. The gate holds the device's
   work, so trip 1's store waits for trip 0's work. *)
let test_trip_follows_work () =
  let d, pd = polled "trip-order" in
  let copy_byte =
    Launch
      {
        image = 0;
        name = "copy";
        label = "copy";
        params = le64 0 ^ le64 0 ^ le64 1;
        scalars = [| scalar 8 (int_at 0) |];
        refs =
          [|
            { G.slot = 0; at = 0; use = Read };
            { G.slot = 1; at = 8; use = Write };
          |];
        groups = one;
        threads = one;
        shared = fixed 0;
      }
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          Loop
            {
              trips = fixed 2;
              trip = Some 0;
              flag = None;
              body = [| submit 0 ~slots:[| Ints; Input 0 |] [| copy_byte |] |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let out = of_bytes d (String.make 8 '\255') in
  P.gate pd;
  let opener =
    Domain.spawn (fun () ->
        Rig_support.await "a sleep at the gate" (fun () -> P.sleepers pd > 0);
        P.open_gate pd)
  in
  ignore (G.run p { inputs = [| out |]; ints = [||] });
  let bytes = read out in
  Domain.join opener;
  equal ~msg:"each trip's work read its own trip" string "\000\001"
    (String.sub bytes 0 2)

(* A launch writes int 0 through an Ints slot, and a later launch's scalar of
   int 0 reads what it wrote. *)
let test_int_follows_writer () =
  let d, _ = polled "int-order" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          submit 0 ~slots:[| Ints |] [| fill ~image:0 ~groups:1 42 |];
          submit 0 ~slots:[| Input 0 |]
            [| fill ~scalars:[| scalar 8 (int_at 0) |] ~image:0 ~groups:1 0 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let out = one_word d in
  ignore (G.run p { inputs = [| out |]; ints = [| 0 |] });
  equal ~msg:"the word the first launch wrote" int 42 (word_of out)

type ints_case = { n : int; a : int; c : int }

let gen_ints =
  let open Gen in
  let+ n = int_range 0 8 and+ a = int_range 0 3 and+ c = int_range 1 4 in
  { n; a; c }

let gen_ints =
  Gen.with_pp
    (fun ppf i -> Format.fprintf ppf "n %d, a %d, c %d" i.n i.a i.c)
    gen_ints

(* The frame's int [n] becomes [a * n + c] through host code; a launch over as
   many groups writes as many words. *)
let ints_law i =
  let d = fst (Lazy.force devices) in
  let t =
    with_host d
      {
        G.devices = [||];
        ints = 1;
        memory = [||];
        images = [| { device = 0; binary = functions } |];
        inputs = [| { device = 0; bytes = 256 } |];
        steps =
          [|
            step_affine Ints ~a:(fixed i.a) ~c:(fixed i.c);
            submit 0 ~slots:[| Input 0 |] [| fill_at 0 100 |];
          |];
      }
  in
  let p = load_ok t [ d; Rig.host ] in
  let x = of_bytes d (String.make 256 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [| i.n |] });
  let m = (i.a * i.n) + i.c in
  equal ~msg:"a launch over the computed groups" string
    (String.concat "" (List.init m (fun j -> le64 (100 + j)))
    ^ String.make (256 - (8 * m)) '\000')
    (read x)

(* run reads its frame once, before its first step: a launch writes input 0, a
   move copies it into input 1 and waits for the launch at the gate, while
   another domain puts a buffer of no bytes in the frame as input 2. The run
   goes on with the buffer the frame held, which its last launch writes. *)
let test_frame_read_once () =
  let d, pd = polled "read-once" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 0;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = Iarray.init 3 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          submit 0 ~slots:[| Input 0 |] [| fill ~image:0 ~groups:1 42 |];
          move (Input 0) (Input 1) 8;
          submit 0 ~slots:[| Input 2 |] [| fill ~image:0 ~groups:1 7 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let frame =
    { G.inputs = [| one_word d; one_word d; one_word d |]; ints = [||] }
  in
  let z = frame.inputs.(2) in
  P.gate pd;
  let changer =
    Domain.spawn (fun () ->
        Rig_support.await "the move at the gate" (fun () -> P.sleepers pd > 0);
        frame.inputs.(2) <- B.create d 0;
        P.open_gate pd)
  in
  ignore (G.run p frame);
  Domain.join changer;
  equal ~msg:"the move ran" int 42 (word_of frame.inputs.(1));
  equal ~msg:"the frame's earlier input 2" int 7 (word_of z)

(* Run 2 waits for run 0's work on its copy of the ints before it writes its
   ints: the frame's ints changed during that wait change nothing. A copy of int
   0 into input 0 is that work. *)
let test_frame_ints_read_once () =
  let d, pd = polled "ints-once" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps = [| submit 0 ~slots:[| Ints; Input 0 |] [| kcopy ~image:0 8 |] |];
    }
  in
  let p = load_ok t [ d ] in
  for _ = 1 to 2 do
    ignore (G.run p { inputs = [| one_word d |]; ints = [| 1 |] })
  done;
  let frame = { G.inputs = [| one_word d |]; ints = [| 5 |] } in
  P.gate pd;
  let changer =
    Domain.spawn (fun () ->
        Rig_support.await "run 2 at the gate" (fun () -> P.sleepers pd > 0);
        frame.ints.(0) <- 99;
        P.open_gate pd)
  in
  let last = G.run p frame in
  Domain.join changer;
  Array.iter Rig.Point.wait last;
  equal ~msg:"the frame's int when run was called" int 5
    (word_of frame.inputs.(0))

(* test/fixtures' [waiting], image 0 on the host (device 1), over input 0: it
   sets word 1 of the input, then spins until word 0 is set, for at most 2^34
   rounds. *)
let waiting_program d =
  {
    G.devices = [| Rig.arch d; Rig.arch Rig.host |];
    ints = 0;
    memory = [||];
    images = [| host_image 1 "waiting" |];
    inputs = [| { G.device = 0; bytes = 16 } |];
    steps =
      [|
        submit 1 ~slots:[| Input 0 |]
          [|
            Launch
              {
                image = 0;
                name = "waiting";
                label = "waiting";
                params = String.make 16 '\000';
                scalars = [| scalar 8 (fixed (1 lsl 34)) |];
                refs = [| { G.slot = 0; at = 0; use = Write } |];
                groups = one;
                threads = one;
                shared = fixed 0;
              };
          |];
      |];
  }

(* run keeps its inputs until it returns: while the host code runs on input 0,
   the frame holds another buffer there and collections run, and the input's
   finaliser has not run. Once the run returned, it runs. *)
let test_input_kept () =
  let d, _ = polled "kept" in
  let p = load_ok (waiting_program d) [ d; Rig.host ] in
  let collected = Atomic.make 0 in
  let[@inline never] [@local never] frame () =
    let x = of_bytes d (String.make 16 '\000') in
    Gc.finalise_last (fun () -> Atomic.incr collected) x;
    ({ G.inputs = [| x |]; ints = [||] }, Rig_support.at x)
  in
  let frame, at = frame () in
  let runner = Domain.spawn (fun () -> ignore (G.run p frame)) in
  Rig_support.await "the code running" (fun () -> Rig_support.load (at + 8) = 1);
  frame.inputs.(0) <- B.create d 16;
  Gc.full_major ();
  Gc.full_major ();
  let while_running = Atomic.get collected in
  Rig_support.store at 1;
  Domain.join runner;
  equal ~msg:"collections of the input while the code ran" int 0 while_running;
  Gc.full_major ();
  Gc.full_major ();
  equal ~msg:"collections of the input once the run returned" int 1
    (Atomic.get collected)

let test_frame_refusals () =
  let d = fst (polled "frame") and e = fst (polled "frame-other") in
  let p = load_ok (base d) [ d ] in
  let before = Rig.submitted d in
  let refused msg inputs =
    raises_match ~msg Exn.invalid_arg (fun () ->
        G.run p { inputs; ints = [||] })
  in
  refused "no input" [||];
  refused "two inputs" [| B.create d 64; B.create d 64 |];
  refused "an input of another device" [| B.create e 64 |];
  refused "an input of fewer bytes" [| B.create d 63 |];
  equal ~msg:"no step ran" int 0 (Rig.submitted d - before);
  Array.iter Rig.Point.wait
    (G.run p { inputs = [| B.create d 65 |]; ints = [||] })

(* run checks its frame before its first step: with its last input too small, it
   raises having submitted nothing. *)
let test_frame_checked_first () =
  let d, _ = polled "checked-first" in
  let t =
    {
      (base d) with
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          submit 0 ~slots:[| Input 0 |] [| fill ~image:0 ~groups:1 1 |];
          submit 0 ~slots:[| Input 1 |] [| fill ~image:0 ~groups:1 2 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let inputs = [| one_word d; B.create d 7 |] in
  let submitted = Rig.submitted d in
  raises_match ~msg:"input 1 of 7 bytes" Exn.invalid_arg (fun () ->
      G.run p { inputs; ints = [||] });
  equal ~msg:"submissions" int submitted (Rig.submitted d)

let test_ints_refusal () =
  let d = fst (polled "ints") in
  let p = load_ok { (base d) with ints = 1 } [ d ] in
  raises_match ~msg:"two ints for one" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| B.create d 64 |]; ints = [| 1; 2 |] })

(* A read-only borrow on [d] of a file of one word, 5, and the file's path. *)
let read_only d =
  let path = Filename.concat (temp_dir ()) "word" in
  Out_channel.with_open_bin path (fun oc -> output_string oc (le64 5));
  let file = Rig_disk.buffer (Result.get_ok (Rig_disk.of_file path)) in
  (path, require_some (B.borrow d file))

(* A device and a program on it whose step 1 writes its input 0, of 8 bytes, by
   each way a run writes an input, after a step 0 that writes the same way into
   input 1. *)
let writers =
  let word memory = G.Memory { memory; offset = 0; length = 8 } in
  let inputs : G.input iarray =
    [| { G.device = 0; bytes = 8 }; { G.device = 0; bytes = 8 } |]
  in
  [
    ( "a host launch's write ref",
      fun () ->
        let affine x =
          step_affine ~host:0 ~image:0 x ~a:(fixed 2) ~c:(fixed 1)
        in
        ( Rig.host,
          {
            G.devices = [| Rig.arch Rig.host |];
            ints = 0;
            memory = [||];
            images = [| host_image 0 "affine" |];
            inputs;
            steps = [| affine (Input 1); affine (Input 0) |];
          } ) );
    ( "a copy part's destination",
      fun () ->
        let d, _ = polled "read-only-copy" in
        let copy x = submit 0 ~slots:[| word 0; x |] [| pcopy 8 |] in
        ( d,
          {
            G.devices = [| Rig.arch d |];
            ints = 0;
            memory = [| alloc ~init:(le64 7) 0 8 |];
            images = [||];
            inputs;
            steps = [| copy (Input 1); copy (Input 0) |];
          } ) );
    ( "a move's destination",
      fun () ->
        let d, _ = polled "read-only-move" in
        ( d,
          {
            G.devices = [| Rig.arch d |];
            ints = 0;
            memory = [| alloc ~init:(le64 7) 0 8 |];
            images = [||];
            inputs;
            steps = [| move (word 0) (Input 1) 8; move (word 0) (Input 0) 8 |];
          } ) );
  ]

(* A read-only input that the run writes is refused before any step: step 0,
   which writes the writable input 1, did not run, the device was given no work,
   and the file keeps its bytes. *)
let test_read_only (_, writer) =
  let d, t = writer () in
  let path, x = read_only d in
  let p = load_ok t [ d ] in
  let y = of_bytes d (le64 5) in
  let before = Rig.submitted d in
  raises_match Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x; y |]; ints = [||] });
  equal ~msg:"the device's submitted value" int before (Rig.submitted d);
  equal ~msg:"input 1, which step 0 writes" string (le64 5) (read y);
  equal ~msg:"the file's bytes" string (le64 5)
    (In_channel.with_open_bin path In_channel.input_all)

(* An input that a device naming it cannot borrow is refused before any step:
   [a]'s input 0, which [b] fills at step 1, on devices that map no peer. Step
   0, a fill of [b]'s own input 1, did not run. *)
let test_unborrowable_input () =
  let a, _ = polled ~peers:false "lone-input-a"
  and b, _ = polled ~peers:false "lone-input-b" in
  let filled x = submit 1 ~slots:[| x |] [| fill ~image:0 ~groups:1 42 |] in
  let t =
    {
      G.devices = [| Rig.arch a; Rig.arch b |];
      ints = 0;
      memory = [||];
      images = [| { device = 1; binary = functions } |];
      inputs = [| { G.device = 0; bytes = 64 }; { G.device = 1; bytes = 64 } |];
      steps = [| filled (Input 1); filled (Input 0) |];
    }
  in
  let p = load_ok t [ a; b ] in
  let y = of_bytes b (String.make 64 '\000') in
  let before = Rig.submitted b in
  raises_match Exn.invalid_arg (fun () ->
      G.run p { inputs = [| B.create a 64; y |]; ints = [||] });
  equal ~msg:"b's submitted value" int before (Rig.submitted b);
  equal ~msg:"input 1, which step 0 fills" string (String.make 64 '\000')
    (read y)

(* A raise at a step comes after the earlier steps ran: a fill of input 0, then
   a copy part whose offset the run's int puts past its slot. *)
let test_raise_after_steps () =
  let d, _ = polled "raise-after" in
  let t =
    {
      (base d) with
      ints = 1;
      steps =
        [|
          submit 0 ~slots:[| Input 0 |] [| fill ~image:0 ~groups:1 42 |];
          submit 0
            ~slots:[| all 0; Input 0 |]
            [| pcopy ~src_offset:(int_at 0) 8 |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 64 '\000') in
  raises_match Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x |]; ints = [| 60 |] });
  equal ~msg:"input 0's first word" int 42 (word_of x)

(* A loop's flag past its input's bytes raises at the run, which load cannot
   know. *)
let test_flag_outside () =
  let d, _ = polled "flag-outside" in
  let t =
    {
      (base d) with
      steps =
        [|
          Loop
            {
              trips = fixed 1;
              trip = None;
              flag = Some (Input 0, 8);
              body = [||];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let ones = of_bytes d (String.make 64 '\001') in
  ignore (G.run p { inputs = [| ones |]; ints = [||] });
  raises_match ~msg:"byte 8 of 8" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| of_bytes d (le64 1) |]; ints = [||] })

(* A copy's end that a run's ints put outside its slot, or whose offset is past
   an int, is refused at that step: [load] cannot know where it lies. *)
let test_copy_outside () =
  let d, _ = polled "copy-outside" in
  let load copy =
    load_ok
      {
        (base d) with
        ints = 1;
        steps = [| submit 0 ~slots:[| all 0; Input 0 |] [| copy |] |];
      }
      [ d ]
  in
  let input = of_bytes d (String.make 64 '\000') in
  let run p v = G.run p { inputs = [| input |]; ints = [| v |] } in
  let src = load (pcopy ~src_offset:(int_at 0) 8)
  and dst = load (pcopy ~dst_offset:(int_at 0) 8)
  and twice = load (pcopy ~src_offset:{ G.add = 0; terms = [| (0, 2) |] } 8) in
  ignore (run src 56);
  ignore (run dst 56);
  raises_match ~msg:"a source from byte 60 of 64" Exn.invalid_arg (fun () ->
      run src 60);
  raises_match ~msg:"a destination from byte 60 of 64" Exn.invalid_arg
    (fun () -> run dst 60);
  raises_match ~msg:"a negative offset" Exn.invalid_arg (fun () -> run src (-8));
  raises_match ~msg:"an offset past an int" Exn.invalid_arg (fun () ->
      run twice max_int)

(* A move's end that a run's ints put outside its slot, or whose offset is past
   an int, is refused at that step, with no move queued; ends at the slot's last
   bytes move. *)
let test_move_outside_slot () =
  let d, _ = polled ~host_visible:false "move-outside-slot" in
  let load ?src_offset ?dst_offset () =
    load_ok
      {
        (base d) with
        ints = 1;
        steps = [| move ?src_offset ?dst_offset (all 0) (Input 0) 8 |];
      }
      [ d ]
  in
  let input = of_bytes d (String.make 64 '\000') in
  let run p v = G.run p { inputs = [| input |]; ints = [| v |] } in
  let src = load ~src_offset:(int_at 0) ()
  and dst = load ~dst_offset:(int_at 0) ()
  and twice = load ~src_offset:{ G.add = 0; terms = [| (0, 2) |] } () in
  ignore (run src 56);
  ignore (run dst 56);
  let queued = Rig.submitted d in
  raises_match ~msg:"a source from byte 60 of 64" Exn.invalid_arg (fun () ->
      run src 60);
  raises_match ~msg:"a destination from byte 60 of 64" Exn.invalid_arg
    (fun () -> run dst 60);
  raises_match ~msg:"a negative offset" Exn.invalid_arg (fun () -> run src (-8));
  raises_match ~msg:"an offset past an int" Exn.invalid_arg (fun () ->
      run twice max_int);
  equal ~msg:"no move queued" int queued (Rig.submitted d)

(* A run whose submission's hand-over fails raises Lost, and so does every run
   after it: the run that raised left the program to them. The hand-over fails
   at the program's first run, or after [runs] runs. *)
let test_run_lost runs =
  let d, pd = polled "run-lost" in
  let p = load_ok (one_device_fill d) [ d ] in
  for _ = 1 to runs do
    Array.iter Rig.Point.wait (G.run p empty_frame)
  done;
  P.fail pd;
  for i = 1 to 3 do
    raises_match ~msg:(strf "run %d from the failure" i)
      (lost d "the submission failed") (fun () -> G.run p empty_frame)
  done

(* Run 2 waits for run 0's work on its copy of the ints: a fault that wait finds
   loses the device, and the run raises Lost. *)
let test_wait_lost () =
  let d, pd = polled "wait-lost" in
  let p = load_ok (ints_fill d) [ d ] in
  ignore (G.run p empty_frame);
  ignore (G.run p empty_frame);
  P.fault pd "the engine hung";
  raises_match (lost d "the engine hung") (fun () -> G.run p empty_frame)

(* A run interrupted from its [n]-th allocation by an exception at every
   allocation from that one on, then another run: the first leaves no lock held,
   its program's lock included, so the second runs, or raises [Rig.Lost] where
   the exception lost the device. Is whether the allocation was reached. *)
let interrupt_run n =
  let d, _ = polled "interrupt" in
  let p = load_ok (one_device_fill d) [ d ] in
  let reached =
    Rig_support.interrupted n (fun () ->
        Array.iter Rig.Point.wait (G.run p empty_frame))
  in
  equal
    ~msg:(strf "from allocation %d, locks held after" n)
    int 0
    (Rig_edge_primitive.holds ());
  (match G.run p empty_frame with
  | points -> Array.iter Rig.Point.wait points
  | exception Rig.Lost _ when Option.is_some (Rig.lost d) -> ());
  reached

let test_interrupt_run () =
  let rec from n = if interrupt_run n then from (n + 1) else n - 1 in
  greater ~msg:"allocations interrupted" int ~than:0 (from 1)

(* Time *)

(* On device 0: a fill of memory 0, a move of it into input 0, a submission of
   no parts, and a copy part of memory 0's first word into memory 1 at the byte
   word 0 of the run's ints holds, which a run refuses past memory 1's last
   word. *)
let timed_program d =
  {
    G.devices = [| Rig.arch d |];
    ints = 1;
    memory = [| alloc 0 size; alloc 0 size |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { G.device = 0; bytes = size } |];
    steps =
      [|
        submit 0 ~slots:[| all 0 |] [| fill ~image:0 ~groups:1 7 |];
        move (all 0) (Input 0) size;
        submit 0 [||];
        submit 0 ~slots:[| all 0; all 1 |] [| pcopy ~dst_offset:(int_at 0) 8 |];
      |];
  }

(* [timed_program]'s fill and copy part alone, which wait for nothing on the
   host. *)
let fill_and_copy d =
  let t = timed_program d in
  {
    t with
    inputs = [||];
    steps = [| Iarray.get t.steps 0; Iarray.get t.steps 3 |];
  }

let timed_frame ?(at = 0) inputs = { G.inputs; ints = [| at |] }

(* A device that times no work has [-1] for each step of parts, and a move and a
   submission of no parts have [0]. *)
let test_time_untimed () =
  let d, _ = P.open_ ~timed:false (fresh "untimed") in
  let p = load_ok (timed_program d) [ d ] in
  equal (array int) [| -1; 0; 0; -1 |]
    (G.time p (timed_frame [| B.create d size |]))

(* A profile taken around a time records each timed step's work with the time's
   own times. *)
let test_time_profiled () =
  let d, _ = polled "time-profiled" in
  let p = load_ok (timed_program d) [ d ] in
  let times, events =
    Rig.Profile.take (fun () -> G.time p (timed_frame [| B.create d size |]))
  in
  let spans =
    List.filter_map
      (function
        | Rig.Profile.Work w when Rig.equal w.device d -> Some (w.stop - w.start)
        | _ -> None)
      events
  in
  mem ~msg:"the fill's time" int times.(0) spans;
  mem ~msg:"the copy part's time" int times.(3) spans

(* A send has no work to time, and runs as in a run. *)
let test_send_time () =
  let d, _ = polled "send-time" in
  let rails, seen = recording_end () in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load ~rails (send_program d) [| d |])
  in
  let times = G.time p { inputs = [||]; ints = [| 5 |] } in
  equal ~msg:"the send's time" int 0 times.(1);
  equal ~msg:"the send's count" (list int) [ 5 ] (List.map fst !seen)

(* A close of a device whose work a time waits for makes the time raise the
   loss: the time's sleep holds until the close has lost the device. *)
let test_time_closed () =
  let d, dp = polled "time-closed" in
  let p = load_ok (fill_and_copy d) [ d ] in
  let sleeping = Atomic.make false in
  P.before dp "sleep" (fun () ->
      Atomic.set sleeping true;
      Rig_support.await "the close's loss" (fun () -> Rig.lost d <> None));
  let timing = Domain.spawn (fun () -> G.time p (timed_frame [||])) in
  Rig_support.await "the time's sleep" (fun () -> Atomic.get sleeping);
  Rig.close d;
  raises_match
    (function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false)
    (fun () -> Domain.join timing);
  equal ~msg:"its loss" (option string) (Some "closed") (Rig.lost d)

(* The pairs of times a device holds, against the pairs a time that raised left
   owed: a time that raises gives its pairs back once their work is done, at the
   device's next timed submit, and a time that returns gives back its own.
   Polled runs a time's work when the time waits for it, and the work a raised
   time left at the next time's wait. *)
type owed = { mutable reached : int; mutable unreached : int }

let timing = abstract "p" ~release:(fun (d, _) -> Rig.close d)

let pairs_commands =
  [
    command "load"
      (Gen.unit @-> makes timing)
      (fun () -> { reached = 0; unreached = 0 })
      (fun () ->
        let d, _ = polled "pairs" in
        (d, load_ok (fill_and_copy d) [ d ]));
    command "time"
      (timing ^-> returns int)
      (fun o ->
        o.reached <- o.unreached;
        o.unreached <- 0;
        o.reached)
      (fun (d, p) ->
        let times = G.time p (timed_frame [||]) in
        Array.iteri
          (fun i t -> at_least ~msg:(strf "step %d's time" i) int ~than:0 t)
          times;
        Rig.time_pairs d);
    command "time that raises"
      (timing ^-> returns int)
      (fun o ->
        o.reached <- 0;
        o.unreached <- o.unreached + 1;
        o.unreached)
      (fun (d, p) ->
        raises_match (Exn.invalid_arg ~sub:"a copy of 8 bytes from byte 64")
          (fun () -> G.time p (timed_frame ~at:size [||]));
        equal ~msg:"no pair reused before a settle" (list string) []
          (P.violations ());
        Rig.time_pairs d);
  ]

(* The most groups of a sized fill, and the most groups a core runs in a
   nanosecond: Polled calls its [fill] once per group, and no core makes more
   than one call a cycle at 8 GHz. A fill of [large] groups thus takes at least
   [large / per_ns] ns. *)
let large = 1 lsl 14
let per_ns = 8

(* On device 0: a fill of memory 0, of [large] words, over the groups word 0 of
   the run's ints holds, then a copy part of its first word into memory 1. *)
let sized_fill d =
  let words = G.Memory { memory = 0; offset = 0; length = 8 * large } in
  {
    G.devices = [| Rig.arch d |];
    ints = 1;
    memory = [| alloc 0 (8 * large); alloc 0 size |];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    steps =
      [|
        submit 0 ~slots:[| words |] [| fill_at 0 7 |];
        submit 0 ~slots:[| all 0; all 1 |] [| pcopy 8 |];
      |];
  }

(* Times of one program from two domains each answer their own run's times,
   within the host's window around their call. A large time's fill is timed, at
   least [large / per_ns] ns: one that answered another run's times, or left its
   fill untimed (0), falls short of it, as a fill of one group does unless the
   machine stalls it. *)
let shared_device = lazy (fst (polled "time-shared"))
let shared = abstract "q"

let shared_time ~least = function
  | Error e -> failf "raised %s" (Printexc.to_string e)
  | Ok (times, window) ->
      equal ~msg:"one per step" int 2 (Array.length times);
      Array.iteri
        (fun i t -> at_least ~msg:(strf "step %d's time" i) int ~than:0 t)
        times;
      at_least ~msg:"the fill's time" int ~than:least times.(0);
      at_most ~msg:"the times together" int ~than:window (times.(0) + times.(1))

let time_fill groups p =
  let t0 = Rig.Profile.now () in
  let times = G.time p (timed_frame ~at:groups [||]) in
  (times, Rig.Profile.now () - t0)

let shared_commands =
  [
    command "load"
      (Gen.unit @-> makes shared)
      (fun () -> ())
      (fun () ->
        let d = Lazy.force shared_device in
        load_ok (sized_fill d) [ d ]);
    command "time"
      (shared ^-> judges (pair (array int) int))
      (fun () -> shared_time ~least:0)
      (time_fill 1);
    command "time large"
      (shared ^-> judges (pair (array int) int))
      (fun () -> shared_time ~least:(large / per_ns))
      (time_fill large);
  ]

let tests =
  [
    group "order"
      [
        prop ~count:200
          "runs and times leave the bytes of their steps run in order, on any \
           device"
          (gen_case some_loops) bytes_law;
        test "a host step's launches are done when run returns"
          test_host_step_done;
      ];
    group "rails"
      [
        test "Rail memory is the rail's end's areas, which every run uses"
          test_rail_areas;
        test
          "a send calls its end's ready with its count once the work that \
           writes the outbound area is done"
          test_send;
        test "a send follows the run's waits" test_send_waits;
        test
          "a send's count below 1 raises at load where it is constant, and at \
           the run before ready otherwise"
          test_send_count;
        test "load raises for a rail that rails does not give" test_no_rail;
      ];
    group "domains"
      [
        stateful ~domains:2 ~steps:2 ~count:30
          "a run on either of two domains reads what the run before it wrote"
          swap_commands;
      ];
    group "value"
      [
        prop ~examples:sum_examples
          "a run computes add plus each word times its factor, in order from \
           add, and raises for a product or partial sum past an int or a sum \
           past its scalar's width"
          gen_sum sum_law;
      ];
    group "scalar"
      [
        prop
          "its little-endian word replaces its bytes of the launch's \
           parameters, every other byte kept"
          gen_beside beside_law;
        cases
          "a W32 scalar takes a value from -2^31 to 2^32 - 1 and writes its \
           low 32 bits"
          ~id:(fun (v, _) -> string_of_int v)
          w32_values scalar_range;
      ];
    group "ref"
      [
        test "its parameter bytes hold its slot's buffer's address in each run"
          test_input_ref;
      ];
    group "part"
      [
        test "a Copy part copies between its step's slots" test_copy_part;
        test "a Copy part in a loop moves with its trips" test_copy_loop;
        test "a Copy part copies within the ints at offsets they hold"
          test_copy_ints;
        test "a slot that only copies name leaves the refs' slots in place"
          test_renumbered;
        test "a launch's label names its work in profiles" test_label;
      ];
    group "step"
      [
        test "a move a device runs is queued, its point the run's end there"
          test_move_queued;
        test "moves through the ints a device runs are queued"
          test_move_ints_queued;
        test "moves through host memory a device runs are queued"
          test_move_host_queued;
        test "an int a queued move wrote is read once the move is done"
          test_move_writes_int;
        test "a move in a loop moves with its trips" test_move_loop;
        test "a move a device outside the program's runs is done at its step"
          test_move_outside;
        prop
          "a loop runs its body as many times as its trips, none for trips \
           below 1, each trip's index in the ints"
          (Gen.int_range (-2) 6) loop_law;
        test "a loop stops at a flag a step cleared" test_flag;
      ];
    group "bytes"
      [
        prop "of_string reads back what to_string wrote" gen_t round_trip;
        prop "of_string answers Error for a cut description" gen_cut cut_law;
        prop "of_string answers a changed byte without raising" gen_changed
          changed_law;
        test "of_string names both versions of another version's description"
          test_version;
        test
          "of_string says where a description cut in its version is malformed"
          test_cut_version;
      ];
    group "footprint"
      [
        cases "a load makes in each budget what footprint counts there"
          ~id:(fun (n, _, _, _, _) -> n)
          footprints test_footprint;
      ];
    group "loaded"
      [
        test
          "an unreachable program returns its memory and images after its last \
           run's work"
          test_collected;
      ];
    group "load"
      [
        cases "raises before it makes anything for a description that breaks"
          ~id:(fun (n, _, _) -> n)
          (breaks @ host_breaks) test_break;
        cases "raises for integers whose sums would wrap"
          ~id:(Format.asprintf "%a" pp_extreme)
          extremes extreme_law;
        cases
          "a device runs code of its arch, of a lower x86-64 level, or of \
           fewer arm64 features"
          ~id:(fun (code, device) -> strf "%s on %s" code device)
          (List.concat_map
             (fun code -> List.map (fun device -> (code, device)) archs)
             archs)
          test_arch_order;
        prop "an arm64 device runs code whose features are among its own"
          arm64_pairs inclusion;
        cases "answers Error, naming the refusing device or the step, for"
          ~id:(fun (n, _, _, _) -> n)
          (refusals @ host_refusals) test_refusal;
        test "answers Error for memory a device cannot borrow" test_unborrowable;
        test "answers Error for host code over memory the host does not address"
          test_host_unaddressed;
        test
          "accepts the two ends of a move or a copy in disjoint views of one \
           memory"
          test_disjoint_views;
        test "loads empty memory of every kind on an io device" test_io_empty;
        test "raises for devices of several machines" test_machines;
        test "raises for rails given with another machine's devices"
          test_far_rails;
        cases "raises Out_of_memory for memory refused by" ~id:fst refusers
          test_load_refused;
        test "raises Lost for a device that faults, and on the lost device"
          test_load_lost;
        test "raises having made nothing from a callback inside a lock"
          test_load_reentry;
      ];
    group "run"
      [
        prop
          "its ends are the last point of each device that ran a step, in the \
           order of the program's devices"
          (gen_case some_loops) ends_law;
        test "its first submission on a device follows its waits" test_after;
        test "its first move on a device follows its waits" test_after_move;
        test "run n first waits for run n - 2's work on its copy of the ints"
          test_ints_wait;
        test "a program without ints waits on the host for no earlier run"
          test_memory_waits_not;
        test "a value is read once the work that writes the ints is done"
          test_int_follows_writer;
        prop "a value read from ints host code wrote sets a launch's groups"
          gen_ints ints_law;
        test "a loop's trip is stored once the work that reads the ints is done"
          test_trip_follows_work;
        test "it reads its frame's inputs once, before its first step"
          test_frame_read_once;
        test "it reads its frame's ints once, before it waits for run n - 2"
          test_frame_ints_read_once;
        test "it keeps its inputs until it returns" test_input_kept;
        test
          "it raises before any step for another number of inputs, an input of \
           another device or of fewer bytes"
          test_frame_refusals;
        test
          "it raises before its first step for an input only a later step names"
          test_frame_checked_first;
        test "it raises for more ints than the program's" test_ints_refusal;
        cases "it raises before any step for a read-only input it writes by"
          ~id:fst writers test_read_only;
        test
          "it raises before any step for an input a naming device cannot borrow"
          test_unborrowable_input;
        test
          "it raises from a callback inside a lock, and on the thread that \
           runs the program"
          test_run_reentry;
        test "a raise at a step comes after the earlier steps ran"
          test_raise_after_steps;
        test "it raises at the step for a copy its ints put outside its slot"
          test_copy_outside;
        test "it raises at the step for a move its ints put outside its slot"
          test_move_outside_slot;
        test "it raises at the step for a loop's flag past its input"
          test_flag_outside;
        cases "it raises Lost for a hand-over that fails, and every run after"
          ~id:(strf "after %d runs") [ 0; 1 ] test_run_lost;
        test "it raises Lost for a fault its wait for run n - 2 finds"
          test_wait_lost;
        test "a run ended by an exception at any allocation leaves no lock held"
          test_interrupt_run;
      ];
    group "time"
      [
        prop
          "each step's time is 0 for a move or a submission of no parts, and \
           lies within the host's window around the call"
          (gen_case (Gen.constant false))
          times_law;
        prop "a program with a loop raises before any step"
          (gen_case (Gen.constant true))
          loop_refused_law;
        test "a send's time is 0" test_send_time;
        test
          "a device that times nothing has -1 for a step of parts, and 0 for a \
           move and no parts"
          test_time_untimed;
        test "a profile around a time records the time's own times"
          test_time_profiled;
        test "a close while a time waits makes it raise the loss"
          test_time_closed;
        stateful ~steps:8
          "a time gives back its pairs, and one that raised gives them back \
           once their work is done"
          pairs_commands;
        stateful ~domains:2 ~steps:2 ~count:30
          "times on two domains each answer their own run's steps"
          shared_commands;
      ];
  ]

let () = exit (Windtrap.run "rig.program" tests)
