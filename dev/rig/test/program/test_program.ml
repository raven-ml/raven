(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module P = Rig_support.Polled
module G = Rig_program
open Rig_program

let timeout = 60.
let strf = Printf.sprintf

(* Fresh Polled devices: a name opens once. *)
let opened = Atomic.make 0

let polled ?peers name =
  P.open_ ?peers (strf "program:%s-%d" name (Atomic.fetch_and_add opened 1))

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

let data bytes = { G.bytes; holes = [||] }
let functions = data "functions"

let hole ?(width = G.W64) ?(add = 0) ?(shift = 0) at leaf =
  { G.at; width; leaf; add; shift }

(* A launch of Polled's [fill]: the 64-bit word [i] of its write slot [0] is
   [base + i], for [groups] groups. *)
let fill ?(holes : _ iarray = [||]) ~image ~groups base =
  let bytes = String.make 8 '\000' ^ le64 base in
  {
    G.queue = "COMPUTE:0";
    after = [||];
    work =
      Launch
        {
          image;
          kernel = "fill";
          params = { bytes; holes };
          refs = [| { Rig.Submission.at = 0; slot = 0 } |];
          groups = (Fixed groups, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
  }

(* A launch of Polled's [copy]: [bytes] bytes from its slot [0] to its slot
   [1]. *)
let kcopy ~image bytes =
  {
    G.queue = "COMPUTE:0";
    after = [||];
    work =
      Launch
        {
          image;
          kernel = "copy";
          params = data (String.make 16 '\000' ^ le64 bytes);
          refs =
            [|
              { Rig.Submission.at = 0; slot = 0 };
              { Rig.Submission.at = 8; slot = 1 };
            |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
  }

let submit ?(reads : _ iarray = [||]) ?(writes : _ iarray = [||]) device parts =
  G.Submit { device; parts; reads; writes; fixed = [||] }

let alloc ?(copies = G.One) ?(init = data "") device bytes =
  G.Alloc { device; kind = B.Device; bytes; init; copies }

let all memory = G.Memory { memory; offset = 0; length = 64 }

let load_ok t ds =
  require_ok ~pp:Format.pp_print_string (G.load t (Iarray.of_list ds))

(* Steps against a model *)

(* Three memories and two inputs of 64 bytes, each on one of two devices, and
   steps over them. *)

let size = 64

type loc = Mem of int | In of int

type op =
  | Fill of { dev : int; dst : loc; groups : int; base : int }
  | Kcopy of { dev : int; src : loc; dst : loc; bytes : int }
  | Move of { src : loc; dst : loc }

type case = {
  mem_dev : int array;
  two : bool array;
  init : string array;
  in_dev : int array;
  inputs : string array;
  ops : op list;
}

let pp_loc ppf = function
  | Mem i -> Format.fprintf ppf "m%d" i
  | In i -> Format.fprintf ppf "x%d" i

let pp_op ppf = function
  | Fill { dev; dst; groups; base } ->
      Format.fprintf ppf "fill@%d %a %d groups from %d" dev pp_loc dst groups
        base
  | Kcopy { dev; src; dst; bytes } ->
      Format.fprintf ppf "copy@%d %a->%a %d bytes" dev pp_loc src pp_loc dst
        bytes
  | Move { src; dst } -> Format.fprintf ppf "move %a->%a" pp_loc src pp_loc dst

let pp_case ppf c =
  Format.fprintf ppf "memory on [%s] two [%s]; inputs on [%s]; [%a]"
    (String.concat ";" (Array.to_list (Array.map string_of_int c.mem_dev)))
    (String.concat ";" (Array.to_list (Array.map string_of_bool c.two)))
    (String.concat ";" (Array.to_list (Array.map string_of_int c.in_dev)))
    (Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ") pp_op)
    c.ops

let gen_case =
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
        ( 1,
          let+ src, dst = two_locs in
          Move { src; dst } );
      ]
  in
  let bytes = string_of ~size:(constant size) char in
  let+ mem_dev = array ~size:(constant 3) dev
  and+ two = array ~size:(constant 3) bool
  and+ init = array ~size:(constant 3) bytes
  and+ in_dev = array ~size:(constant 2) dev
  and+ inputs = array ~size:(constant 2) bytes
  and+ ops = list ~size:(int_range 1 6) op in
  { mem_dev; two; init; in_dev; inputs; ops }

let gen_case = Gen.with_pp pp_case gen_case
let devices = lazy (fst (polled "model-a"), fst (polled "model-b"))

let describe c =
  let slot = function Mem m -> all m | In i -> G.Input i in
  let image d = d in
  let step = function
    | Fill { dev; dst; groups; base } ->
        submit dev
          ~writes:[| slot dst |]
          [| fill ~image:(image dev) ~groups base |]
    | Kcopy { dev; src; dst; bytes } ->
        submit dev
          ~reads:[| slot src |]
          ~writes:[| slot dst |]
          [| kcopy ~image:(image dev) bytes |]
    | Move { src; dst } -> G.Move { src = slot src; dst = slot dst }
  in
  let a, b = Lazy.force devices in
  {
    G.devices = [| Rig.arch a; Rig.arch b |];
    code = [||];
    ints = 0;
    memory =
      Iarray.init 3 (fun m ->
          alloc
            ~copies:(if c.two.(m) then Two else One)
            ~init:(data c.init.(m))
            c.mem_dev.(m) size);
    images =
      [|
        { device = 0; binary = functions }; { device = 1; binary = functions };
      |];
    inputs =
      Iarray.map
        (fun device -> { G.device; bytes = size })
        (Iarray.of_array c.in_dev);
    steps = Iarray.of_list (List.map step c.ops);
  }

(* The model: memory of two copies is one per run parity. *)
let model c =
  let mem =
    Array.init 3 (fun m -> Array.init 2 (fun _ -> Bytes.of_string c.init.(m)))
  in
  fun n ->
    let inputs = Array.map Bytes.of_string c.inputs in
    let cell = function
      | Mem m -> mem.(m).(if c.two.(m) then n land 1 else 0)
      | In i -> inputs.(i)
    in
    List.iter
      (function
        | Fill { dst; groups; base; _ } ->
            for i = 0 to groups - 1 do
              Bytes.set_int64_le (cell dst) (8 * i) (Int64.of_int (base + i))
            done
        | Kcopy { src; dst; bytes; _ } ->
            Bytes.blit (cell src) 0 (cell dst) 0 bytes
        | Move { src; dst } -> Bytes.blit (cell src) 0 (cell dst) 0 size)
      c.ops;
    Array.map Bytes.to_string inputs

let model_law c =
  let a, b = Lazy.force devices in
  let ds = [| a; b |] in
  let p = load_ok (describe c) [ a; b ] in
  let expect = model c in
  for n = 0 to 2 do
    let xs = Array.map2 (fun d s -> of_bytes ds.(d) s) c.in_dev c.inputs in
    let points = G.run p { inputs = xs; ints = [||] } in
    let ran =
      List.filter
        (fun d ->
          List.exists
            (function
              | Fill { dev; _ } | Kcopy { dev; _ } -> dev = d | Move _ -> false)
            c.ops)
        [ 0; 1 ]
    in
    equal ~msg:"one point per device that ran, in their order" (list string)
      (List.map
         (fun d -> strf "%s:%d" (Rig.name ds.(d)) (Rig.submitted ds.(d)))
         ran)
      (List.map
         (fun pt -> Format.asprintf "%a" Rig.Point.pp pt)
         (Array.to_list points));
    equal
      ~msg:(strf "the inputs after run %d" n)
      (array string) (expect n) (Array.map read xs)
  done;
  (* A step that names memory another device's step wrote before. *)
  let rec crossed last = function
    | [] -> false
    | op :: ops ->
        let dev, reads, writes =
          match op with
          | Fill { dev; dst; _ } -> (Some dev, [], [ dst ])
          | Kcopy { dev; src; dst; _ } -> (Some dev, [ src ], [ dst ])
          | Move { src; dst } -> (None, [ src ], [ dst ])
        in
        let seen l =
          match (List.assoc_opt l last, dev) with
          | Some (Some d'), Some d -> d' <> d
          | _ -> false
        in
        List.exists seen (reads @ writes)
        || crossed (List.map (fun l -> (l, dev)) writes @ last) ops
  in
  cover "a step follows another device's step" (crossed [] c.ops);
  cover "memory of two copies" (Array.exists Fun.id c.two);
  cover "an input another device borrows"
    (List.exists
       (function
         | Fill { dev; dst = In i; _ } | Kcopy { dev; dst = In i; _ } ->
             dev <> c.in_dev.(i)
         | _ -> false)
       c.ops);
  cover "a move" (List.exists (function Move _ -> true | _ -> false) c.ops)

(* Holes *)

type hole_case = { width : G.width; add : int; shift : int }

let gen_hole =
  let open Gen in
  let+ width = of_list [ G.W32; G.W64 ]
  and+ add = int_range 0 (1 lsl 20)
  and+ shift = int_range 0 62 in
  { width; add; shift }

let gen_hole =
  Gen.with_pp
    (fun ppf h ->
      Format.fprintf ppf "%s, add %d, shift %d"
        (match h.width with W32 -> "W32" | W64 -> "W64")
        h.add h.shift)
    gen_hole

let shaped h at leaf =
  { G.at; width = h.width; leaf; add = h.add; shift = h.shift }

let word s i = Int64.to_int (String.get_int64_le s (8 * i))

(* A fill of one group whose base is a hole: it stores the base at word [i] of
   its write slot. *)
let fill_word i hole =
  {
    G.queue = "COMPUTE:0";
    after = [||];
    work =
      Launch
        {
          image = 0;
          kernel = "fill";
          params = { bytes = le64 (8 * i) ^ le64 0; holes = [| hole |] };
          refs = [| { Rig.Submission.at = 0; slot = 0 } |];
          groups = (Fixed 1, Fixed 1, Fixed 1);
          threads = (Fixed 1, Fixed 1, Fixed 1);
          shared = Fixed 0;
        };
  }

(* Memory 0's address lands in word 0 of input 0, through a plain hole; the same
   through [h], in its word 1; and [h] over it in memory 1's init, then memory
   0's handle, both moved into input 1. *)
let hole_law h =
  let d = fst (Lazy.force devices) in
  let address = G.Address { memory = 0; on = 0 } in
  let init =
    {
      G.bytes = String.make 16 '\000';
      holes = [| shaped h 0 address; hole 8 (G.Handle 0) |];
    }
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [||];
      ints = 0;
      memory = [| alloc 0 8; alloc ~init 0 16 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 16 }; { device = 0; bytes = 16 } |];
      steps =
        [|
          submit 0 ~writes:[| Input 0 |]
            [| fill_word 0 (hole 8 (G.Leaf address)) |];
          submit 0 ~writes:[| Input 0 |]
            [| fill_word 1 (shaped h 8 (G.Leaf address)) |];
          Move
            {
              src = Memory { memory = 1; offset = 0; length = 16 };
              dst = Input 1;
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = B.create d 16 and y = B.create d 16 in
  ignore (G.run p { inputs = [| x; y |]; ints = [||] });
  let x = read x and y = read y in
  let a = word x 0 in
  let w = (a + h.add) lsr h.shift in
  let low32 = w land 0xffff_ffff in
  equal ~msg:"a launch's hole" int
    (match h.width with W32 -> low32 | W64 -> w)
    (word x 1);
  let memory_word =
    match h.width with
    | W32 -> low32 (* the init's other 4 bytes stay 0 *)
    | W64 -> w
  in
  equal ~msg:"a memory's hole" int memory_word (word y 0);
  equal ~msg:"Polled's handle is the address" int a (word y 1);
  cover "a shift past the address's bits" (h.shift > 40);
  cover "a W32 hole" (h.width = W32)

(* Two copies *)

let one_device_fill ?(copies = G.One) d =
  {
    G.devices = [| Rig.arch d |];
    code = [||];
    ints = 0;
    memory = [| alloc ~copies 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    steps = [| submit 0 ~writes:[| all 0 |] [| fill ~image:0 ~groups:1 7 |] |];
  }

let test_two_waits () =
  let d, pd = polled "two" in
  let p = load_ok (one_device_fill ~copies:Two d) [ d ] in
  ignore (P.launches pd);
  let empty = { G.inputs = [||]; ints = [||] } in
  ignore (G.run p empty);
  ignore (G.run p empty);
  equal ~msg:"runs 0 and 1 handed over, neither run" int 0
    (List.length (P.launches pd));
  ignore (G.run p empty);
  greater ~msg:"run 2 waited for run 0's work on its copy" int ~than:0
    (List.length (P.launches pd));
  Rig.wait d (Rig.submitted d)

let test_one_waits_not () =
  let d, pd = polled "one" in
  let p = load_ok (one_device_fill d) [ d ] in
  ignore (P.launches pd);
  let empty = { G.inputs = [||]; ints = [||] } in
  for _ = 1 to 3 do
    ignore (G.run p empty)
  done;
  equal ~msg:"memory of one copy: runs wait only by its stamps, in the queue"
    int 0
    (List.length (P.launches pd));
  Rig.wait d (Rig.submitted d)

(* After *)

let test_after () =
  let a, _ = polled "after-a" and b, _ = polled "after-b" in
  let pa = load_ok (one_device_fill a) [ a ] in
  let pb = load_ok (one_device_fill b) [ b ] in
  let empty = { G.inputs = [||]; ints = [||] } in
  let pt = (G.run pa empty).(0) in
  equal ~msg:"a's run not reached" bool true
    (Rig.signaled a < Rig.Point.value pt);
  ignore (G.run ~after:[| pt |] pb empty);
  greater ~msg:"b's run followed it" int
    ~than:(Rig.Point.value pt - 1)
    (Rig.signaled a);
  Rig.wait b (Rig.submitted b)

(* Refusals *)

let base d =
  {
    G.devices = [| Rig.arch d |];
    code = [||];
    ints = 0;
    memory = [| alloc 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 64 } |];
    steps = [| submit 0 ~writes:[| Input 0 |] [| fill ~image:0 ~groups:1 0 |] |];
  }

let with_params bytes (t : G.t) =
  {
    t with
    steps =
      [|
        submit 0 ~writes:[| Input 0 |]
          [|
            {
              G.queue = "COMPUTE:0";
              after = [||];
              work =
                Launch
                  {
                    image = 0;
                    kernel = "fill";
                    params = data bytes;
                    refs = [||];
                    groups = (Fixed 1, Fixed 1, Fixed 1);
                    threads = (Fixed 1, Fixed 1, Fixed 1);
                    shared = Fixed 0;
                  };
            };
          |];
      |];
  }

let addr memory = G.Address { memory; on = 0 }

let holed ?width ?shift n h =
  { G.bytes = String.make n '\000'; holes = [| hole ?width ?shift 0 h |] }

let refusals =
  let one d = [ d ] in
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
    ( "an init longer than its memory",
      (fun d ->
        {
          (base d) with
          memory = [| alloc ~init:(data (String.make 65 'x')) 0 64 |];
        }),
      one );
    ( "a hole past its bytes",
      (fun d ->
        { (base d) with memory = [| alloc ~init:(holed 4 (addr 0)) 0 64 |] }),
      one );
    ( "a shift of 63",
      (fun d ->
        {
          (base d) with
          memory = [| alloc ~init:(holed ~shift:63 8 (addr 0)) 0 64 |];
        }),
      one );
    ( "memory of two copies named in memory of one",
      (fun d ->
        {
          (base d) with
          memory =
            [| alloc ~copies:Two 0 8; alloc ~init:(holed 8 (addr 0)) 0 8 |];
        }),
      one );
    ( "memory of two copies named in an image",
      (fun d ->
        {
          (base d) with
          memory = [| alloc ~copies:Two 0 8 |];
          images = [| { device = 0; binary = holed 16 (addr 0) } |];
        }),
      one );
    ( "an entry of a later image",
      (fun d ->
        {
          (base d) with
          images =
            [|
              {
                device = 0;
                binary = holed 16 (G.Entry { image = 1; name = "fill" });
              };
              { device = 0; binary = functions };
            |];
        }),
      one );
    ( "a view past its memory",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0
                ~writes:[| Memory { memory = 0; offset = 60; length = 8 } |]
                [| fill ~image:0 ~groups:1 0 |];
            |];
        }),
      one );
    ( "an input's index",
      (fun d ->
        {
          (base d) with
          steps =
            [| submit 0 ~writes:[| Input 3 |] [| fill ~image:0 ~groups:1 0 |] |];
        }),
      one );
    ( "an image's index",
      (fun d ->
        {
          (base d) with
          steps =
            [| submit 0 ~writes:[| Input 0 |] [| fill ~image:5 ~groups:1 0 |] |];
        }),
      one );
    ( "parameters of no whole word",
      (fun d -> with_params (String.make 6 '\000') (base d)),
      one );
    ( "parameters over 4096 bytes",
      (fun d -> with_params (String.make 4100 '\000') (base d)),
      one );
    ( "an address on no device",
      (fun d ->
        {
          (base d) with
          memory =
            [| alloc ~init:(holed 8 (G.Address { memory = 0; on = 2 })) 0 64 |];
        }),
      one );
    ( "holes that share a byte",
      (fun d ->
        {
          (base d) with
          memory =
            [|
              alloc
                ~init:
                  {
                    bytes = String.make 16 '\000';
                    holes = [| hole 0 (addr 0); hole ~width:W32 4 (addr 0) |];
                  }
                0 64;
            |];
        }),
      one );
    ( "a launch on a queue that runs no launch",
      (fun d ->
        let f = fill ~image:0 ~groups:1 0 in
        {
          (base d) with
          steps =
            [|
              submit 0 ~writes:[| Input 0 |] [| { f with queue = "COPY:0" } |];
            |];
        }),
      one );
    ( "a grid a run refuses",
      (fun d ->
        {
          (base d) with
          steps =
            [|
              submit 0 ~writes:[| Input 0 |]
                [| fill ~image:0 ~groups:(1 lsl 33) 0 |];
            |];
        }),
      one );
    ( "an image its device refuses",
      (fun d ->
        {
          (base d) with
          images = [| { device = 0; binary = data "nonsense" } |];
        }),
      one );
    ( "a function its image lacks",
      (fun d ->
        {
          (base d) with
          memory =
            [|
              alloc ~init:(holed 8 (G.Entry { image = 0; name = "nope" })) 0 64;
            |];
        }),
      one );
  ]

let test_refusal (_, t, ds) =
  let d = fst (polled "refused") in
  is_error
    ~pp:(fun _ _ -> ())
    ~msg:"load answers Error"
    (G.load (t d) (Iarray.of_list (ds d)))

let test_unborrowable () =
  let a, _ = polled ~peers:false "lone-a"
  and b, _ = polled ~peers:false "lone-b" in
  let t =
    {
      G.devices = [| Rig.arch a; Rig.arch b |];
      code = [||];
      ints = 0;
      memory = [| alloc 1 64 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [||];
      steps = [| submit 0 ~writes:[| all 0 |] [| fill ~image:0 ~groups:1 0 |] |];
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

let test_frame_refusals () =
  let d = fst (polled "frame") and e = fst (polled "frame-other") in
  let p = load_ok (base d) [ d ] in
  let refused msg inputs =
    raises_match ~msg Exn.invalid_arg (fun () ->
        G.run p { inputs; ints = [||] })
  in
  refused "no input" [||];
  refused "two inputs" [| B.create d 64; B.create d 64 |];
  refused "an input of another device" [| B.create e 64 |];
  refused "an input of fewer bytes" [| B.create d 63 |];
  ignore (G.run p { inputs = [| B.create d 65 |]; ints = [||] });
  Rig.wait d (Rig.submitted d)

(* Host code and loops *)

(* test/host's fixtures: [affine] stores [a * in[i] + c] at [out[i]] for [i]
   below [n], over buffers [out], [in] and values [n], [a], [c]; [loop] calls
   the program at its value 0 as many times as its value 1, directly when its
   value 2 is [2], on its buffers and the values after its eighth. *)
let affine =
  {
    obj = Rig_host_support.fixture ~dir:"../host/fixtures" "affine";
    entry = "affine";
  }

let loop =
  {
    obj = Rig_host_support.fixture ~dir:"../host/fixtures" "loop";
    entry = "loop";
  }

(* [x.(0) <- a * x.(0) + c], for host code [code]'s [affine]. *)
let step_affine ?(code = 0) x ~a ~c =
  G.Host
    {
      code;
      buffers = [| (x, B.Read_write); (x, B.Read) |];
      values = [| Fixed 1; a; c |];
      split = None;
    }

let one_word d = of_bytes d (le64 0)
let word_of b = word (read b) 0

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
    {
      G.devices = [| Rig.arch d |];
      code = [| affine |];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 256 } |];
      steps =
        [|
          step_affine Ints ~a:(Fixed i.a) ~c:(Fixed i.c);
          G.Submit
            {
              device = 0;
              reads = [||];
              writes = [| Input 0 |];
              fixed = [||];
              parts =
                [|
                  (let f = fill ~image:0 ~groups:1 100 in
                   match f.work with
                   | Launch l ->
                       {
                         f with
                         work =
                           Launch { l with groups = (Int 0, Fixed 1, Fixed 1) };
                       }
                   | _ -> f);
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let x = of_bytes d (String.make 256 '\000') in
  ignore (G.run p { inputs = [| x |]; ints = [| i.n |] });
  let m = (i.a * i.n) + i.c in
  equal ~msg:"a launch over the computed groups" string
    (String.concat "" (List.init m (fun j -> le64 (100 + j)))
    ^ String.make (256 - (8 * m)) '\000')
    (read x)

(* A loop of [k] trips: host code counts the trips in input 0, and a launch
   stores each trip's index in input 1. *)
let loop_law k =
  let d = fst (Lazy.force devices) in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [| affine |];
      ints = 2;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = Iarray.init 2 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          Loop
            {
              trips = Int 0;
              trip = Some 1;
              flag = None;
              body =
                [|
                  step_affine (Input 0) ~a:(Fixed 1) ~c:(Fixed 1);
                  submit 0 ~writes:[| Input 1 |]
                    [|
                      fill ~image:0 ~groups:1 0
                        ~holes:[| hole 8 (G.Int 1 : value) |];
                    |];
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let counter = one_word d and last = one_word d in
  ignore (G.run p { inputs = [| counter; last |]; ints = [| k |] });
  equal ~msg:"trips" int (max 0 k) (word_of counter);
  equal ~msg:"the last trip's index" int (max 0 (k - 1)) (word_of last)

(* A loop whose flag a device's step clears in its first trip runs once: the
   flag is read after that work. *)
let test_flag () =
  let d, _ = polled "flag" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [| affine |];
      ints = 0;
      memory = [| alloc ~init:(data (le64 1)) 0 8 |];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          Loop
            {
              trips = Fixed 10;
              trip = None;
              flag = Some { memory = 0; offset = 0; length = 8 };
              body =
                [|
                  submit 0
                    ~writes:[| Memory { memory = 0; offset = 0; length = 8 } |]
                    [| fill ~image:0 ~groups:1 0 |];
                  step_affine (Input 0) ~a:(Fixed 1) ~c:(Fixed 1);
                |];
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let counter = one_word d in
  ignore (G.run p { inputs = [| counter |]; ints = [||] });
  equal ~msg:"trips" int 1 (word_of counter)

(* Host code calls other host code through its address, a [Code] leaf. *)
let test_code_leaf () =
  let d = fst (Lazy.force devices) in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [| loop; affine |];
      ints = 0;
      memory = [||];
      images = [||];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          Host
            {
              code = 0;
              buffers = [| (Input 0, B.Read_write); (Input 0, B.Read) |];
              values =
                [|
                  Leaf (Code 1);
                  Fixed 3;
                  Fixed 2;
                  Fixed 0;
                  Fixed 0;
                  Fixed 0;
                  Fixed 0;
                  Fixed 3;
                  Fixed 1;
                  Fixed 1;
                  Fixed 1;
                |];
              split = None;
            };
        |];
    }
  in
  let p = load_ok t [ d ] in
  let counter = one_word d in
  ignore (G.run p { inputs = [| counter |]; ints = [||] });
  equal ~msg:"three calls of affine" int 3 (word_of counter)

let host_refusals =
  let one d = [ d ] in
  let with_steps ?(ints = 0) ?(code : _ iarray = [| affine |]) d steps =
    { (base d) with code; ints; steps }
  in
  [
    ( "host code that does not link",
      (fun d -> with_steps ~code:[| { obj = "nonsense"; entry = "f" } |] d [||]),
      one );
    ( "a host value of a rail this machine has not",
      (fun d ->
        with_steps d
          [| step_affine (Input 0) ~a:(Leaf (Ready 9)) ~c:(Fixed 0) |]),
      one );
    ( "a host value of a function its image lacks",
      (fun d ->
        with_steps d
          [|
            step_affine (Input 0)
              ~a:(Leaf (Entry { image = 0; name = "nope" }))
              ~c:(Fixed 0);
          |]),
      one );
    ( "a loop's trips of a rail this machine has not",
      (fun d ->
        with_steps d
          [|
            Loop
              {
                trips = Leaf (Ready_arg 9);
                trip = None;
                flag = None;
                body = [||];
              };
          |]),
      one );
    ( "an int past the ints",
      (fun d ->
        with_steps ~ints:1 d [| step_affine Ints ~a:(Int 1) ~c:(Fixed 0) |]),
      one );
    ( "a trip past the ints",
      (fun d ->
        with_steps ~ints:1 d
          [|
            Loop { trips = Fixed 1; trip = Some 1; flag = None; body = [||] };
          |]),
      one );
    ( "a flag of no byte",
      (fun d ->
        with_steps d
          [|
            Loop
              {
                trips = Fixed 1;
                trip = None;
                flag = Some { memory = 0; offset = 0; length = 0 };
                body = [||];
              };
          |]),
      one );
    ( "a split over one value",
      (fun d ->
        with_steps d
          [|
            Host
              {
                code = 0;
                buffers = [||];
                values = [| Fixed 0; Fixed 0 |];
                split = Some { extent = Fixed 1; blocks = 1; lo = 1; hi = 1 };
              };
          |]),
      one );
    ( "a code's index",
      (fun d ->
        with_steps d [| step_affine ~code:3 Ints ~a:(Fixed 1) ~c:(Fixed 0) |]),
      one );
  ]

(* Memory a host step names that the host does not address. *)
let test_host_unaddressed () =
  let d, _ =
    P.open_ ~host_visible:false
      (strf "program:hidden-%d" (Atomic.fetch_and_add opened 1))
  in
  let t =
    {
      (base d) with
      code = [| affine |];
      steps = [| step_affine (all 0) ~a:(Fixed 1) ~c:(Fixed 0) |];
    }
  in
  is_error ~pp:(fun _ _ -> ()) (G.load t [| d |])

(* A host step writes its input 0: a read-only borrow of a file there is refused
   before the code runs, and the file keeps its bytes. *)
let test_read_only_input () =
  let path = Filename.temp_file "rig-program" ".bin" in
  Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
  Out_channel.with_open_bin path (fun oc -> output_string oc (le64 5));
  let file = Result.get_ok (Rig_disk.of_file path) in
  let x = Option.get (B.borrow Rig.host file) in
  let t =
    {
      G.devices = [| Rig.arch Rig.host |];
      code = [| affine |];
      ints = 0;
      memory = [||];
      images = [||];
      inputs = [| { device = 0; bytes = 8 } |];
      steps = [| step_affine (Input 0) ~a:(Fixed 2) ~c:(Fixed 1) |];
    }
  in
  let p = load_ok t [ Rig.host ] in
  raises_match ~msg:"a read-only input" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the file's bytes" string (le64 5)
    (In_channel.with_open_bin path In_channel.input_all)

let test_ints_refusal () =
  let d = fst (polled "ints") in
  let p = load_ok { (base d) with ints = 1 } [ d ] in
  raises_match ~msg:"two ints for one" Exn.invalid_arg (fun () ->
      G.run p { inputs = [| B.create d 64 |]; ints = [| 1; 2 |] });
  Rig.wait d (Rig.submitted d)

(* Bytes *)

(* Descriptions of any shape, well formed or not: the format holds every value
   of the type. *)
let gen_t =
  let open Gen in
  let n = int_range (-2) 5000 in
  let few g = map Iarray.of_array (array ~size:(int_range 0 3) g) in
  let str = string_of ~size:(int_range 0 12) char in
  let leaf =
    one_of
      [
        (let+ memory = n and+ on = n in
         Address { memory; on });
        map (fun m -> Handle m) n;
        (let+ image = n and+ name = str in
         Entry { image; name });
        map (fun i -> Code i) n;
        map (fun r -> Ready r) n;
        map (fun r -> Ready_arg r) n;
      ]
  in
  let value =
    one_of
      [
        map (fun x -> Fixed x) int;
        map (fun i -> Int i) n;
        (let+ input = n and+ on = n in
         (Input { input; on } : value));
        map (fun l -> Leaf l) leaf;
      ]
  in
  let data f =
    let+ bytes = str
    and+ holes =
      few
        (let+ at = n
         and+ width = of_list [ W32; W64 ]
         and+ leaf = f
         and+ add = int
         and+ shift = n in
         { at; width; leaf; add; shift })
    in
    { bytes; holes }
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
  let access = of_list [ B.Read; B.Read_write ] in
  let triple = triple value value value in
  let work =
    one_of
      [
        map (fun v -> Words v) view;
        (let+ fill = leaf
         and+ arg = view
         and+ ring_units = n
         and+ segment_bytes = n in
         G.Fill { fill; arg; ring_units; segment_bytes });
        (let+ src = view and+ dst = view in
         (Copy { src; dst } : work));
        (let+ image = n
         and+ kernel = str
         and+ params = data value
         and+ refs =
           few
             (let+ at = n and+ slot = n in
              { Rig.Submission.at; slot })
         and+ groups = triple
         and+ threads = triple
         and+ shared = value in
         Launch { image; kernel; params; refs; groups; threads; shared });
      ]
  in
  let part =
    let+ queue = str and+ after = few n and+ work = work in
    { queue; after; work }
  in
  let rec step depth =
    let leaves =
      [
        (let+ device = n
         and+ parts = few part
         and+ reads = few slot
         and+ writes = few slot
         and+ fixed = few (pair view access) in
         Submit { device; parts; reads; writes; fixed });
        (let+ src = slot and+ dst = slot in
         G.Move { src; dst });
        (let+ code = n
         and+ buffers = few (pair slot access)
         and+ values = few value
         and+ split =
           option
             (let+ extent = value and+ blocks = n and+ lo = n and+ hi = n in
              { extent; blocks; lo; hi })
         in
         Host { code; buffers; values; split });
      ]
    in
    if depth = 0 then one_of leaves
    else
      one_of
        ((let+ trips = value
          and+ trip = option n
          and+ flag = option view
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
            and+ init = data leaf
            and+ copies = of_list [ One; Two ] in
            Alloc { device; kind; bytes; init; copies });
           (let+ rail = n and+ area = of_list [ Outbound; Inbound; Counts ] in
            Rail { rail; area });
         ])
  and+ images =
    few
      (let+ device = n and+ binary = data leaf in
       { device; binary })
  and+ code =
    few
      (let+ obj = str and+ entry = str in
       { obj; entry })
  and+ inputs =
    few
      (let+ device = n and+ bytes = n in
       { device; bytes })
  and+ ints = n
  and+ steps = few (step 2) in
  { devices; memory; images; code; inputs; ints; steps }

let pp_t ppf t =
  Format.fprintf ppf "a description of %d bytes, %d steps"
    (String.length (G.to_string t))
    (Iarray.length t.steps)

let gen_t = Gen.with_pp pp_t gen_t
let description = Testable.make ~pp:pp_t ~equal:( = )

let round_trip t =
  equal (result description string) (Ok t) (G.of_string (G.to_string t))

(* A cut or a changed byte of a description's bytes decodes or answers [Error]:
   [of_string] raises nothing. *)
let damaged (t, cut, at, byte) =
  let s = G.to_string t in
  let n = String.length s in
  let cut = String.sub s 0 (cut mod (n + 1)) in
  let changed =
    String.mapi (fun i c -> if i = at mod n then Char.chr byte else c) s
  in
  ignore (G.of_string cut : (G.t, string) result);
  ignore (G.of_string changed : (G.t, string) result);
  cover "a cut inside" (String.length cut < n)

let gen_damage =
  let open Gen in
  let+ t = gen_t and+ cut = nat and+ at = nat and+ byte = int_range 0 255 in
  (t, cut, at, byte)

let test_version () =
  let s = Bytes.of_string (G.to_string (base (fst (polled "version")))) in
  Bytes.set_int64_le s 12 1L;
  let why = require_error (G.of_string (Bytes.to_string s)) in
  contains ~msg:"names the versions" ~sub:"version 1, not 2" why

(* Rails *)

(* test/program's fixture [ready] stores its value 2 at its buffer 0, then calls
   the function at its value 0 with its values 1 and 3: a rail's ready function,
   its argument and a count. *)
let ready =
  { obj = Rig_host_support.fixture ~dir:"fixtures" "ready"; entry = "ready" }

let area n = Bigarray.(Array1.create char c_layout n)

(* An end of a rail of this process whose ready function is the support's
   [ready]: a call stores its count at its argument, the word [stamp]. *)
let fake_end stamp =
  {
    Rig_remote_abi.outbound = area 512;
    inbound = area 512;
    counts = area 384;
    ready = ignore;
    ready_fn = Rig_support.ready;
    ready_arg = Nativeint.of_int (B.address stamp);
  }

let rail_program =
  {
    G.devices = [| Rig.arch Rig.host |];
    memory = [| Rail { rail = 7; area = Outbound } |];
    images = [||];
    code = [| ready |];
    inputs = [||];
    ints = 1;
    steps =
      [|
        Host
          {
            code = 0;
            buffers =
              [|
                (Memory { memory = 0; offset = 0; length = 8 }, B.Read_write);
              |];
            values = [| Leaf (Ready 7); Leaf (Ready_arg 7); Int 0; Fixed 1 |];
            split = None;
          };
      |];
  }

(* Host code writes a rail's outbound area and calls its ready function, which a
   [Ready] leaf names, with its argument. *)
let test_rail () =
  let stamp = B.of_string (le64 0) in
  let e = fake_end stamp in
  let rails id = if id = 7 then Some e else None in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load ~rails rail_program [| Rig.host |])
  in
  ignore (G.run p { inputs = [||]; ints = [| 42 |] });
  let outbound = String.init 8 (Bigarray.Array1.get e.outbound) in
  equal ~msg:"the outbound area" int 42 (word outbound 0);
  equal ~msg:"the ready function's count" int 1 (word (read stamp) 0)

let test_no_rail () =
  let why = require_error (G.load rail_program [| Rig.host |]) in
  contains ~msg:"names the rail" ~sub:"rail 7" why

(* Constants beside holes *)

type beside = {
  bytes : int64;  (** The word the description holds. *)
  value : int;
  width : G.width;
  shift : int;
  per_run : bool;  (** The value is the run's int, else fixed. *)
}

let gen_beside =
  let open Gen in
  let+ width = of_list [ G.W32; G.W64 ]
  and+ shift = int_range 0 62
  and+ value = int_range 0 max_int
  and+ raw = int64
  and+ clash = bool
  and+ per_run = bool in
  let field =
    let w = Int64.of_int (value lsr shift) in
    match width with W32 -> Int64.logand w 0xffff_ffffL | W64 -> w
  in
  let raw =
    match width with W32 -> Int64.logand raw 0xffff_ffffL | W64 -> raw
  in
  let bytes = if clash then raw else Int64.logand raw (Int64.lognot field) in
  { bytes; value; width; shift; per_run }

let gen_beside =
  Gen.with_pp
    (fun ppf b ->
      Format.fprintf ppf "bytes %Lx, value %x >> %d, %s, %s" b.bytes b.value
        b.shift
        (match b.width with W32 -> "W32" | W64 -> "W64")
        (if b.per_run then "per run" else "fixed"))
    gen_beside

(* A launch of [main] whose 8 parameter bytes hold [b.bytes] with a hole over
   [b.value]: a value that meets their set bits is refused, at load where it is
   fixed and at run where it is the run's; otherwise the launch reads the bytes
   with the value's bits ORed in, every other bit kept. *)
let beside_law b =
  let d, pd = polled "beside" in
  let leaf : value = if b.per_run then Int 0 else Fixed b.value in
  let bytes = Bytes.create 8 in
  Bytes.set_int64_le bytes 0 b.bytes;
  let t =
    {
      G.devices = [| Rig.arch d |];
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      code = [||];
      inputs = [||];
      ints = 1;
      steps =
        [|
          submit 0
            [|
              {
                G.queue = "COMPUTE:0";
                after = [||];
                work =
                  Launch
                    {
                      image = 0;
                      kernel = "main";
                      params =
                        {
                          bytes = Bytes.to_string bytes;
                          holes =
                            [|
                              {
                                G.at = 0;
                                width = b.width;
                                leaf;
                                add = 0;
                                shift = b.shift;
                              };
                            |];
                        };
                      refs = [||];
                      groups = (Fixed 1, Fixed 1, Fixed 1);
                      threads = (Fixed 1, Fixed 1, Fixed 1);
                      shared = Fixed 0;
                    };
              };
            |];
        |];
    }
  in
  let field =
    let w = Int64.of_int (b.value lsr b.shift) in
    match b.width with W32 -> Int64.logand w 0xffff_ffffL | W64 -> w
  in
  let clashes = Int64.logand b.bytes field <> 0L in
  cover "a value that meets set bits" clashes;
  cover "a value beside set bits" ((not clashes) && b.bytes <> 0L);
  cover "a value read per run" b.per_run;
  match (G.load t [| d |], clashes && not b.per_run) with
  | Error _, true -> ()
  | Ok _, true -> fail "load took a fixed value that meets set bits"
  | Error why, false -> failf "load: %s" why
  | Ok p, false ->
      ignore (P.launches pd);
      let frame = { G.inputs = [||]; ints = [| b.value |] } in
      if clashes then
        raises_match ~msg:"run refuses it" Exn.invalid_arg (fun () ->
            G.run p frame)
      else begin
        let pt = (G.run p frame).(0) in
        Rig.Point.wait pt;
        let l = List.hd (P.launches pd) in
        equal ~msg:"the bytes with the value ORed in" int64
          (Int64.logor b.bytes field)
          (String.get_int64_le l.params 0)
      end

(* Integers at their extremes *)

(* A description's integers may come from another process's bytes: each field
   whose sum with another bounds an access is drawn where a sum wraps. *)
type extreme =
  | Ints of int
  | Hole_at of int
  | View_at of int
  | View_length of int

let pp_extreme ppf = function
  | Ints n -> Format.fprintf ppf "ints %d" n
  | Hole_at n -> Format.fprintf ppf "a hole at %d" n
  | View_at n -> Format.fprintf ppf "a view at %d" n
  | View_length n -> Format.fprintf ppf "a view of %d bytes" n

let gen_extreme =
  let open Gen in
  let n =
    of_list
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
  in
  Gen.with_pp pp_extreme
    (one_of
       [
         map (fun n -> Ints n) (such_that (fun n -> n < 0 || n > max_int / 8) n);
         map (fun n -> Hole_at n) n;
         map (fun n -> View_at n) n;
         map (fun n -> View_length n) n;
       ])

(* Each answers [Error] at load: none loads, and none raises. *)
let extreme_law e =
  let d = fst (Lazy.force devices) in
  let t = base d in
  let view offset length =
    submit 0
      ~writes:[| Memory { memory = 0; offset; length } |]
      [| fill ~image:0 ~groups:1 0 |]
  in
  let t =
    match e with
    | Ints n -> { t with ints = n }
    | Hole_at n ->
        {
          t with
          memory =
            [|
              alloc
                ~init:
                  {
                    bytes = String.make 8 '\000';
                    holes = [| hole n (addr 0) |];
                  }
                0 64;
            |];
        }
    | View_at n -> { t with steps = [| view n 8 |] }
    | View_length n -> { t with steps = [| view 8 n |] }
  in
  is_error ~pp:(fun _ _ -> ()) ~msg:"load answers Error" (G.load t [| d |])

(* The ints and the run's work *)

(* A loop of two trips stores its trip in int 0, and each trip's launch copies
   that word's low byte, read through an Ints slot, to byte [trip] of the input:
   the launch's offset is a hole over the trip. The gate holds the device's
   work, so trip 1's store waits for trip 0's work. *)
let test_trip_follows_work () =
  let d, pd = polled "trip-order" in
  let copy_byte =
    {
      G.queue = "COMPUTE:0";
      after = [||];
      work =
        Launch
          {
            image = 0;
            kernel = "copy";
            params =
              {
                bytes = le64 0 ^ le64 0 ^ le64 1;
                holes = [| hole 8 (Int 0 : value) |];
              };
            refs =
              [|
                { Rig.Submission.at = 0; slot = 0 };
                { Rig.Submission.at = 8; slot = 1 };
              |];
            groups = (Fixed 1, Fixed 1, Fixed 1);
            threads = (Fixed 1, Fixed 1, Fixed 1);
            shared = Fixed 0;
          };
    }
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [||];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          Loop
            {
              trips = Fixed 2;
              trip = Some 0;
              flag = None;
              body =
                [|
                  submit 0 ~reads:[| Ints |] ~writes:[| Input 0 |]
                    [| copy_byte |];
                |];
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

(* A launch writes int 0 through an Ints slot, and a later launch's hole over
   int 0 reads what it wrote. *)
let test_int_follows_writer () =
  let d, _ = polled "int-order" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [||];
      ints = 1;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = [| { device = 0; bytes = 8 } |];
      steps =
        [|
          submit 0 ~writes:[| Ints |] [| fill ~image:0 ~groups:1 42 |];
          submit 0 ~writes:[| Input 0 |]
            [| fill ~image:0 ~groups:1 0 ~holes:[| hole 8 (Int 0 : value) |] |];
        |];
    }
  in
  let p = load_ok t [ d ] in
  let out = one_word d in
  ignore (G.run p { inputs = [| out |]; ints = [| 0 |] });
  equal ~msg:"the word the first launch wrote" int 42 (word_of out)

(* What run reads of the caller's *)

(* run checks an input where a step uses it: a launch writes input 0, a move
   copies it into input 1 and waits for the launch at the gate, while another
   domain puts a buffer of no bytes in the frame as input 2. The launch that
   writes input 2 is refused, and the buffer the frame held keeps its word. *)
let test_input_checked_at_use () =
  let d, pd = polled "checked-at-use" in
  let t =
    {
      G.devices = [| Rig.arch d |];
      code = [||];
      ints = 0;
      memory = [||];
      images = [| { device = 0; binary = functions } |];
      inputs = Iarray.init 3 (fun _ -> { G.device = 0; bytes = 8 });
      steps =
        [|
          submit 0 ~writes:[| Input 0 |] [| fill ~image:0 ~groups:1 42 |];
          G.Move { src = Input 0; dst = Input 1 };
          submit 0 ~writes:[| Input 2 |] [| fill ~image:0 ~groups:1 7 |];
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
  raises_match ~msg:"input 2 of no bytes" Exn.invalid_arg (fun () ->
      G.run p frame);
  Domain.join changer;
  equal ~msg:"the move ran" int 42 (word_of frame.inputs.(1));
  equal ~msg:"the frame's earlier input 2" int 0 (word_of z)

let tests =
  [
    group ~timeout "steps"
      [
        prop
          "a run leaves the bytes of its steps run in order, on any device, \
           memory of two copies alternating by run, and answers one point per \
           device that ran"
          gen_case model_law;
      ];
    group ~timeout "holes"
      [
        prop
          "a hole holds the low bits of its leaf plus add, shifted, in memory \
           and in a launch"
          gen_hole hole_law;
        prop
          "a value that meets set bits of its bytes is refused, and the bits \
           outside each value keep the description's bytes"
          gen_beside beside_law;
      ];
    group ~timeout "bytes"
      [
        prop "of_string reads back what to_string wrote" gen_t round_trip;
        prop "of_string answers a cut or a changed byte without raising"
          gen_damage damaged;
        test "of_string names another version" test_version;
      ];
    group ~timeout "host code"
      [
        prop
          "host code computes ints from the frame's, which a launch's geometry \
           reads"
          gen_ints ints_law;
        prop
          "a loop runs its body as many times as its trips, none for a \
           negative count, each trip's index in the ints"
          (Gen.int_range (-2) 6) loop_law;
        test "a loop stops at a flag a device cleared" test_flag;
        test "host code calls host code a Code leaf names" test_code_leaf;
      ];
    group ~timeout "rails"
      [
        test "host code fills a rail's area and calls its ready function"
          test_rail;
        test "load answers Error for a rail this machine has not" test_no_rail;
      ];
    group ~timeout "order"
      [
        test "run n of memory of two copies follows run n - 2 on its copy"
          test_two_waits;
        test "memory of one copy waits for nothing on the host"
          test_one_waits_not;
        test "a run's first submission follows its after points" test_after;
        test "a loop's trip is stored once the work that reads the ints is done"
          test_trip_follows_work;
        test "an Int is read once the work that writes the ints is done"
          test_int_follows_writer;
      ];
    group ~timeout "refusals"
      [
        prop "load answers Error for integers whose sums would wrap" gen_extreme
          extreme_law;
        cases
          ~name:(fun (n, _, _) -> n)
          "load answers Error for" (refusals @ host_refusals) test_refusal;
        test "load answers Error for memory a device cannot borrow"
          test_unborrowable;
        test
          "load answers Error for host code over memory the host does not \
           address"
          test_host_unaddressed;
        test "load raises for devices of several machines" test_machines;
        test "run raises for a frame that does not fit" test_frame_refusals;
        test "run raises for more ints than the program's" test_ints_refusal;
        test "run raises for a read-only input a step writes"
          test_read_only_input;
        test "run checks an input where a step uses it"
          test_input_checked_at_use;
      ];
  ]

let () = exit (Windtrap.run "rig.program" tests)
