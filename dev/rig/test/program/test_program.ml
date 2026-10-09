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
let fill ?(holes = [||]) ~image ~groups base =
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

let submit ?(reads = [||]) ?(writes = [||]) device parts =
  G.Submit { device; parts; reads; writes; fixed = [||] }

let alloc ?(copies = G.One) ?(init = data "") device bytes =
  G.Alloc { device; kind = B.Device; bytes; init; copies }

let all memory = G.Memory { memory; offset = 0; length = 64 }

let load_ok t ds =
  require_ok ~pp:Format.pp_print_string (G.load t (Array.of_list ds))

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
    memory =
      Array.init 3 (fun m ->
          alloc
            ~copies:(if c.two.(m) then Two else One)
            ~init:(data c.init.(m))
            c.mem_dev.(m) size);
    images =
      [|
        { device = 0; binary = functions }; { device = 1; binary = functions };
      |];
    inputs =
      Array.map
        (fun device -> { G.device; bytes = size; access = B.Read_write })
        c.in_dev;
    steps = Array.of_list (List.map step c.ops);
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
    let points = G.run p { inputs = xs } in
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
      memory = [| alloc 0 8; alloc ~init 0 16 |];
      images = [| { device = 0; binary = functions } |];
      inputs =
        [|
          { device = 0; bytes = 16; access = B.Read_write };
          { device = 0; bytes = 16; access = B.Read_write };
        |];
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
  ignore (G.run p { inputs = [| x; y |] });
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
    memory = [| alloc ~copies 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [||];
    steps = [| submit 0 ~writes:[| all 0 |] [| fill ~image:0 ~groups:1 7 |] |];
  }

let test_two_waits () =
  let d, pd = polled "two" in
  let p = load_ok (one_device_fill ~copies:Two d) [ d ] in
  ignore (P.launches pd);
  let empty = { G.inputs = [||] } in
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
  let empty = { G.inputs = [||] } in
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
  let empty = { G.inputs = [||] } in
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
    memory = [| alloc 0 64 |];
    images = [| { device = 0; binary = functions } |];
    inputs = [| { device = 0; bytes = 64; access = B.Read_write } |];
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
    (G.load (t d) (Array.of_list (ds d)))

let test_unborrowable () =
  let a, _ = polled ~peers:false "lone-a"
  and b, _ = polled ~peers:false "lone-b" in
  let t =
    {
      G.devices = [| Rig.arch a; Rig.arch b |];
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
    raises_match ~msg Exn.invalid_arg (fun () -> G.run p { inputs })
  in
  refused "no input" [||];
  refused "two inputs" [| B.create d 64; B.create d 64 |];
  refused "an input of another device" [| B.create e 64 |];
  refused "an input of fewer bytes" [| B.create d 63 |];
  ignore (G.run p { inputs = [| B.create d 65 |] });
  Rig.wait d (Rig.submitted d)

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
      ];
    group ~timeout "order"
      [
        test "run n of memory of two copies follows run n - 2 on its copy"
          test_two_waits;
        test "memory of one copy waits for nothing on the host"
          test_one_waits_not;
        test "a run's first submission follows its after points" test_after;
      ];
    group ~timeout "refusals"
      [
        cases
          ~name:(fun (n, _, _) -> n)
          "load answers Error for" refusals test_refusal;
        test "load answers Error for memory a device cannot borrow"
          test_unborrowable;
        test "load raises for devices of several machines" test_machines;
        test "run raises for a frame that does not fit" test_frame_refusals;
      ];
  ]

let () = exit (Windtrap.run "rig.program" tests)
