(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver's work runs as programs hand it over, through rig. *)

open Windtrap
module N = Rig_nv
module A = Rig_nv_abi
module B = Rig.Buffer
module S = Rig_nv_support
module H = Rig_gpu_support.Host

let strf = Printf.sprintf

(* The code of the function an entry answered, if it answered one. *)
let code_of : Rig_edge.entry_answer -> int option = function
  | Entry e -> Some e.code
  | Absent | No_memory _ -> None

let second = 1_000_000_000
let host = S.host
let address = S.address
let alloc g kind n = require_some (N.alloc g kind n)
let still = Rig_gpu_support.still

(* The [n] unsigned 32-bit words at [a]. *)
let words32 a n = List.init n (fun i -> H.get32 (a + (4 * i)))

(* [f ()], the host word [w] set to [x] after it whether it returns or raises,
   so that work held on [w] ends. *)
let releasing w x f = Fun.protect ~finally:(fun () -> H.set64 (host w) x) f

(* The entry of a segment that writes [x] at [a] once the channel's earlier work
   completed, and of one that waits until the word at [a] is at least [x]. *)
let release l a x = S.segment l (A.Method.release (Int a) (Int x))
let acquire l a x = S.segment l (A.Method.acquire (Int a) (Int x))

(* A kernel that copies [n] bytes from the address [src] to [dst], starting [ns]
   nanoseconds late. *)
let copy_after l k ?(ns = 0) ~dst ~src n =
  S.launch l k "copy_after" ~blocks:1 [ ns; dst; src; n ]

(* Driver devices the tests stop themselves *)

(* The GPU in rig under a name of its own, for the tests of the driver's own
   stop: rig hands it work, and the test stops its handle itself. *)
let handed =
  let n = Atomic.make 0 in
  fun () -> S.handed (strf "NV:own-%d" (Atomic.fetch_and_add n 1))

(* GPUs *)

let gpus =
  group ~timeout:10. "GPUs"
    [
      test "an NVIDIA display controller is a GPU" (fun () ->
          equal (list bool)
            [ true; true; false; false ]
            (List.map
               (fun (vendor, class_) -> N.is_gpu ~vendor ~class_)
               [
                 (0x10de, 0x030000);
                 (0x10de, 0x030200);
                 (0x10de, 0x040300);
                 (0x1002, 0x030000);
               ]));
    ]

(* A path's locks *)

(* What a take of [l] answers: ["refused"] for the [Invalid_argument] of a
   take inside a path's lock, ["taken"] once it returns. *)
let take l =
  match N.Lock.protect l ignore with
  | () -> "taken"
  | exception Invalid_argument why
    when String.starts_with
           ~prefix:"Rig_nv: a lock taken inside one of its rank or a later one"
           why ->
      "refused"

(* A lock counts among its thread's locks while it is held. Its holder's
   take of it or of another path's lock raises having taken nothing, and its
   holder's try of either answers [false]; it is given back as its function
   returns or raises. *)
let test_lock () =
  let l = N.Lock.create () and other = N.Lock.create () in
  let before = Rig_support.holds () in
  let held, again, nested, tried, tried_other =
    N.Lock.protect l (fun () ->
        let held = Rig_support.holds () - before in
        let again = take l and nested = take other in
        let tried = N.Lock.try_protect l (fun () -> fail "ran") in
        let tried_other = N.Lock.try_protect other (fun () -> fail "ran") in
        (held, again, nested, tried, tried_other))
  in
  equal ~msg:"locks held inside" int 1 held;
  equal ~msg:"the holder's take" string "refused" again;
  equal ~msg:"the holder's take of another path's lock" string "refused"
    nested;
  equal ~msg:"the holder's try" bool false tried;
  equal ~msg:"the holder's try of another path's lock" bool false tried_other;
  equal ~msg:"the other lock outside" string "taken" (take other);
  equal ~msg:"locks held after" int 0 (Rig_support.holds () - before);
  raises Exit (fun () -> N.Lock.protect l (fun () -> raise Exit));
  equal ~msg:"locks held after a raise" int 0 (Rig_support.holds () - before);
  equal ~msg:"a take after" string "taken" (take l);
  equal ~msg:"a try after" bool true (N.Lock.try_protect l ignore)

(* A try answers [false], having run nothing, while another thread holds the
   lock. *)
let test_try_held () =
  let l = N.Lock.create () in
  let taken = Atomic.make false and release = Atomic.make false in
  let holder =
    Domain.spawn (fun () ->
        N.Lock.protect l (fun () ->
            Atomic.set taken true;
            while not (Atomic.get release) do
              Domain.cpu_relax ()
            done))
  in
  while not (Atomic.get taken) do
    Domain.cpu_relax ()
  done;
  let tried = N.Lock.try_protect l (fun () -> fail "ran") in
  Atomic.set release true;
  Domain.join holder;
  equal ~msg:"a try while another thread holds it" bool false tried;
  equal ~msg:"a try after" bool true (N.Lock.try_protect l ignore)

let locks =
  group ~timeout:10. "a path's locks"
    [
      test "count while held, refuse their holder's take, and go back"
        test_lock;
      test "answer a try while another thread holds them" test_try_held;
    ]

(* Parameters *)

let widths = [ 1; 2; 4; 8 ]
let bytes_of p = List.init (Bigarray.Array1.dim p) (fun i -> Char.code p.{i})

(* A field of 1, 2, 4 or 8 bytes in parameters of up to 32 bytes, its last
   possible place drawn at a fixed weight, and a value to write to it. *)
let field =
  let open Gen in
  let* n = of_list ~pp:Format.pp_print_int widths in
  let* size = int_range n 32 in
  let+ at = frequency [ (3, int_range 0 (size - n)); (1, constant (size - n)) ]
  and+ x = int in
  (size, (at, n), x)

let field_law (size, (at, n), x) =
  cover "the field ends the parameters" (at + n = size);
  cover "8 bytes" (n = 8);
  let p = N.params size in
  N.set p (at, n) x;
  let byte i =
    if i < at || i >= at + n then 0 else (x asr (8 * (i - at))) land 0xff
  in
  equal (list int) ~msg:"the bytes, little-endian" (List.init size byte)
    (bytes_of p);
  let unsigned = if n = 8 then x else x land ((1 lsl (8 * n)) - 1) in
  equal int ~msg:"the field read back" unsigned (N.get p (at, n))

let refused msg f = raises_match ~msg Exn.invalid_arg f

let parameters =
  group ~timeout:10. "parameters"
    [
      cases ~name:string_of_int "params n is n zero bytes" [ 0; 1; 17 ]
        (fun n ->
          equal (list int) (List.init n (fun _ -> 0)) (bytes_of (N.params n)));
      prop "a field reads back the low bytes written to it, little-endian"
        field field_law;
      cases ~name:string_of_int "a field of another width is refused"
        [ 0; 3; 5; 16; -1 ]
        (fun n ->
          let p = N.params 32 in
          refused "get" (fun () -> N.get p (0, n));
          refused "set" (fun () -> N.set p (0, n) 0));
      cases ~name:string_of_int "a field past the parameters is refused" widths
        (fun n ->
          let p = N.params 8 in
          List.iter
            (fun at ->
              refused "get" (fun () -> N.get p (at, n));
              refused "set" (fun () -> N.set p (at, n) 0))
            [ -1; 8 - n + 1 ]);
    ]

(* Facts *)

let digits s = String.for_all (function '0' .. '9' -> true | _ -> false) s

let facts () =
  S.with_driver @@ fun g ->
  let f = N.facts g in
  starts_with ~affix:"sm_" f.arch;
  equal bool
    ~msg:(strf "%s ends in digits" f.arch)
    true
    (String.length f.arch > 3
    && digits (String.sub f.arch 3 (String.length f.arch - 3)));
  greater int ~msg:"budget" ~than:0 f.budget;
  equal
    (list (pair string (list string)))
    [ ("COMPUTE:0", [ "Words"; "Launch" ]); ("COPY:0", [ "Words"; "Copy" ]) ]
    (List.map
       (fun (q : Rig_edge.queue) ->
         ( q.name,
           List.map
             (function
               | Rig_edge.Words -> "Words"
               | Fill -> "Fill"
               | Copy -> "Copy"
               | Launch -> "Launch")
             q.runs ))
       f.queues);
  equal (option int) ~msg:"the copy engine" (Some 1) f.copy_engine;
  equal int ~msg:"alignment" 4096 f.alignment;
  equal bool ~msg:"completion is the store" true (f.completion = Store);
  equal (list bool) ~msg:"waits on stores, hosts" [ true; true ]
    [ f.waits.stores; f.waits.hosts ];
  equal bool ~msg:"may block" false f.may_block;
  let w = N.locate f.word in
  equal nativeint ~msg:"the word's handle is its address"
    (Nativeint.of_int (Option.get w.address))
    w.handle;
  equal int ~msg:"the word starts at 0" 0 (H.get64 (Option.get w.host));
  equal int ~msg:"signaled reads the word" 0 (N.signaled g)

let capability () =
  S.with_driver @@ fun g ->
  let c = N.capability g in
  mem int ~msg:"compute class" c.compute_class [ 0xc7c0; 0xc9c0; 0xcec0 ];
  List.iter
    (fun (what, n) -> greater int ~msg:what ~than:0 n)
    [
      ("gpcs", c.gpcs);
      ("tpcs_per_gpc", c.tpcs_per_gpc);
      ("sms_per_tpc", c.sms_per_tpc);
      ("warps_per_sm", c.warps_per_sm);
    ];
  List.iter
    (fun (what, w) ->
      at_least int ~msg:what ~than:N.windows w;
      less int ~msg:what ~than:(1 lsl 40) w)
    [ ("shared window", c.shared_window); ("local window", c.local_window) ];
  equal (result unit string) ~msg:"local 0" (Ok ()) (c.local 0)

(* rig finds the record the device's facts declare, and none on the host. *)
let declared () =
  S.with_ @@ fun t ->
  (match Rig.capability t.d A.key with
  | Some c -> equal bool ~msg:"the record" true (c == N.capability t.g)
  | None -> fail "no record under the key");
  is_none ~msg:"the host" (Rig.capability Rig.host A.key)

(* The host addresses every kind but Device memory. The GPU addresses the
   kinds the host addresses below the windows, and Device memory outside
   them. *)
let kinds_of_memory () =
  S.with_driver @@ fun g ->
  List.iter
    (fun (name, kind) ->
      let r = alloc g kind 4096 in
      equal bool
        ~msg:(name ^ ": the host addresses it")
        (kind <> Device)
        (Option.is_some (N.locate r).host);
      let first = address r and last = address r + 4095 in
      if kind <> Device then
        less int ~msg:(name ^ ": its last byte") ~than:N.windows last
      else if last >= N.windows then
        at_least int ~msg:"Device: its first byte, its last past the windows"
          ~than:(1 lsl 40) first;
      N.free g r)
    [ ("Device", Device); ("Pinned", Pinned); ("Mapped", Mapped) ];
  if (N.facts g).maps_host then begin
    let p = H.pages H.page in
    let r = require_some ~msg:"map_host" (N.map_host g p H.page) in
    less int ~msg:"map_host: its last byte" ~than:N.windows
      (address r + H.page - 1);
    N.free g r;
    H.free_pages p H.page
  end

let facts =
  group ~timeout:60. "facts"
    [
      test "states a GPU's facts" facts;
      test "declares the GPU to compiled code" capability;
      test "rig finds its record" declared;
      test "gives memory of each kind where the host and the GPU reach it"
        kinds_of_memory;
    ]

(* Memory *)

(* A copy of more bytes than the copy engine moves at once, out to GPU memory at
   an odd offset and back. *)
let long_copy () =
  S.with_ @@ fun t ->
  let n = A.Method.max_copy + 4099 in
  let h, ha = S.shared t n in
  let d = B.view (B.create t.d (n + 1)) ~first:1 ~length:n in
  S.pattern ha n 3;
  S.run t [| S.copy ~dst:d h |];
  S.pattern ha n 4;
  S.run t [| S.copy ~dst:h d |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch ha n 3)

let refusal () =
  S.with_driver @@ fun g ->
  for _ = 1 to 100 do
    is_none ~msg:"twice the budget" (N.alloc g Device (2 * (N.facts g).budget))
  done;
  N.free g (require_some ~msg:"64 MiB after" (N.alloc g Device (64 lsl 20)))

(* Host memory no page backs is refused: map_host answers None. *)
let refused_host () =
  S.with_driver @@ fun g ->
  let p = H.pages H.page in
  H.free_pages p H.page;
  equal bool ~msg:"an unmapped page" true (Option.is_none (N.map_host g p 8));
  equal bool ~msg:"address 0" true (Option.is_none (N.map_host g 0 8))

(* So is a page of a file mapped shared, where Linux refuses the kernel
   driver's long-term pin for writing: on x86-64 from Linux 6.5 on, for a file
   whose file system tracks its dirty pages. The file lies beside the suite. *)
let kernel () =
  In_channel.with_open_text "/proc/sys/kernel/osrelease" input_line

let pins_refused () =
  Rig.arch Rig.host = "x86_64"
  && Scanf.sscanf (kernel ()) "%d.%d" (fun major minor ->
         major > 6 || (major = 6 && minor >= 5))

(* The type of the file system that holds [path]: that of the mount point in
   /proc/self/mounts that is its longest prefix. *)
let file_system path =
  let mounts =
    In_channel.with_open_text "/proc/self/mounts" In_channel.input_lines
  in
  let within (best, kind) l =
    match String.split_on_char ' ' l with
    | _ :: dir :: k :: _ when String.starts_with ~prefix:dir path ->
        if String.length dir > String.length best then (dir, k)
        else (best, kind)
    | _ -> (best, kind)
  in
  snd (List.fold_left within ("", "") mounts)

let refused_file () =
  S.with_driver @@ fun g ->
  if not (pins_refused ()) then
    skip ~reason:"Linux takes long-term pins of shared file mappings" ();
  if file_system (Sys.getcwd ()) = "tmpfs" then
    skip ~reason:"tmpfs tracks no dirty pages" ();
  let path = Filename.temp_file ~temp_dir:(Sys.getcwd ()) "rig-nv" ".map" in
  Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
  let fd = Unix.openfile path [ O_RDWR ] 0o600 in
  let file =
    Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
    Unix.map_file fd Bigarray.char Bigarray.c_layout true [| H.page |]
  in
  let a = H.data (Bigarray.array1_of_genarray file) in
  equal bool ~msg:"a file's shared mapping" true
    (Option.is_none (N.map_host g a 8));
  ignore (Sys.opaque_identity file)

(* Two host ranges that end in one page at different places both map, and
   each frees on its own: [0, 1.5) pages and [1.25, 3) pages. *)
let shared_page () =
  S.with_driver @@ fun g ->
  if (N.facts g).maps_host then begin
    let p = H.pages (3 * H.page) in
    let q = H.page / 4 in
    let a = require_some ~msg:"the first" (N.map_host g p (6 * q)) in
    let b =
      require_some ~msg:"the second, which shares a page"
        (N.map_host g (p + (5 * q)) (7 * q))
    in
    N.free g a;
    N.free g b;
    H.free_pages p (3 * H.page)
  end

(* Views a GPU's reopened device makes of memory the GPU already maps: each
   view keeps it mapped until its own free, whichever view goes first, and the
   memory's own region goes after them, as rig frees them. *)
let views_of_mapped () =
  let n = 4096 in
  let a = S.driver () in
  let src = alloc a Pinned n in
  S.pattern (host src) n 5;
  S.stop_driver a;
  S.with_ @@ fun t ->
  let view () =
    require_some ~msg:"a view of memory its GPU maps" (N.map_peer t.g a src)
  in
  let v = view () in
  let v' = view () in
  N.free t.g v;
  let k = S.kernels t in
  let l = S.launches t.g in
  let out = alloc t.g Pinned n in
  S.run t [| S.words (copy_after l k ~dst:(address out) ~src:(address v') n) |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch (host out) n 5);
  S.free_launches l;
  N.free t.g out;
  N.free t.g v';
  N.free a src

(* A stopped device's Mapped memory lies behind a BAR word that no viewer's
   submissions read any more, so nothing would order the host's writes
   through it before a viewer's work: no view of it is made. *)
let views_of_bar () =
  let a = S.driver () in
  let src = alloc a Mapped 4096 in
  S.stop_driver a;
  S.with_ @@ fun t ->
  is_none ~msg:"a view of a stopped device's Mapped memory"
    (N.map_peer t.g a src);
  N.free a src

let memory =
  group ~timeout:120. "memory"
    [
      test "a view of memory its GPU maps lasts until its own free"
        views_of_mapped;
      test "map_peer of a stopped device's Mapped memory is None"
        views_of_bar;
      test "a copy longer than the copy engine's is the identity" long_copy;
      test
        "an allocation past the GPU's memory is None, and gives back what it \
         took"
        refusal;
      test "map_host answers None for host memory no page backs" refused_host;
      test
        "map_host answers None for a file's shared mapping Linux will not pin"
        refused_file;
      test "map_host maps host ranges that share a page at different ends"
        shared_page;
    ]

(* Another function's DMA

   The GPU's own function stands for the other one, as every machine has it.
   Through NVIDIA's kernel driver the device exports a dma-buf. Its bytes are
   not checked here: the export leaves bAllowMmap off, so the host cannot map
   the dma-buf (nv-dmabuf.c), and no other API of this process imports one.
   Only a NIC that registers it can read them. Over PCI, a GPU taken
   physically gives bus ranges and stays taken until its process exits, so a
   child of the suite makes that export, and the GPU opens again once the
   child is gone. Behind an IOMMU that translates, which a device that maps
   host memory shows, no other function reaches the memory at a bus address:
   the export is refused, naming the GPU, and keeps nothing, so the GPU opens
   again in the process. *)

let dma_child = "--dma-over-pci"
let dma_sizes = [ 1; 4095; 4096; 4097; 1 lsl 22 ]
let small_bar = "memory BAR is smaller than its memory"

(* An [fn] that is no bus address, or no function of the machine, is refused
   naming it. *)
let refuses_strangers d =
  List.iter
    (fun fn ->
      match B.dma (B.create d 64) ~fn with
      | Ok _ -> failf "%s reaches the memory" fn
      | Error why -> contains ~msg:fn ~sub:fn why)
    [ "41:00.0"; "ffff:ff:1f.7" ]

let gpu_bus d =
  let fn = require_some ~msg:"the device's bus" (Rig.bus d) in
  equal ~msg:"GPU 0's function" (option string)
    (List.nth_opt (require_ok (Rig_nv_pci.buses ())) 0)
    (Some fn);
  fn

let exports_dmabuf () =
  S.with_ @@ fun { d; _ } ->
  let fn = gpu_bus d in
  List.iter
    (fun n ->
      match B.dma (B.create d n) ~fn with
      | Error why when Rig_gpu_support.has ~sub:"open kernel modules" why ->
          skip ~reason:why ()
      | Error why -> failf "%d bytes: %s" n why
      | Ok (Bus _) -> fail "the kernel driver's path gives a dma-buf"
      | Ok (Dmabuf { fd; offset }) -> (
          Fun.protect ~finally:(fun () -> Rig_gpu_support.Dmabuf.close fd)
          @@ fun () ->
          equal ~msg:"close-on-exec" bool true
            (Rig_gpu_support.Dmabuf.cloexec fd);
          equal ~msg:"the region's offset" int 0 offset;
          match Rig_gpu_support.Dmabuf.size fd with
          | Some bytes -> at_least ~msg:"its bytes" int ~than:n bytes
          | None -> fail "the descriptor is no dma-buf"))
    dma_sizes;
  refuses_strangers d

(* In the child: the bytes at the ranges are the buffer's, through BAR 1, the
   memory BAR, and a reopen of the GPU in the process is refused. The export is
   refused, and the test skips, where the memory BAR is smaller than the GPU's
   memory, as on a host that caps the BAR. *)
let exports_bus () =
  let t = S.open_ () in
  let fn = gpu_bus t.d in
  let n = 1 lsl 16 in
  let b = B.create t.d n in
  B.fill b 'x';
  (match B.dma b ~fn with
  | Error why when Rig_gpu_support.has ~sub:small_bar why ->
      S.close t;
      Rig_gpu_support.skipped why
  | Error why -> failf "refused: %s" why
  | Ok (Dmabuf _) -> fail "the path over PCI gives bus ranges"
  | Ok (Bus runs) ->
      satisfies ~msg:"the ranges" ~claim:"pages that hold the buffer"
        (list (pair int int))
        (fun runs ->
          List.for_all (fun (a, k) -> a mod 4096 = 0 && k mod 4096 = 0) runs
          && List.fold_left (fun s (_, k) -> s + k) 0 runs >= n)
        runs;
      equal ~msg:"the bytes through the BAR" string (String.make n 'x')
        (Rig_gpu_support.bar_bytes fn ~bar:1 runs n));
  refuses_strangers t.d;
  S.close t;
  match Rig.open_ (S.opener 0) with
  | Ok d ->
      Rig.close d;
      fail "the GPU opened again in the process"
  | Error why -> contains ~sub:"given to another PCI function" why

(* Whether the GPU is behind an IOMMU that translates; if it is, [dma] refuses
   and the export is checked here. *)
let refused_behind_iommu () =
  S.with_ @@ fun t ->
  if not (N.facts t.g).maps_host then false
  else begin
    let fn = gpu_bus t.d in
    let why = require_error (B.dma (B.create t.d (1 lsl 16)) ~fn) in
    contains ~msg:"names the GPU" ~sub:fn why;
    contains ~msg:"names the IOMMU" ~sub:"IOMMU" why;
    refuses_strangers t.d;
    true
  end

let exports () =
  match S.path () with
  | Some Pci ->
      if not (refused_behind_iommu ()) then Rig_gpu_support.child dma_child;
      S.with_ ignore
  | Some Kernel | None -> exports_dmabuf ()

let dma =
  group ~timeout:120. "another function's DMA"
    [
      test
        "dma exports a dma-buf through the kernel driver, or over PCI bus \
         ranges that keep the GPU until the process exits, refused behind an \
         IOMMU"
        exports;
    ]

(* Work *)

let launch () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1000 in
  let out = alloc t.g Pinned (4 * n) in
  S.run t
    [| S.words (S.launch l k "double_index" ~blocks:4 [ address out; n ]) |];
  equal (list int) (List.init n (fun i -> 2 * i)) (words32 (host out) n);
  S.free_launches l;
  N.free t.g out

let misuse () =
  S.with_driver @@ fun g ->
  let raises name f = raises_match ~msg:name Exn.invalid_arg f in
  raises "alloc of 0 bytes" (fun () -> N.alloc g Device 0);
  raises "map_host of 0 bytes" (fun () -> N.map_host g (host (S.word g)) 0)

(* A wait on another device's word in host memory wherever the process placed
   it, such as above 2^40, the widest address a channel's semaphore names: a
   Polled device's. *)
let high_word () =
  S.with_ @@ fun t ->
  let module P = Rig_support.Polled in
  let pd, p = P.open_ "POLLED:high" in
  Fun.protect ~finally:(fun () -> Rig.close pd) @@ fun () ->
  let word = require_some (P.locate (P.facts p).word).host in
  at_least int ~msg:"the word's host address" ~than:(1 lsl 40) word;
  let empty = Rig.Submission.make pd [||] in
  let run = Rig.Submission.Run.make empty in
  let point = Rig.submit run ~buffers:[||] in
  let b = alloc t.g Pinned 16 in
  H.set64 (host b) 0;
  let l = S.launches t.g in
  let s = Rig.Submission.make t.d [| S.words (release l (address b) 9) |] in
  S.watchdog "a wait on a high host word" (fun () ->
      let v =
        Rig.submit (Rig.Submission.Run.make s) ~waits:[| point |] ~buffers:[||]
      in
      still ~msg:"the word" int
        (Rig.Point.value v - 1)
        (fun () -> N.signaled t.g)
        ~ms:20;
      ignore (P.run p);
      Rig.Point.wait v);
  equal int ~msg:"released" 9 (H.get64 (host b));
  S.free_launches l;
  N.free t.g b

(* A submission waits in its channels on every producer it follows, however
   many: 300 Polled devices' values, more than a fixed reserve of waits held,
   none of them reached when the submit returns. *)
let many_waits () =
  S.with_ @@ fun t ->
  let module P = Rig_support.Polled in
  let producers = List.init 300 (fun i -> P.open_ (strf "POLLED:wait-%d" i)) in
  let close () = List.iter (fun (d, _) -> Rig.close d) producers in
  Fun.protect ~finally:close @@ fun () ->
  let point (d, _) =
    let run = Rig.Submission.Run.make (Rig.Submission.make d [||]) in
    Rig.submit run ~buffers:[||]
  in
  let points = Array.of_list (List.map point producers) in
  let s = Rig.Submission.make t.d [||] in
  S.watchdog "a submission of 300 waits" (fun () ->
      let v =
        Rig.submit (Rig.Submission.Run.make s) ~waits:points ~buffers:[||]
      in
      let reached = List.filter (fun (_, p) -> P.queued p = 0) producers in
      equal int ~msg:"producers reached on the host" 0 (List.length reached);
      List.iter (fun (_, p) -> ignore (P.run p)) producers;
      Rig.Point.wait v)

(* A ring entry cut in half is refused when submitted, and no value is
   assigned. *)
let refused () =
  S.with_ @@ fun t ->
  let base = Rig.submitted t.d in
  raises_match ~msg:"one ring word"
    (Exn.invalid_arg ~substring:"never fit")
    (fun () -> S.submit t [| S.words [| 0 |] |]);
  equal int ~msg:"values assigned" base (Rig.submitted t.d)

let work =
  group ~timeout:60. "work"
    [
      test "a kernel scheduled from a ring entry computes" launch;
      test "an allocation or mapping of no bytes raises" misuse;
      test "a wait holds work on another device's host word above 2^40"
        high_word;
      test "a submission waits in its channels on 300 producers" many_waits;
      test "a submission of half a ring entry raises" refused;
    ]

(* Launches, through rig. The conformance suite states the laws every driver's
   launches keep; these are NV's. *)

module Sub = Rig.Submission
module Run = Rig.Submission.Run

let launch_image =
  Rig_gpu_support.loader (fun () -> S.fixture "launch_sm89.cubin")

let kernel_image =
  Rig_gpu_support.loader (fun () -> S.fixture "kernels_sm89.cubin")

let grid_image =
  Rig_gpu_support.loader (fun () -> S.fixture "grid_params_sm89.cubin")

(* A ref at [at] to [slot], which the work writes unless [use] says. *)
let refer ?(use = Sub.write) at slot = use slot ~at

(* [kernel] of [image], with [params] bytes of parameters. *)
let launch_part image kernel ~params refs =
  Sub.launch ~refs (require_some (Rig.Image.resolve image kernel)) ~params
let copy_part ~dst src = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))

(* The [n] 32-bit words of [b] from byte [at], copied to the host. *)
let words b ~at n =
  let h = B.create Rig.host (4 * n) in
  B.copy ~src:(B.view b ~first:at ~length:(4 * n)) ~dst:h;
  B.read h Bigarray.int32 (fun a ->
      Array.init n (fun i -> Int32.to_int a.{i} land 0xffff_ffff))

let submitted run ~buffers = Rig.submit run ~buffers

(* Stores [ids]'s parameters into the block [k] of [run]: its words from [out]
   [offset] bytes on hold [a + b * k]. *)
let ids_params run k ~offset ~a ~b =
  Run.int64 run k 0 (Int64.of_int offset);
  Run.int64 run k 8 (Int64.of_int a);
  Run.int32 run k 16 b;
  Run.float32 run k 20 0.

(* A launch of [kernel] (fixtures/launch.cu) over [groups] of [threads] with
   [shared] bytes of dynamic shared memory, on [d], into a buffer of its own:
   a run of its submission. *)
let launching ?(kernel = "ids") ?(params = 24) d ~groups:(gx, gy, gz)
    ~threads:(tx, ty, tz) ~shared =
  let image = launch_image d in
  let s =
    Sub.make d [| launch_part image kernel ~params [| refer 0 0 |] |]
  in
  let run = Run.make s in
  let b = Sub.block s 0 in
  Run.groups run b gx gy gz;
  Run.threads run b tx ty tz;
  Run.shared run b shared;
  run

(* The most dynamic shared memory a block of [rotate] takes on [g]. *)
let dynamic_shared g =
  let c =
    require_ok ~pp:Format.pp_print_string
      (A.Cubin.of_string (S.fixture "launch_sm89.cubin"))
  in
  let k = require_some (A.Cubin.kernel c "rotate") in
  A.Launch.dynamic_shared
    (require_ok ~pp:Format.pp_print_string (A.Launch.make (N.capability g) k))

(* Launches past NV's limits are refused before any value, and the device stays
   live. The image's load takes values of its own. *)
let refused_launches () =
  S.with_ @@ fun { d; g } ->
  let out = B.create d 4096 in
  ignore (launch_image d);
  let loaded = Rig.submitted d in
  let refuses msg ?kernel ?params ?(groups = (1, 1, 1)) ?(threads = (1, 1, 1))
      ?(shared = 0) () =
    let run = launching ?kernel ?params d ~groups ~threads ~shared in
    raises_match ~msg (Exn.invalid_arg ~substring:"never fit") (fun () ->
        submitted run ~buffers:[| out |])
  in
  (* More threads than the function has, which the setter refuses. *)
  List.iter
    (fun threads ->
      raises_match
        (Exn.invalid_arg ~substring:"threads per group")
        (fun () -> launching d ~groups:(1, 1, 1) ~threads ~shared:0))
    [ (1025, 1, 1); (1024, 2, 1) ];
  refuses "65536 groups along y" ~groups:(1, 65536, 1) ();
  refuses "65536 groups along z" ~groups:(1, 1, 65536) ();
  refuses "2^31 groups along x" ~groups:(1 lsl 31, 1, 1) ();
  refuses "65 threads along z" ~threads:(1, 1, 65) ();
  refuses "a byte of shared memory past the most, rounded to 128"
    ~kernel:"rotate" ~params:12
    ~shared:(dynamic_shared g + 1)
    ();
  equal int ~msg:"values assigned" loaded (Rig.submitted d);
  equal (option string) ~msg:"the device's loss" None (Rig.lost d);
  let run =
    launching d ~groups:(1, 1, 65535) ~threads:(1024, 1, 1) ~shared:0
  in
  ids_params run (Sub.block (Run.submission run) 0) ~offset:0 ~a:0 ~b:1;
  let out = B.create d (4 * 65535 * 1024) in
  Rig.Point.wait (submitted run ~buffers:[| out |])

(* A launch takes as much dynamic shared memory as a block may beside its
   own. *)
let most_shared_memory () =
  S.with_ @@ fun { d; g } ->
  let shared = dynamic_shared g in
  let run =
    launching ~kernel:"rotate" ~params:12 d ~groups:(2, 1, 1)
      ~threads:(256, 1, 1) ~shared
  in
  let b = Sub.block (Run.submission run) 0 in
  Run.int64 run b 0 0L;
  Run.int32 run b 8 5;
  let out = B.create d (4 * 512) in
  Rig.Point.wait (submitted run ~buffers:[| out |]);
  let value i g = (5 + (3 * i) + (7 * g)) land 0xffff_ffff in
  equal (array int) ~msg:"the words"
    (Array.init 512 (fun i ->
         let g = i / 256 in
         value ((g * 256) + ((i + 1) mod 256)) g))
    (words out ~at:0 512)

(* Functions with static shared memory run with no dynamic shared memory, alone
   and chained, over one group and over several. *)
let static_shared () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let value k i g = (k + (3 * i) + (7 * g)) land 0xffff_ffff in
  let expect ~groups ~threads k =
    Array.init (groups * threads) (fun i ->
        let g = i / threads in
        value k ((g * threads) + ((i + 1) mod threads)) g)
  in
  List.iter
    (fun (groups, threads) ->
      let n = groups * threads in
      let a = B.create d (4 * n) and b = B.create d (4 * n) in
      let turn slot = launch_part image "turn" ~params:12 [| refer 0 slot |] in
      let s =
        Sub.make d
          [| turn 0; turn 1 |]
      in
      let run = Run.make s in
      Array.iteri
        (fun i k ->
          let blk = Sub.block s i in
          Run.groups run blk groups 1 1;
          Run.threads run blk threads 1 1;
          Run.shared run blk 0;
          Run.int64 run blk 0 0L;
          Run.int32 run blk 8 k)
        [| 5; 9 |];
      Rig.Point.wait (submitted run ~buffers:[| a; b |]);
      let msg = strf "%d groups of %d" groups threads in
      equal (array int) ~msg (expect ~groups ~threads 5) (words a ~at:0 n);
      equal (array int) ~msg (expect ~groups ~threads 9) (words b ~at:0 n))
    [ (1, 1); (1, 32); (4, 256) ]

(* A function whose switch is a jump table in its constant bank 2 reads it: pick
   (fixtures/launch.cu) over one group of 32 threads, for each type of element.
   Bank 0 is not the only bank a launch binds. *)
let jump_table () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let n = 32 in
  let data = Bytes.create (8 * n) in
  for i = 0 to n - 1 do
    Bytes.set_int64_le data (8 * i)
      (Int64.of_int (i * 0x0123_4567_89ab lxor (-1 lsl 40)))
  done;
  let floats = Bytes.create (8 * n) and doubles = Bytes.create (8 * n) in
  for i = 0 to n - 1 do
    let x = (float_of_int i *. 1.5) -. 7. in
    Bytes.set_int32_le floats (4 * i) (Int32.bits_of_float x);
    Bytes.set_int64_le doubles (8 * i) (Int64.bits_of_float x)
  done;
  let u32 x = x land 0xffff_ffff in
  let integer b i = function
    | 0 -> Bytes.get_uint8 b i
    | 1 -> Bytes.get_uint16_le b (2 * i)
    | 2 -> Int32.to_int (Bytes.get_int32_le b (4 * i))
    | 3 -> Int64.to_int (Bytes.get_int64_le b (8 * i))
    | 4 -> Bytes.get_int8 b i
    | 5 -> Bytes.get_int16_le b (2 * i)
    | 6 -> Int32.to_int (Bytes.get_int32_le b (4 * i))
    | _ -> Int64.to_int (Bytes.get_int64_le b (8 * i))
  in
  let truncated x = Int64.to_int (Int64.of_float x) in
  List.iter
    (fun t ->
      let src, value =
        match t with
        | 8 ->
            ( floats,
              fun i ->
                truncated
                  (Int32.float_of_bits (Bytes.get_int32_le floats (4 * i))) )
        | 9 ->
            ( doubles,
              fun i ->
                truncated
                  (Int64.float_of_bits (Bytes.get_int64_le doubles (8 * i))) )
        | t -> (data, fun i -> integer data i t)
      in
      let input = B.create d (Bytes.length src) in
      B.copy ~src:(B.of_string (Bytes.to_string src)) ~dst:input;
      let out = B.create d (4 * n) in
      let s =
        Sub.make d
          [|
            launch_part image "pick" ~params:20
              [| refer 0 1; refer ~use:Sub.read 8 0 |];
          |]
      in
      let run = Run.make s in
      let b = Sub.block s 0 in
      Run.groups run b 1 1 1;
      Run.threads run b n 1 1;
      Run.int64 run b 0 0L;
      Run.int64 run b 8 0L;
      Run.int32 run b 16 t;
      Rig.Point.wait (submitted run ~buffers:[| input; out |]);
      equal (array int) ~msg:(strf "type %d" t)
        (Array.init n (fun i -> u32 (value i)))
        (words out ~at:0 n))
    (List.init 10 Fun.id)

(* A cached launch submitted with a warm run, its block stored anew each time
   through every setter, allocates nothing. *)
let no_allocation () =
  S.with_ @@ fun { d; _ } ->
  let run = launching d ~groups:(1, 1, 1) ~threads:(32, 1, 1) ~shared:0 in
  Run.release run;
  let buffers = [| B.create d 4096 |] in
  let b = Sub.block (Run.submission run) 0 in
  let once i =
    if not (Run.take run) then fail "the run is taken";
    Run.groups run b 1 1 1;
    Run.threads run b 32 1 1;
    Run.shared run b 0;
    Run.int64 run b 0 0L;
    Run.int64 run b 8 (Int64.of_int i);
    Run.int32 run b 16 1;
    Run.float32 run b 20 1.5;
    Run.float64 run b 16 2.5;
    Run.int32 run b 16 1;
    Sys.opaque_identity (submitted run ~buffers)
  in
  Rig.Point.wait (once 0);
  let last = ref (once 1) in
  let bytes =
    Rig_support.allocated_bytes (fun () ->
        for i = 2 to 100 do
          last := once i
        done)
  in
  Rig.Point.wait !last;
  equal int ~msg:"bytes of 99 launches" 0 bytes

(* Public launches and a private copy kernel share the compute channel: a
   copy of 256 KiB, from which the driver's kernel runs it. The edges their
   memory derives still put the copy between the launches. *)
let launch_copy_launch () =
  S.with_ @@ fun { d; _ } ->
  let image = launch_image d in
  let n = 64 * 1024 in
  let out = B.create d (4 * n) and mid = B.create d (4 * n) in
  let dst = B.create d (4 * n) in
  let s =
    Sub.make d
      [|
        launch_part image "ids" ~params:24 [| refer 0 0 |];
        copy_part ~dst:mid out;
        launch_part image "twice" ~params:20
          [| refer 0 2; refer ~use:Sub.read 8 1 |];
      |]
  in
  let run = Run.make s in
  let b0 = Sub.block s 0 and b2 = Sub.block s 2 in
  Run.groups run b0 (n / 256) 1 1;
  Run.threads run b0 256 1 1;
  ids_params run b0 ~offset:0 ~a:11 ~b:3;
  Run.groups run b2 (n / 256) 1 1;
  Run.threads run b2 256 1 1;
  Run.int64 run b2 0 0L;
  Run.int64 run b2 8 0L;
  Run.int32 run b2 16 1;
  Rig.Point.wait (submitted run ~buffers:[| out; mid; dst |]);
  equal (array int)
    (Array.init n (fun k -> (2 * (11 + (3 * k))) + 1))
    (words dst ~at:0 n)

(* Launches past the end of the compute channel's launch ring: each writes its
   own word, through a ref whose offset names it. *)
let launches_wrap () =
  S.with_ @@ fun { d; _ } ->
  let n = 4000 in
  let out = B.create d (4 * n) in
  let run = launching d ~groups:(1, 1, 1) ~threads:(1, 1, 1) ~shared:0 in
  Run.release run;
  let b = Sub.block (Run.submission run) 0 in
  let last = ref None in
  for i = 0 to n - 1 do
    if not (Run.take run) then fail "the run is taken";
    ids_params run b ~offset:(4 * i) ~a:i ~b:0;
    last := Some (submitted run ~buffers:[| out |])
  done;
  Option.iter Rig.Point.wait !last;
  equal (array int) (Array.init n Fun.id) (words out ~at:0 n)

(* A submission of [n] launches on [d], launch [i] writing [i] into the 32-bit
   word [i] of its buffer: a run of it. *)
let numbered d n =
  let image = launch_image d in
  let s =
    Sub.make d
      (Array.init n (fun _ ->
           launch_part image "ids" ~params:24 [| refer 0 0 |]))
  in
  let run = Run.make s in
  for i = 0 to n - 1 do
    let b = Sub.block s i in
    Run.groups run b 1 1 1;
    Run.threads run b 1 1 1;
    ids_params run b ~offset:(4 * i) ~a:i ~b:0
  done;
  run

(* [n] numbered launches submitted on [d], into a buffer of their own: their
   point, and the words they wrote, read once the point is reached. *)
let numbered_submit d n =
  let out = B.create d (4 * n) in
  let run = numbered d n in
  let p = submitted run ~buffers:[| out |] in
  let written () =
    Rig.Point.wait p;
    words out ~at:0 n
  in
  (p, written)

(* One submission of more launches than the launch and segment rings hold
   runs once they grew, as one value, each launch writing its word. *)
let many_launches n =
  S.with_ @@ fun { d; _ } ->
  ignore (launch_image d);
  let before = Rig.submitted d in
  let p, written = numbered_submit d n in
  equal int ~msg:"its value" (before + 1) (Rig.Point.value p);
  equal (array int) (Array.init n Fun.id) (written ())

(* While a growth runs, holding no turn, a submission of another domain that
   needs a growth too takes the device's next value. The first submission
   takes the one after, and every launch of both writes its word. *)
let growth_interleaved () =
  let held = Atomic.make None and growths = Atomic.make 0 in
  let module Held = struct
    include N

    let grow g =
      Atomic.incr growths;
      Option.iter (fun f -> f ()) (Atomic.exchange held None);
      N.grow g
  end in
  let t = S.open_as (module Held) in
  Fun.protect ~finally:(fun () -> S.close t) @@ fun () ->
  ignore (launch_image t.d);
  let before = Rig.submitted t.d in
  let other = ref None in
  let submits () = numbered_submit t.d 2_000 in
  Atomic.set held
    (Some (fun () -> other := Some (Domain.join (Domain.spawn submits))));
  let p, written = numbered_submit t.d 4_096 in
  let q, other_written = Option.get !other in
  equal int ~msg:"the other domain's value" (before + 1) (Rig.Point.value q);
  equal int ~msg:"its value" (before + 2) (Rig.Point.value p);
  equal (array int) ~msg:"its words" (Array.init 4_096 Fun.id) (written ());
  equal (array int) ~msg:"the other's words" (Array.init 2_000 Fun.id)
    (other_written ());
  equal int ~msg:"growths" 2 (Atomic.get growths)

(* The bytes of the GPU's mappings that are not reservations ([---p]). *)
let mapped_bytes () =
  List.fold_left
    (fun n line ->
      Scanf.sscanf line "%x-%x %s" (fun a b perms ->
          if perms = "---p" then n else n + (b - a)))
    0
    (Rig_nv_nvidia_support.mappings ())

(* The bytes [t]'s process maps for the GPU once [n] numbered launches ran
   after [sizes] did, each in a submission of its own, and its collected and
   cached memory went back. A driver allocation then runs, the device's
   first since the launches. *)
let mapped_after t sizes n =
  ignore (launch_image t.S.d);
  List.iter (fun k -> ignore (snd (numbered_submit t.d k) ())) (sizes @ [ n ]);
  for _ = 1 to 3 do
    Gc.full_major ();
    Rig.free_cache t.d
  done;
  N.free t.g (alloc t.g Pinned H.page);
  mapped_bytes ()

(* The rings a growth replaces go back once the device reached the work on
   them: a device whose rings grew in steps maps what one whose rings grew at
   once to the same size maps. Submissions the rings hold make nothing
   more. *)
let rings_given_back () =
  if S.path () = Some S.Pci then
    skip
      ~reason:
        "it counts mappings of NVIDIA's kernel driver's files; GPU 0 is over \
         PCI"
      ();
  let n = 16_384 and last = 8_192 in
  S.with_ (fun t -> ignore (mapped_after t [] 1 : int));
  let at_once = S.with_ (fun t -> mapped_after t [ last ] n) in
  S.with_ @@ fun t ->
  let stepped = mapped_after t [ 1_024; 2_048; 4_096; last ] n in
  equal int ~msg:"grown in steps" at_once stepped;
  equal int ~msg:"after three more" stepped (mapped_after t [ n; n; n ] n)

(* A function whose threads keep 2 KiB of local memory runs as a launch: its
   entry made the channel's local memory serve it. *)
let local_launch () =
  S.with_ @@ fun { d; _ } ->
  let n = 1024 in
  let out = B.create d (4 * n) in
  let s =
    Sub.make d
      [| launch_part (kernel_image d) "stack" ~params:12 [| refer 0 0 |] |]
  in
  let run = Run.make s in
  let b = Sub.block s 0 in
  Run.groups run b (n / 256) 1 1;
  Run.threads run b 256 1 1;
  Run.int64 run b 0 0L;
  Run.int32 run b 8 n;
  Rig.Point.wait (submitted run ~buffers:[| out |]);
  equal (array int)
    (Array.init n (fun i -> (512 * i) + 130816))
    (words out ~at:0 n)

(* Each kernel reads through a generic pointer into its own parameters. Two
   chained launches and fresh values on the next submission distinguish their
   parameter blocks from one another and from an earlier run. *)
let grid_parameters (groups, threads) =
  S.with_ @@ fun { d; _ } ->
  let image = grid_image d in
  let n = groups * threads in
  let buffers = Array.init 2 (fun _ -> B.create d (4 * n)) in
  let s =
    Sub.make d
      (Array.init 2 (fun i ->
           launch_part image "grid_params" ~params:24 [| refer 0 i |]))
  in
  let run = Run.make s in
  Run.release run;
  let values = [| 0; 0x7fff_ffff; 0x8000_0000; 0xffff_ffff |] in
  let value round part i = values.(i mod 4) lxor ((round * 7) + part) in
  for round = 0 to 2 do
    if not (Run.take run) then fail "the run is taken";
    Array.iteri
      (fun part _ ->
        let b = Sub.block s part in
        Run.groups run b groups 1 1;
        Run.threads run b threads 1 1;
        Run.int64 run b 0 0L;
        for i = 0 to 3 do
          Run.int32 run b (8 + (4 * i)) (value round part i)
        done)
      buffers;
    Rig.Point.wait (submitted run ~buffers);
    Array.iteri
      (fun part out ->
        equal (array int)
          ~msg:(strf "round %d, part %d" round part)
          (Array.init n (value round part))
          (words out ~at:0 n))
      buffers
  done

(* The release of the last launch of a chain follows every launch of the
   chain: the host sees it only once the earlier launches' writes are
   visible, which copy 4 KiB each after 2 ms and 1 ms. *)
let chain_delays = [ 2_000_000; 1_000_000 ]

let chain_release () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 4096 in
  let src = alloc t.g Pinned n and word = alloc t.g Pinned 8 in
  let dsts = List.map (fun _ -> alloc t.g Pinned n) chain_delays in
  S.pattern (host src) n 11;
  List.iter (fun d -> S.pattern (host d) n 12) dsts;
  H.set64 (host word) 0;
  let copies =
    List.map2
      (fun ns d -> ("copy_after", 1, [ ns; address d; address src; n ]))
      chain_delays dsts
  in
  let e =
    S.chained l k (copies @ [ ("empty", 1, []) ]) ~release:(address word, 7)
  in
  let t0 = Rig.Profile.now () in
  let p = S.submit t [| S.words e |] in
  S.watchdog "the chain's release" (fun () ->
      while H.get64 (host word) <> 7 do
        Domain.cpu_relax ()
      done);
  let waited = Rig.Profile.now () - t0 in
  List.iteri
    (fun i d ->
      equal int ~msg:(strf "copy %d's first byte that differs" i) (-1)
        (S.mismatch (host d) n 11))
    dsts;
  at_least int ~msg:"the ns until the release"
    ~than:(List.fold_left ( + ) 0 chain_delays)
    waited;
  Rig.Point.wait p;
  S.free_launches l;
  List.iter (N.free t.g) (src :: word :: dsts)

let launches =
  group ~timeout:120. "launches"
    [
      test "a chain's last release follows the chain's launches" chain_release;
      test "launches past NV's limits are refused" refused_launches;
      test "a launch takes the most dynamic shared memory" most_shared_memory;
      test "functions with static shared memory run alone and chained"
        static_shared;
      test "a function reads the jump table in its constant bank 2" jump_table;
      test "a cached launch allocates nothing" no_allocation;
      test "a launch after a copy after a launch reads the copy"
        launch_copy_launch;
      test "4,000 launches pass the launch ring's end" launches_wrap;
      cases ~name:(strf "%d launches")
        "a submission of launches runs whatever their number"
        [ 1; 1_000; 4_096; 16_384 ] many_launches;
      test "a submission of another domain runs while a growth runs"
        growth_interleaved;
      test "rings a growth replaced go back" rings_given_back;
      test "a function with local memory runs as a launch" local_launch;
      cases
        ~name:(fun (g, t) -> strf "%d groups of %d" g t)
        "a kernel addresses its own parameters"
        [ (1, 1); (3, 32); (5, 256) ]
        grid_parameters;
    ]

(* Room *)

(* 40,000 submissions, empty ones on COMPUTE:0 between one-byte copies on
   COPY:0, wrap both channels' rings and segments; every byte arrives. *)
let wraps () =
  S.with_ @@ fun t ->
  let n = 20_000 in
  let src, sa = S.shared t n in
  let dst, da = S.shared t n in
  S.pattern sa n 5;
  S.pattern da n 6;
  let byte b i = B.view b ~first:i ~length:1 in
  let base = Rig.submitted t.d and last = ref None in
  for i = 0 to n - 1 do
    ignore (S.submit t [||]);
    last := Some (S.submit t [| S.copy ~dst:(byte dst i) (byte src i) |])
  done;
  Option.iter Rig.Point.wait !last;
  equal int ~msg:"the word" (base + (2 * n)) (N.signaled t.g);
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch da n 5)

(* 40,000 copies of 16 bytes on COPY:0, each waited for, pass the end of the
   selected channel's segment and descriptor rings. *)
let sequential () =
  S.with_ @@ fun t ->
  let h, ha = S.shared t 16 in
  let a = B.create t.d 16 and b = B.create t.d 16 in
  S.pattern ha 16 7;
  S.run t [| S.copy ~dst:a h |];
  for i = 1 to 40_000 do
    if i mod 2 = 1 then S.run t [| S.copy ~dst:b a |]
    else S.run t [| S.copy ~dst:a b |]
  done;
  S.pattern ha 16 8;
  S.run t [| S.copy ~dst:h a |];
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch ha 16 7)

(* Copies switch channels within their logical queue; alternating last
   channels must release every copy and the dependent words in order. The
   copy channel's join word, which the copy engine writes as two 32-bit
   words, crosses its 32-bit carry within its first thousand joins. *)
let join_alternation () =
  S.with_ @@ fun t ->
  let l = S.launches t.g in
  let scratch = alloc t.g Pinned 8 in
  let segment = release l (address scratch) 1 in
  let bytes = 1 lsl 20 in
  let h, ha = S.shared t bytes in
  let a = B.create t.d bytes and b = B.create t.d bytes in
  let parts length =
    let h = B.view h ~first:0 ~length in
    let a = B.view a ~first:0 ~length in
    let b = B.view b ~first:0 ~length in
    [|
      [|
        S.copy ~dst:a h;
        S.copy ~dst:b a;
        S.words segment;
        S.copy ~dst:h b;
      |];
      [|
        S.copy ~dst:a h;
        S.copy ~dst:b a;
        S.copy ~dst:h b;
        S.words segment;
      |];
    |]
  in
  let small = parts 16 and large = parts bytes in
  (* The words follow the copies by memory they name: their release stands for
     a use of [h] and [b], which the copies write. *)
  let fixed = [ (h, Rig.Read); (b, Rig.Read) ] in
  S.pattern ha bytes 7;
  let n = 2_000 in
  S.watchdog "work across alternating joins" (fun () ->
      for i = 1 to n do
        (* Large copies exercise the private kernel without transferring a
           whole image at every value. *)
        let parts = if i > n - 16 then large else small in
        let s = Sub.make ~fixed t.d parts.(i land 1) in
        Rig.Point.wait (Rig.submit (Run.make s) ~buffers:[||])
      done);
  equal int ~msg:"the compute part's release" 1 (H.get64 (host scratch));
  equal int ~msg:"the copied bytes" (-1) (S.mismatch ha bytes 7);
  S.free_launches l;
  N.free t.g scratch

(* The tear probe's unordered rounds. *)
let tear_rounds = 20_000

(* The copy engine's release of a 64-bit value and the compute engine's,
   unordered on one word across its 32-bit carry: whichever lands last, the word
   holds one of the two values, never a half of each. rig places no words on the
   copy queue, so the releases go to the driver's C entries as words, one part
   on each queue. Both channels announce they are ready before the host opens
   their gates. Two ordered rounds first establish each complete value;
   unordered rounds may choose either winner. *)
let carry_tear () =
  S.with_driver @@ fun g ->
  let l = S.launches g in
  let w = alloc g Pinned 8 in
  let gates = Array.init 2 (fun _ -> alloc g Pinned 8) in
  let ready = Array.init 2 (fun _ -> alloc g Pinned 8) in
  let below = (1 lsl 32) - 1 and above = 1 lsl 32 in
  let gated i command =
    S.segment l
      (A.Template.concat
         [
           A.Method.release (Int (address ready.(i))) (Int 1);
           A.Method.acquire (Int (address gates.(i))) (Int 1);
           command;
         ])
  in
  let parts =
    [|
      (1, gated 0 (A.Method.copy_release (Int (address w)) (Int below)));
      (0, gated 1 (A.Method.release (Int (address w)) (Int above)));
    |]
  in
  let v = ref 0 in
  let wait_word r x =
    while H.get64 (host r) <> x do
      Domain.cpu_relax ()
    done
  in
  let words = (w :: Array.to_list gates) @ Array.to_list ready in
  let start () =
    List.iter (fun r -> H.set64 (host r) 0) words;
    incr v;
    S.Edge.submit g ~v:!v parts;
    Array.iter (fun r -> wait_word r 1) ready
  in
  let reached () =
    while N.signaled g < !v do
      Domain.cpu_relax ()
    done
  in
  let open_gate i = H.set64 (host gates.(i)) 1 in
  let torn = ref [] in
  S.watchdog "unordered releases across a carry" (fun () ->
      Array.iteri
        (fun first value ->
          start ();
          open_gate first;
          wait_word w value;
          open_gate (1 - first);
          reached ();
          equal int
            ~msg:(strf "engine %d releases last" (1 - first))
            (if first = 0 then above else below)
            (H.get64 (host w)))
        [| below; above |];
      for i = 1 to tear_rounds do
        start ();
        open_gate (i land 1);
        open_gate (1 - (i land 1));
        reached ();
        let x = H.get64 (host w) in
        if x <> below && x <> above then torn := x :: !torn
      done);
  equal (list int) ~msg:"torn values" [] !torn;
  S.free_launches l;
  List.iter (N.free g) words

(* Words parts of no words take no ring entry: after a part that writes a
   word, more of them than a channel's ring holds run as one value, and the
   first part's write lands. *)
let empty_words () =
  S.with_ @@ fun t ->
  let l = S.launches t.g and w = alloc t.g Pinned 8 in
  H.set64 (host w) 0;
  let none = Sub.words (B.view (B.create Rig.host 8) ~first:0 ~length:0) in
  let parts = Array.make 20_001 none in
  parts.(0) <- S.words (release l (address w) 1);
  let before = Rig.submitted t.d in
  let p = S.submit t parts in
  S.watchdog "20,000 empty Words parts" (fun () -> Rig.Point.wait p);
  equal int ~msg:"its value" (before + 1) (Rig.Point.value p);
  equal int ~msg:"the first part's word" 1 (H.get64 (host w));
  equal (option string) ~msg:"the device's loss" None (Rig.lost t.d);
  S.free_launches l;
  N.free t.g w

(* One submission of more parts than 16 bits number, alternating channels:
   launch [j] writes [j] into a word, which copy [j] then copies into word [j]
   of [out]. The edges their memory derives join each part to the one before
   it, which ran on the other channel, and the joins cross each join word's
   32-bit carry. *)
let joined_parts () =
  S.with_ @@ fun t ->
  let image = launch_image t.d in
  let n = 33_000 in
  let cell = B.create t.d 4 and out = B.create t.d (4 * n) in
  let part i =
    if i land 1 = 0 then launch_part image "ids" ~params:24 [| refer 0 0 |]
    else copy_part ~dst:(B.view out ~first:(4 * (i / 2)) ~length:4) cell
  in
  let s = Sub.make t.d (Array.init (2 * n) part) in
  let run = Run.make s in
  for j = 0 to n - 1 do
    let b = Sub.block s (2 * j) in
    Run.groups run b 1 1 1;
    Run.threads run b 1 1 1;
    ids_params run b ~offset:0 ~a:j ~b:0
  done;
  let p = submitted run ~buffers:[| cell |] in
  S.watchdog "66,000 joined parts" (fun () -> Rig.Point.wait p);
  equal (array int) (Array.init n Fun.id) (words out ~at:0 n)

let room =
  group ~timeout:300. "room"
    [
      test "20,000 Words parts of no words run as one value" empty_words;
      test "40,000 submissions wrap both channels and every copy arrives" wraps;
      test "40,000 copies one after the other pass the segment ring's end"
        sequential;
      test "work waiting on copies runs in order as their channels alternate"
        join_alternation;
      test "66,000 parts, each joined to the one before, run in order"
        joined_parts;
      test "two engines' releases across a 32-bit carry never tear" carry_tear;
    ]

(* Local memory *)

(* The 32-bit word i of the stack kernels' output. *)
let stacked i = (512 * i) + 130816

let local () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let out = alloc t.g Pinned (4 * n) in
  let compute () =
    H.write (host out) (String.make (4 * n) '\000');
    S.run t [| S.words (S.launch l k "stack" ~blocks:4 [ address out; n ]) |];
    S.reset l;
    equal (list int) (List.init n stacked) (words32 (host out) n)
  in
  compute ();
  let { A.Gpu.local; _ } = N.capability t.g in
  equal (result unit string) ~msg:"a smaller one" (Ok ()) (local 1024);
  compute ();
  S.free_launches l;
  N.free t.g out

(* Ring entries past the 262,144 of a channel's ring at open run once the
   device made its channels anew, as one value. Then a kernel with local
   memory still computes, and a copy, words and a copy, joined across the
   channels, run in order: the new channels get the device's setup and local
   memory again. *)
let channels_grown () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let out = alloc t.g Pinned (4 * n) in
  let b = alloc t.g Pinned 8 in
  H.set64 (host b) 0;
  let compute msg =
    H.write (host out) (String.make (4 * n) '\000');
    S.run t [| S.words (S.launch l k "stack" ~blocks:4 [ address out; n ]) |];
    S.reset l;
    equal (list int) ~msg (List.init n stacked) (words32 (host out) n)
  in
  compute "before";
  (* 300,000 entries that do nothing. *)
  let nops = S.words (Array.make 600_000 0) in
  let before = Rig.submitted t.d in
  S.run t [| nops; S.words (release l (address b) 9) |];
  S.reset l;
  equal int ~msg:"its value" (before + 1) (Rig.submitted t.d);
  equal int ~msg:"the release after the entries" 9 (H.get64 (host b));
  compute "after";
  let h, ha = S.shared t 16 and back, ba = S.shared t 16 in
  let a = B.create t.d 16 in
  S.pattern ha 16 3;
  S.run t
    [|
      S.copy ~dst:a h; S.words (release l (address b) 10); S.copy ~dst:back a;
    |];
  S.reset l;
  equal int ~msg:"the words between the copies" 10 (H.get64 (host b));
  equal int ~msg:"the first byte that differs" (-1) (S.mismatch ba 16 3);
  S.free_launches l;
  List.iter (N.free t.g) [ out; b ]

(* A growth of the channels while a value is held on a host word: once a
   room asked for new channels and the growth made them, every room answers
   LATER until the device reached every value, after a second growth
   replaced the first set before its install too. The submission then
   starts on the new channels; after an allocation, which gives the replaced
   sets back where the RM frees objects, words on both channels, joined, run
   on the new ones. *)
let drained_growth () =
  S.with_driver @@ fun g ->
  let l = S.launches g in
  let gate = alloc g Pinned 8 and out = alloc g Pinned 16 in
  List.iter (fun a -> H.set64 a 0) [ host gate; host out; host out + 8 ];
  let base = N.signaled g in
  let reach v =
    S.watchdog (strf "the value %d" v) (fun () ->
        while N.signaled g < v do
          Domain.cpu_relax ()
        done)
  in
  let grown = function
    | Rig_edge.Grown -> "Grown"
    | Short n -> strf "Short %d" n
  in
  let nops n = [| (0, Array.make (2 * n) 0) |] in
  (* The set [big] grows takes rings of 2^20 entries, which [bigger]
     exceeds. *)
  let big = nops 300_000 and bigger = nops 1_100_000 in
  let copy_release a x = S.segment l (A.Method.copy_release (Int a) (Int x)) in
  let small = [| (1, copy_release (address out) 1) |] in
  S.Edge.submit g ~v:(base + 1) [| (0, acquire l (address gate) 1) |];
  releasing gate 1 (fun () ->
      equal string ~msg:"past the rings" "GROW" (S.Edge.room g big);
      equal string ~msg:"the first growth" "Grown" (grown (N.grow g));
      equal string ~msg:"past the set handed" "GROW" (S.Edge.room g bigger);
      equal string ~msg:"the second growth" "Grown" (grown (N.grow g));
      equal string ~msg:"held" "LATER" (S.Edge.room g big);
      equal string ~msg:"a small room while held" "LATER" (S.Edge.room g small);
      H.set64 (host gate) 1;
      reach (base + 1);
      equal string ~msg:"once reached" "FITS" (S.Edge.room g bigger));
  S.Edge.submit g ~v:(base + 2) bigger;
  reach (base + 2);
  N.free g (alloc g Pinned H.page);
  S.Edge.submit g ~v:(base + 3)
    [| (1, copy_release (address out) 7); (0, release l (address out + 8) 8) |];
  reach (base + 3);
  equal (list int) ~msg:"the releases" [ 7; 8 ]
    [ H.get64 (host out); H.get64 (host out + 8) ];
  S.free_launches l;
  List.iter (N.free g) [ gate; out ]

(* The driver holds 32,768 values in flight: behind a value held on a host
   word, 32,767 more submissions return, and the next waits until the held
   value is reached. One submission of many launches first grows the compute
   channel's segment ring past what the empty submissions take. *)
let in_flight () =
  S.with_ @@ fun t ->
  let l = S.launches t.g in
  let gate = alloc t.g Pinned 8 in
  H.set64 (host gate) 0;
  ignore (snd (numbered_submit t.d 16_384) ());
  S.watchdog "32,768 values in flight" (fun () ->
      releasing gate 1 (fun () ->
          ignore (S.submit t [| S.words (acquire l (address gate) 1) |]);
          for _ = 2 to 32_768 do
            ignore (S.submit t [||])
          done;
          let next = Atomic.make false in
          let d =
            Domain.spawn (fun () ->
                Rig.Point.wait (S.submit t [||]);
                Atomic.set next true)
          in
          still ~msg:"the next submission" bool false
            (fun () -> Atomic.get next)
            ~ms:20;
          H.set64 (host gate) 1;
          Domain.join d));
  S.free_launches l;
  N.free t.g gate

(* A kernel holds its local memory while the device grows it and schedules a
   kernel on the new one: both compute right. *)
let growth () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1024 in
  let flag = alloc t.g Pinned 8 in
  let first = alloc t.g Pinned (4 * n) in
  let next = alloc t.g Pinned (4 * n) in
  H.set64 (host flag) 0;
  releasing flag 1 (fun () ->
      let held =
        S.launch l k "stack_held" ~blocks:4
          [ address flag; 10 * second; address first; n ]
      in
      ignore (S.submit t [| S.words held |]);
      let { A.Gpu.local; _ } = N.capability t.g in
      equal (result unit string) ~msg:"grown" (Ok ()) (local 4096);
      let v =
        S.submit t
          [| S.words (S.launch l k "stack" ~blocks:4 [ address next; n ]) |]
      in
      still ~msg:"the word while held" int (Rig.Point.value v - 2)
        (fun () -> N.signaled t.g)
        ~ms:20;
      H.set64 (host flag) 1;
      Rig.Point.wait v);
  let want = List.init n stacked in
  equal (list int) ~msg:"the held kernel" want (words32 (host first) n);
  equal (list int) ~msg:"the next kernel" want (words32 (host next) n);
  S.free_launches l;
  List.iter (N.free t.g) [ flag; first; next ]

(* Local memory handed over: each growth replaces it while kernels placed
   before it still keep 2 KiB a thread in the old one, each held until the test
   raises a flag of its own. The test then runs them one at a time and grows
   again after each, so the word moves on by single values and every growth
   frees what the word passed. Every kernel computes right and the device
   lives. An RTX 5000 Ada still reads a memory freed under a held kernel, so
   on it the test does not see when each memory goes back. Each memory takes
   some 600 MiB of the GPU. *)
let handover () =
  S.with_ @@ fun t ->
  let k = S.kernels t and l = S.launches t.g in
  let kernels = 4 and n = 1024 in
  let flags = alloc t.g Pinned (8 * kernels) in
  let out = alloc t.g Pinned (4 * n * kernels) in
  H.write (host flags) (String.make (8 * kernels) '\000');
  let held = Queue.create () and next = ref 0 and bytes = ref 3072 in
  let hand () =
    let i = !next in
    incr next;
    let flag = address flags + (8 * i) and at = address out + (4 * n * i) in
    let launch =
      S.launch l k "stack_held" ~blocks:4 [ flag; 10 * second; at; n ]
    in
    Queue.push (i, S.submit t [| S.words launch |]) held
  in
  let grow () =
    bytes := !bytes + 32;
    equal (result unit string)
      ~msg:(strf "local %d" !bytes)
      (Ok ())
      ((N.capability t.g).local !bytes)
  in
  let run () =
    let i, p = Queue.pop held in
    H.set64 (host flags + (8 * i)) 1;
    Rig.Point.wait p
  in
  let raise_all () = H.write (host flags) (String.make (8 * kernels) '\001') in
  Fun.protect ~finally:raise_all (fun () ->
      for _ = 2 to kernels do
        hand ();
        grow ()
      done;
      hand ();
      while not (Queue.is_empty held) do
        run ();
        grow ()
      done);
  let want = List.init n stacked in
  for i = 0 to kernels - 1 do
    equal (list int) ~msg:(strf "kernel %d" i) want
      (words32 (host out + (4 * n * i)) n)
  done;
  equal (option string) ~msg:"the device's loss" None (Rig.lost t.d);
  S.free_launches l;
  List.iter (N.free t.g) [ flags; out ]

let deep_image = Rig_gpu_support.loader (fun () -> S.fixture "deep_sm89.cubin")

(* The bytes of the local memory that serves [g]'s launches of deep
   (fixtures/deep.cu). *)
let deep_local g =
  let pp = Format.pp_print_string in
  let c = require_ok ~pp (A.Cubin.of_string (S.fixture "deep_sm89.cubin")) in
  let k = require_some (A.Cubin.kernel c "deep") in
  let l = require_ok ~pp (A.Launch.make (N.capability g) k) in
  (A.Local_memory.make (N.capability g) (A.Launch.local_bytes l)).bytes

(* A function the GPU has no local memory for: [deep] keeps 480 KiB a thread
   (fixtures/deep.cu), and its local memory holds that for every thread the
   GPU may run at once, past the GPU's memory. Its resolve runs the rounds of
   reclamation, which give a dropped buffer back to the driver, then raises
   Out_of_memory for the bytes of that local memory. It made nothing: the
   device lives, and [shallow], of the same image, grows the local memory
   and computes right. The driver refuses one allocation past the GPU's
   memory, so no memory runs short; a GPU that holds it skips. *)
let no_local_memory () =
  S.with_ @@ fun { d; g } ->
  let bytes = deep_local g in
  if bytes <= Rig.budget d then
    skip
      ~reason:(strf "the GPU holds deep's %d bytes of local memory" bytes)
      ();
  let image = deep_image d in
  let dropped = 64 * 1024 * 1024 in
  let[@inline never] drop () =
    ignore (Sys.opaque_identity (B.create d dropped))
  in
  drop ();
  Gc.full_major ();
  let before = Rig.allocated d in
  let refused =
    match Rig.Image.resolve image "deep" with
    | _ -> fail "deep's local memory was made"
    | exception Rig.Out_of_memory (d', n) -> (Rig.equal d d', n)
  in
  equal (pair bool int) ~msg:"Out_of_memory's device and bytes" (true, bytes)
    refused;
  at_most ~msg:"the device's bytes, the dropped buffer reclaimed" int
    ~than:(before - dropped) (Rig.allocated d);
  equal (option string) ~msg:"the device's loss" None (Rig.lost d);
  (* shallow's threads keep [stack] words: thread i writes i + i mod stack. *)
  let n = 2048 and stack = 1024 in
  let s =
    Sub.make d [| launch_part image "shallow" ~params:12 [| refer 0 0 |] |]
  in
  let run = Run.make s in
  let b = Sub.block s 0 in
  Run.groups run b (n / 256) 1 1;
  Run.threads run b 256 1 1;
  Run.shared run b 0;
  Run.int32 run b 8 n;
  let out = B.create d (4 * n) in
  Rig.Point.wait (submitted run ~buffers:[| out |]);
  equal (array int) ~msg:"shallow's words"
    (Array.init n (fun i -> i + (i mod stack)))
    (words out ~at:0 n)

let local =
  group ~timeout:60. "local memory"
    [
      test "local memory serves kernels that keep 2 KiB a thread" local;
      test "a growth leaves a running kernel's local memory intact" growth;
      test "a replaced local memory stays until the kernels on it ran"
        handover;
      test
        "a function the GPU has no local memory for raises Out_of_memory, \
         having made nothing"
        no_local_memory;
    ]

let channels =
  group ~timeout:120. "channels"
    [
      test "300,000 ring entries run, and kernels and joins run after"
        channels_grown;
      test "a growth of the channels waits for the work in flight"
        drained_growth;
      test "32,768 values stay in flight, and the next waits" in_flight;
    ]

(* Images *)

let images () =
  S.with_driver @@ fun g ->
  let bin = S.fixture "kernels_sm89.cubin" in
  let size = A.Cubin.size (require_ok (A.Cubin.of_string bin)) in
  let n, lay =
    match require_ok (N.image g bin) with
    | Place (n, lay) -> (n, lay)
    | Loaded _ -> fail "an image with nothing to place"
  in
  equal int ~msg:"the region's bytes" size n;
  let r = alloc g Device n in
  let i, bytes = lay r in
  equal int ~msg:"the image's bytes" size (String.length bytes);
  is_some ~msg:"double_index" (code_of (N.entry i "double_index"));
  equal (option int) ~msg:"a missing kernel" None
    (code_of (N.entry i "missing"));
  is_error ~msg:"not a cubin" (N.image g "not a cubin");
  N.unload g i;
  N.free g r

(* A cubin loads where another one's code ran and was unloaded: its launch runs
   its own code. The second cubin is laid over the first one's region, so that
   its code is where the first one's ran whatever the allocator answers. Each
   image goes to its region by a kernel that copies it from Mapped memory. *)
let reloaded () =
  S.with_ @@ fun t ->
  let k = S.kernels t in
  let l = S.launches t.g in
  let n = 1000 in
  let out = alloc t.g Pinned (4 * n) in
  let copy_in code bytes =
    let staging = alloc t.g Mapped (String.length bytes) in
    H.write (host staging) bytes;
    S.run t
      [|
        S.words
          (copy_after l k ~dst:(address code) ~src:(address staging)
             (String.length bytes));
      |];
    N.free t.g staging
  in
  let compute bin code factor =
    H.write (host out) (String.make (4 * n) '\000');
    S.run t
      [|
        S.words
          (S.launch_at l ~code:(address code) bin "index" ~blocks:4
             [ address out; n ]);
      |];
    S.reset l;
    equal (list int) ~msg:(strf "%d i" factor)
      (List.init n (fun i -> factor * i))
      (words32 (host out) n)
  in
  let twice = S.fixture "twice_sm89.cubin" in
  let i, code, bytes = S.image t.g twice in
  copy_in code bytes;
  compute twice code 2;
  N.unload t.g i;
  let thrice = S.fixture "thrice_sm89.cubin" in
  let i', bytes' =
    match N.image t.g thrice with
    | Ok (Place (size, lay)) ->
        equal int ~msg:"the second's image is the first's size"
          (String.length bytes) size;
        lay code
    | Ok (Loaded _) -> fail "an image with nothing to place"
    | Error e -> failf "loading: %s" e
  in
  copy_in code bytes';
  compute thrice code 3;
  N.unload t.g i';
  N.free t.g code;
  S.free_launches l;
  N.free t.g out

let images =
  group ~timeout:60. "images"
    [
      test "load, find their kernels and unload" images;
      test "code loaded where other code ran runs as loaded" reloaded;
    ]

(* Timeline and loss *)

let stop_idle () =
  S.with_ @@ fun t ->
  let r = alloc t.g Device 64 in
  let a = H.pages H.page in
  let m = require_some (N.map_host t.g a 64) in
  S.run t [||];
  S.close t;
  equal int ~msg:"the word" (Rig.submitted t.d) (Rig.signaled t.d);
  N.free t.g r;
  N.free t.g m;
  H.free_pages a H.page;
  let { A.Gpu.local; _ } = N.capability t.g in
  is_error ~msg:"local after stop" (local 1024)

(* COMPUTE:0 waits on a host word nobody writes, a release queued behind the
   wait: stop returns with the word at the last value, and the waiting work
   never runs. *)
let stop_waiting () =
  let t = handed () in
  let l = S.launches t.g in
  let w = alloc t.g Pinned 8 in
  let marked = alloc t.g Pinned 8 in
  H.set64 (host w) 0;
  H.set64 (host marked) 0;
  let ws =
    Array.append (acquire l (address w) 1) (release l (address marked) 7)
  in
  let v = Rig.Point.value (S.submit t [| S.words ws |]) in
  still ~msg:"the word" int (v - 1) (fun () -> N.signaled t.g) ~ms:20;
  S.watchdog "stop" (fun () -> S.stop_driver t.g);
  equal int ~msg:"the word" v (N.signaled t.g);
  H.set64 (host w) 1;
  still ~msg:"the waiting work" int 0 (fun () -> H.get64 (host marked)) ~ms:200;
  S.free_launches l;
  List.iter (N.free t.g) [ w; marked ];
  S.with_ (fun t' -> S.run t' [||])

(* A kernel runs with a release queued behind it: stop returns with the word at
   the last value, and the queued work never runs. *)
let stop_running () =
  let t = handed () in
  let k = S.kernels t in
  let l = S.launches t.g in
  let flag = alloc t.g Pinned 8 in
  let marked = alloc t.g Pinned 8 in
  H.set64 (host flag) 0;
  H.set64 (host marked) 0;
  releasing flag 1 (fun () ->
      let spin = S.launch l k "spin" ~blocks:1 [ address flag; 10 * second ] in
      let v1 = Rig.Point.value (S.submit t [| S.words spin |]) in
      let v2 =
        Rig.Point.value (S.submit t [| S.words (release l (address marked) 7) |])
      in
      still ~msg:"the word" int (v1 - 1) (fun () -> N.signaled t.g) ~ms:20;
      S.watchdog "stop" (fun () -> S.stop_driver t.g);
      equal int ~msg:"the word" v2 (N.signaled t.g));
  still ~msg:"the queued work" int 0 (fun () -> H.get64 (host marked)) ~ms:200;
  S.free_launches l;
  List.iter (N.free t.g) [ flag; marked ]

(* Opens and closes keep, of the C heap, each device's edge alone: 80 bytes,
   which glibc's allocator holds in a chunk of 96. Warm opens first make the
   process's client and the GPU's objects, which stay. The runtime's own
   tables may grow by a few hundred bytes over the run, so a leak of more
   than 64 bytes an open fails it. Each open boots the GPU's GSP over PCI,
   in about 3.5 s: the 20 opens take 70 s. *)
let edge_chunk = 96
let runtime_slack = 1024
let warm_opens = 4
let opens = 16
let opens_timeout = 120.

let opens_keep_edges () =
  let heap () =
    match Rig_support.heap_bytes () with
    | Some n -> n
    | None -> skip ~reason:"the system counts no C heap" ()
  in
  let cycle () =
    let t = S.open_ () in
    ignore (B.create t.d 1 : B.t);
    S.run t [||];
    S.close t;
    Gc.full_major ();
    Rig.free_cache t.d
  in
  for _ = 1 to warm_opens do
    cycle ()
  done;
  let c = heap () in
  for _ = 1 to opens do
    cycle ()
  done;
  at_most int ~msg:"the C heap's growth over the opens"
    ~than:((opens * edge_chunk) + runtime_slack)
    (heap () - c)

let timeline =
  group ~timeout:60. "timeline"
    [
      test "a close of an idle device leaves the word at the last value"
        stop_idle;
      test ~timeout:opens_timeout
        "opens and closes keep each device's edge alone of the C heap"
        opens_keep_edges;
      test
        "stop of a device waiting on a word that never moves leaves the word \
         at the last value (sampled)"
        stop_waiting;
      test
        "stop of a running device leaves the word at the last value, and its \
         queued work never runs (sampled)"
        stop_running;
    ]

(* The shared device: the stateful tests' programs use one device, opened by the
   first and closed when the run ends. *)

let shared = fixture ~teardown:S.close S.open_

(* map_host: host ranges. Every range of mapped host memory maps, whatever
   pages it shares with ranges mapped before, and frees on its own. *)

module Registry = struct
  type t = { mutable regions : (int * int) list } (* pages [lo, hi) *)

  let arena = 4
  let pages (a, n) = (a / H.page, ((a + n - 1) / H.page) + 1)
  let make () = { regions = [] }

  let map m (a, n) =
    let lo, hi = pages (a, n) in
    let same (lo', hi') = lo = lo' && hi = hi' in
    let inside (lo', hi') = lo' <= lo && hi <= hi' && not (same (lo', hi')) in
    let partly (lo', hi') =
      lo < hi' && lo' < hi && not (inside (lo', hi') || same (lo', hi'))
    in
    cover "a range of the same pages as a mapped one"
      (List.exists same m.regions);
    cover "a range inside a mapped one's pages" (List.exists inside m.regions);
    cover "a range that shares only some of a mapped one's pages"
      (List.exists partly m.regions);
    m.regions <- m.regions @ [ (lo, hi) ];
    true

  let free m i =
    let lo, hi = List.nth m.regions i in
    m.regions <- List.filteri (fun j _ -> j <> i) m.regions;
    cover "a free of pages another range still maps"
      (List.exists (fun (lo', hi') -> lo < hi' && lo' < hi) m.regions)

  (* The system *)

  type sys = {
    t : S.t;
    base : int;
    scratch : N.region;
    kernels : S.kernels;
    launches : S.launches;
    lock : Mutex.t;
    mutable live : (N.region * int * int) list;
  }

  let start () =
    let t = shared () in
    {
      t;
      base = H.pages (arena * H.page);
      scratch = Option.get (N.alloc t.g Pinned (arena * H.page));
      kernels = S.kernels t;
      launches = S.launches t.g;
      lock = Mutex.create ();
      live = [];
    }

  let release s =
    List.iter (fun (r, _, _) -> N.free s.t.g r) s.live;
    N.free s.t.g s.scratch;
    S.free_launches s.launches;
    H.free_pages s.base (arena * H.page)

  let map_sys s (a, n) =
    match N.map_host s.t.g (s.base + a) n with
    | Some r ->
        Mutex.protect s.lock (fun () -> s.live <- s.live @ [ (r, a, n) ]);
        true
    | None -> false

  let free_sys s i =
    let r, _, _ = List.nth s.live i in
    N.free s.t.g r;
    Mutex.protect s.lock (fun () ->
        s.live <- List.filteri (fun j _ -> j <> i) s.live)

  (* Each region shares its bytes both ways: a kernel reads what the host wrote,
     and the host reads what a kernel wrote. *)
  let invariant _ s =
    let across ~dst ~src n =
      S.run s.t [| S.words (copy_after s.launches s.kernels ~dst ~src n) |];
      S.reset s.launches
    in
    List.iteri
      (fun i (r, a, n) ->
        let at = s.base + a in
        S.pattern at n (2 * i);
        across ~dst:(address s.scratch) ~src:(address r) n;
        equal int
          ~msg:(strf "the GPU reads [%d, %d)" a (a + n))
          (-1)
          (S.mismatch (host s.scratch) n (2 * i));
        S.pattern (host s.scratch) n ((2 * i) + 1);
        across ~dst:(address r) ~src:(address s.scratch) n;
        equal int
          ~msg:(strf "the host reads [%d, %d)" a (a + n))
          (-1)
          (S.mismatch at n ((2 * i) + 1)))
      s.live

  let range =
    let point = [ 0; 8; 2048; 4088; 4096; 4104; 8192; 12280 ] in
    let length = [ 8; 64; 2048; 4096; 4104; 8192 ] in
    let scale x = x * H.page / 4096 in
    Gen.with_pp
      (fun ppf (a, n) -> Format.fprintf ppf "(%d, %d)" a n)
      (Gen.frequency
         [
           ( 8,
             Gen.such_that
               (fun (a, n) -> a + n <= arena * H.page)
               (Gen.map
                  (fun (a, n) -> (scale a, scale n))
                  (Gen.pair (Gen.of_list point) (Gen.of_list length))) );
           (* One range drawn often, so that ranges of the same pages are drawn
              too. *)
           (2, Gen.constant (0, H.page));
         ])
end

let registry =
  abstract "r" ~release:Registry.release ~invariant:Registry.invariant

let mapped_regions =
  among int registry (fun m ->
      List.init (List.length m.Registry.regions) Fun.id)

let registry_commands =
  [
    command "start" (Gen.unit @-> makes registry) Registry.make Registry.start;
    command "map_host"
      (registry ^-> Registry.range @-> returns bool)
      Registry.map Registry.map_sys;
    command "free"
      (registry ^-> mapped_regions ^-> returns unit)
      Registry.free Registry.free_sys;
  ]

let stateful =
  group ~timeout:300. "stateful"
    [
      stateful ~count:100 ~steps:20 "map_host shares a host range both ways"
        registry_commands;
    ]

let () =
  if Array.length Sys.argv = 2 && Sys.argv.(1) = dma_child then
    exit
      (match exports_bus () with
      | () -> 0
      | exception e ->
          prerr_endline (Printexc.to_string e);
          1);
  exit
    (run "rig_nv"
       [
         gpus;
         locks;
         parameters;
         facts;
         memory;
         dma;
         work;
         launches;
         room;
         local;
         channels;
         images;
         timeline;
         stateful;
       ])
