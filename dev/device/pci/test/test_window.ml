(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Device_pci
open Device_pci_support

let strf = Printf.sprintf

external window_of : Window.t -> int * int * int * bool
  = "device_pci_test_window_of"

external c_store32 : Window.t -> int -> int -> unit = "device_pci_test_store32"

external c_store64 : Window.t -> int -> int64 -> unit
  = "device_pci_test_store64"

external c_load32 : Window.t -> int -> int = "device_pci_test_load32"
external c_load64 : Window.t -> int -> int64 = "device_pci_test_load64"
external c_write : Window.t -> int -> string -> unit = "device_pci_test_write"
external c_failed : Window.t -> string option = "device_pci_test_failed"
external c_flush : Window.t -> unit = "device_pci_test_flush"
external c_combines : Window.t -> bool = "device_pci_test_combines"

(* Far machines put their bytes at [base], away from 0 so that an access sent to
   its offset in place of its address misses them. *)
let base = 0x4000_0000

(* [mapped skew n] and [through skew n] are [n] fresh zero bytes whose address
   is [skew] modulo 8. A far machine holds exactly the window's bytes. *)
let mapped skew n = Window.v (memory (n + 8) + skew) n

(* [combining skew n] is [mapped skew n] on a window that combines, whose copies
   go 16 bytes at a time. *)
let combining skew n = Window.v ~combines:true (memory (n + 8) + skew) n

let through skew n =
  let a = base + skew in
  Window.through (Window.unsafe_transport (far a n)) a n

let far_window n =
  let f = far base n in
  (f, Window.through (Window.unsafe_transport f) base n)

(* The model

   A window is its bytes inside its root window's. Alignment is of the address
   on the machine: [skew] is the root's address modulo 8. Counts below zero are
   misuse like bytes outside the window. *)

type view = { bytes : Bytes.t; skew : int; off : int; len : int }

let inside r o n = o >= 0 && n >= 0 && o <= r.len - n
let aligned r o w = (r.skew + r.off + o) mod w = 0
let check r o n = if not (inside r o n) then invalid_arg "outside"

let check_word r o w =
  check r o w;
  cover "a word that ends the window" (o = r.len - w);
  if not (aligned r o w) then invalid_arg "unaligned"

module Model = struct
  let window skew n = { bytes = Bytes.make n '\000'; skew; off = 0; len = n }

  let sub r o n =
    check r o n;
    { r with off = r.off + o; len = n }

  let get8 r o =
    check r o 1;
    Bytes.get_uint8 r.bytes (r.off + o)

  let set8 r o b =
    check r o 1;
    Bytes.set_uint8 r.bytes (r.off + o) (b land 0xff)

  let get32 r o =
    check_word r o 4;
    Int32.to_int (Bytes.get_int32_le r.bytes (r.off + o)) land 0xffff_ffff

  let set32 r o x =
    check_word r o 4;
    Bytes.set_int32_le r.bytes (r.off + o) (Int32.of_int x)

  let get64 r o =
    check_word r o 8;
    Bytes.get_int64_le r.bytes (r.off + o)

  let set64 r o x =
    check_word r o 8;
    Bytes.set_int64_le r.bytes (r.off + o) x

  let read r o n =
    check r o n;
    Bytes.sub_string r.bytes (r.off + o) n

  let blit_string r s so o n =
    if so < 0 || n < 0 || so > String.length s - n then invalid_arg "string";
    check r o n;
    Bytes.blit_string s so r.bytes (r.off + o) n

  let write r o s =
    check r o (String.length s);
    Bytes.blit_string s 0 r.bytes (r.off + o) (String.length s)

  let fill r o n c =
    check r o n;
    Bytes.fill r.bytes (r.off + o) n c

  let flush r = check_word r 0 4
end

(* Offsets and counts around every window's bounds, and the extremes. *)
let index =
  Gen.frequency
    [
      (8, Gen.int_range (-2) 42);
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ min_int; min_int + 1; max_int - 1; max_int ] );
    ]

let skew = Gen.int_range 0 7
let length = Gen.int_range 0 40
let inner = Gen.int_range 0 40
let bytes = Gen.string_of ~size:(Gen.int_range 0 12) Gen.char

(* Every access of the interface and of device_pci.h, on windows made by [make].
   The C accesses check no bounds, so they are called inside. *)
let commands ~is_mapped make =
  let win =
    abstract "w" ~invariant:(fun r w ->
        equal ~msg:"length" int r.len (Window.length w);
        equal ~msg:"mapped" bool is_mapped (Window.mapped w);
        equal ~msg:"bytes" string
          (Bytes.sub_string r.bytes r.off r.len)
          (Window.read w 0 r.len))
  in
  let word w r o = inside r o w && aligned r o w in
  [
    command "window" (skew @-> length @-> makes win) Model.window make;
    command "sub" (win ^-> index @-> index @-> makes win) Model.sub Window.sub;
    command "get8" (win ^-> index @-> returns int) Model.get8 Window.get8;
    command "set8"
      (win ^-> index @-> Gen.int @-> returns unit)
      Model.set8 Window.set8;
    command "get32" (win ^-> index @-> returns int) Model.get32 Window.get32;
    command "set32"
      (win ^-> index @-> Gen.int @-> returns unit)
      Model.set32 Window.set32;
    command "get64" (win ^-> index @-> returns int64) Model.get64 Window.get64;
    command "set64"
      (win ^-> index @-> Gen.int64 @-> returns unit)
      Model.set64 Window.set64;
    command "read"
      (win ^-> index @-> index @-> returns string)
      Model.read Window.read;
    command "write"
      (win ^-> index @-> bytes @-> returns unit)
      Model.write Window.write;
    command "blit_string"
      (win ^-> bytes @-> index @-> index @-> index @-> returns unit)
      Model.blit_string
      (fun w s so o n -> Window.blit_string s so w o n);
    command "fill"
      (win ^-> index @-> index @-> Gen.char @-> returns unit)
      Model.fill Window.fill;
    command "flush" (win ^-> returns unit) Model.flush Window.flush;
    command "device_pci_store32"
      ~pre:(fun r o _ -> word 4 r o)
      (win ^-> inner @-> Gen.int @-> returns unit)
      Model.set32 c_store32;
    command "device_pci_store64"
      ~pre:(fun r o _ -> word 8 r o)
      (win ^-> inner @-> Gen.int64 @-> returns unit)
      Model.set64 c_store64;
    command "device_pci_load32"
      ~pre:(fun r o -> word 4 r o)
      (win ^-> inner @-> returns int)
      Model.get32 c_load32;
    command "device_pci_load64"
      ~pre:(fun r o -> word 8 r o)
      (win ^-> inner @-> returns int64)
      Model.get64 c_load64;
    command "device_pci_write"
      ~pre:(fun r o s -> inside r o (String.length s))
      (win ^-> inner @-> bytes @-> returns unit)
      Model.write c_write;
    command "device_pci_flush"
      ~pre:(fun r -> word 4 r 0)
      (win ^-> returns unit)
      Model.flush c_flush;
  ]

(* A window one byte past an aligned address: its words at offsets 3 and 7 are
   aligned on the machine, those at offset 0 are not. *)
let address_alignment make () =
  let w = Window.sub (make 0 24) 1 20 in
  equal ~msg:"get32 at an aligned address" int 0 (Window.get32 w 3);
  equal ~msg:"get64 at an aligned address" int64 0L (Window.get64 w 7);
  raises_match ~msg:"get32 at an unaligned address" Exn.invalid_arg (fun () ->
      Window.get32 w 0);
  raises_match ~msg:"get64 at an unaligned address" Exn.invalid_arg (fun () ->
      Window.get64 w 0)

let same_bytes =
  group ~timeout:patience "accesses"
    [
      stateful ~count:300 "a mapped window behaves as its bytes"
        (commands ~is_mapped:true mapped);
      stateful ~count:300 "a combining window behaves as its bytes"
        (commands ~is_mapped:true combining);
      stateful ~count:300 "a window through a transport behaves as its bytes"
        (commands ~is_mapped:false through);
      test "a mapped word is aligned by its address" (address_alignment mapped);
      test "a far word is aligned by its address" (address_alignment through);
    ]

(* Windows *)

let test_v () =
  let a = memory 16 in
  let w = Window.v a 16 in
  equal ~msg:"address" int a (Window.address w);
  equal ~msg:"length" int 16 (Window.length w);
  equal ~msg:"mapped" bool true (Window.mapped w)

let test_through () =
  let w = through 0 16 in
  equal ~msg:"address" int base (Window.address w);
  equal ~msg:"length" int 16 (Window.length w);
  equal ~msg:"mapped" bool false (Window.mapped w)

let test_negative () =
  let tr = Window.unsafe_transport (far base 16) in
  let a = memory 16 in
  List.iter
    (fun n ->
      let msg = string_of_int n in
      raises_match ~msg Exn.invalid_arg (fun () -> Window.v a n);
      raises_match ~msg Exn.invalid_arg (fun () -> Window.through tr base n))
    [ -1; min_int ];
  raises_match ~msg:"no transport" Exn.invalid_arg (fun () ->
      Window.through (Window.unsafe_transport 0) base 16)

(* A sub-window's place, for [(skew, len, off, n)] with [off, n] in [len]. *)
let place =
  let open Gen in
  bind (pair skew length) (fun (s, len) ->
      bind (int_range 0 len) (fun off ->
          map (fun n -> (s, len, off, n)) (int_range 0 (len - off))))
  |> with_pp (fun ppf (s, len, off, n) ->
      Format.fprintf ppf "skew %d, length %d, sub %d %d" s len off n)

let sub_address make (s, len, off, n) =
  let w = make s len in
  let x = Window.sub w off n in
  equal ~msg:"address" int (Window.address w + off) (Window.address x);
  equal ~msg:"length" int n (Window.length x)

(* A window combines as it was made, as device_pci.h sees it; a sub-window as
   its parent; one through a transport never. *)
let test_combines () =
  let a = memory 64 in
  List.iter
    (fun (msg, w, want) -> equal ~msg bool want (c_combines w))
    [
      ("mapped", Window.v a 64, false);
      ("mapped, combining", Window.v ~combines:true a 64, true);
      ("a sub-window", Window.sub (Window.v ~combines:true a 64) 8 16, true);
      ("through", through 0 16, false);
    ]

let windows =
  group ~timeout:patience "windows"
    [
      test "v is the bytes at the address it is given" test_v;
      test "through is the bytes at the address it is given" test_through;
      test "a window combines as made, a sub-window as its parent" test_combines;
      test "v and through refuse a negative length, and through no transport"
        test_negative;
      prop "a mapped sub-window starts its offset past its parent" place
        (sub_address mapped);
      prop "a far sub-window starts its offset past its parent" place
        (sub_address through);
    ]

(* Bigarrays *)

let test_bigarray () =
  let w = Window.sub (mapped 0 16) 3 10 in
  let b = Window.bigarray w in
  equal ~msg:"length" int 10 (Bigarray.Array1.dim b);
  Window.set8 w 0 0x41;
  equal ~msg:"a store is in the bigarray" char 'A' b.{0};
  b.{9} <- 'z';
  equal ~msg:"the bigarray is the window" int (Char.code 'z') (Window.get8 w 9)

let bigarrays =
  group ~timeout:patience "bigarrays"
    [
      test "a mapped window's bigarray is its bytes" test_bigarray;
      test "a far window has no bigarray" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Window.bigarray (through 0 16)));
    ]

(* Long copies

   A copy of 8 KiB or more moves in pieces, with the runtime released for each,
   while another domain compacts the heap and moves the strings it copies. *)

let piece = 8192

(* [(skew, n, off)]: the bytes from [off] of a mapped window of [n] bytes. *)
let long_place =
  let open Gen in
  bind
    (pair skew (int_range (piece - 8) ((3 * piece) + 9)))
    (fun (s, n) -> map (fun off -> (s, n, off)) (int_range 0 9))
  |> with_pp (fun ppf (s, n, off) ->
      Format.fprintf ppf "skew %d, length %d, from %d" s n off)

(* [f ()] while another domain compacts the heap, and whether a compaction ran
   meanwhile. *)
let compacting f =
  let compactions () = (Gc.quick_stat ()).compactions in
  let stop = Atomic.make false in
  let before = compactions () in
  let other =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          Gc.compact ()
        done)
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set stop true;
      Domain.join other)
    f;
  compactions () > before

let long_copies make (skew, n, off) =
  let w = make skew n and k = n - off in
  let s = String.init k (fun i -> Char.chr (((i * 7) + skew) land 0xff)) in
  let t =
    "ab" ^ String.map (fun c -> Char.chr ((Char.code c + 1) land 0xff)) s ^ "cd"
  in
  cover "8 KiB or more" (k >= piece);
  cover "a part of a piece at the end" (k > piece && k mod piece <> 0);
  let moved =
    compacting (fun () ->
        Window.write w off s;
        equal ~msg:"read after write" string s (Window.read w off k);
        Window.blit_string t 2 w off k;
        equal ~msg:"read after blit_string" string (String.sub t 2 k)
          (Window.read w off k);
        Window.fill w off k 'q';
        equal ~msg:"read after fill" string (String.make k 'q')
          (Window.read w off k);
        equal ~msg:"the bytes before" string (String.make off '\000')
          (Window.read w 0 off))
  in
  cover "the heap compacted during the copies" moved

let long =
  group ~timeout:patience "long copies"
    [
      prop
        "read, write, blit_string and fill agree with the bytes while the heap \
         moves"
        long_place (long_copies mapped);
      prop
        "on a combining window, read, write, blit_string and fill agree with \
         the bytes while the heap moves"
        long_place (long_copies combining);
    ]

(* Transports *)

let access = triple bool int int

let test_one_access () =
  let f, w = far_window 32 in
  let once msg kind off n run =
    run ();
    equal ~msg (list access) [ (kind, base + off, n) ] (log f)
  in
  once "get32" false 4 4 (fun () -> ignore (Window.get32 w 4));
  once "set32" true 28 4 (fun () -> Window.set32 w 28 1);
  once "get64" false 8 8 (fun () -> ignore (Window.get64 w 8));
  once "set64" true 24 8 (fun () -> Window.set64 w 24 1L);
  once "device_pci_load32" false 4 4 (fun () -> ignore (c_load32 w 4));
  once "device_pci_store32" true 28 4 (fun () -> c_store32 w 28 1);
  once "device_pci_load64" false 8 8 (fun () -> ignore (c_load64 w 8));
  once "device_pci_store64" true 24 8 (fun () -> c_store64 w 24 1L);
  once "flush" false 0 4 (fun () -> Window.flush w);
  once "device_pci_flush" false 0 4 (fun () -> c_flush w)

(* Stores through a far sub-window [4 * off] bytes into its machine's, then a
   flush: the machine sees the stores in their order, then the flush's one load
   of the sub-window's first word. *)
let flushes =
  let stores = Gen.list ~size:(Gen.int_range 0 6) (Gen.int_range 0 7) in
  Gen.pair (Gen.pair (Gen.int_range 0 7) stores) Gen.bool

let test_flush_order ((off, words), in_c) =
  let f = far base 64 in
  let w =
    Window.sub (Window.through (Window.unsafe_transport f) base 64) (4 * off) 32
  in
  List.iter (fun i -> Window.set32 w (4 * i) i) words;
  if in_c then c_flush w else Window.flush w;
  cover "stores before the flush" (words <> []);
  let a = base + (4 * off) in
  equal (list access)
    (List.map (fun i -> (true, a + (4 * i), 4)) words @ [ (false, a, 4) ])
    (log f)

(* A store of [n] bytes at [off] of a far window reaches no other byte of the
   machine: a neighbouring register keeps its value and its side effects. *)
let stores_only write (skew, len, off, n) =
  let f = far (base + skew) len in
  let w = Window.through (Window.unsafe_transport f) (base + skew) len in
  write w off (String.make n 'x');
  let a = base + skew + off in
  cover "a store between two other bytes" (off > 0 && off + n < len);
  List.iter
    (fun (_, x, m) ->
      if x < a || x + m > a + n then
        failf "a store of %d bytes at %#x is outside [%#x, %#x)" m x a (a + n))
    (log f)

let bulk =
  [
    prop "write stores only the bytes it is given" place
      (stores_only Window.write);
    prop "blit_string stores only the bytes it is given" place
      (stores_only (fun w off s ->
           Window.blit_string ("ab" ^ s ^ "cd") 2 w off (String.length s)));
    prop "device_pci_write stores only the bytes it is given" place
      (stores_only c_write);
  ]

(* Two domains each store ascending words of their half of a far machine. *)
let test_order () =
  let words = 256 in
  let f, w = far_window (8 * words) in
  let store d () =
    for i = 0 to words - 1 do
      Window.set32 w (((d * words) + i) * 4) i
    done
  in
  let other = Domain.spawn (store 1) in
  store 0 ();
  Domain.join other;
  let seen = List.map (fun (_, a, _) -> (a - base) / 4) (log f) in
  List.iter
    (fun d ->
      let mine = List.filter (fun i -> i / words = d) seen in
      equal ~msg:(strf "domain %d" d) (list int)
        (List.init words (fun i -> (d * words) + i))
        mine)
    [ 0; 1 ]

(* A transport access that blocks holds no other domain: one collects while it
   waits. Holding the runtime, the collection would wait for the access, and the
   access would fail once its hold ran out. *)
let test_blocking () =
  let f, w = far_window 8 in
  hold f;
  let other =
    Domain.spawn (fun () ->
        while not (waiting f) do
          Domain.cpu_relax ()
        done;
        Gc.full_major ();
        let_go f)
  in
  let x = Window.get32 w 0 in
  Domain.join other;
  equal ~msg:"the word" int 0 x

(* Through a failed transport a read gives all ones, as a function that left the
   bus answers: each read as the bytes it gives. *)
let failed_reads =
  [
    ("get8", fun w -> (strf "%x" (Window.get8 w 1), "ff"));
    ("get32", fun w -> (strf "%x" (Window.get32 w 4), "ffffffff"));
    ("get64", fun w -> (strf "%Lx" (Window.get64 w 8), "ffffffffffffffff"));
    ("read", fun w -> (Window.read w 1 9, String.make 9 '\xff'));
    ("device_pci_load32", fun w -> (strf "%x" (c_load32 w 4), "ffffffff"));
    ( "device_pci_load64",
      fun w -> (strf "%Lx" (c_load64 w 8), "ffffffffffffffff") );
  ]

let failed_writes =
  [
    ("set8", fun w -> Window.set8 w 1 0);
    ("set32", fun w -> Window.set32 w 4 0);
    ("set64", fun w -> Window.set64 w 8 0L);
    ("write", fun w -> Window.write w 1 "123456789");
    ("fill", fun w -> Window.fill w 1 9 'x');
    ("device_pci_store32", fun w -> c_store32 w 4 0);
    ("device_pci_store64", fun w -> c_store64 w 8 0L);
    ("device_pci_write", fun w -> c_write w 1 "123456789");
  ]

let broken () =
  let f, w = far_window 32 in
  break f;
  ignore (log f : _ list);
  (f, w)

let transports =
  group ~timeout:patience "transports"
    ([
       test "a 32- or 64-bit access is one access of its width" test_one_access;
       test "each domain's accesses arrive in the order it makes them"
         test_order;
       test "a domain whose transport access blocks holds no other"
         test_blocking;
       prop "a flush loads its window's first word after the stores before it"
         flushes test_flush_order;
       cases ~name:fst "a read through a failed transport gives all ones"
         failed_reads (fun (_, read) ->
           let got, want = read (snd (broken ())) in
           equal string want got);
       cases ~name:fst "a write through a failed transport is dropped"
         failed_writes (fun (_, write) ->
           let f, w = broken () in
           write w;
           equal (list access) [] (log f));
       cases ~name:fst "a flush through a failed transport reaches nothing"
         [ ("flush", Window.flush); ("device_pci_flush", c_flush) ]
         (fun (_, flush) ->
           let f, w = broken () in
           flush w;
           equal (list access) [] (log f));
     ]
    @ bulk)

(* The C view *)

let test_window_of () =
  let w = mapped 0 16 in
  let a = Window.address w in
  let shown = quad int int int bool in
  equal ~msg:"mapped" shown (a, 16, a, false) (window_of w);
  equal ~msg:"a mapped sub-window" shown
    (a + 3, 9, a + 3, false)
    (window_of (Window.sub w 3 9));
  let w = through 0 16 in
  equal ~msg:"through" shown (base, 16, 0, true) (window_of w);
  equal ~msg:"a far sub-window" shown
    (base + 3, 9, 0, true)
    (window_of (Window.sub w 3 9))

let test_c_failed () =
  equal ~msg:"mapped" (option string) None (c_failed (mapped 0 16));
  let f, w = far_window 32 in
  equal ~msg:"through a live transport" (option string) None (c_failed w);
  break f;
  equal ~msg:"through a failed transport" (option string)
    (Some "far: the link broke") (c_failed w)

let c_view =
  group ~timeout:patience "device_pci.h"
    [
      test "device_pci_window_of reads a window's place and side" test_window_of;
      test
        "device_pci_failed is NULL until the window's transport fails, then \
         its reason"
        test_c_failed;
    ]

let () =
  exit
  @@ run "device_pci.window"
       [ same_bytes; windows; bigarrays; long; transports; c_view ]
