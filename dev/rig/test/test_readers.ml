(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The C readers of rig.h, from native code and from bytecode, where another
   library's stubs resolve them at load. *)

open Windtrap
module B = Rig.Buffer
module R = Rig_support.Reader

let timeout = 60.

let test_live () =
  let b = B.create Rig.host 40 in
  equal int (B.address b) (R.host b);
  equal int 40 (R.bytes b);
  equal (option string) None (R.why b)

let test_view () =
  let b = B.create Rig.host 64 in
  let v = B.view b ~first:8 ~length:6 in
  equal int (B.address b + 8) (R.host v);
  equal int 6 (R.bytes v)

let test_dead () =
  let b = B.create Rig.host 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" b));
  equal (option string) (Some "donated") (R.why b)

(* A claim from C resolves and claims, and its release ends it. *)
let test_claim () =
  let b = B.create Rig.host 8 in
  let answer = Testable.make ~pp:R.pp_answer ~equal:( = ) in
  equal ~msg:"claimed" answer R.Claimed (R.claim b B.Read);
  R.release b;
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal ~msg:"released" bool true (Rig.Claim.exclusive c b))

(* Spans *)

let opaque = lazy (fst (Rig_support.Polled.open_ ~host_visible:false "readers:span"))

let far =
  lazy
    (require_ok ~pp:Format.pp_print_string
       (Rig.open_
          (module Rig_support.Polled)
          ~machine:(Rig_support.machine "readers:far")
          ~name:"readers:far-gpu"
          (fun () ->
            Ok (Rig_support.Polled.make ~host_visible:false ~peers:false ()))))

(* Host memory lies at host addresses in space 0, a view at its own first
   byte. *)
let test_span_host () =
  let b = B.create Rig.host 64 in
  equal ~msg:"the buffer" (pair int int) (0, B.address b) (R.span b);
  let v = B.view b ~first:8 ~length:6 in
  equal ~msg:"a view" (pair int int) (0, B.address b + 8) (R.span v)

(* Memory the host does not address is a space of its own, at offsets; a
   borrow lies where the memory it maps does. *)
let test_span_device () =
  let d = Lazy.force opaque in
  let m = B.create d 64 and m' = B.create d 64 in
  let h = B.create Rig.host (1 lsl 16) in
  let space, first = R.span m in
  not_equal ~msg:"not the host's" int 0 space;
  equal ~msg:"its first byte" int 0 first;
  equal ~msg:"a view" (pair int int) (space, 8)
    (R.span (B.view m ~first:8 ~length:6));
  not_equal ~msg:"another memory" int space (fst (R.span m'));
  equal ~msg:"a borrow of host memory" (pair int int) (R.span h)
    (R.span (require_some (B.borrow d h)))

(* The buffers the law draws views of: two host memories, two bigarray
   buffers over shared bytes, a borrow of the first, two memories the host
   does not address, and one of another machine. *)
let world () =
  let h = B.create Rig.host (1 lsl 16) in
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 32 in
  let d = Lazy.force opaque in
  [|
    h;
    B.create Rig.host 32;
    B.of_bigarray ba;
    B.of_bigarray (Bigarray.Array1.sub ba 8 16);
    require_some (B.borrow d h);
    B.create d 32;
    B.create d 32;
    B.create (Lazy.force far) 32;
  |]

let pp_views =
  Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (i, o, l) ->
      Format.fprintf ppf "%d[%d,+%d]" i o l)

let views =
  let open Gen in
  with_pp pp_views
    (list ~size:(int_range 2 6)
       (triple (int_range 0 7) (int_range 0 32) (int_range 0 32)))

let shares (s, f, n) (s', f', n') =
  n > 0 && n' > 0 && s = s' && f < f' + n' && f' < f + n

(* Two buffers share a byte iff their spans are in one space and their ranges
   meet, as Buffer.overlaps says. *)
let law_spans vs =
  let w = world () in
  let bs =
    List.map
      (fun (i, o, l) ->
        let b = w.(i) in
        let first = Int.min o (B.length b) in
        (i, B.view b ~first ~length:(Int.min l (B.length b - first))))
      vs
  in
  let place b =
    let s, f = R.span b in
    (s, f, B.length b)
  in
  List.iteri
    (fun k (i, b) ->
      List.iteri
        (fun j (i', b') ->
          if j > k then begin
            let o = B.overlaps b b' in
            cover "views that share a byte" o;
            cover "views of two buffers over one byte" (o && i <> i');
            equal ~msg:"overlaps" bool o (shares (place b) (place b'))
          end)
        bs)
    bs

let tests =
  [
    group ~timeout "readers"
      [
        test "a live buffer reads as its address and length" test_live;
        test "a view reads as its own first byte and elements" test_view;
        test "a dead buffer reads as its reason" test_dead;
        test "a claim from C claims and its release ends it" test_claim;
      ];
    group ~timeout "spans"
      [
        test "host memory lies at host addresses" test_span_host;
        test "other memory lies in a space of its own, at offsets"
          test_span_device;
        prop "spans share a byte as Buffer.overlaps says" views law_spans;
      ];
  ]

let () = exit (run "rig.readers" tests)
