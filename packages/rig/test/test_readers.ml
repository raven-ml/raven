(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The C readers of rig.h, from native code and from bytecode, where another
   library's stubs resolve them at load. *)

open Windtrap
module B = Rig.Buffer
module R = Rig_support.Reader

(* Whether two buffers or ranges share memory. *)
let sharing =
  Testable.make
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf (if o then "share" else "apart"))
    ~equal:Bool.equal

let test_live () =
  let b = B.create Rig.host 40 in
  not_equal ~msg:"its address" int 0 (R.host b);
  equal ~msg:"its bytes" int 40 (R.bytes b);
  equal ~msg:"its reason" (option string) None (R.why b)

let test_view () =
  let b = B.create Rig.host 64 in
  let v = B.view b ~first:8 ~length:6 in
  equal ~msg:"its address" int (R.host b + 8) (R.host v);
  equal ~msg:"its bytes" int 6 (R.bytes v)

let test_dead () =
  let b = B.create Rig.host 8 in
  ignore (require_some (Rig_support.donated b));
  equal (option string) (Some "donated") (R.why b)

(* A take of the memory is refused while the claim holds, and made once it is
   released. *)
let test_claim () =
  let b = B.create Rig.host 8 in
  let answer = Testable.make ~pp:R.pp_answer ~equal:( = ) in
  equal ~msg:"the claim" answer R.Claimed (R.claim b Rig.Read);
  ignore (B.take ~why:"taken" [ b ]);
  equal ~msg:"after a take while claimed" (option string) None (B.dead b);
  R.release b;
  ignore (B.take ~why:"taken" [ b ]);
  equal ~msg:"after a take once released" (option string) (Some "taken")
    (B.dead b)

(* Spans *)

let opaque =
  lazy (fst (Rig_support.Polled.open_ ~host_visible:false "readers:span"))

let far =
  lazy
    (require_ok ~pp:Format.pp_print_string
       (Rig.open_
          ~machine:(Rig_support.machine "readers:far")
          {
            name = "readers:far-gpu";
            driver = (module Rig_support.Polled);
            make =
              (fun ~word ->
                Ok
                  (Rig_support.Polled.bind ~word
                     (Rig_support.Polled.make ~host_visible:false ~peers:false
                        ())));
          }))

(* Host memory lies at host addresses in space 0, a view at its own first
   byte. *)
let test_span_host () =
  let b = B.create Rig.host 64 in
  equal ~msg:"the buffer" (pair int int) (0, R.host b) (R.span b);
  let v = B.view b ~first:8 ~length:6 in
  equal ~msg:"a view" (pair int int) (0, R.host b + 8) (R.span v)

(* Memory the host does not address is a space of its own, at offsets; a borrow
   lies where the memory it maps does. *)
let test_span_device () =
  let d = Lazy.force opaque in
  let m = B.create d 64 and m' = B.create d 64 in
  let h = B.create Rig.host B.paged_from in
  let space, first = R.span m in
  not_equal ~msg:"not the host's" int 0 space;
  equal ~msg:"its first byte" int 0 first;
  equal ~msg:"a view" (pair int int) (space, 8)
    (R.span (B.view m ~first:8 ~length:6));
  not_equal ~msg:"another memory" int space (fst (R.span m'));
  equal ~msg:"a borrow of host memory" (pair int int) (R.span h)
    (R.span (require_some (B.borrow d h)))

(* The buffers the law draws views of: two host memories, two bigarray buffers
   over shared bytes, a borrow of the first, two memories the host does not
   address, and one of another machine. *)
let world () =
  let h = B.create Rig.host B.paged_from in
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

(* Views of two buffers over one word: the host memory and its borrow, and the
   two bigarrays, the second 8 bytes into the first. *)
let spans_examples = [ [ (0, 0, 8); (4, 0, 8) ]; [ (2, 8, 8); (3, 0, 8) ] ]

(* Whether two spans' ranges, rounded out to whole words of their space,
   meet. *)
let shares (s, f, n) (s', f', n') =
  let w = Rig.word in
  let lo f = f / w * w and hi f n = (f + n + w - 1) / w * w in
  n > 0 && n' > 0 && s = s' && lo f < hi f' n' && lo f' < hi f n

(* Two buffers share memory iff their spans are in one space and their ranges
   meet there once rounded out to words, as Buffer.overlaps says: the pairs of
   drawn views, by their places in the list, that share. *)
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
  let pairs = ref [] in
  List.iteri
    (fun k (i, b) ->
      List.iteri
        (fun j (i', b') ->
          if j > k then pairs := ((k, j), (i, b), (i', b')) :: !pairs)
        bs)
    bs;
  let pairs = List.rev !pairs in
  let overlapping =
    List.filter (fun (_, (_, b), (_, b')) -> B.overlaps b b') pairs
  in
  cover "views that share a word" (overlapping <> []);
  cover "views of two buffers over one word"
    (List.exists (fun (_, (i, _), (i', _)) -> i <> i') overlapping);
  let sharing_pairs f =
    List.filter_map
      (fun (kj, (_, b), (_, b')) -> if f b b' then Some kj else None)
      pairs
  in
  equal ~msg:"the pairs that share memory"
    (list (pair int int))
    (sharing_pairs (fun b b' -> shares (place b) (place b')))
    (sharing_pairs B.overlaps)

(* Ranges at a buffer's bounds lie within it: an empty range at its end overlaps
   nothing, its last byte overlaps itself, and two halves of it share no byte. A
   range past the bounds aborts the process, which no test in the process can
   observe. *)
let bounds =
  [
    ("an empty range at the end", (64, 0, 0, 64), false);
    ("the last byte", (63, 1, 63, 1), true);
    ("two halves", (0, 32, 32, 32), false);
    ("the whole buffer", (0, 64, 0, 64), true);
  ]

let test_overlap_bound (_, (af, an, bf, bn), expected) =
  let a = B.create Rig.host 64 in
  equal sharing expected (R.overlap a af an a bf bn)

let tests =
  [
    group "low level"
      [
        test "a live buffer reads as its address and length" test_live;
        test "a view reads as its own first byte and elements" test_view;
        test "a dead buffer reads as its reason" test_dead;
        test "a claim from C claims and its release ends it" test_claim;
        test "host memory lies at host addresses" test_span_host;
        test "other memory lies in a space of its own, at offsets"
          test_span_device;
        prop ~examples:spans_examples
          "spans share a word as Buffer.overlaps says" views law_spans;
        cases
          ~id:(fun (what, _, _) -> what)
          "rig_buffers_overlap of ranges at a buffer's bounds is whether they \
           share a byte"
          bounds test_overlap_bound;
      ];
  ]

let () = exit (run "rig.readers" tests)
