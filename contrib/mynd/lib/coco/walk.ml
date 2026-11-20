(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let columns ~height c =
  let first = ref (-1) and last = ref (-1) and at = ref 0 in
  Array.iteri
    (fun i n ->
      if i land 1 = 1 && n > 0 then begin
        if !first < 0 then first := !at;
        last := !at + n - 1
      end;
      at := !at + n)
    c;
  if !first < 0 || height = 0 then (1, 0) else (!first / height, !last / height)

let area c =
  let a = ref 0 in
  Array.iteri (fun i n -> if i land 1 = 1 then a := !a + n) c;
  !a

(* rleIou's walk (pycocotools 2.0.11, common/maskApi.c): both runs at once,
   counting the pixels where either and where both are [true]. *)
let iou d g ~crowd =
  let kd = Array.length d and kg = Array.length g in
  let cd = ref d.(0) and cg = ref g.(0) in
  let id = ref 1 and ig = ref 1 in
  let vd = ref false and vg = ref false in
  let i = ref 0 and u = ref 0 and ct = ref 1 in
  while !ct > 0 do
    let c = Int.min !cd !cg in
    if !vd || !vg then begin
      u := !u + c;
      if !vd && !vg then i := !i + c
    end;
    cd := !cd - c;
    if !cd = 0 && !id < kd then begin
      cd := d.(!id);
      incr id;
      vd := not !vd
    end;
    cg := !cg - c;
    if !cg = 0 && !ig < kg then begin
      cg := g.(!ig);
      incr ig;
      vg := not !vg
    end;
    ct := !cd + !cg
  done;
  if !i = 0 then 0.
  else
    let u = if crowd then area d else !u in
    Float.of_int !i /. Float.of_int u
