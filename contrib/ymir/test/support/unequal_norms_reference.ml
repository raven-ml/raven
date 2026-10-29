(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Written by test/gen/unequal_norms.py; do not edit. *)

[@@@ocamlformat "disable"]

(* A label, a unit row a, a row b of another norm, and their separation and
   b's position angle from a, from the rows' exact values at 60 digits,
   rounded once. *)
let pairs = [
  ("1 uas, generic, norm 1.3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.ec26ff386ba02p-1; 0x1.79a44ace0d6eep-1; -0x1.e27a3e867ac98p-2 |], 0x1.5528730409865p-38, 0x1.197cbceae1846p+0);
  ("1 uas, generic, norm 0.8", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.2edcebd3f3763p-1; 0x1.d0ca34aed574cp-2; -0x1.28e8c403fccacp-2 |], 0x1.552903b44352ap-38, 0x1.197d248b504eep+0);
  ("1 uas, generic, norm 3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.1bef1d16b43edp+1; 0x1.b3bd9163e81d7p+0; -0x1.165a37c3bcfe1p+0 |], 0x1.5528ae38e6ba8p-38, 0x1.197cb1be0e964p+0);
  ("1 mas, generic, norm 1.3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.ec26ff2225064p-1; 0x1.79a44af9b3944p-1; -0x1.e27a3e58b5f6dp-2 |], 0x1.4d29532a3c586p-28, 0x1.197c9899357c4p+0);
  ("1 mas, generic, norm 0.8", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.2edcebc63e2b4p-1; 0x1.d0ca34e48e405p-2; -0x1.28e8c3e7d2708p-2 |], 0x1.4d29531aecf09p-28, 0x1.197c9885d5fc3p+0);
  ("1 mas, generic, norm 3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.1bef1d09da488p+1; 0x1.b3bd9196455c5p+0; -0x1.165a37a955497p+0 |], 0x1.4d295377382b2p-28, 0x1.197c98a257efcp+0);
  ("1 arcsec, generic, norm 1.3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.ec26a81e433f9p-1; 0x1.79a4f57aa298bp-1; -0x1.e2798b8fc69bfp-2 |], 0x1.455a5b2ff6f9bp-18, 0x1.197c987c95825p+0);
  ("1 arcsec, generic, norm 0.8", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.2edcb63a01ffbp-1; 0x1.d0cb06be51f70p-2; -0x1.28e855e252d62p-2 |], 0x1.455a5b3001144p-18, 0x1.197c987ca31fcp+0);
  ("1 arcsec, generic, norm 3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.1beeead661dfcp+1; 0x1.b3be56526cd79p+0; -0x1.1659d0842da8cp+0 |], 0x1.455a5b2ff3de8p-18, 0x1.197c987c98931p+0);
  ("pi/2, generic, norm 1.3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.1223e5c6644b4p-1; 0x1.0c9622f2b8ce2p+0; 0x1.19a167395fc0fp-1 |], 0x1.921fb54442d19p+0, 0x1.197c987c952c4p+0);
  ("pi/2, generic, norm 0.8", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.5167422f405cap-2; 0x1.4a9166170ad66p-1; 0x1.5a9f43f7d84ffp-2 |], 0x1.921fb54442d18p+0, 0x1.197c987c952c4p+0);
  ("pi/2, generic, norm 3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.3c50ce0c4c56dp+0; 0x1.35e84fb59a28fp+1; 0x1.44f54fb85acafp+0 |], 0x1.921fb54442d18p+0, 0x1.197c987c952c4p+0);
  ("pi - 1e-9, generic, norm 1.3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.ec26ff3d0ac1dp-1; -0x1.79a44ac4ff1b8p-1; 0x1.e27a3e8ff9b40p-2 |], 0x1.921fb5421d100p+1, 0x1.197c9843403d4p+0);
  ("pi - 1e-9, generic, norm 0.8", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.2edcebd6cb8afp-1; -0x1.d0ca34a3b021ep-2; 0x1.28e8c409d4bd8p-2 |], 0x1.921fb5421d101p+1, 0x1.197c984af515dp+0);
  ("pi - 1e-9, generic, norm 3", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.1bef1d195ed24p+1; -0x1.b3bd9159751fcp+0; 0x1.165a37c93771bp+0 |], 0x1.921fb5421d101p+1, 0x1.197c97b84d74ep+0);
  ("1 uas, axis, norm 1.3", [| 0x1p+0; 0.; 0. |], [| 0x1.4cccccccccccdp+0; -0x1.2f5ffbe8c40bep-39; -0x1.a0c2056d66864p-38 |], 0x1.552844bf45540p-38, 0x1.becde5da115a9p+1);
  ("1 uas, axis, norm 0.8", [| 0x1p+0; 0.; 0. |], [| 0x1.999999999999ap-1; -0x1.7562711e7b225p-40; -0x1.007765cd2b665p-38 |], 0x1.552844bf45540p-38, 0x1.becde5da115a9p+1);
  ("1 uas, axis, norm 3", [| 0x1p+0; 0.; 0. |], [| 0x1.8000000000000p+1; -0x1.5e0c4a0c93703p-38; -0x1.e0dfdee0b15fdp-37 |], 0x1.552844bf45540p-38, 0x1.becde5da115a9p+1);
  ("1 mas, axis, norm 1.3", [| 0x1p+0; 0.; 0. |], [| 0x1.4cccccccccccdp+0; -0x1.2843bc014f73ap-29; -0x1.96fd794cd61f1p-28 |], 0x1.4d295322c9b40p-28, 0x1.becde5da115a9p+1);
  ("1 mas, axis, norm 0.8", [| 0x1p+0; 0.; 0. |], [| 0x1.999999999999ap-1; -0x1.6ca22277c43f8p-30; -0x1.f4e932d4b8c3dp-29 |], 0x1.4d295322c9b40p-28, 0x1.becde5da115a9p+1);
  ("1 mas, axis, norm 3", [| 0x1p+0; 0.; 0. |], [| 0x1.8000000000000p+1; -0x1.55d8005047fb9p-28; -0x1.d59a9fa76d379p-27 |], 0x1.4d295322c9b41p-28, 0x1.becde5da115a9p+1);
  ("1 arcsec, axis, norm 1.3", [| 0x1p+0; 0.; 0. |], [| 0x1.4cccccccbc007p+0; -0x1.2152259942b8cp-19; -0x1.8d738875026a7p-18 |], 0x1.455a5b2ff8f9ep-18, 0x1.becde5da115a9p+1);
  ("1 arcsec, axis, norm 0.8", [| 0x1p+0; 0.; 0. |], [| 0x1.9999999984ecep-1; -0x1.641655a8efa85p-20; -0x1.e92bbba3b4344p-19 |], 0x1.455a5b2ff8f9dp-18, 0x1.becde5da115a9p+1);
  ("1 arcsec, axis, norm 3", [| 0x1p+0; 0.; 0. |], [| 0x1.7fffffffec9e1p+1; -0x1.4dd4f04e60adcp-18; -0x1.ca98ffe978f0fp-17 |], 0x1.455a5b2ff8f9dp-18, 0x1.becde5da115a9p+1);
  ("pi/2, axis, norm 1.3", [| 0x1p+0; 0.; 0. |], [| 0x1.e8949cccc9359p-204; -0x1.c74c1644d0f8cp-2; -0x1.38bacde4f03b8p+0 |], 0x1.921fb54442d18p+0, 0x1.becde5da115a9p+1);
  ("pi/2, axis, norm 0.8", [| 0x1p+0; 0.; 0. |], [| 0x1.2caa391b8f837p-204; -0x1.182ed2a080991p-2; -0x1.80e5e9b74f0e3p-1 |], 0x1.921fb54442d18p+0, 0x1.becde5da115a9p+1);
  ("pi/2, axis, norm 3", [| 0x1p+0; 0.; 0. |], [| 0x1.19df9589d68b3p-202; -0x1.06abe576788f8p+0; -0x1.68d78b1bda1d5p+1 |], 0x1.921fb54442d18p+0, 0x1.becde5da115a9p+1);
  ("pi - 1e-9, axis, norm 1.3", [| 0x1p+0; 0.; 0. |], [| -0x1.4cccccccccccdp+0; -0x1.e8df253b93a22p-32; -0x1.4fca7c8d2ac69p-30 |], 0x1.921fb5421d100p+1, 0x1.becde5da115a9p+1);
  ("pi - 1e-9, axis, norm 0.8", [| 0x1p+0; 0.; 0. |], [| -0x1.999999999999ap-1; -0x1.2cd816e995edap-32; -0x1.9d47fbc16fb95p-31 |], 0x1.921fb5421d100p+1, 0x1.becde5da115a9p+1);
  ("pi - 1e-9, axis, norm 3", [| 0x1p+0; 0.; 0. |], [| -0x1.8000000000000p+1; -0x1.1a0a957afc8ecp-30; -0x1.83737c0558bdbp-29 |], 0x1.921fb5421d100p+1, 0x1.becde5da115a9p+1);
]
