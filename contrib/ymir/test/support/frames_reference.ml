(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Written by test/gen/frames.py; do not edit. *)

[@@@ocamlformat "disable"]

(* Each fixed frame's orientation from ICRS at 60 digits, row-major, as
   double-doubles (hi, lo): hi is the frame's literal. *)
let orientations = [
  ("fk5_j2000", [| (0x1.fffffffffffc0p-1, -0x1.1ba4fdfd7b605p-55); (-0x1.dcd657fd3c8f3p-24, 0x1.b36bbfb503b2bp-78); (-0x1.7af89e61cb788p-25, -0x1.6b4957576732bp-80); (0x1.dcd6592ff199bp-24, -0x1.f42d592f659d5p-80); (0x1.fffffffffff9fp-1, -0x1.b8f9ae4ae72ffp-55); (0x1.9e5e984f985f0p-24, -0x1.d973090f965a9p-78); (0x1.7af8985a25caap-25, -0x1.6fb9889ddaf00p-79); (-0x1.9e5e99b08a26ep-24, -0x1.4640177a27d30p-78); (0x1.fffffffffffcdp-1, 0x1.4247c7d5c0841p-55) |]);
  ("galactic", [| (-0x1.c18a642acb4d6p-5, 0x1.bccaf418c7608p-59); (-0x1.bf332573582edp-1, -0x1.89e84c54acb94p-56); (-0x1.ef727241c3f7cp-2, -0x1.b3fb9b40b3e54p-58); (0x1.f9f7d2657bcdep-2, 0x1.9e0c02dc1d014p-56); (-0x1.c7816b23e14f8p-2, 0x1.1b746cb8f75cdp-57); (0x1.7e7474ed9dd07p-1, 0x1.da435541623edp-55); (-0x1.bc3ebccbc409ep-1, 0x1.64f78b976b50fp-55); (-0x1.95a910cffa753p-3, -0x1.542ed2e66db7ap-57); (0x1.d2ed6938b6c7cp-2, 0x1.601caef1f790fp-57) |]);
  ("ecliptic_j2000", [| (0x1.fffffffffffccp-1, 0x1.a5c9cc2915adep-56); (-0x1.30037d6267513p-24, 0x1.d924a8a29f5cbp-79); (0x1.5a03026ad0b89p-24, -0x1.72123ca8e488fp-78); (0x1.1a9546e9bfe11p-25, -0x1.62086d5525767p-79); (0x1.d5c037bd4c600p-1, 0x1.9b56692c2d1f6p-57); (0x1.9752da8eda502p-2, 0x1.2944502db7682p-60); (-0x1.b663a4c38f2b3p-24, 0x1.b1a07c450e34ep-78); (-0x1.9752da8eda4a9p-2, -0x1.20a1bc90ec96ep-58); (0x1.d5c037bd4c5e6p-1, -0x1.c87b99d378cfcp-55) |]);
  ("supergalactic", [| (0x1.80040eb2b69d7p-2, 0x1.590d952456eb1p-56); (0x1.5d8d3425023aap-2, -0x1.eebcc779b509bp-57); (0x1.b9485c01c6acep-1, 0x1.694d85d187e14p-60); (-0x1.cbf0a8a466cb9p-1, -0x1.f0fa6dee93c51p-55); (-0x1.88192eac0df82p-4, -0x1.8baaf58837802p-61); (0x1.b71371ddc875fp-2, -0x1.e585f7adc79d7p-56); (0x1.d4bc660a11819p-3, 0x1.f3564f716ea1bp-58); (-0x1.debe4dca298bbp-1, -0x1.78e4705f82ab4p-56); (0x1.153fa3c392c72p-2, -0x1.7e4c0e881471bp-56) |]);
]

(* pyerfa's ecm06 at J2000 TT, row-major. *)
let ecm06_j2000 = [| 0x1.fffffffffffcbp-1; -0x1.30037d6267514p-24; 0x1.5a03026ad0b89p-24; 0x1.1a9546e9bfe15p-25; 0x1.d5c037bd4c5ffp-1; 0x1.9752da8eda502p-2; -0x1.b663a4c38f2b4p-24; -0x1.9752da8eda4a9p-2; 0x1.d5c037bd4c5e5p-1 |]

(* ICRS (ra, dec) in degrees and astropy's Galactic unit vector. *)
let astropy_galactic = [
  (0., 0., [| -0x1.c18a98671b359p-5; 0x1.f9f7d3058d094p-2; -0x1.bc3ebc6954ab9p-1 |]);
  (0x1.4e883126e978dp+6, 0x1.603b645a1cac1p+4, [| -0x1.fbc849e7ede47p-1; -0x1.43cd95786030ap-4; -0x1.9cd1916ed3b78p-4 |]);
  (0x1.0a67ae147ae14p+8, -0x1.cef9db22d0e56p+4, [| 0x1.fffffffff5619p-1; 0x1.6ff006356409ap-19; 0x1.88ef5d54ca385p-20 |]);
  (0x1.81b80dc33721dp+7, 0x1.b20d4fdf3b646p+4, [| 0x1.b097e11b1a3e3p-25; -0x1.2e58c6fc0233fp-26; 0x1.ffffffffffff3p-1 |]);
  (0x1.55e81450efdcap+3, 0x1.4a274299d883cp+5, [| -0x1.ecef5e7387f80p-2; 0x1.976123c86b5a0p-1; -0x1.7882499ffb511p-2 |]);
  (0x1.2bde353f7ced9p+8, 0x1.45df3b645a1cbp+5, [| 0x1.e6672494c5695p-3; 0x1.eeb157709ba30p-1; 0x1.9ac37f9bc8ef7p-4 |]);
  (0x1.2c00000000000p+7, -0x1.e000000000000p+5, [| 0x1.cb9aa58a9c1b3p-3; -0x1.f1b2fb169cce7p-1; -0x1.19677845298f7p-4 |]);
  (0x1p+0, 0x1.6600000000000p+6, [| -0x1.f00e04b2034f6p-2; 0x1.809ceb653d9d0p-1; 0x1.cb20620203095p-2 |]);
]

(* ICRS (ra, dec) in degrees and astropy's barycentric mean ecliptic at J2000 unit vector. *)
let astropy_ecliptic = [
  (0., 0., [| 0x1.fffffffffffccp-1; 0x1.1a9546e9bfe15p-25; -0x1.b663a4c38f2b5p-24 |]);
  (0x1.4e883126e978dp+6, 0x1.603b645a1cac1p+4, [| 0x1.a51d13f304becp-4; 0x1.fd27d0010953ap-1; -0x1.721d23120922dp-6 |]);
  (0x1.0a67ae147ae14p+8, -0x1.cef9db22d0e56p+4, [| -0x1.c18a5ecdd2508p-5; -0x1.fcd63137270bap-1; -0x1.8b2813a239ba8p-4 |]);
  (0x1.81b80dc33721dp+7, 0x1.b20d4fdf3b646p+4, [| -0x1.bc3ebb17c7458p-1; -0x1.70c99ed3a8a10p-12; 0x1.fd142d3758450p-2 |]);
  (0x1.55e81450efdcap+3, 0x1.4a274299d883cp+5, [| 0x1.7a283ade3e1ebp-1; 0x1.8f9744044a607p-2; 0x1.1977399a678b2p-1 |]);
  (0x1.2bde353f7ced9p+8, 0x1.45df3b645a1cbp+5, [| 0x1.826ae3852814ap-2; -0x1.5f8c3a313a004p-2; 0x1.b85c72475c506p-1 |]);
  (0x1.2c00000000000p+7, -0x1.e000000000000p+5, [| -0x1.bb67b46426f60p-2; -0x1.d78244cc17113p-4; -0x1.c9bb4a5a86a98p-1 |]);
  (0x1p+0, 0x1.6600000000000p+6, [| 0x1.1de9030e468f8p-7; 0x1.9773835ec243bp-2; 0x1.d5b3b2b9c188cp-1 |]);
]

(* ERFA's t_s2c at lon 3.0123, lat -0.999, rounded once. *)
let s2c = [| -0x1.12c0be5a7e447p-1; 0x1.1dc84ffd986ffp-4; -0x1.ae8e6949c824cp-1 |]

(* A label, two float64 vectors a and b, and the separation and position
   angle of b from a, from the vectors' exact values, rounded once. *)
let pairs = [
  ("1 uas, generic", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.7a9426c8f053cp-1; 0x1.227e60ed4568fp-1; -0x1.7322f504fbfd7p-2 |], 0x1.5528006116b7ap-38, 0x1.197cd1f7ba4c9p+0);
  ("1 mas, generic", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| 0x1.7a9426b7cdb61p-1; 0x1.227e610ed8e83p-1; -0x1.7322f4e1c70cap-2 |], 0x1.4d29530e03429p-28, 0x1.197c9883d3206p+0);
  ("pi/2, generic", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.a5c112bb1073cp-2; 0x1.9d35bf9ccd8bfp-1; 0x1.b14714f5ce63fp-2 |], 0x1.921fb54442d18p+0, 0x1.197c987c952c4p+0);
  ("pi - 1e-9, generic", [| 0x1.7a9426c8f4b7ep-1; 0x1.227e60ed3cce5p-1; -0x1.7322f50505031p-2 |], [| -0x1.7a9426cc7e6dbp-1; -0x1.227e60e64e153p-1; 0x1.7322f50c49ecep-2 |], 0x1.921fb5421d101p+1, 0x1.197c9876f1b87p+0);
  ("1 uas, axis", [| 0x1p+0; 0.; 0. |], [| 0x1p+0; -0x1.d2bb0d6619eafp-40; -0x1.40953f40763fep-38 |], 0x1.552844bf45540p-38, 0x1.becde5da115a9p+1);
  ("1 mas, axis", [| 0x1p+0; 0.; 0. |], [| 0x1p+0; -0x1.c7caab15b54f6p-30; -0x1.3911bfc4f37a6p-28 |], 0x1.4d295322c9b41p-28, 0x1.becde5da115a9p+1);
  ("pi/2, axis", [| 0x1p+0; 0.; 0. |], [| 0x1.77d4c76273645p-204; -0x1.5e3a8748a0bf5p-2; -0x1.e11f642522d1cp-1 |], 0x1.921fb54442d18p+0, 0x1.becde5da115a9p+1);
  ("pi - 1e-9, axis", [| 0x1p+0; 0.; 0. |], [| -0x1p+0; -0x1.780e1ca3fb690p-32; -0x1.024cfd58e5d3dp-30 |], 0x1.921fb5421d100p+1, 0x1.becde5da115a9p+1);
  ("1 uas, near the north pole", [| -0x1.fe55f4c7d8001p-21; 0x1.88ec90b6bf0b5p-20; 0x1.fffffffffca69p-1 |], [| -0x1.fe55b22158288p-21; 0x1.88ec423461a9dp-20; 0x1.fffffffffca69p-1 |], 0x1.552844bf354c6p-38, 0x1.657184ae60a2ep-3);
  ("1 mas, near the north pole", [| -0x1.fe55f4c7d8001p-21; 0x1.88ec90b6bf0b5p-20; 0x1.fffffffffca69p-1 |], [| -0x1.fd519a6475f3ep-21; 0x1.87b9e379fa234p-20; 0x1.fffffffffcab4p-1 |], 0x1.4d295322c9b4fp-28, 0x1.657184ae74419p-3);
  ("pi/2, near the north pole", [| -0x1.fe55f4c7d8001p-21; 0x1.88ec90b6bf0b5p-20; 0x1.fffffffffca69p-1 |], [| 0x1.901bd2298c62ep-2; -0x1.d74c6982c3a2cp-1; 0x1.cd63fbc645e53p-20 |], 0x1.921fb54442d18p+0, 0x1.657184ae74487p-3);
  ("pi - 1e-9, near the north pole", [| -0x1.fe55f4c7d8001p-21; 0x1.88ec90b6bf0b5p-20; 0x1.fffffffffca69p-1 |], [| 0x1.fe8ba868f91b9p-21; -0x1.892bd2680faeep-20; -0x1.fffffffffca5ap-1 |], 0x1.921fb5421d100p+1, 0x1.657184ae742adp-3);
  ("1 uas, near the south pole", [| 0x1.5140a7e71a972p-25; -0x1.6d33491600044p-23; -0x1.fffffffffff77p-1 |], [| 0x1.5138dbd97a434p-25; -0x1.6d351a6c6247bp-23; -0x1.fffffffffff77p-1 |], 0x1.552844bf469e3p-38, 0x1.4f1a6c638d176p+2);
  ("1 mas, near the south pole", [| 0x1.5140a7e71a972p-25; -0x1.6d33491600044p-23; -0x1.fffffffffff77p-1 |], [| 0x1.32cb92acd308dp-25; -0x1.744d0285d7686p-23; -0x1.fffffffffff73p-1 |], 0x1.4d295322c9b3cp-28, 0x1.4f1a6c638d03ep+2);
  ("pi/2, near the south pole", [| 0x1.5140a7e71a972p-25; -0x1.6d33491600044p-23; -0x1.fffffffffff77p-1 |], [| -0x1.7673fe0c86992p-1; -0x1.5d2ee398c9be8p-1; 0x1.76ce7d8722e87p-24 |], 0x1.921fb54442d18p+0, 0x1.4f1a6c638d03fp+2);
  ("pi - 1e-9, near the south pole", [| 0x1.5140a7e71a972p-25; -0x1.6d33491600044p-23; -0x1.fffffffffff77p-1 |], [| -0x1.5788eb6f5cd93p-25; 0x1.6bbc5a5984a2bp-23; 0x1.fffffffffff78p-1 |], 0x1.921fb5421d100p+1, 0x1.4f1a6c638d03ap+2);
  ("1 mas from the north pole", [| 0.; 0.; 0x1p+0 |], [| 0x1.2086b8cebfea5p-28; 0x1.4d295322c9b41p-29; 0x1p+0 |], 0x1.4d295322c9b41p-28, 0x1.4f1a6c638d03fp+1);
  ("pi/2 from the north pole", [| 0.; 0.; 0x1p+0 |], [| -0x1.5e3a8748a0bf5p-2; -0x1.e11f642522d1cp-1; 0. |], 0x1.921fb54442d18p+0, 0x1.43eee03e1961bp+2);
  ("1 deg from the north pole", [| 0.; 0.; 0x1p+0 |], [| -0x1.8d3957b96cce7p-9; 0x1.19989d7383b69p-6; 0x1.ffec097f5af8ap-1 |], 0x1.1df46a2529d39p-6, 0x1.657184ae74487p+0);
  ("1 mas from the south pole", [| 0.; 0.; -0x1p+0 |], [| 0x1.2086b8cebfea5p-28; 0x1.4d295322c9b41p-29; -0x1p+0 |], 0x1.4d295322c9b41p-28, 0x1.0c152382d7366p-1);
  ("pi/2 from the south pole", [| 0.; 0.; -0x1p+0 |], [| -0x1.5e3a8748a0bf5p-2; -0x1.e11f642522d1cp-1; 0. |], 0x1.921fb54442d18p+0, 0x1.1740afa84ad8ap+2);
  ("1 deg from the south pole", [| 0.; 0.; -0x1p+0 |], [| -0x1.8d3957b96cce7p-9; 0x1.19989d7383b69p-6; -0x1.ffec097f5af8ap-1 |], 0x1.1df46a2529d39p-6, 0x1.becde5da115a9p+0);
]
