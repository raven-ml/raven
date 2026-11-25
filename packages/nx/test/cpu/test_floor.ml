(* What nx.cpu requires on x86-64: AVX2, FMA, F16C and BMI2, which its units
   are compiled for, with the SSE4.2 and POPCNT the C compiler takes beside
   them, whose instructions (CRC32, POPCNT) have no VEX form that AVX would
   gate, and the XMM and YMM state the system saves (Intel SDM vol. 1,
   §14.3). Bits from the SDM: CPUID.01H:ECX and CPUID.07H.00H:EBX. *)

open Windtrap

type word = Ecx1 | Ebx7 | Xcr0

let features =
  [
    ("FMA", Ecx1, 12);
    ("SSE4_2", Ecx1, 20);
    ("POPCNT", Ecx1, 23);
    ("OSXSAVE", Ecx1, 27);
    ("AVX", Ecx1, 28);
    ("F16C", Ecx1, 29);
    ("AVX2", Ebx7, 5);
    ("BMI2", Ebx7, 8);
    ("XMM state", Xcr0, 1);
    ("YMM state", Xcr0, 2);
  ]

(* A processor's words: CPUID.01H:ECX, CPUID.07H.00H:EBX and XCR0. *)
type words = { ecx1 : int; ebx7 : int; xcr0 : int }

let words =
  Testable.structural ~pp:(fun ppf w ->
      Format.fprintf ppf "{ ecx1 = 0x%08x; ebx7 = 0x%08x; xcr0 = 0x%08x }"
        w.ecx1 w.ebx7 w.xcr0)

(* The words of a processor of [fs] alone, every other bit clear. *)
let only fs =
  let bits w =
    List.fold_left
      (fun acc (_, w', b) -> if w = w' then acc lor (1 lsl b) else acc)
      0 fs
  in
  { ecx1 = bits Ecx1; ebx7 = bits Ebx7; xcr0 = bits Xcr0 }

(* Every bit of the three 32-bit words set: a floor that demanded a bit clear
   would refuse it. *)
let every = { ecx1 = 0xFFFF_FFFF; ebx7 = 0xFFFF_FFFF; xcr0 = 0xFFFF_FFFF }

(* [w] with feature [f]'s bit clear. *)
let lacking (_, word, bit) w =
  let clear x = x land lnot (1 lsl bit) in
  match word with
  | Ecx1 -> { w with ecx1 = clear w.ecx1 }
  | Ebx7 -> { w with ebx7 = clear w.ebx7 }
  | Xcr0 -> { w with xcr0 = clear w.xcr0 }

let runs w = Nx_cpu.floor_holds ~ecx1:w.ecx1 ~ebx7:w.ebx7 ~xcr0:w.xcr0

let () =
  exit
    (run "nx_cpu.floor"
       [
         group "floor_holds"
           [
             cases ~id:fst "a processor of every feature runs nx.cpu"
               [ ("those features alone", only features); ("every bit", every) ]
               (fun (_, w) -> satisfies ~claim:"runs nx.cpu" words runs w);
             cases
               ~id:(fun (name, _, _) -> name)
               "a processor lacking one feature does not run nx.cpu" features
               (fun f ->
                 satisfies ~claim:"does not run nx.cpu" words
                   (fun w -> not (runs w))
                   (lacking f every));
           ];
       ])
