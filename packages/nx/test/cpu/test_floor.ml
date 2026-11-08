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

let holds fs =
  let bits w =
    List.fold_left
      (fun acc (_, w', b) -> if w = w' then acc lor (1 lsl b) else acc)
      0 fs
  in
  Nx_cpu.floor_holds ~ecx1:(bits Ecx1) ~ebx7:(bits Ebx7) ~xcr0:(bits Xcr0)

let () =
  exit
    (run "Nx_cpu floor"
       [
         test "a processor of every feature runs nx.cpu" (fun () ->
             equal bool true (holds features));
         cases ~id:(fun (n, _, _) -> n)
           "a processor without one feature does not run nx.cpu:" features
           (fun f ->
             equal bool false (holds (List.filter (( != ) f) features)));
       ])
