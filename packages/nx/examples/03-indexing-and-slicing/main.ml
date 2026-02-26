(** Select, slice, and mask — extract exactly the data you need.

    A grade book of 5 students across 4 subjects. We'll pull out individual
    scores, entire rows and columns, ranges, and use boolean masks to find top
    performers. *)

open Nx
open Nx.Infix

let () =
  (* Grade book: 5 students × 4 subjects (Math, Science, English, Art). *)
  let grades =
    create float64 [| 5; 4 |]
      [|
        88.0;
        72.0;
        95.0;
        83.0;
        45.0;
        90.0;
        67.0;
        78.0;
        92.0;
        85.0;
        91.0;
        70.0;
        76.0;
        63.0;
        80.0;
        95.0;
        60.0;
        78.0;
        55.0;
        82.0;
      |]
  in
  Printf.printf "Grade book (students × subjects):\n%s\n\n"
    (data_to_string grades);

  (* Single element: student 0's Science score (row 0, col 1). *)
  let score = item [ 0; 1 ] grades in
  Printf.printf "Student 0, Science: %.0f\n\n" score;

  (* Entire row: all of student 2's grades. *)
  let student_2 = grades.${[ I 2; A ]} in
  Printf.printf "Student 2 (all subjects): %s\n\n" (data_to_string student_2);

  (* Entire column: everyone's Math scores (column 0). *)
  let math = grades.${[ A; I 0 ]} in
  Printf.printf "Math scores (all students): %s\n\n" (data_to_string math);

  (* Range: students 1-3, first two subjects. *)
  let subset = grades.${[ R (1, 4); R (0, 2) ]} in
  Printf.printf "Students 1-3, Math & Science:\n%s\n\n" (data_to_string subset);

  (* Strided: every other student, every other subject. *)
  let strided = grades.${[ Rs (0, 5, 2); Rs (0, 4, 2) ]} in
  Printf.printf "Every other student & subject:\n%s\n\n"
    (data_to_string strided);

  (* Boolean mask: which students scored above 85 in Math? *)
  let math_scores = grades.${[ A; I 0 ]} in
  let high_math = greater_s math_scores 85.0 in
  Printf.printf "Math > 85 mask: %s\n" (data_to_string high_math);

  let top_students = compress ~axis:0 ~condition:high_math grades in
  Printf.printf "Students with Math > 85:\n%s\n\n" (data_to_string top_students);

  (* where: replace failing grades (<60) with 60. *)
  let passing =
    where (less_s grades 60.0) (full float64 [| 5; 4 |] 60.0) grades
  in
  Printf.printf "After floor at 60:\n%s\n\n" (data_to_string passing);

  (* take: select specific students by index. *)
  let picks = take ~axis:0 (create int32 [| 3 |] [| 0l; 2l; 4l |]) grades in
  Printf.printf "Students 0, 2, 4:\n%s\n" (data_to_string picks)
