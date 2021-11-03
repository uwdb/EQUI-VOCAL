- Check the generated Datalog code: 
```
--show=transformed-datalog
```
<!-- # q1
- input: 140 frames 
- person, then car; person at edge corner, car at intersection 
- duration of person > 30, duration of car > 30
- Window size 100
- output: only fid. 420 rows
```
SouffleProf
    runtime  loadtime  savetime relations     rules    tuples generated
        64s     .340s     .000s         8        16          626K

Slowest relations to fully evaluate
 ----- Relation Table -----
   TOT_T  NREC_T   REC_T  COPY_T  LOAD_T  SAVE_T  TUPLES   READS   TUP/s    ID NAME

     63s     63s   .000s   .000s   .000s   .000s    194K    195K   3.05K    R8 q_neg
   .612s   .612s   .000s   .000s   .000s   .000s    195K    585K    318K    R6 q
   .044s   .044s   .000s   .000s   .000s   .000s     420       0   9.48K    R7 q_filtered
Slowest rules to fully evaluate
  ----- Rule Table -----
   TOT_T  NREC_T   REC_T  TUPLES   TUP/s      ID RELATION

     63s     63s   .000s    194K  3.0512    N8.1 q_neg
   .612s   .612s   .000s    195K 318.822    N6.1 q
   .044s   .044s   .000s     420 9.49281    N7.1 q_filtered
  cpu total
        64s
```
# q1.1
- Similar to q1
- for q_neg: strict subset
- output: 5467 rows
```
SouffleProf
    runtime  loadtime  savetime relations     rules    tuples generated
      3.37s     .332s     .001s         8        16          626K

Slowest relations to fully evaluate
 ----- Relation Table -----
   TOT_T  NREC_T   REC_T  COPY_T  LOAD_T  SAVE_T  TUPLES   READS   TUP/s    ID NAME

   2.36s   2.36s   .000s   .000s   .000s   .000s    189K    195K   80.2K    R8 q_neg
   .604s   .604s   .000s   .000s   .000s   .000s    195K    580K    322K    R6 q
   .044s   .044s   .000s   .000s   .000s   .001s   5.46K       0    121K    R7 q_filtered
Slowest rules to fully evaluate
  ----- Rule Table -----
   TOT_T  NREC_T   REC_T  TUPLES   TUP/s      ID RELATION

   2.36s   2.36s   .000s    189K 80.2237    N8.1 q_neg
   .604s   .604s   .000s    195K   322.9    N6.1 q
   .044s   .044s   .000s   5.46K 121.982    N7.1 q_filtered
  cpu total
      3.39s
``` -->

# q1.2
```
SouffleProf
    runtime  loadtime  savetime relations     rules    tuples generated
        19m     .332s     .016s         8        16          626K

Slowest relations to fully evaluate
 ----- Relation Table -----
   TOT_T  NREC_T   REC_T  COPY_T  LOAD_T  SAVE_T  TUPLES   READS   TUP/s    ID NAME

     19m     19m   .000s   .000s   .000s   .000s    135K    195K     117    R8 q_neg
   2.27s   2.27s   .000s   .000s   .000s   .000s    195K    526K   85.7K    R6 q
   .055s   .055s   .000s   .000s   .000s   .016s   59.5K       0   1.08M    R7 q_filtered
Slowest rules to fully evaluate
  ----- Rule Table -----
   TOT_T  NREC_T   REC_T  TUPLES   TUP/s      ID RELATION

     19m     19m   .000s    135K0.117267    N8.1 q_neg
   2.27s   2.27s   .000s    195K 85.7858    N6.1 q
   .055s   .055s   .000s   59.5K 1082.52    N7.1 q_filtered
  cpu total
        19m
```

# q2
- More constraints on object attributes.
- q_neg is the correct one.
```
SouffleProf
    runtime  loadtime  savetime relations     rules    tuples generated
      3.09s     .333s     .001s         8        16          345K

Slowest relations to fully evaluate
 ----- Relation Table -----
   TOT_T  NREC_T   REC_T  COPY_T  LOAD_T  SAVE_T  TUPLES   READS   TUP/s    ID NAME

   1.77s   1.77s   .000s   .000s   .000s   .000s   4.26K   10.1K   2.40K    R6 q
   .818s   .818s   .000s   .000s   .000s   .000s   1.57K   4.26K   1.92K    R8 q_neg
   .107s   .000s   .085s   .021s   .000s   .000s    104K    326K    971K    R3 g1seq
Slowest rules to fully evaluate
  ----- Rule Table -----
   TOT_T  NREC_T   REC_T  TUPLES   TUP/s      ID RELATION

   1.77s   1.77s   .000s   4.26K 2.40319    N6.1 q
   .818s   .818s   .000s   1.57K 1.92535    N8.1 q_neg
   .083s   .000s   .083s    100K 1205.38    C3.1 g1seq
  cpu total
      3.11s
```

# q2.1 
```
SouffleProf
    runtime  loadtime  savetime relations     rules    tuples generated
      8.40s     .334s     .000s         8        16          353K

Slowest relations to fully evaluate
 ----- Relation Table -----
   TOT_T  NREC_T   REC_T  COPY_T  LOAD_T  SAVE_T  TUPLES   READS   TUP/s    ID NAME

   6.12s   6.12s   .000s   .000s   .000s   .000s   7.66K   22.9K   1.25K    R6 q
   1.77s   1.77s   .000s   .000s   .000s   .000s   7.64K   7.66K   4.29K    R8 q_neg
   .108s   .000s   .085s   .021s   .000s   .000s    104K    326K    968K    R3 g1seq
Slowest rules to fully evaluate
  ----- Rule Table -----
   TOT_T  NREC_T   REC_T  TUPLES   TUP/s      ID RELATION

   6.12s   6.12s   .000s   7.66K 1.25148    N6.1 q
   1.77s   1.77s   .000s   7.64K 4.29729    N8.1 q_neg
   .083s   .000s   .083s    100K 1204.51    C3.1 g1seq
  cpu total
      8.42s
```

# q2.2
```
SouffleProf
    runtime  loadtime  savetime relations     rules    tuples generated
      8.32s     .351s     .000s         6        17          369K

Slowest relations to fully evaluate
 ----- Relation Table -----
   TOT_T  NREC_T   REC_T  COPY_T  LOAD_T  SAVE_T  TUPLES   READS   TUP/s    ID NAME

   4.80s   4.80s   .000s   .000s   .000s   .000s   9.66K   9.69K   2.00K    R6 q_neg
   2.91s   2.91s   .000s   .000s   .000s   .000s      30       0      10    R5 q_filtered
   .186s   .026s   .141s   .018s   .000s   .000s    104K    104K    562K    R2 g1seq
Slowest rules to fully evaluate
  ----- Rule Table -----
   TOT_T  NREC_T   REC_T  TUPLES   TUP/s      ID RELATION

   4.80s   4.80s   .000s   9.66K  2.0095    N6.1 q_neg
   2.91s   2.91s   .000s      300.0102921    N5.1 q_filtered
   .138s   .000s   .138s    100K 722.912    C2.1 g1seq
  cpu total
      8.33s
```

# q2.3
- didn't finish