"""
Given the result lists from using two differnt fps settings, compute the accuracy. 
"""
gt_list = [(0, 36.45453311166528, 40.123333333333335), (1, 0.0, 4.266843853820598), (1, 0.0, 5.233550664451827), (1, 0.0, 5.866910299003322), (1, 0.0, 6.766947674418605), (1, 0.0, 7.166964285714286), (1, 0.0, 8.600357142857144), (1, 0.0, 9.433725083056478), (1, 0.0, 9.967080564784053), (1, 0.0, 11.600481727574751), (1, 0.0, 12.600523255813954), (1, 0.0, 12.633857973421927), (1, 1.3667234219269104, 5.233550664451827), (1, 1.3667234219269104, 5.866910299003322), (1, 1.3667234219269104, 6.766947674418605), (1, 1.3667234219269104, 7.166964285714286), (1, 1.3667234219269104, 8.600357142857144), (1, 1.3667234219269104, 9.433725083056478), (1, 1.3667234219269104, 9.967080564784053), (1, 1.3667234219269104, 11.600481727574751), (1, 1.3667234219269104, 12.600523255813954), (1, 1.3667234219269104, 12.633857973421927), (1, 1.3667234219269104, 12.700527408637875), (1, 10.333762458471762, 12.700527408637875), (1, 10.333762458471762, 14.467267441860466), (1, 10.333762458471762, 15.167296511627908), (1, 10.333762458471762, 16.000664451827245), (1, 10.333762458471762, 16.500685215946845), (1, 10.333762458471762, 17.500726744186046), (1, 10.333762458471762, 18.067416943521597), (1, 10.333762458471762, 18.634107142857143), (1, 10.333762458471762, 19.96749584717608), (1, 10.333762458471762, 20.500851328903654), (1, 10.333762458471762, 20.934202657807308), (1, 10.333762458471762, 21.000872093023258), (1, 10.333762458471762, 21.800905315614617), (1, 10.333762458471762, 22.434264950166114), (1, 10.333762458471762, 22.767612126245847), (1, 10.333762458471762, 25.134377076411962), (1, 16.96737126245847, 19.96749584717608), (1, 16.96737126245847, 20.500851328903654), (1, 16.96737126245847, 20.934202657807308), (1, 16.96737126245847, 21.000872093023258), (1, 16.96737126245847, 21.800905315614617), (1, 16.96737126245847, 22.434264950166114), (1, 16.96737126245847, 22.767612126245847), (1, 16.96737126245847, 23.734318936877077), (1, 16.96737126245847, 23.767653654485052), (1, 16.96737126245847, 24.134335548172757), (1, 16.96737126245847, 24.701025747508307), (1, 16.96737126245847, 25.134377076411962), (1, 16.96737126245847, 25.367720099667775), (1, 16.96737126245847, 26.801112956810634), (1, 16.96737126245847, 27.534476744186048), (1, 16.96737126245847, 27.63448089700997), (1, 16.96737126245847, 27.73448504983389), (1, 16.96737126245847, 27.86782392026578), (1, 16.96737126245847, 28.667857142857144), (1, 16.96737126245847, 29.667898671096346), (1, 16.96737126245847, 30.334593023255817), (1, 17.967412790697676, 20.934202657807308), (1, 17.967412790697676, 21.800905315614617), (1, 17.967412790697676, 22.434264950166114), (1, 17.967412790697676, 22.767612126245847), (1, 17.967412790697676, 23.734318936877077), (1, 17.967412790697676, 23.767653654485052), (1, 17.967412790697676, 24.134335548172757), (1, 17.967412790697676, 24.701025747508307), (1, 17.967412790697676, 25.134377076411962), (1, 17.967412790697676, 25.367720099667775), (1, 17.967412790697676, 26.801112956810634), (1, 17.967412790697676, 27.534476744186048), (1, 17.967412790697676, 27.63448089700997), (1, 17.967412790697676, 27.73448504983389), (1, 17.967412790697676, 27.86782392026578), (1, 17.967412790697676, 28.667857142857144), (1, 17.967412790697676, 29.667898671096346), (1, 17.967412790697676, 29.867906976744187), (1, 17.967412790697676, 30.334593023255817), (1, 24.93436877076412, 27.73448504983389), (1, 24.93436877076412, 27.86782392026578), (1, 24.93436877076412, 29.667898671096346), (1, 24.93436877076412, 29.867906976744187), (1, 24.93436877076412, 31.334634551495018), (1, 24.93436877076412, 31.66798172757475), (1, 24.93436877076412, 32.501349667774086), (1, 24.93436877076412, 35.334800664451826), (1, 24.93436877076412, 35.96816029900332), (1, 24.93436877076412, 36.168168604651164), (1, 24.93436877076412, 36.334842192691035), (1, 24.93436877076412, 36.93486710963455), (1, 24.93436877076412, 36.968201827242524), (1, 24.93436877076412, 40.135000000000005), (1, 24.93436877076412, 40.135000000000005), (2, 7.565094208922139, 11.131019673039622), (2, 7.565094208922139, 12.130811859240787), (2, 7.565094208922139, 13.230583264062068), (2, 7.565094208922139, 13.4305417013023), (2, 7.565094208922139, 13.730479357162649), (2, 7.565094208922139, 15.363473261291217), (2, 7.565094208922139, 15.396799667497922), (2, 7.565094208922139, 15.69673732335827), (2, 7.565094208922139, 16.529897478525907), (2, 7.565094208922139, 17.896280133000833), (2, 7.565094208922139, 17.896280133000833), (2, 7.565094208922139, 17.929606539207537), (2, 7.565094208922139, 18.962725131615407), (2, 7.565094208922139, 20.062496536436687), (2, 7.565094208922139, 23.195178719867), (2, 27.327653089498476, 33.02646855084511), (2, 27.327653089498476, 34.392851205320035), (2, 27.327653089498476, 35.89253948462178), (2, 27.327653089498476, 38.4586727625381), (2, 27.327653089498476, 38.558651981158214), (2, 27.327653089498476, 38.991895261845386), (2, 27.327653089498476, 39.02522166805209), (2, 27.327653089498476, 39.79172901080632), (2, 27.327653089498476, 40.09166666666667), (4, 0.03324793388429752, 4.189239669421488), (4, 0.03324793388429752, 4.787702479338843), (4, 0.03324793388429752, 5.5191570247933885), (4, 0.03324793388429752, 5.618900826446281), (4, 0.03324793388429752, 6.948818181818182), (4, 0.03324793388429752, 7.24804958677686), (4, 0.03324793388429752, 7.281297520661157), (4, 0.03324793388429752, 8.112495867768596), (4, 0.03324793388429752, 8.611214876033058), (4, 0.03324793388429752, 9.242925619834711), (4, 0.03324793388429752, 9.808140495867768), (4, 0.03324793388429752, 10.539595041322315), (4, 0.03324793388429752, 11.769768595041322), (4, 2.2608595041322315, 5.5191570247933885), (4, 2.2608595041322315, 5.618900826446281), (4, 2.2608595041322315, 6.948818181818182), (4, 2.2608595041322315, 7.24804958677686), (4, 2.2608595041322315, 7.281297520661157), (4, 2.2608595041322315, 8.112495867768596), (4, 2.2608595041322315, 8.611214876033058), (4, 2.2608595041322315, 9.242925619834711), (4, 2.2608595041322315, 9.808140495867768), (4, 2.2608595041322315, 10.539595041322315), (4, 2.2608595041322315, 11.769768595041322), (4, 2.2608595041322315, 15.393793388429753)]

compare_list = [(0, 36.45453311166528, 40.123333333333335), (1, 0.0, 4.266843853820598), (1, 0.0, 5.233550664451827), (1, 0.0, 5.866910299003322), (1, 0.0, 6.766947674418605), (1, 0.0, 7.166964285714286), (1, 0.0, 8.600357142857144), (1, 0.0, 9.433725083056478), (1, 0.0, 9.967080564784053), (1, 0.0, 11.600481727574751), (1, 0.0, 12.600523255813954), (1, 0.0, 12.633857973421927), (1, 1.3667234219269104, 5.233550664451827), (1, 1.3667234219269104, 5.866910299003322), (1, 1.3667234219269104, 6.766947674418605), (1, 1.3667234219269104, 7.166964285714286), (1, 1.3667234219269104, 8.600357142857144), (1, 1.3667234219269104, 9.433725083056478), (1, 1.3667234219269104, 9.967080564784053), (1, 1.3667234219269104, 11.600481727574751), (1, 1.3667234219269104, 12.600523255813954), (1, 1.3667234219269104, 12.633857973421927), (1, 1.3667234219269104, 12.700527408637875), (1, 10.333762458471762, 12.700527408637875), (1, 10.333762458471762, 14.467267441860466), (1, 10.333762458471762, 15.167296511627908), (1, 10.333762458471762, 16.000664451827245), (1, 10.333762458471762, 16.500685215946845), (1, 10.333762458471762, 17.500726744186046), (1, 10.333762458471762, 18.067416943521597), (1, 10.333762458471762, 18.634107142857143), (1, 10.333762458471762, 19.96749584717608), (1, 10.333762458471762, 20.500851328903654), (1, 10.333762458471762, 20.934202657807308), (1, 10.333762458471762, 21.000872093023258), (1, 10.333762458471762, 21.800905315614617), (1, 10.333762458471762, 22.434264950166114), (1, 10.333762458471762, 22.767612126245847), (1, 10.333762458471762, 25.134377076411962), (1, 16.96737126245847, 19.96749584717608), (1, 16.96737126245847, 20.500851328903654), (1, 16.96737126245847, 20.934202657807308), (1, 16.96737126245847, 21.000872093023258), (1, 16.96737126245847, 21.800905315614617), (1, 16.96737126245847, 22.434264950166114), (1, 16.96737126245847, 22.767612126245847), (1, 16.96737126245847, 23.734318936877077), (1, 16.96737126245847, 23.767653654485052), (1, 16.96737126245847, 24.134335548172757), (1, 16.96737126245847, 24.701025747508307), (1, 16.96737126245847, 25.134377076411962), (1, 16.96737126245847, 25.367720099667775), (1, 16.96737126245847, 26.801112956810634), (1, 16.96737126245847, 27.534476744186048), (1, 16.96737126245847, 27.63448089700997), (1, 16.96737126245847, 27.73448504983389), (1, 16.96737126245847, 27.86782392026578), (1, 16.96737126245847, 28.667857142857144), (1, 16.96737126245847, 29.667898671096346), (1, 16.96737126245847, 30.334593023255817), (1, 17.967412790697676, 20.934202657807308), (1, 17.967412790697676, 21.800905315614617), (1, 17.967412790697676, 22.434264950166114), (1, 17.967412790697676, 22.767612126245847), (1, 17.967412790697676, 23.734318936877077), (1, 17.967412790697676, 23.767653654485052), (1, 17.967412790697676, 24.134335548172757), (1, 17.967412790697676, 24.701025747508307), (1, 17.967412790697676, 25.134377076411962), (1, 17.967412790697676, 25.367720099667775), (1, 17.967412790697676, 26.801112956810634), (1, 17.967412790697676, 27.534476744186048), (1, 17.967412790697676, 27.63448089700997), (1, 17.967412790697676, 27.73448504983389), (1, 17.967412790697676, 27.86782392026578), (1, 17.967412790697676, 28.667857142857144), (1, 17.967412790697676, 29.667898671096346), (1, 17.967412790697676, 29.867906976744187), (1, 17.967412790697676, 30.334593023255817), (1, 24.93436877076412, 27.73448504983389), (1, 24.93436877076412, 27.86782392026578), (1, 24.93436877076412, 29.667898671096346), (1, 24.93436877076412, 29.867906976744187), (1, 24.93436877076412, 31.334634551495018), (1, 24.93436877076412, 31.66798172757475), (1, 24.93436877076412, 32.501349667774086), (1, 24.93436877076412, 35.334800664451826), (1, 24.93436877076412, 35.96816029900332), (1, 24.93436877076412, 36.168168604651164), (1, 24.93436877076412, 36.334842192691035), (1, 24.93436877076412, 36.93486710963455), (1, 24.93436877076412, 36.968201827242524), (1, 24.93436877076412, 40.135000000000005), (1, 24.93436877076412, 40.135000000000005), (2, 7.565094208922139, 10.831082017179273), (2, 7.565094208922139, 12.130811859240787), (2, 7.565094208922139, 12.76401357716819), (2, 7.565094208922139, 13.4305417013023), (2, 7.565094208922139, 13.730479357162649), (2, 7.565094208922139, 15.363473261291217), (2, 7.565094208922139, 15.396799667497922), (2, 7.565094208922139, 15.463452479911332), (2, 7.565094208922139, 16.529897478525907), (2, 7.565094208922139, 17.59634247714048), (2, 7.565094208922139, 17.59634247714048), (2, 7.565094208922139, 17.59634247714048), (2, 27.327653089498476, 32.72653089498476), (2, 27.327653089498476, 33.02646855084511), (2, 27.327653089498476, 34.792768079800496), (2, 27.327653089498476, 34.85942089221391), (2, 27.327653089498476, 36.49241479634248), (2, 27.327653089498476, 38.558651981158214), (2, 27.327653089498476, 38.991895261845386), (2, 27.327653089498476, 39.02522166805209), (2, 27.327653089498476, 39.79172901080632), (2, 27.327653089498476, 40.09166666666667), (4, 0.03324793388429752, 4.189239669421488), (4, 0.03324793388429752, 4.787702479338843), (4, 0.03324793388429752, 5.5191570247933885), (4, 0.03324793388429752, 5.618900826446281), (4, 0.03324793388429752, 6.948818181818182), (4, 0.03324793388429752, 7.24804958677686), (4, 0.03324793388429752, 7.281297520661157), (4, 0.03324793388429752, 8.112495867768596), (4, 0.03324793388429752, 8.611214876033058), (4, 0.03324793388429752, 9.242925619834711), (4, 0.03324793388429752, 9.808140495867768), (4, 0.03324793388429752, 10.539595041322315), (4, 0.03324793388429752, 11.769768595041322), (4, 2.2608595041322315, 5.5191570247933885), (4, 2.2608595041322315, 5.618900826446281), (4, 2.2608595041322315, 6.948818181818182), (4, 2.2608595041322315, 7.24804958677686), (4, 2.2608595041322315, 7.281297520661157), (4, 2.2608595041322315, 8.112495867768596), (4, 2.2608595041322315, 8.611214876033058), (4, 2.2608595041322315, 9.242925619834711), (4, 2.2608595041322315, 9.808140495867768), (4, 2.2608595041322315, 10.539595041322315), (4, 2.2608595041322315, 11.769768595041322)]

TP = 0

for tup in compare_list:
    for gt_tup in gt_list:
        if tup[0] == gt_tup[0] and abs(tup[1] - gt_tup[1]) <= 0.08 and abs(tup[2] - gt_tup[2]) <= 0.08:
            TP += 1
            break

FP = len(compare_list) - TP
FN = len(gt_list) - TP
print(TP, FP, FN)
precision = TP * 1.0 / (TP + FP)
recall = TP * 1.0 / (TP + FN)
print(precision, recall)
print(2 * precision * recall / (precision + recall))

"""
fps 15:
TP, FP, FN: 50 161 93
precision, recall: 0.23696682464454977 0.34965034965034963
F1: 0.28248587570621475

skip 1 frame for every 5 frames, 5 videos:
TP, FP, FN: 91 63 52
precision, recall: 0.5909090909090909 0.6363636363636364
F1: 0.6127946127946128

skip 1 frame for every 10 frames, 5 videos:
106 42 37
0.7162162162162162 0.7412587412587412
0.7285223367697594
"""