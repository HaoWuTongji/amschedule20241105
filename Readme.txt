For instances with the same number of machines (m) and parts (n), the sets of machines and parts are consistent, meaning that we use a common "m-n.txt" file for instances with the same m and n. However, the due-date properties vary, as the due dates for the parts are specified in a separate "Due-Dates.txt" file.

The machine parameters are based on the specifications of four commonly used SLM machines, namely:

	EOS M290 (https://www.eos.info/en-us/metal-solutions/metal-printers/eos-m-290.)
	SLM 280 2.0 (https://slm280.slm-solutions.com/.)
	Concept Laser M2 Series 5 (https://www.colibriumadditive.com/printers/l-pbf-printers/m2-series-5.)
	Sisma EVEMET 200 (https://www.sisma-laser.fr/evemet-200.)

The data format is as follows, with parameter definitions and units illustrated in our paper.
==============================================================================

types_machine types_parts
m n

1 num_machine v^c v^h t^{s1} t^{s2} T^{ini} T^{fin} t^{d1} t^{d2} L W H n^l {\delta} D^p D^s t^r  
2 num_machine v^c v^h t^{s1} t^{s2} T^{ini} T^{fin} t^{d1} t^{d2} L W H n^l {\delta} D^p D^s t^r  
3 num_machine v^c v^h t^{s1} t^{s2} T^{ini} T^{fin} t^{d1} t^{d2} L W H n^l {\delta} D^p D^s t^r  

1 num_part num_orientation v a
l w h s
l w h s
l w h s
l w h s
l w h s

2 num_part num_orientation v a
l w h s
l w h s
l w h s
l w h s
l w h s

3 num_part num_orientation v a
l w h s
l w h s
l w h s
l w h s
l w h s

==============================================================================

Suppose we have an instance below :

2 2
2 3

1 1 810 1944 1800 3600 10 100 21600 28800 250 250 325 1 0.06 0.09 0.11 11
2 1 730 1650 1512 5112 10 150 8100 13500 280 280 365 2 0.03 0.13 0.18 10

0 1 5 6744.0 8607.8
57.5 24.6 18.0 1724.0
38.8 24.5 41.7 2596.0
22.1 32.0 46.0 2174.0
18.0 24.5 47.6 1489.0
42.1 28.1 36.5 2667.0

1 2 3 37635.0 17532.0
73.0 64.0 51.9 23352.0
78.3 72.7 76.4 14668.0
87.7 70.5 74.4 3396.0

==============================================================================

The illustration is as below:

types_machine=2 types_parts=2
num_machine=2 num_parts=3

machine_id=1 num_machine=1 v^c=810 v^h=1944 t^{s1}=1800 t^{s2}=3600 T^{ini}=10 T^{fin}=100 t^{d1}=21600 t^{d2}=28800 L=250 W=250 H=325 n^l=1 {\delta}=0.06 D^p=0.09 D^s=0.11 t^r=11  
machine_id=2 num_machine=1 v^c=730 v^h=1650 t^{s1}=1800 t^{s2}=3600 T^{ini}=10 T^{fin}=100 t^{d1}=21600 t^{d2}=28800 L=250 W=250 H=325 n^l=1 {\delta}=0.06 D^p=0.09 D^s=0.11 t^r=11  
 
part_id=0 num_part1=1 num_orientation=5 v=6744.0 a=8607.8
l=57.5 w=24.6 h=18.0 s=1724.0 // orientation_1
l=38.8 w=24.5 h=41.7 s=2596.0 // orientation_2
l=22.1 w=32.0 h=46.0 s=2174.0 // orientation_3
l=18.0 w=24.5 h=47.6 s=1489.0 // orientation_4
l=42.1 w=28.1 h=36.5 s=2667.0 // orientation_5

part_id=1 num_part=2 num_orientation=3 v=37635.0 a=17532.0
l=73.0 w=64.0 h=51.9 s=23352.0 // orientation_1
l=78.3 w=72.7 h=76.4 s=14668.0 // orientation_2
l=87.7 w=70.5 h=74.4 s=3396.0 // orientation_3

