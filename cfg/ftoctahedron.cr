#
#       44              53              62              71
#    41 42 43        50 51 52        59 60 61        68 69 70
# 36 37 38 39 40  45 46 47 48 49  54 55 56 57 58  63 64 65 66 67
# --------------  --------------  --------------  --------------
# 0  1  2  3  4    9 10 11 12 13  18 19 20 21 22  27 28 29 30 31
#    5  6  7         14 15 16        23 24 25        32 33 34
#       8               17              26              35

scheme2d
	0:0,0'2,0'1,1.732  1:2,0'3,1.732'1,1.732  2:2,0'4,0'3,1.732  3:4,0'5,1.732'3,1.732  4:4,0'6,0'5,1.732
	               5:1,1.732'3,1.732'2,3.464  6:3,1.732'4,3.464'2,3.464  7:3,1.732'5,1.732'4,3.464
	                              8:2,3.464'4,3.464'3,5.196
	$9-17=0-8:6,0  $18-26=0-8:12,0  $27-35=0-8:18,0
	$36-44=0-8:0,0*1,-1,0,0  $45-53=0-8:6,0*1,-1,0,0  $54-62=0-8:12,0*1,-1,0,0  $63-71=0-8:18,0*1,-1,0,0

model3d
	36-71:y=0/0,35.264  0-35:y=0/0,-35.264  63-71,27-35:x=63/0,90  36-44,0-8:x=45/0,-90  54-71,18-35:x=54/0,90
	camera= -5,40,0 ,0,0,0, -50

turns
	0::/0-9-18-27,1-10-19-28,2-11-20-29,3-12-21-30,4-13-22-31,5-14-23-32,6-15-24-33,7-16-25-34,8-17-26-35,36-45-54-63,37-46-55-64,38-47-56-65,39-48-57-66,40-49-58-67,41-50-59-68,42-51-60-69,43-52-61-70,44-53-62-71
	1::/0-40,1-39,2-38,3-37,4-36,5-43,6-42,7-41,8-44,18-58,19-57,20-56,21-55,22-54,23-61,24-60,25-59,26-62,9-67,10-66,11-65,12-64,13-63,14-70,15-69,16-68,17-71,27-49,28-48,29-47,30-46,31-45,32-52,33-51,34-50,35-53
	2::/0-17-49-44,1-15-48-42,2-14-47-43,3-10-46-39,4-9-45-40,5-16-52-41,6-12-51-37,7-11-50-38,8-13-53-36,27-22-58-63,28-21-57-64,29-25-56-68,30-24-55-69,31-26-54-71,32-20-61-65,33-19-60-66,34-23-59-70,35-18-62-67 
	3::/$0,$0          4::/$0,$0,$0  
	5::/$1,$0          6::/$1,$0,$0   7::/$1,$0,$0,$0
	8::/$2,$0          9::/$2,$0,$0  10::/$2,$0,$0,$0  
	11::/$2,$2,$2     12::/$11,$0    13::/$11,$0,$0    14::/$11,$0,$0,$0
	15::/$0,$2        16::/$15,$0    17::/$15,$0,$0    18::/$15,$0,$0,$0
	19::/$0,$0,$0,$2  20::/$19,$0    21::/$19,$0,$0    22::/$19,$0,$0,$0

moves
	 0:LD:1,0,1,	12,13	/8-0-4,5-2-7,6-1-3,9-35-36,10-33-37,14-34-38,15-30-39,17-31-40,26-67-45/8-0,4-8,0-4  1:LD':0,1,0,	12,13/$0,$0/0-8,4-0,8-4
	 2:ld:3,2,3,	14,15	/t0,$0,t0,t0,t0/0-8,36-0,8-17		 3:ld':2,3,2,	 14,15	/$2,$2/8-0,17-8,0-36
	 4:rd:5,4,5,	8,9		/t0,t0,$0,t0,t0/17-13,8-17,13-49	 5:rd':4,5,4,	 8,9	/$4,$4/13-17,49-13,17-8
	 6:RD:7,6,7,	10,11	/t0,t0,t0,$0,t0/13-17,9-13,17-9		 7:RD':6,7,6,	 10,11	/$6,$6/17-13,9-17,13-9
	 8:LU:9,8,9,	4,5		/t1,$0,t1/36-44,44-40,40-36			 9:LU':8,9,8,	 4,5	/$8,$8/44-36,36-40,40-44
	10:lu:11,10,11,	6,7		/t1,$6,t1/44-36,53-44,36-0			11:lu':10,11,10, 6,7	/$10,$10/36-44,0-36,44-53
	12:ru:13,12,13,	0,1		/t1,$4,t1/49-53,13-49,53-44			13:ru':12,13,12, 0,1	/$12,$12/53-49,44-53,49-13  
	14:RU:15,14,15, 2,3 	/t1,$2,t1/53-49,45-53,49-45			15:RU':14,15,14, 2,3	/$14,$14/49-53,45-49,53-45

algo
	0:
		hash_depth=6
		max_sol=12 max_weight=12 sol_time=15000
		s: 0=1=3=4=6=8 9=10=12=13=15=17 18=19=21=22=24=26 27=28=30=31=33=35 36=37=39=40=42=44 45=46=48=49=51=53 54=55=57=58=60=62
		w: 42=43&51=50 51=52&60=59 60=61&69=68 69=70&42=41 42=38&6=2 51=47&15=11 60=56&24=20 69=65&33=29 6=7&15=14 15=16&24=23 24=25&33=32 33=34&6=5

		p0: 0=8|0=26 0=8|0=53 1=8|1=26 1=8|1=53 3=8|3=26 3=8|3=53 4=8|4=26 4=8|4=53 6=8|6=26 6=8|6=53
		p0: 18=8|18=26 18=8|18=53 19=8|19=26 19=8|19=53 21=8|21=26 21=8|21=53 22=8|22=26 22=8|22=53 24=8|24=26 24=8|24=53
		p0: 45=8|45=26 45=8|45=53 46=8|46=26 46=8|46=53 48=8|48=26 48=8|48=53 49=8|49=26 49=8|49=53 51=8|51=26 51=8|51=53

		p1: 9=17|9=35 9=17|9=44 10=17|10=35 10=17|10=44 12=17|12=35 12=17|12=44 13=17|13=35 13=17|13=44 15=17|15=35 15=17|15=44 
		p1: 27=17|27=35 27=17|27=44 28=17|28=35 28=17|28=44 30=17|30=35 30=17|30=44 31=17|31=35 31=17|31=44 33=17|33=35 33=17|33=44 
		p1: 36=17|36=35 36=17|36=44 37=17|37=35 37=17|37=44 39=17|39=35 39=17|39=44 40=17|40=35 40=17|40=44 42=17|42=35 42=17|42=44 

		p2: 63=8|63=26 63=8|63=53 64=8|64=26 64=8|64=53 66=8|66=26 66=8|66=53 67=8|67=26 67=8|67=53 69=8|69=26 69=8|69=53
		p2: 54=17|54=35 54=17|54=44 55=17|55=35 55=17|55=44 57=17|57=35 57=17|57=44 58=17|58=35 58=17|58=44 60=17|60=35 60=17|60=44 
		p2: 0=8|0=26 0=8|0=53 1=8|1=26 1=8|1=53 3=8|3=26 3=8|3=53 4=8|4=26 4=8|4=53 6=8|6=26 6=8|6=53
	1:
		start_depth=8 max_depth=12 max_weight=12 max_sol=10000 step_time=15000
		param=6
#		param=7,0,0, 15,2.0, 99, 99,99, 99,99, 99,99,99, 99,99,99
		s: 0=1=3=4=6=8 9=10=12=13=15=17 18=19=21=22=24=26 27=28=30=31=33=35 36=37=39=40=42=44 45=46=48=49=51=53 54=55=57=58=60=62
		w: 42=43&51=50 51=52&60=59 60=61&69=68 69=70&42=41 42=38&6=2 51=47&15=11 60=56&24=20 69=65&33=29 6=7&15=14 15=16&24=23 24=25&33=32 33=34&6=5
			moves= 0-15; 0-15

zero
	wwwwwwwwwgggggggggrrrrrrrrrpppppppppbbbbbbbbbsssssssssyyyyyyyyyooooooooo

faces
	0:0-8   *0,-90,0*0,-180,0*0,-180,-45/0-8/-1,1,-1
	1:9-17  *0,-180,0*0,-270,0*0,-270,-45/9-17/-1,1,1
	2:18-26 *0,-270,0*0,-360,0*0,-360,-45/18-26/1,1,1
	3:27-35 *0,-360,0*0,-440,0*0,-440,-45/27-35/1,1,-1
	4:36-44 *0,-440,0*0,-520,0*0,-530,45/36-44/-1,-1,-1
	5:45-53 *0,-530,0*0,-620,0*0,-620,45/45-53/-1,-1,1
	6:54-62 *0,-620,0*0,-710,0*0,-710,45/54-62/1,-1,1
	7:63-71 *0,-710,0*0,-800,0*0,-800,45/63-71/1,-1,-1

colors
	R:(32,32,255):red G:(64,160,64):green B:(255,64,64):blue O:(64,128,255):orange Y:(0,200,255):yellow W:(224,224,224):white P:(255,0,255):purple S:(128,128,128):silver
