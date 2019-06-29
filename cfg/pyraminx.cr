#       18               0               9
#    19 20 21         1  2  3        10 11 12
# 22 23 24 25 26   4  5  6  7  8  13 14 15 16 17
#
#                 27 28 29 30 31  
#                    32 33 34
#                       35

scheme2d
	27:0,0'1,0'0.5,0.867   28:1,0'1.5,0.866'0.5,0.867
	$29=27:1,0   $30=28:1,0   $31=27:2,0   $32=27:0.5,0.867   $33=28:0.5,0.867   $34=32:1,0   $35=27:1,1.734
	$0,1,2,3,4,5,6,7,8=35,32,33,34,27,28,29,30,31:0,0*1,-1,0,0   $9-17=0-8:3,0   $18-26=0-8:-3,0

model3d
	0-26:y=4/0,19.502  27-35:y=27/0,-90  9-17:x=13/0,120  18-26:x=26/1,-120
	camera=-90,0,0, 0,0.068,0.096

turns
	0::/27-31-35,28-30-33,29-34-32, 0-9-18,1-10-19,2-11-20,3-12-21,4-13-22,5-14-23,6-15-24,7-16-25,8-17-26
	1::/8-0-4,3-1-6,7-2-5, 13-18-27,14-20-28,15-19-32,16-23-33,17-22-35,10-21-29,11-25-30,12-24-34,9-26-31
	2::/$1,$1   3::/$0,$1   4::/$0,$1,$1   5::/$0,$0   6::/$0,$0,$1   7::/$0,$0,$1,$1   8::/$1,$0,$0   9::/$1,$0,$0,$1   10::/$1,$0,$0,$1,$1

moves
	 0:R:1,0/29-3-15,34-6-10,30-7-14,8-13-31/6-3,10-15   1:R':0,1/$0,$0/3-6,15-10
	 2:L:3,2/t0,$0,t0,t0/1-6,24-21			3:L':2,3/$2,$2/6-1,21-24
	 4:U:5,4/t0,t0,$0,t0/15-12,19-24		5:U':4,5/$4,$4/12-15,24-19
	 6:F:7,6/t1,t1,$0,t1/3-1,12-10,21-19	7:F':6,7/$6,$6/1-3,10-12,19-21
	 8:r:9,8/8-13-31/8-13					9:r':8,9/31-13-8/13-8
	10:l:11,10/4-27-26/26-4 				11:l':10,11/26-27-4/4-26
	12:f:13,12/9-0-18/0-18,9-0,18-9 		13:f':12,13/18-0-9/9-18,0-9,18-0
	14:u:15,14/17-22-35/17-22  				15:u':14,15/22-17-35/22-17

faces
	0:0-8 *0,180,0/0-8/0,0,1
	1:9-17 *0,60,0/9-17/0.866,0,-0.35
	2:18-26 *0,-60,0/18-26/-0.866,0,-0.5
	3:27-35 *0,-180,0*110,-180,0/27-35/0,1,0

algo
	0: 
		s: 0=2 8=7 5=4 35=33
		moves= 8-15
	1: 
		moves= 0-7

zero
	rrrrrrrrrbbbbbbbbboooooooooggggggggg

colors
	R:(0,0,255):red G:(64,160,64):green O:(0,106,255):orange B:(0,0,0):black
