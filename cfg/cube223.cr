scheme2d
				24|2,0	25|3,0
				26|2,1	27|3,1
18|0,2	19|1,2	 0|2,2	 1|3,2	 6|4,2	 7|5,2	12|6,2	13|7,2
20|0,3	21|1,3	 2|2,3	 3|3,3	 8|4,3	 9|5,3	14|6,3	15|7,3
22|0,4	23|1,4	 4|2,4	 5|3,4	10|4,4	11|5,4	16|6,4	17|7,4
				28|2,5	29|3,5
				30|2,6	31|3,6

model3d
	12-17:x=12/0,90  18-23:x=0/0,-90  6-17:x=6/0,90  24-27:y=26/3,90  28-31:y=28/0,-90

moves
	0:R2:0,1/1-16,3-14,5-12,27-31,25-29,6-11,7-10,8-9/5-1,27-25
	1:F2:1,0/0-5,1-4,2-3,26-29,27-28,6-23,8-21,10-19/26-27,6-10
	2:U:3,2,6/0-18-12-6,1-19-13-7,24-25-27-26/1-0,7-6		3:U':2,3,6/$2,$2,$2/0-1,6-7		6:U2:6,2,3/$2,$2/1-0,7-6×2
	4:D:5,4,7/4-10-16-22,5-11-17-23,28-29-31-30/4-5,10-11	5:D':4,5,7/$4,$4,$4/5-4,11-10	7:D2:7,4,5/$4,$4/4-5,10-11×2
									
turns
	0::/0-18-12-6,1-19-13-7,2-20-14-8,3-21-15-9,4-22-16-10,5-23-17-11,24-25-27-26,28-30-31-29
	1::/0-17,1-16,2-15,3-14,4-13,5-12,6-11,7-10,8-9,19-22,18-23,20-21,24-28,25-29,26-30,27-31
	2::/$0,$0   3::/$0,$0,$0   4::/$1,$0   5::/$1,$0,$0   6::/$1,$0,$0,$0

algo
	hash_depth=6
	p:useMask bits=26 withSymmetry

zero
	rrrrrrbbbbbbooooooggggggwwwwyyyy

faces
	0:0-5:rgbo *0,-180,0/0-5/0,0,2
	1:6-11:rgbo *0,-270,0/6-11/2,0,0
	2:12-17:rgbo *0,-360,0/12-17/0,0,-2 
	3:18-23:rgbo *0,-450,0/18-23/-2,0,0
	4:24-27:yw *0,-540,0*-90,-540,0/24-27/0,-2,0
	5:28-31:yw *90,-540,0/28-31/0,2,0

colors
	R:(32,32,255):red G:(64,160,64):green B:(255,64,64):blue O:(64,128,255):orange Y:(0,200,255):yellow W:(192,192,192):white
