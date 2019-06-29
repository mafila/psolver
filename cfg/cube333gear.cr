#          36 37 38
#          39 40 41
#          42 43 44
# 27 28 29  0  1  2   9 10 11  18 19 20 
# 30 31 32  3  4  5  12 13 14  21 22 23
# 33 34 35  6  7  8  15 16 17  24 25 26
#          45 46 47
#          48 49 50
#          51 52 53

scheme2d
	0:0,0'1,0'1.3,1.3'0,1  1:1.9,0'2,0.6'2.5,0.9'3,0.6'3.1,0  2:4,0'5,0'5,1'3.7,1.3
	3:0,1.9'0.6,2'0.9,2.5'0.6,3'0,3.1    4|2,2    5:5,1.9'4.4,2'4.1,2.5'4.4,3'5,3.1
	6:0,5'1,5'1.3,3.7'0,4  7:1.9,5'2,4.4'2.5,4.1'3,4.4'3.1,5  8:4,5'5,5'5,4'3.7,3.7
	$9-17=0-8:5,0  $18-26=0-8:10,0  $27-35=0-8:-5,0  $36-44=0-8:0,-5  $45-53=0-8:0,5

	*$101=1:0,0*0.866,0.866,-0.5,0.5,2.5,0 *$105=5:0,0*0.866,0.866,-0.5,0.5,5,2.5 *$107=7:0,0*0.866,0.866,-0.5,0.5,2.5,5 *$103=3:0,0*0.866,0.866,-0.5,0.5,0,2.5
	*$201=1:0,0*0.866,0.866,0.5,-0.5,2.5,0 *$205=5:0,0*0.866,0.866,0.5,-0.5,5,2.5 *$207=7:0,0*0.866,0.866,0.5,-0.5,2.5,5 *$203=3:0,0*0.866,0.866,0.5,-0.5,0,2.5
	*$110,114,116,112,210,214,216,212=101,105,107,103,201,205,207,203:5,0
	*$119,123,125,121,219,223,225,221=101,105,107,103,201,205,207,203:10,0
	*$128,132,134,130,228,232,234,230=101,105,107,103,201,205,207,203:-5,0
	*$137,141,143,139,237,241,243,239=101,105,107,103,201,205,207,203:0,-5
	*$146,150,152,148,246,250,252,248=101,105,107,103,201,205,207,203:0,5

model3d
	18-26,119,123,125,121,219,223,225,221:x=18/0,90	  
	27-35,128,132,134,130,228,232,234,230:x=0/0,-90  
	9-26,110,114,116,112,210,214,216,212,119,123,125,121,219,223,225,221:x=9/0,90  
	36-44,137,141,143,139,237,241,243,239:y=42/0,90  
	45-53,146,150,152,148,246,250,252,248:y=45/0,-90

turns
	0::/0-6-8-2,1-3-7-5,18-20-26-24,19-23-25-21,9-42-35-47,12-43-32-46,15-44-29-45,10-39-34-50,13-40-31-49,16-41-28-48,11-36-33-53,14-37-30-52,17-38-27-51,4-4,22-22, 101-103-107-105,201-203-207-205,119-123-125-121,219-223-225-221,112-143-132-146,212-243-232-246,110-139-134-150,210-239-234-250,116-141-128-148,216-241-228-248,114-137-130-152,214-237-230-252
	1::/0-36-26-45,1-37-25-46,2-38-24-47,3-39-23-48,4-40-22-49,5-41-21-50,6-42-20-51,7-43-19-52,8-44-18-53,9-11-17-15,10-14-16-12,29-27-33-35,28-30-34-32,13-13,31-31, 101-137-125-146,201-237-225-246,103-139-123-148,203-239-223-248,105-141-121-150,205-241-221-250,107-143-119-152,207-243-219-252,110-114-116-112,210-214-216-212,128-130-134-132,228-230-234-232
	2::/t0,t0			3::/t0,t0,t0
	4::/t1,t0			5::/t1,t0,t0		6::/t1,t0,t0,t0
	7::/t1,t1			8::/t1,t1,t0		9::/t1,t1,t0,t0			10::/t1,t1,t0,t0,t0
	11::/t1,t1,t1		12::/t1,t1,t1,t0	13::/t1,t1,t1,t0,t0		14::/t1,t1,t1,t0,t0,t0
	15::/t0,t1			16::/t0,t1,t0		17::/t0,t1,t0,t0		18::/t0,t1,t0,t0,t0
	19::/t0,t0,t0,t1	20::/t0,t0,t0,t1,t0	21::/t0,t0,t0,t1,t0,t0	22::/t0,t0,t0,t1,t0,t0,t0

moves
	0:R2:1/8-18,44-53,5-21,41-50,2-24,38-47,4-40-22-49, 9-17,10-16,11-15,14-12, 125-207-43-119-225-46-101-219-52-107-201-37, 243-19-152-246-1-137-252-7-143-237-25-146, 105-121,205-221,141-150,241-250/8-2,44-38×2
	1:R2':0/$0,$0,$0,$0,$0,$0,$0,$0,$0,$0,$0/2-8,38-44×2
	2:U2:3/t0,t0,t0,$0,t0/2-0,11-9×2
	3:U2':2/$2,$2,$2,$2,$2,$2,$2,$2,$2,$2,$2/0-2,9-11×2
	5:F2':4/t1,t0,t0,t0,$1,t0,t1,t1,t1/44-42,15-9×2
	4:F2:5/$5,$5,$5,$5,$5,$5,$5,$5,$5,$5,$5/42-44,9-15×2

faces
	0:0-8 *0,-180,0/0-8/0,0,2
	1:9-17 *0,-270,0/9-17/2,0,0
	2:18-26 *0,-360,0/18-26/0,0,-2
	3:27-35 *0,-450,0/27-35/-2,0,0
	4:36-44 *0,-540,0*-90,-540,0/36-44/0,-2,0
	5:45-53 *90,-540,0/45-53/0,2,0

algo
	hash_depth=7
	p:useMask bits=26 withSymmetry

zero
	rrrrrrrrrbbbbbbbbbooooooooogggggggggwwwwwwwwwyyyyyyyyy

colors
	R:(64,64,255):red G:(64,160,64):green B:(255,64,64):blue O:(64,128,255):orange Y:(0,200,255):yellow W:(192,192,192):white
