#!/usr/bin/python3
import numpy as np, subprocess as sp, cv2, re, os, glob, sys, math, datetime, threading, hashlib, copy, screeninfo, pickle
sin,cos= lambda z: math.sin(math.radians(z)), lambda z: math.cos(math.radians(z)) # sin,cos in degrees

isWinRelease= False
models= ['cube223','cube222','cube333','cube444','cube555','pyraminx','skewb','cube333gear','ftoctahedron']
allcolors= {} # colors dictionary {'cube333':{'R':[[0,0,255],[0,0,200],...],'G':[[0,255,0],...]}}
try: pkl= open('colors.pkl','rb'); allcolors= pickle.load(pkl); pkl.close()
except: pass


# Cube on plane
# self.poly - dictionary of lists with points for each item
# self.colors - dictionary of colors for each item
# self.hidden - dictionary of hidden items
class Scheme:
	def __init__(self): self.poly= {}; self.colors= {}; self.hidden= {}
	def showInfo(self): print(f'scheme: {len(self.poly)} items')
	def nlist(self, a, b): # parse string like '1-5', or '1,2,3,4' to list
		return range(int(a[0:-len(b)]), int(a[len(a)-len(b)+1:])+1) if b and b[0]=='-' else list(map(int,a.split(',')))

	def parseLine(self, line, cm=None):
		for a in re.findall( r'(\d+)\|([\d\.-]+),([\d\.-]+)(?:\s|$)', line ): # like 2|0.5,0.5
			n= int(a[0]) # item number
			a= [ float(a[1]), float(a[2]), float(a[1])+1, float(a[2])+1 ] # (x,y)=>(x+1,y+1) square
			self.poly[n]= [ (a[0],a[1]), (a[2],a[1]), (a[2],a[3]), (a[0],a[3]) ] # add poly

		for a in re.findall( r'([\d-]+)\:([\d\.-]+,[\d\.-]+(\'[\d\.-]+,[\d\.-]+)+)', line ): # like 2:0,0'1,0'1,1
			n= int(a[0]) # item number
			if n not in self.poly: self.poly[n]= [] # initialize poly
			for b in re.findall( r'([\d\.-]+),([\d\.-]+)', a[1]): # like 2.0,-3.5
				self.poly[n]+= ( (float(b[0]), float(b[1])), )

		r= (r'(\*){0,1}\$(\d+([,-]\d+)*)=(\d+([,-]\d+)*):([\d\.-]+),([\d\.-]+)'
			r'(\*([\d\.-]+),([\d\.-]+),([\d\.-]+),([\d\.-]+)(,([\d\.-]+),([\d\.-]+)){0,1}){0,1}')
		for a in re.findall(r, line): # like $3,4,5=0-2:-1,1*0.5,0.5
			dx,dy = float(a[5]),float(a[6]) # delta to move the item
			xx,yy,xy,yx,x0,y0 = ( # rotation cos,cos,-sin,sin,point for the item
				(float(v) if v else (1 if i<2 else 0)) for i,v in enumerate((a[8],a[9],a[10],a[11],a[13],a[14])) )
			for i,t in enumerate( self.nlist(a[1], a[2]) ): # iterate destination items list
				sl= self.poly[ self.nlist(a[3], a[4])[i] ]
				self.poly[t]= [ ( dx+x0+(x[0]-x0)*xx+(x[1]-y0)*xy , dy+y0+(x[0]-x0)*yx+(x[1]-y0)*yy ) for x in sl ]
				self.hidden[int(t)]= a[0] # hidden mark - '*' is present at the begging of line

	def draw(self, scr, x0=0, y0=0, w=0, h=0, colmap= {}, hl= []): # draw in rectangle with color map, highlight items
		show= [ n for n in self.poly.keys() if n not in self.hidden or self.hidden[n]=='' ] # list of items to show
		pl= [ p for pp in [ self.poly[n] for n in show ] for p in pp ] # all points from all faces to draw
		minX,maxX,minY,maxY = ( # model's rectangle
			min(p[0] for p in pl), max(p[0] for p in pl), min(p[1] for p in pl), max(p[1] for p in pl) )
		cf= 30 if w==0 and h==0 else min( w/(maxX-minX) if w>0 else 1e9 , h/(maxY-minY) if h>0 else 1e9 ) # ratio
		hll,cen = [],{} # polys to highlight, centers of blocks
		for n,pp in filter( lambda x: not show or x[0] in show, self.poly.items() ): # iterates polys to show
			pp= [ ((p[0]-minX)*cf+x0, (p[1]-minY)*cf+y0) for p in pp ] # points in screen coordinates
			col= colmap[self.colors[n]] if n in self.colors and self.colors[n] in colmap else (64,64,64) # item color
			cv2.fillPoly(scr.img, [np.array(pp,dtype=np.int32)], col)
			cv2.polylines(scr.img, [np.array(pp,dtype=np.int32)], 1, (32,32,32), 1, cv2.LINE_AA) # show item with ribs
			cen[n]= ( int(sum(p[0] for p in pp)/len(pp)), int(sum(p[1] for p in pp)/len(pp)) ) # item center
			if n in hl: hll.append(np.array(pp,dtype=np.int32)) # remember items to highlight
			#scr.putTextCenter(str(n), cen[n], (255,255,255), fsz=0.3) # show number on the item
		for x in hll: cv2.polylines(scr.img, [x], 1, (192,192,192), 1, cv2.LINE_AA) # and now show highlighted items


# 3d model of the cube
# self.poly - current state of compiled model: start with 2d scheme in constructor, then init(), then origami()
# self.colors - dictionary of colors for each item
# self.faces - dictionary of faces with the data to rotate face; self.plan - origami plan; self.cang - default eye angle
# self.poly0 - initial state to use with origami
# self.t0 - animation start time; self.face, self.facefrom - keep current and previous face
class Model:
	def showInfo(self): print(f'model: {len(self.plan)} origami turns')
	def __init__(self, cm): #
		self.poly= {}; self.colors= cm.sch2d.colors; self.faces= {}; self.plan= []; self.cang=(-30,-30,0)
		ppl= [p for pp in cm.sch2d.poly.values() for p in pp] # all points of 2d model
		x1,x2,y1,y2 = min(p[0] for p in ppl), max(p[0] for p in ppl), min(p[1] for p in ppl), max(p[1] for p in ppl)
		xd,yd,xycf = (x1+x2)/2, (y1+y2)/2, 1/max(x2-x1,y2-y1) # move coordinates and scale
		for n in cm.sch2d.poly.keys(): # center and resize scheme
			self.poly[n]= np.array([ ((p[0]-xd)*xycf, (p[1]-yd)*xycf, 0, 1) for p in cm.sch2d.poly[n]], dtype=np.double)
		self.poly0= self.poly.copy() # save prepared 2d scheme for use when origami
		self.t0= self.face= self.facefrom= None

	def	parseLine(self, line, cm=None):
		for a in re.findall( r'((,?(\d+)-?(\d+)?)+):(\w)=(\d+)/(\d+),([0-9\-\.]+)',line ): # 63-71,27-35:x=63/0,90
			rg,xy,a5,a6,a7 = [], a[4], int(a[5]), int(a[6]), float(a[7])
			for b in re.findall( r'(\d+)-?(\d+)?',a[0] ):
				rg.extend( range(int(b[0]),int(b[1])+1) if b[1] else [int(b[0])] )
			self.plan.append((
				rg,a7 if xy=='y' else 0, a7 if xy=='x' else 0,
				self.poly[a5][a6][0] if xy=='x' else 0, self.poly[a5][a6][1] if xy=='y' else 0 ))

		for a in re.findall( r'camera\s*=([\d\-\.\,\s]+)',line ): # camera angle
			self.cang= list(map(float, re.findall(r'[\d\-\.]+', a)))

	def parseFacesLine(self, line, cm=None): # num-items-colors + 3d processing 0:0-8:rgbo *0,180/0-5;6-12/0,0,1;1,0,0
		r= r'([\d-]+):([\d\-\,]+){0,1}(\:([a-z]+)){0,1}(\s+((\*[\d\-\.\s,]+){1,})/([^/]+)/([^/]+)){0,1}'
		for b in re.findall(r, line):
			n= int(b[0]) # face number
			if b[2]: cm.facecol[n]= set(b[3].upper()) # colors allowed for the face
			for a in re.findall(r'(\d+)-(\d+)', b[1]): cm.faces[n]= range(int(a[0]), int(a[1])+1) # 0-8
			if b[4]: # *180,0,0*0,90,0/0-5;6-12/0,0,1;1,0,0=>[[(180,0,0),(0,90,0)],None,((0-5,(0,0,1)),(6-12,(1,0,0)))]
				self.faces[n]= (
					[ list(map(float,re.findall(r'[\d\-\.]+', sta))) for sta in re.findall(r'\*[\d\-\.\s,]+', b[5]) ],
					[ (range(int(c[0]), int(c[1])+1), list(map(float,b[8].split(';')[j].split(','))))
						for j,c in enumerate(re.findall(r'(\d+)-(\d+)',b[7])) ]
				)

	def init(self): # prepare model for origami
		ca= self.cang
		if len(ca)>3 and (ca[3] or ca[4] or ca[5]): # center of 3d model defined in camera param in model3d section
			self.move= np.array([[1,0,0,ca[3]], [0,1,0,ca[4]], [0,0,1,ca[5]], [0,0,0,1]],dtype=np.double)
		else: # calculate model's center
			self.origami(1) # compile scheme to model
			ppl= [p for pp in self.poly.values() for p in pp] # all points of 3d model
			x1,x2,y1,y2,z1,z2 = ( # bounding parallelepiped
				min(p[0] for p in ppl), max(p[0] for p in ppl),
				min(p[1] for p in ppl), max(p[1] for p in ppl),
				min(p[2] for p in ppl), max(p[2] for p in ppl)
			)
			self.move= np.array( # move matrix to the defined center of 3d model
				[[1,0,0,-(x1+x2)/2], [0,1,0,-(y1+y2)/2], [0,0,1,-(z1+z2)/2], [0,0,0,1]], dtype=np.double)
		self.proj= np.array([[1,0,0,0],[0,1,0,0],[0,0,-1.001001,-0.1001001],[0,0,-1,1]],dtype=np.double) # projection matrix
		self.poly= self.poly0.copy() # back to scheme

	def rot3d(self, ax=0, ay=0, az=0, sx=0, sy=0, sz=0, rev= False): # rotation point sx,sy,sz; angles ax,ay,az
		rx= [[1,0,0,0], [0,cos(ax),-sin(ax),0], [0,sin(ax),cos(ax),0], [0,0,0,1]]
		ry= [[cos(ay),0,sin(ay),0], [0,1,0,0], [-sin(ay),0,cos(ay),0], [0,0,0,1]]
		rz= [[cos(az),-sin(az),0,0], [sin(az),cos(az),0,0], [0,0,1,0], [0,0,0,1]]
		if sx or sy or sz: # move origin to the point sx,sy,sz and back
			m1,m2 = [[1,0,0,-sx],[0,1,0,-sy],[0,0,1,-sz],[0,0,0,1]], [[1,0,0,sx],[0,1,0,sy],[0,0,1,sz],[0,0,0,1]]
			if rev: return np.array(m2,dtype=np.double).dot(rz).dot(ry).dot(rx).dot(m1)
			else: return np.array(m2,dtype=np.double).dot(rx).dot(ry).dot(rz).dot(m1)
		else:
			if rev: return np.array(rz,dtype=np.double).dot(ry).dot(rx)
			else: return np.array(rx,dtype=np.double).dot(ry).dot(rz)

	def origami(self, stage): # compile 2d scheme to 3d model, stage is between 0 and 1.0, 1=100%
		if stage>1: self.origami(1); return
		self.poly= self.poly0.copy() # take 2d scheme
		for s in self.plan: # step of origami plan
			m= self.rot3d(ax=s[1]*stage, ay=s[2]*stage, sx=s[3], sy=s[4])
			for n in s[0]: self.poly[n]= [ m.dot(p) for p in self.poly[n] ]

	def moveToFace(self, sec): # turn model to self.face
		if not self.faces: return None,None,None,None,True # no face movements
		f,f0,df = self.faces[self.face][0],(0,0,0),None # current face angles; previous face angles; actual distance vector
		ppi,ax,ay,az,d = self.poly.copy(),0,0,0,0 # current poly; rotation angles; actual distance for "floating" face
		# timing: move previous face back; turn 1,2,3; move face to cam
		t0,t1,t2,t3,t4 = 0, 1.0 if len(f)>0 else 0, 1.0 if len(f)>1 else 0, 1.0 if len(f)>2 else 0, 0.5
		if self.facefrom is not None: t0= 0.5; f0= self.faces[self.facefrom][0][-1] # f0 is previous face angles
		if t0 and sec<t0: # return previous face back
			ax,ay,az= f0; d= (t0-sec)*0.05; df= self.faces[self.facefrom][1]
		elif t1 and sec<t0+t1: # turn 1
			ax,ay,az = ( f0[i]*(t1-sec+t0)+f[0][i]*(sec-t0) for i in (0,1,2) )
		elif t2 and sec<t0+t1+t2: # turn 2
			tt= t0+t1; ax,ay,az = ( f[0][i]*(tt+t2-sec)+f[1][i]*(sec-tt) for i in (0,1,2) )
		elif t3 and sec<t0+t1+t2+t3: # turn 3
			tt= t0+t1+t2; ax,ay,az = ( f[1][i]*(tt+t3-sec)+f[2][i]*(sec-tt) for i in (0,1,2) )
		elif t4 and sec<t0+t1+t2+t3+t4: # move current face to cam
			ax,ay,az= f[-1]; d= (sec-t0-t1-t2-t3)*0.05; df= self.faces[self.face][1]
		elif sec<t0+t1+t2+t3+t4+0.3: ax,ay,az= f[-1]; d= 0.025; df= self.faces[self.face][1] # wait 0.3 sec
		else: return None,None,None,None,True # done: face moved to camera
		if df: # update poly to move active face to camera
			for x in df:
				for n in x[0]: ppi[n]= [ ([p[0]+x[1][0]*d,p[1]+x[1][1]*d,p[2]+x[1][2]*d,p[3]]) for p in self.poly[n] ]
		return ppi,ax,ay,az,False # return adjusted poly, camera angles and finished flag

	def draw(self, cm, scr, x0=0, y0=0, w=0, h=0, cmap= {}, start=False, face=None, mshow= [], mmark= '', showcam= True): # show 3d model
		x0= self.x0= x0 or self.x0; y0= self.y0= y0 or self.y0; w= self.w= w or self.w; h= self.h= h or self.h
		cmap= self.cmap= cmap or self.cmap; ca= self.cang
		bx,by,bz,cby= ca[0], ca[1], ca[2], ca[6] if len(ca)>6 else ca[1] # eye angles; cby camera's rotation on Y-axis
		ax,ay,az,s,con,ppi= 0,0,0,3.3,True,self.poly # ax,ay,az rotate angles; s- scale; con- camera on; model's poly

		if showcam: # cam present => we are in color detection mode
			if start or self.t0 is None: self.t0= datetime.datetime.now() # start a cartoon
			if face is not None: self.facefrom = self.face; self.face= face; # move to new face
			sec= (datetime.datetime.now()-self.t0).total_seconds() # time since start in seconds
			if self.face is None: # initial model origami & camera turns
				if sec<1.5:
					r= sec/1.5; self.origami(r); bx,by,cby,bz,s,con = bx*r, 3*by*r/4, 3*cby*r/4, bz*r, 1.3+2*r, False
				elif (by or cby) and sec<2:
					self.origami(1); by,cby,con= sec*by/2, sec*cby/2, False
				elif sec<2.5:
					self.origami(1); con= False
				else:
					return True # we prepared everything - origami and positioning done
			else: # move to face animation
				ppi,ax,ay,az,ret= self.moveToFace(sec)
				if ret: return True
		else: s= 5.5 # if no cam => we are in show the solution mode => scale to bigger size
		s*= (0.8 if cm.file=='cube223' else 1.1 if cm.file=='pyraminx' else 0.85 if showcam else 1.0) # adjust scale

		scale= np.array([[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,1]]) # scale, use it after all other transformations
		tsc= scale.dot(self.proj).dot(self.rot3d(bx,cby,bz)).dot(self.move) # transform matrix without turns for camera
		tsf= scale.dot(self.proj).dot(self.rot3d(bx,by,bz)).dot(self.rot3d(ax,ay,az,rev=True)).dot(self.move) # +turns
		scf= min(w,h)/2 # screen scale
		scr.img[y0:y0+h,x0:x0+w]= (32,32,32) # clean screen rectangle for 3d model

		if showcam: # show camera
			cam= [ [[-2,-4,-56],[2,-4,-56],[2,0,-56],[-2,0,-56]],[[-2,-4,-63],[-2,-4,-56],[2,-4,-56],[2,-4,-63]],
				[[2,-4,-63],[2,-4,-56],[2,0,-56],[2,0,-63]],[[-1,-3,-53],[1,-3,-53],[1,-1,-53],[-1,-1,-53]],
				[[-1,-3,-56],[-1,-3,-53],[1,-3,-53],[1,-3,-56]],[[1,-3,-56],[1,-3,-53],[1,-1,-53],[1,-1,-56]] ]
			for i,cp in enumerate(cam):
				zscale= 0.006 if cm.file in ('pyraminx','ftoctahedron') else 0.01 # camera scale pyraminx,ftoctahedron
				for j,p in enumerate(cp):
					xy= tsc.dot([p[0]*0.01,p[1]*0.01,p[2]*zscale,1]); xy/= xy[3]
					cp[j]= (x0+(xy[0]+1)*scf, y0+(xy[1]+1)*scf)
				cc= 60-i%3*10; col= (192,128,128) if i==3 and con else (cc,cc,cc)
				cv2.fillPoly(scr.img, [np.array(cp,dtype=np.int32)], col, cv2.LINE_AA)

		pp,ppz = {},{} # dictionary of screen polys; dictionary of z-depth with poly index
		for n,pi in ppi.items():
			pp2,zz = [],0.0
			for p in pi: xy= tsf.dot(p); xy/= xy[3]; zz-= xy[2]; pp2.append((x0+(xy[0]+1)*scf, y0+(xy[1]+1)*scf))
			zz/=len(pi)
			while zz in ppz: zz+=1e-9 # avoid equal z-depth
			ppz[zz]= n; pp[n]= np.array(pp2,dtype=np.int32)

		cen,fah= {},{} # centers of items, used to draw arrows & show item number; shown faces dictionary
		for i,n in enumerate([n for _,n in sorted(ppz.items(),key= lambda z:z[0])]): # sort items with z-depth
			if not showcam and cm.file=='cube333gear': # show 333gear in a special way, hide back faces
				if n%100//9 not in fah:
					fah[n%100//9]= 1; fa= n%100//9*9
					fap= np.array([ pp[fa][0], pp[fa+2][1], pp[fa+8][1], pp[fa+6][0] ], dtype=np.int32)
					cv2.fillPoly(scr.img, [fap], (48,48,48), cv2.LINE_AA)
					cv2.polylines(scr.img, [fap], 1, (32,32,32), 1, cv2.LINE_AA)

			cen[n]= (int(sum([p[0] for p in pp[n]])/len(pp[n])), int(sum([p[1] for p in pp[n]])/len(pp[n])))
			col= ( cmap[self.colors[n]] if self.colors[n] in cmap else None
				) if n in self.colors and cmap else (64,64,64)
			if col and con and self.face is not None and n in self.faces[self.face][1][0][0]:
				col= list(map(lambda z:int(z*1.2), col))
			if col:
				cv2.fillPoly(scr.img, [pp[n]], col, cv2.LINE_AA)
				cv2.polylines(scr.img, [pp[n]], 1, (32,32,32), 1, cv2.LINE_AA)
			#scr.putTextCenter(str(n), cen[n], (0,0,0), fsz=0.3) # show item number, good for debugging

		for ar in mshow: # draw arrows to show the move
			tip= 0.2 if cm.file in ('pyraminx','ftoctahedron','skewb') else 0.1 # adjust arrow size for certain models
			cv2.arrowedLine(scr.img,cen[ar[0]],cen[ar[1]],(255,255,255),3,cv2.LINE_AA,0,tip) # show the move
			cv2.arrowedLine(scr.img,cen[ar[0]],cen[ar[1]],(0,0,0),1,cv2.LINE_AA,0,tip)
			if mmark: # show mark like ×2
				for p in [(-1,-1),(-1,1),(1,1),(1,-1)]: # bug in openCV, draw text width by hand
					scr.putTextCenter(mmark.replace('×','x'),(cen[ar[0]][0]+p[0],cen[ar[0]][1]+p[1]),(255,255,255),0.5)
				scr.putTextCenter(mmark.replace('×','x'), (cen[ar[0]][0],cen[ar[0]][1]), (0,0,0), 0.5) # in black color

		return False

	def doTheMove(self, move): # do the move
		c2= self.colors.copy() # copy to avoid using already copied color
		for k,v in move.items(): self.colors[k]= c2[v] if v in c2 else ''


# Cube transformation: Move or Turn
# self.moves - dictionary of hashes for each move [move][from]=to
# self.revmoves - reverse moves; self.movename - names of moves
# self.moveshow, self.movemark - dictionary with list of arrows for each move and move marks
class Transform:
	def showInfo(self):
		print(f'{self.name}: {len(self.moves.keys())} transformations {" ".join(sorted(set(self.movename.values())))}')
	def __init__(self, s):
		self.name= s; self.moves= {}; self.revmoves= {}; self.movename= {}; self.moveshow= {}; self.movemark= {}

	def parseLine(self, line, cm): # 0:R:1,0,2,3,4/0-1,2-3/0-1,3-4
		# number : name : back_move,skip_move2,skip_move3,...  / items to transfer / show move arrows
		r= (r'(\d*):([\d\w\']*):\s*([\d\-,\s]*?)\s*/\s*([t\$]?\d+(-\d+)*(,\s*[t\$]?\d+(-\d+)*)*)'
			r'(\s*/\s*(\d+(-\d+)*(,\s*\d+(-\d+)*)*))?([×xX]\d+)?')
		for a in re.findall(r, line):
			n= int(a[0]) if a[0] else len(self.moves.keys()) # number of the move
			if a[1]: self.movename[n]= a[1] # move name
			if n not in self.moves: self.moves[n]= {} # initialize
			if a[2]: # reverse, skip and special moves
				self.revmoves[n]= re.split(r'[,\s]+', a[2]); self.revmoves[n]+= ['-1']*(16-len(self.revmoves[n]))

			for b in re.findall( r'(([t\$])?(\d+)+(-\d+)*)', a[3]): # parse move as a hash from=>where
				if b[1]=='$' or b[1]=='t': # use another transformation
					tranSrc= self.moves if b[1]=='$' else cm.turns.moves # another move of this set or from turns
					if self.moves[n]: # do the transformations again and again
						m2= self.moves[n].copy() # next transformation to apply
						for k,v in tranSrc[int(b[2])].items():
							self.moves[n][k]= m2[v] if v in m2 else v # if no info for an item - keep it
					else:
						self.moves[n]= tranSrc[int(b[2])].copy() # first time - copy another move
				else:
					pl= [int(c) for c in re.findall( r'(\d+)', b[0])]; pl.append(pl[0]) # always cycle last=>first
					for i in range(1,len(pl)): self.moves[n][pl[i]]= pl[i-1] # save move transformation
			for b in a[8].split(','): # show move
				if n not in self.moveshow: self.moveshow[n]= [] # initialize show move arrows list
				if b: c= b.split('-'); self.moveshow[n].append([int(c[0]),int(c[1])]) # add an arrow
			self.movemark[n]= a[12] # movemark like '×2'

	def writeC(self, file): # write include C file with the transformations
		f= open(file,'w')
		for n in self.moves.keys():
			f.write(f'case {n}:'+';'.join([ f'b[{k}]=a[{v}]' for k,v in self.moves[n].items() if k!=v ]))
			for i in range(0,3):
				f.write(f';Cb->m{i}={self.revmoves[n][i] if n in self.revmoves and i<len(self.revmoves[n]) else -1}')
			f.write(';break;\n')
		f.close()


# Algorithm to find the solution
# self.data - dictionary of steps with data like max_depth=7
# self.prune - dictionary of steps with dictionary of prune tables for the step
# self.mask - dictionary of steps with dictionary of prune tables with mask to calc prune table hash value
# self.symmetry - dictionary of steps with dictionary of prune tables with use turn symmetry flag
# self.solved - dictionary of steps with solved conditions for the step
# self.weight - dictionary of steps with weight formula for the step
# self.moves - dictionary of steps with the list of allowed moves
# self.keywrd - keywords to parse and process data values
# self.revcol - defines reverse colors, colors on opposite side
# self.n - number of current step
class Algo:
	def __init__(self): # initialize structures for 0 step, if there is only one step, "0:" line is not necessary
		self.data= {0:{}}; self.prune= {0:{}}; self.mask= {0:{}}; self.symmetry= {0:{}}
		self.solved= {0:['',0]}; self.weight= {0:['',0]}; self.moves= {0:[]}; self.param= {0:[]}
		self.keywrd= { 	'hash_depth':('stepHashDepth',0), 'start_depth':('stepStartDepth',0),
			'max_depth':('stepMaxDepth',255), 'max_weight':('stepMaxWeights',0), 'seq':('stepSeq',0),
			'max_sol':('stepMaxSol',1), 'sol_time':('solTime',0), 'sol_time_hard':('solTimeHard',0),
			'step_time':('stepTime',0), 'link_step':('stepLink',-1), 'link_step_local':('stepLinkLocal',0) }
		self.revcol= []; self.n= 0

	def parseLine(self, line, cm=None): # parse the line of the algorithm with step=self.n
		for a in re.findall( r'(^|\s)(\d+)\:', line ): # new step, like '1:'
			self.n= n= int(a[1]) # init step data structures
			self.data[n]= {}; self.prune[n]= {}; self.mask[n]= {}; self.symmetry[n]= {}
			self.solved[n]= ['',0]; self.weight[n]= ['',0]; self.moves[n]= []; self.param[n]= []

		for b in re.findall( r'step_(\d+)_turn:\s*([\d\s,]*)', line ): # step_N_turn => copy step N and do turn symmetry
			for a in ('data','prune','solved','param'): # deepcopy .data, .prune, .solved, .param from linked step
				at= getattr(self,a); at[self.n]= copy.deepcopy( at[int(b[0])] )
			self.data[self.n]['seq']= 0 # don't copy sequence state
			for t in map(int, re.findall(r'\d+',b[1]) ): # list of turns for the new step
				turn= lambda z: f'[{ cm.turns.moves[ int(t) ][ int(z.group()[1:-1]) ] }]' # turn t: '[5]' => '[25]'
				ps= self.solved[self.n]; ps[0]= re.sub( r'\[\d+\]', turn, ps[0] ) # turn solved state
				for p in self.prune[self.n].keys(): # turn all prunes
					ps= self.prune[self.n][p]; ps[0]= re.sub( r'\[\d+\]', turn, ps[0] )

		for a in re.findall( r'param\s*=\s*([\d\.\-]+\s*(,\s*[\d\.\-]+)*)', line ): # param=1.0,2.0
			self.param[self.n]= re.findall( r'[\d\.\-]+',a[0] )

		for b in re.findall( r'([psw](\d*))\:(.*)', line ): # prune table by bits, solved status or weight; "p0: 1=2=3 4=5&6=7 4=5|6"
			npsw= b[1] or '0'
			if b[0]=='s': ps= self.solved[self.n]
			elif b[0]=='w': ps= self.weight[self.n]
			elif npsw in self.prune[self.n]: ps= self.prune[self.n][npsw]
			else: ps= self.prune[self.n][npsw]= ['',0]; self.mask[self.n][npsw]= []; self.symmetry[self.n][npsw]= 0
			s= b[2].strip()
			if 'useMask' in s: # like "useMask bits=31 withSymmetry 1,2,3"
				match= re.search( r'useMask\s+(bits=(\d+)){0,1}(\s+withSymmetry){0,1}([\s\d\,]+){0,1}', s )
				if match:
					if match.group(3): self.symmetry[self.n][npsw]= 1 # like "withSymmetry"
					if match.group(4): self.mask[self.n][npsw]= re.findall( r'\d+', match.group(4) ) # like "1,2,3"
					if b[0]=='p':
						bits= ps[1]= int(match.group(2)); ps[0]= f'hashFromMask(a,prunei)/*{bits}*/'
			elif 'include' in s: ps[0]+= '\n#'+s; ps[1]+= 1
			else:
				while True:
					s2= s; s= re.sub( r'([\:\d]+)(=[\:\d]+)(=.*)', r'\1\2 \1\3', s )
					if s2==s: break
				for i,a in enumerate(re.split( r'\s+', s )):
					a= re.sub(r'([\:\*]?\d+)!=([\:\*]?\d+)',r'a[\1]!=a[\2]', a)
					a= re.sub(r'([\:\*]?\d+)=([\:\*]?\d+)',r'a[\1]==a[\2]', a)
					a= re.sub(r'a\[\:(\d+)\]', r'revColor[(int)a[\1]]', a)
					a= re.sub(r'a\[\*(\d+)\]', r'ZERO_CUBE[\1]', a)
					a= a.replace('&','&&').replace('|','||')
					if b[0]=='s': ps[0]+= ('&&(' if ps[1]+i else '(')+ a +')'
					elif b[0]=='w': ps[0]+= ('+(' if ps[1]+i else '(')+ a +')'
					else: ps[0]+= ('|(' if ps[1]+i else '(')+ a +(')<<%d\n'%(ps[1]+i)) # using uint32 so bits<=31
				ps[1]+= i+1

		for a in re.findall( r'(%s)\s*=\s*([^\s]+)'%('|'.join(self.keywrd.keys())), line ): # like max_depth=11
			self.data[self.n][a[0]]= a[1]

		for a in re.findall( r'moves\s*=\s*(.*)', line ): # process allowed moves or move-set template
			mvset= []
			for b in re.split( r'\s*;\s*', a.strip() ):
				mvlist= []
				for c in re.split( r'\s*,\s*', b):
					d= re.split( r'\s*-\s*', c)
					if len(d)>1: x= mvlist.extend( range(int(d[0]), int(d[1])+1) )
					elif d[0][0:2]=='*=': mvlist.append( '255,'+d[0][2:] )
					elif d[0][0:2]=='*!': mvlist.append( '254,'+d[0][2:] )
					elif d[0][0:2]=='*^': mvlist.append( '253,'+d[0][2:] )
					elif d[0][0:2]=='*#': mvlist.append( '252,'+d[0][2:] )
					else: mvlist.append( int(d[0]) )
				mvset.append(mvlist)
			self.moves[self.n].append(mvset)

	def Dic1str(self, dic): return '"'+'","'.join([ dic[k] for k in sorted(dic.keys()) ])+'"'
	def Dic2int(self, dic): return ','.join([ '{'+','.join(dic[k])+'}' for k in sorted(dic.keys()) ])
	def StepDic(self, sk, dic):
		return ''.join([ f'case {st}: ' +
			("" if dic[st][0][0]=="\n" else "return ") + dic[st][0] + '\n;\n' for st in sk if dic[st][0]
		])

	def compile(self, cm, force=False): # prepare headers and compile solve.c
		print(); cm.sch2d.showInfo(); cm.mod3d.showInfo(); cm.moves.showInfo(); cm.turns.showInfo()
		print(f'faces: {len(cm.faces.keys())}'); print()
		if isWinRelease: return
		fbin,fcfg = f'bin/{cm.algname}', f'cfg/{cm.file}.cr'
		if not force and os.path.isfile(fbin) and os.path.getctime(fcfg)<os.path.getctime(fbin): return # don't compile

		cm.moves.writeC('include/moves.h'); cm.turns.writeC('include/turns.h');  # write header files for moves & turns
		f= open('include/movenames.h','w'); f.write(self.Dic1str(cm.moves.movename)); f.close() # move names
		f= open('include/specmoves.h','w'); f.write(self.Dic2int(cm.moves.revmoves)); f.close() # special moves
		sk= sorted(self.data.keys()) # sorted steps of algo
		for st in sk: # iterate alog steps
			if self.solved[st][1]==0: # empty solved -> use face items
				self.solved[st][0]= '&&'.join([  f'a[{f[0]}]==a[{x}]' for f in cm.faces.values() for x in f[1:] ])
			if not self.moves[st]: # empty moves -> use all moves
				self.moves[st]= [[range(0,len(cm.moves.moves))]]
		f= open('include/solved.h','w'); f.write(self.StepDic(sk, self.solved)); f.close() # solved state per step
		f= open('include/weight.h','w'); f.write(self.StepDic(sk, self.weight)); f.close() # calc weights per step
		f= open('include/prune.h','w')
		for st in self.data.keys(): # case <step>: switch(<prine_index>): { case <index>: return <value>; ... }
			f.write(f'case {st}: switch(prunei){{\n' + '\n'.join([
					f'case {p}:return {self.prune[st][p][0]};' for p in self.prune[st]
				]) + '\n}\n' if self.prune[st] else ''
			)
		f.close()

		f= open('include/data.h','w')
		f.write(f'#define ZERO_CUBE "{cm.zero}"\n') # zero cube definition
		f.write(f'#define N_STEPS {len(self.data.keys())}\n') # number of algo steps
		nblk= 1+max([ z for y in [x.keys() for x in cm.moves.moves.values()] for z in y ])
		f.write(f'#define N_BLOCKS {nblk}\n') # number of blocks = max item in moves
		f.write(f'#define ALG_NAME "{cm.algname}"\n')
		f.write(f'#define N_MOVES {len(cm.moves.moves)}\n') # number of moves
		f.write(f'#define N_TURNS {len(cm.turns.moves)}\n') # number of turns
		maxsol= max([ int(self.data[st]["max_sol"]) if "max_sol" in self.data[st] else 1 for st in sk ])
		f.write(f'#define N_MAX_SOL {maxsol}\n')
		f.write(f'#define N_MAX_MOVESETS {max([ len(self.moves[st]) for st in sk ])}\n')
		f.write(f'#define N_MAX_PRUNES {max([ len(self.prune[st].keys()) for st in sk ]+[1])}\n')
		f.write(f'#define N_MAX_PARAMS {max(16,max([ len(self.param[st]) for st in sk ]))}\n')
		if self.revcol: # reverse colors like yog-wrb
			f.write(f'#define REVCOL_FROM "{self.revcol[0]}"\n')
			f.write(f'#define REVCOL_TO "{self.revcol[1]}"\n')
		for kn,kw in self.keywrd.items(): # stepHashDepth,stepStartDepth,stepMaxDepth,stepMaxWeights,stepSeq,...
			f.write('int '+kw[0]+'[N_STEPS]={'
				+ ','.join([ str(self.data[st][kn] if kn in self.data[st] else kw[1]) for st in sk ])
				+ '};\n')
		f.write('float stepParams[N_STEPS][N_MAX_PARAMS]={'
			+ ','.join([ '{'+','.join(self.param[st])+'}' for st in sk ])
			+ '};\n')
		f.write('int nStepPruneSymmetry[N_STEPS][N_MAX_PRUNES]={{'
			+ '},{'.join([ ','.join([ str(v) for v in self.symmetry[st].values() ]) for st in sk ])
			+ '}};\n')
		f.write('int nStepMasks[N_STEPS][N_MAX_PRUNES]={{'
			+ '},{'.join([ ','.join([ str(len(ml)) for ml in self.mask[st].values() ]) for st in sk ])
			+ '}};\n')
		f.write('int stepMasks[N_STEPS][N_MAX_PRUNES][N_BLOCKS]={{'
			+ '},{'.join([ '{'+'},{'.join([ ','.join([ str(m) for m in ml ]) for ml in self.mask[st].values() ])
			+ '}' for st in sk ]) + '}};\n')
		f.write('int nStepPrunes[N_STEPS]={' + ','.join([ str(len(self.prune[st].keys())) for st in sk ]) + '};\n')
		f.write('size_t stepPruneSize[N_STEPS][N_MAX_PRUNES]={{'
			+ '},{'.join([ ','.join([ f'((size_t)1)<<{m[1]}' for m in self.prune[st].values() ]) for st in sk ])
			+ '}};\n')
		f.write('char stepMd5[N_STEPS][N_MAX_PRUNES][33]={{"'
			+ '"},{"'.join([ '","'.join([
				self.md5(st, m) for m in (self.prune[st].values() if self.prune[st] else ['0']) ]) for st in sk ])
			+ '"}};\n')
		f.write('int nStepMoveSets[N_STEPS]={' + ','.join([str(len(self.moves[st])) for st in sk]) + '};\n')
		f.write('uint8_t stepMoveSetsLen[N_STEPS][N_MAX_MOVESETS]={{'
			+ '},{'.join([ ','.join([ str(len(mmm)) for mmm in self.moves[st] ]) for st in sk ])
			+ '}};\n')
		f.write('uint8_t stepMoves[N_STEPS][N_MAX_MOVESETS][N_MAX_MS_MOVES][N_MOVES+1]={{'
			+ '},{'.join([ '{'+'},{'.join([ '{'+'},{'.join([ str(len(mm))
				+ ''.join([ ','+str(m) for m in mm ]) for mm in mmm])+'}' for mmm in self.moves[st] ])
				+ '}' for st in sk ])
			+ '}};\n')
		f.close()

		# unpack algo data; compile solve.c with the prepared headers then save in bin/
		tgzList= sorted(glob.glob( f'data/arch/{cm.algname}.tgz*' ))
		if len(tgzList)>0:
			p1= sp.Popen(['cat']+tgzList, stdout=sp.PIPE)
			p2= sp.Popen(['tar','xvzf','-','-C','data/'], stdin=p1.stdout)
			p2.wait()
		#p= sp.Popen(['gcc','-fopenmp','-O3','-Iinclude','-Icfg','-osolve','-march=native','-Wall','solve.c']); p.wait()
		p= sp.Popen(['gcc','-fopenmp','-O3','-Iinclude','-Icfg','-obin/'+cm.algname,'-march=native','solve.c']); p.wait()
		#p= sp.Popen(['gcc','-fopenmp','-O3','-Iinclude','-Icfg','-obin/'+cm.algname,'solve.c']); p.wait()
		if p.returncode!=0: print(f'Error: solver compilation error {p.returncode}'); exit(-4)
		if force: print(f'{file} compiled'); exit(0)

	def md5(self, st, m): # unique md5 value for the step
		return hashlib.md5(( m[0] + '|'
				+ str(self.data[st]['hash_depth'] if 'hash_depth' in self.data[st] else '') + '|'
				+ str(self.data[st]['max_depth'] if 'max_depth' in self.data[st] else '') + '|'
				+ ','.join([ str(m) for msets in self.moves[st] for msetmove in msets for m in msetmove ])
			).encode('ascii')).hexdigest()

	def random(self, cm): # get random position
		p= sp.Popen(['bin/'+cm.algname,'rand'], stdout=sp.PIPE)
		s= p.communicate()[0].decode('ascii').strip().replace(' ','')
		if p.returncode!=0: print(f"Error: can't run bin/{cm.algname} code {p.returncode}")
		return s

	def run(self, cm, scr, cube): # run algo
		scr.hide()
		p= sp.Popen(['bin/'+cm.algname,cube]); p.wait() # run solver
		if p.returncode!=0: print(f"Error: can't run bin/{cm.algname} code {p.returncode}")
		solution= list(map( int, re.findall(r'\d+',open('.solution','r').read()) )) # read the solution
		perpage= scr.perline*scr.lines
		pages= [ solution[i:i+perpage] for i in range(0,len(solution),perpage) ] # split into pages
		scr.show()
		return pages


# Model of the cube. Read .cr file and store all the data
# self.sch2d,self.mod3d,self.turns,self.moves,self.algo,self.zero,self.faces,self.defcol,self.cmap - data from .cr file
class CubeModel:
	def __init__(self, file): # Read the file with cube model
		self.file= file; print(f'\nloading {file}');  # initialize data structured before loading cube model
		self.algname= file.split("_")[0]
		self.sch2d,self.mod3d= Scheme(),None
		self.turns,self.moves= Transform('turns'),Transform('moves')
		self.algo= Algo()
		self.faces,self.defcol,self.facecol,self.colname = {},{},{},{}
		self.cmap= allcolors[self.algname]= allcolors[self.algname] if self.algname in allcolors else {}
		mode= -1 # currend state: scheme2d, model3d, algo, etc
		for line in [x.rstrip() for x in open(f'cfg/{file}.cr')]: # iterate file line by line and search for patterns
			line= re.sub(r'\s+',' ',line.strip()) # strip and raplace all whitespaces with tab
			if len(line)>0 and line[0]=='#': continue # skip comments
			elif line=='scheme2d': sys.stdout.write(' '+line); cur= self.sch2d; mode= 0
			elif line=='model3d': sys.stdout.write(' '+line); self.mod3d= Model(self); cur= self.mod3d; mode= 0
			elif line=='algo': sys.stdout.write(' '+line); cur= self.algo; mode= 1
			elif line=='turns': sys.stdout.write(' '+line); cur= self.turns; cur.n= len(self.sch2d.poly); mode= 0
			elif line=='moves': sys.stdout.write(' '+line); cur= self.moves; cur.n= len(self.sch2d.poly); mode= 0
			elif line=='zero': sys.stdout.write(' '+line); mode= 2
			elif line=='faces': sys.stdout.write(' '+line); mode= 3
			elif line=='colors': sys.stdout.write(' '+line); mode= 4
			elif line!='':
				if mode in (0,1): cur.parseLine(line, self) # Scheme, Transform or Algo
				elif mode==2: # reverse colors and zero cube string
					for a in re.findall( r'reverse:\s*([a-z]+)-([a-z]+)', line): # reverse: rbogwy-ogrbyw
						if a[0]: self.algo.revcol= [a[0],a[1]]
					else: self.zero= line # zero cube string
				elif mode==3:
					self.mod3d.parseFacesLine(line, self) # faces in 3d model
				elif mode==4: # read default colors in BGR
					for a in re.findall( r'([A-Z])\:\((\d+),(\d+),(\d+)\)(:([^\s]+)){0,1}',line ): # R:(255,0,0):red
						self.defcol[a[0]]= (int(a[1]),int(a[2]),int(a[3]))
						self.colname[a[0]]= a[5] if a[5] else f'color {a[0]}'
		self.mod3d.init(); self.showDefMask= self.followFrame= False # prepare mod3d & init model options
		print()


# MainScreen - initialize window, img array with screen image; draw everything on the screen
# self.img - screen image
# self.with,self.height - screen size
# self.sSz - 1/100 of width for paddings and line heights
# modXX, camXX, mapXX - positions for models, camera and color map
class MainScreen:
	def __init__(self): # initialize screen and define all sizes for the main screen
		monList= screeninfo.get_monitors()
		if len(monList)==0: print("Can't find any monitor") #; exit(-2)
		self.width, self.height = monList[0].width, monList[0].height; print('\nscreen size:', self.width, self.height)
		sSz= self.sSz= int(int(self.width)/100) #
		self.img= np.zeros((self.height,self.width,3), np.uint8); self.img[:]= (32,32,32); # drawing, fill background
		self.mapimgS= self.mapimgV= None # colors maps for Saturation and Value in HSV model
		self.modX1, self.modY1, self.modY2 = sSz*2, sSz*2, 4*sSz+int((self.height-6*sSz)/2) # models position
		self.modW, self.modH = int(self.width*0.3-4*sSz), int((self.height-6*sSz)/2) # models width and height
		self.camX, self.camY1, self.camW = int(self.width*0.3), 2*sSz, int(self.width*0.4) # cam frame position
		self.mapW, self.mapH = int(self.width*0.3-6*sSz), self.height-4*sSz # color map width and height
		self.mapX, self.mapY = self.width-self.mapW-2*sSz, sSz*2 # color map position
		self.perline, self.lines = 8,4 # solution sizes
		#self.perline, self.lines = 2,2 # solution sizes
		self.warning, self.preparing = False,True # warning in colors stats and preparing model flag
		self.shown= False

	def hide(self):
		if self.shown:
			cv2.destroyWindow("window"); self.shown= False
	def show(self):
		self.shown= True
		cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN) # open main window in full-screen mode
		cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		cv2.imshow("window", self.img); cv2.waitKey(33)

	# put colored, multi-line text in the center of a rectangle
	def putTextCenter(self, txt, xy, col=(255,255,255), fsz=None, w=0, h=0, fw=1, ff=cv2.FONT_HERSHEY_SIMPLEX):
		sz= fsz if fsz else 0.5
		lines= [ (re.split('&[0-9]',l), re.findall('&[0-9]',l)) for l in re.split('\n',txt) ] # text lines and blocks
		carr= [col] if type(col)==tuple else col # initialize color array
		while True:
			twhs= [[cv2.getTextSize(b, ff, sz, fw)[0] for b in l[0]] for l in lines] # width,height for blocks in lines
			tw= max(sum(wh[0] for wh in lwh) for lwh in twhs) # text width
			lh= max(wh[1] for lwh in twhs for wh in lwh) # line height
			th= int(lh*len(twhs)+lh*(len(twhs)-1)//2) # text height
			if not fsz: sz*= min(w/tw,h/th); fsz= sz # resize text to fit w-h rectangle
			else: break
		y,c = int(xy[1]+(h-th)/2+lh), carr[0] # y-pos of the text and current color
		for i,l in enumerate(lines): # iterate lines
			x= int(xy[0]+(w-tw)/2) # x-pos of the line
			for j,b in enumerate(l[0]): # iterate blocks in line
				cv2.putText(self.img, b, (x,y), ff, sz, c, fw, cv2.LINE_AA) # write block
				if j<len(l[1]): c= carr[int(l[1][j][1])] # if have a command '&D' switch to color D
				x+= twhs[i][j][0] # next block in line
			y+= lh*3//2 # next line with 1.5 line height

	def chooseModel(self, allModelsDict): # show models
		self.img[:]= (24,24,24); n= len(allModelsDict); nr= 5; nc= int(n/nr)+1 # models, rows & columns
		ww,hh = 3*self.width*0.8/(3*nr-1),3*self.height*0.8/(3*nc-1); d= int(min(ww,hh)*0.75) # square for models
		sSz= self.sSz
		for r,(j,m) in enumerate(allModelsDict.items()): # number of model, model dict key and the model object
			if m.file=='cube333gear': m.mod3d.colors= { n:'' for n in range(54,253) }
			for i,c in enumerate(m.zero.upper()): m.mod3d.colors[i]= c.upper() # use zero cube colors
			x,y= int(self.width*0.09+ww*(r%nr)), int(self.height*0.12+hh*int(r/nr))
			cv2.rectangle(self.img, (x-sSz*3//2, y-2*sSz), (x+d+sSz*3//2, y+d+4*sSz), (32,32,32), -1)
			m.mod3d.origami(1)
			m.mod3d.draw(m, self, x, y, d, d, cmap=cubModDict[j].defcol, showcam=False) # show the model
			self.putTextCenter(str(j+1)+". "+m.file, (x,y+d+2*sSz), (255,255,255), w=d, h=4*sSz//3)
		self.show() # show choose model dialog

	def drawModels(self, cm, face=999): # draw 2d and 3d models
		hl= cm.faces[face] if 0<=face and face<len(cm.faces) else [] # items to highlight
		cm.sch2d.draw(self, self.modX1, self.modY1, self.modW, self.modH, hl=hl, colmap=cm.defcol)
		if face<len(cm.faces): cm.mod3d.draw(cm, scr, face=face, start=True)

	def drawCamFrame(self, cm, cam): # show cam frame & information under it
		self.camY2= cam.drawFrame(self, self.camX, self.camY1, self.camW, cm, cm.mode==1)+self.sSz # show frame & update camY2
		cv2.rectangle(self.img, (self.camX,self.camY2+self.sSz*4),(self.camX+self.camW,self.height), (32,32,32), -1) # clear text under camera
		if self.preparing: txt= 'preparing the model and calibrating the camera...' # message while making preparations
		else: # in work mode => show the menu
			if cm.mode==1: txt= ( # calibration mode
				f'&1ESC&0 - exit from the calibration mode\n'
				f'&1BACKSPACE&0 - remove last defined color\n'
				f'&1DEL&0 - clear all calibrated colors for the model\n'
				)+'\n'.join(list(f'&1{c}&0 - {cm.colname[c.upper()]}' for c in list(set(cm.zero.lower())))) # list all colors by name from the model
			else: txt= (
				f'&1ESC&0 - exit and choose a new model\n'
				f'&1LEFT,RIGHT arrow&0 or &19&0,&10&0 - move along faces\n'
				f'{"&1SPACE&0 - accept suggested colors" if cam.cols and cm.cmap and len(cm.cmap)>0 else ""}\n'
				f'{"&1ENTER&0 - all colors are defined, solve the cube" if scr.showColorsStat(cm) else ""}\n'
				f'{"&11&0 - calibration" if not cm.followFrame else ""}\n'
				f'{"accurately place the cube in the grid and press &11&0" if not cm.followFrame else ""}\n\n'
				f'{"&12&0 - grid auto-detection is &1"+("on" if cm.followFrame else "off")+"&0" if cm.cmap and len(cm.cmap)>0 else ""}\n'
				f'{"&13&0 - contours mask is &1"+("on" if cm.showDefMask else "off")+"&0" if cm.followFrame else ""}\n'
				f'&14&0 - solve random cube'
				)
		self.putTextCenter(txt, (self.camX, self.camY2+self.sSz*4),	[(128,128,128),(192,128,128),(32,32,32)],
			w=self.camW, h=self.height-self.camY2-6*self.sSz)

	def drawColorMap(self, cm, cam=None): # show color map and detected colors
		img,cmap = self.img, cm.cmap
		x0,y0,y1 = self.mapX, self.mapY+self.sSz, self.height-self.mapW-2*self.sSz # color circle position
		d= self.mapW-3*self.sSz; r= d/2; dr= (d-1)/2 # color circle diameter and radius
		cv2.rectangle(img, (self.camX+self.camW,0), (self.width,self.height), (32,32,32), -1) # clear

		ret= 0
		if self.mapimgS is None: # initialize and draw color map in self.mapimgS and self.mapimgV (Saturation and Value)
			self.mapimgS, self.mapimgV = np.zeros((d,d,3),dtype=np.uint8), np.zeros((d,d,3),dtype=np.uint8)
			for i in range(d): # draw color map
				for j in range(d):
					x,y = dr-i, dr-j; vs= int(math.hypot(x,y)/r*256); h= (math.atan2(x,y)/math.pi+1)*90; hi= int(h)
					ret+= h-hi
					if ret>=1: ret-= 1; hi+= 1
					self.mapimgS[i][j]= ( hi, vs, 192 ) if vs<256 else (32,32,32)
					self.mapimgV[i][j]= ( hi, 255, vs ) if vs<256 else (32,32,32)
			self.mapimgS= cv2.cvtColor(self.mapimgS, cv2.COLOR_HSV2BGR)
			self.mapimgV= cv2.cvtColor(self.mapimgV, cv2.COLOR_HSV2BGR)
		img[y0:y0+d,x0:x0+d]= self.mapimgS; cv2.circle(img, (x0+int(r),y0+int(r)), int(r)+2, (48,48,48), 3, cv2.LINE_AA)
		img[y1:y1+d,x0:x0+d]= self.mapimgV; cv2.circle(img, (x0+int(r),y1+int(r)), int(r)+2, (48,48,48), 3, cv2.LINE_AA)

		srtCol= { x[0]:1 for x in sorted([[c,v] for c in cmap for v in cmap[c]], key=lambda z:z[1][0]) }.keys()
		for i,c in enumerate(srtCol): # sorted colors by minimal hue
			col= (255,255,255) if i%2==0 else (0,0,0) # current point-set color - black or white
			xyS,xyV = ( list(
				(int(r+r*p[j]*cos(p[0]*2)/256), int(r+r*p[j]*sin(p[0]*2)/256)) for p in cmap[c] ) for j in (1,2) )
			if len(xyS)==0: continue
			for j,xy in enumerate((xyS,xyV)):
				yy= (y0,y1)[j]
				for x,y in xy: cv2.rectangle(img, (x0+x-1,yy+y-1), (x0+x,yy+y), col, 1)
				xa,xb,ya,yb = (min(p[0] for p in xy), max(p[0] for p in xy), # bounding rectangle
					min(p[1] for p in xy), max(p[1] for p in xy))
				xc,yc = int((xb+xa)/2), int((yb+ya)/2) # bounding circle
				rr= int(max( math.hypot(xc-p[0], yc-p[1]) for p in xy )+5)
				a= math.atan2(yc-dr,xc-dr)
				r1= math.hypot(xc-dr,yc-dr)+rr
				r2= max(r1+1, int(r+ self.sSz*(1 if i%2==0 else 5)//2))
				r3= int(r2+self.sSz)
				(ax,ay),(bx,by),(cx,cy) = ( (int(dr+math.cos(a)*rx), int(dr+math.sin(a)*rx)) for rx in (r1,r2,r3) )
				cv2.circle(img, (x0+xc,yy+yc), rr, col, 1, cv2.LINE_AA);
				cv2.line(img, (x0+ax,yy+ay), (x0+bx,yy+by), col, 1, cv2.LINE_AA)
				cv2.circle(img, (x0+cx,yy+cy), self.sSz, col, 1, cv2.LINE_AA)
				self.putTextCenter(c, (x0+cx-self.sSz//2,yy+cy-self.sSz//2), (128,128,128), w=self.sSz, h=self.sSz)

		if cam: # show list of average HSV colors for each item
			for p in cam.avgHSV.values():
				for j in (0,1):
					yy= (y0,y1)[j]
					x,y= int(r+r*p[j+1]*cos(p[0]*2)/256), int(r+r*p[j+1]*sin(p[0]*2)/256)
					cv2.rectangle(self.img, (x0+x-1,yy+y-1), (x0+x+1,yy+y+1), (128,255,128), 2)

	def showColorsStat(self, cm, maxface=False): # color statistics for current model, like R:3/9, G:5/9
		sSz= self.sSz
		acl= [ cm.sch2d.colors[i] for f in cm.faces for i in cm.faces[f] if i in cm.sch2d.colors] # all colors list
		ccn= { c:acl.count(c) for c in set(acl) } # colors count hash
		csa,isok,isne = [],True,True # list with text stats for each color, stats ok flag and not empty flag
		x1,y1,x2,y2 = self.camX, self.camY2, self.camX+self.camW, self.camY2+sSz*4
		for c in set(cm.zero.upper()): # unique colors from zero cube
			ccn[c]= ccn[c] if c in ccn else 0
			cc= cm.zero.upper().count(c)
			csa.append( c+':'+('&1' if ccn[c]!=cc else '')+str(ccn[c])+'&0/'+str(cc) ) # text for the color
			if ccn[c]>0 and isne: isne= False
			if ccn[c]!=cc: isok= False
		cv2.rectangle(self.img, (x1,y1), (x2,y2), (32,32,32), -1 ) # clear area under the camera
		if not isne: # color statistics text
			self.putTextCenter('  '.join(csa), (x1,y1+sSz//2), [(128,128,128),(128,128,192)], w=self.camW, h=3*sSz//2)
		if not isok and (maxface or self.warning): # if last face (trying to run solver)
			if maxface: self.warning= True
			scr.putTextCenter('something wrong with the colors, recheck',
				(x1,y1+2*sSz), (128,128,192), w=self.camW, h=3*sSz//2)
		return isok

	def showMessage(self, msg):
		x1,y1,x2,y2 = self.camX, self.camY2, self.camX+self.camW, self.camY2+self.sSz*4
		cv2.rectangle(self.img, (x1,y1), (x2,y2), (32,32,32), -1 ) # clear area under the camera
		self.putTextCenter(msg, (x1,y1+2*self.sSz), [(128,128,128),(128,128,192)], w=self.camW)

	def showSolutionPage(self, cm, solution, page, npages): # draw solution page
		self.img[:]= (32,32,32); x0,y0 = self.sSz*3,self.sSz*3
		w,h = int((self.width-2*x0)/self.perline),int((self.height-2*x0)/self.lines)
		x,y = x0,y0; i=0
		for v in solution: # for each position in the solution
			cm.mod3d.draw(cm, self, x, y, int(w*0.8), int(h*0.8), cmap=cm.defcol, # draw the position
				mshow=cm.moves.moveshow[v], mmark=cm.moves.movemark[v], showcam=False)
			self.putTextCenter(cm.moves.movename[v] if v in cm.moves.movename else "", # show move name like Uw2'
				(x+int(w/20),y+int(h/20)-20), fsz=0.75)
			cm.mod3d.doTheMove(cm.moves.moves[v]) # move the model
			if i%self.perline==self.perline-1: x=x0; y+= h; i+= 1 # last in the line
			elif v==-1: # next line
				if i%self.perline>0: x=x0; y+= h; i= (int(i/self.perline)+1)*self.perline
			else: # arrow to next move
				cv2.arrowedLine(self.img, (int(x+0.85*w),int(y+0.4*w)), (int(x+0.95*w),int(y+0.4*w)),
					(64,64,64), 3, cv2.LINE_AA )
				cv2.arrowedLine(self.img, (int(x+0.85*w),int(y+0.4*w)), (int(x+0.95*w),int(y+0.4*w)),
					(0,0,0), 1, cv2.LINE_AA )
				x+= w; i+= 1
		if page==npages-1 or len(solution)==0: # solved cube as a final step
			cm.mod3d.draw(cm, self, x, y, int(w*0.8), int(h*0.8), cmap=cm.defcol, showcam=False)
		if npages>1: # page navigator
			self.putTextCenter('page '+str(page+1)+'/'+str(npages)+' use Left-Right arrows or page number 1,2,... to go through the pages',
				(self.width/2, self.height-0.5*y0), (80,80,80), fsz=0.9)
		self.show()


# Cam Processing - find colors for the cube in the video cam frame
# capture frame, save colors, find nearest color, find colors & draw cam frame
class ProcessCam:
	def __init__(self): # initialize cam and calc sizes
		self.cap= cv2.VideoCapture(0)
		if not self.cap.isOpened(): print('Error: cannot open webcam'); exit(-3)
		self.height=64 # cam image - Width x 64
		self.camw, self.camh = self.cap.get(3), self.cap.get(4) # cam frame size
		self.ratio= self.height/self.camh; self.width= int(self.camw*self.ratio)
		self.ratio*= 2 # we will use centered 1/4 of the captured frame, so ratio is *2
		self.asstack= [] # auto-detect solutions stack

	def getFrame(self): # capture cam frame
		ret,frame= self.cap.read()
		frame= frame[ int(self.camh/4):int(self.camh*3/4) , int(self.camw/4):int(self.camw*3/4) ] # 1/4 of the frame
		self.scrFrame= cv2.flip(frame,1) # flip & save source frame
		self.frame= cv2.resize(self.scrFrame, None, fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_AREA)

	def saveColors(self, cm, updateMap=False): # save colors in the model and write updated color map to colors.pkl
		for n in self.cols: # update model's colors
			cm.sch2d.colors[n]= cm.mod3d.colors[n]= self.cols[n]
		if updateMap: # force update colors.pkl
			for n,c in self.cols.items():
				if c not in cm.cmap: cm.cmap[c]= []
				cm.cmap[c].append([int(self.avgHSV[n][0]), int(self.avgHSV[n][1]), int(self.avgHSV[n][2])])
			cm.cmap[c]= cm.cmap[c][-16:] # keep last 16 color points
			pkl= open('colors.pkl', 'wb'); pickle.dump(allcolors, pkl); pkl.close()
			#f= open('allcolors.py','w'); f.write('data={\n')
			#for fn,cl in allcolors.data.items(): # write out {'cube333.cr': {'R':[[75,125,100],[76,126,101],...],...}
			#	L=[ f"'{cn}':[[{'],['.join( ','.join(str(v) for v in c) for c in lst )}]],\n" for cn,lst in cl.items() ]
			#	f.write( f"'{fn}':{{ \n{''.join(L)}\t}},\n" )
			#f.write('}'); f.close()

	def nearlestColor(self, cm, tc, face=None): # find nearest color: model; color; face of the model, None - every face
		dist= lambda z: ( # my color distance not ideal but fast
			(
				 ( sin(tc[0]*2)*tc[1]*tc[2] - sin(z[0]*2)*z[1]*z[2] )**2
				+( cos(tc[0]*2)*tc[1]*tc[2] - cos(z[0]*2)*z[1]*z[2] )**2
			)/255**2
			+ ( tc[2]-z[2] )**2
		)
		# cncd - list 'R':100,'R':110,'G':120 - distances to the face colors
		cncd= [ (cn,dist(c)) for cn,cl in cm.cmap.items() if face is None or not cm.facecol or cn in cm.facecol[face] for c in cl ]
		return min( cncd, key=lambda z:z[1], default=[''] ) # best (color, min distance)

	def findColors(self, cm, face, calibrate=False): # main function - find colors for each item of the face
		self.faceItems= cm.faces[face] # save items of the face for drawing
		self.cols= {} # defined colors
		pl= [ p for pp in [ cm.sch2d.poly[n] for n in self.faceItems ] for p in pp ] # all points of the face
		minX,maxX= min(p[0] for p in pl), max(p[0] for p in pl) # bounding square of the face in model coordinates
		minY,maxY= min(p[1] for p in pl), max(p[1] for p in pl)
		self.getFrame(); hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV) # initialize & convert to HSV
		cf= self.camP= self.brcont= None
		ww,hh= self.width, self.height

		if cm.followFrame: # auto-detect mode => find best grid
			fmask = np.zeros(self.frame.shape[:2], np.uint8) # create mask
			XY1,XY2,self.brcont = [],[],[]
			for dd in (0.07,0.05,0.03,0.02,0.01): # decrease color delta values to find < number items * 2.5 contours
				dH,dSV = int(180*dd),int(255*dd) # delta values for Hue and Saturation-Value
				for m in cm.cmap.values(): # to build the mask iterate all map's colors, m is a list of all HSV triplets
					minH,maxH= max(min(z[0] for z in m)-dH,0), min(max(z[0] for z in m)+dH,179)
					minS,maxS= max(min(z[1] for z in m)-dSV,0), min(max(z[1] for z in m)+dSV,255)
					minV,maxV= max(min(z[2] for z in m)-dSV,0), min(max(z[2] for z in m)+dSV,255)
					if maxH-minH>90: # Hue is spitted into two diapasons like [167-18] => [167,180],[0,18]
						minH,maxH = min(z[0] for z in m if z[0]>90), max(z[0] for z in m if z[0]<=90)
						rg1= cv2.inRange(hsv, (0,minS,minV), (maxH,maxS,maxV))
						rg2= cv2.inRange(hsv, (minH,minS,minV), (179,maxS,maxV))
						fmask= cv2.bitwise_or(cv2.bitwise_or(fmask, rg1), rg2)
					else:
						rg= cv2.inRange(hsv, (minH,minS,minV), (maxH,maxS,maxV))
						fmask= cv2.bitwise_or(fmask, rg)
				contours= cv2.findContours(fmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] #[1]
				if contours is not None:
					for c in contours:
						x,y,w,h= cv2.boundingRect(c)
						self.brcont.append([x,y,min(x+w+1,ww),min(y+h+1,hh)])
						if ww*0.1<x and x<ww*0.9 and hh*0.1<y and y<hh*0.9: XY1.append((x-2,y-2))
						if ww*0.1<x+w and x+w<ww*0.9 and hh*0.1<y+h and y+h<hh*0.9: XY2.append((x+w+1,y+h+1))
					if min(len(XY1),len(XY2))<=len(self.faceItems)*2.5: break

			bestw= None # bestw is maximum grid width in suitable maxd range
			for X1,Y1 in XY1: # iterate candidates for grid from contours boundaries
				for X2,Y2 in XY2: # then check that w/h ratio is close enough to model's
					if X2-X1>ww*0.25 and Y2-Y1>hh*0.25 and abs((X2-X1)/(Y2-Y1)-(maxX-minX)/(maxY-minY))<0.05:
						_cf= ( (X2-X1)/(maxX-minX) + (Y2-Y1)/(maxY-minY) )/2 # a candidate is (X1,Y1)=>(X2,Y2)
						maxd= None # maximum distance from face item to a map color
						if cm.showDefMask: # green dots are projection of candidate rectangles
							for x in (X1,X2): cv2.line(self.frame, (x,hh-1), (x,hh-1),(0,255,0))
							for y in (Y1,Y2): cv2.line(self.frame, (ww-1,y), (ww-1,y),(0,255,0))
						for n,pp in filter( lambda x: x[0] in self.faceItems, cm.sch2d.poly.items() ): # iterate face items
							pp= [ ((maxX-p[0])*_cf+X1, (p[1]-minY)*_cf+Y1) for p in pp ] # points in cam coordinates
							x1,y1= min([p[0] for p in pp]), min([p[1] for p in pp]) # item's min x,y in cam coordinates
							x2,y2= max([p[0] for p in pp]),max([p[1] for p in pp]) # item's max x,y in cam coordinates
							L= [ (x1+(x2-x1)*0.05+(p[0]-x1)*0.9, y1+(y2-y1)*0.05+(p[1]-y1)*0.9) for p in pp ]
							camCP= [np.array(L, dtype=np.int32 )] # adjusted items in cam coordinates
							mask= np.zeros(self.frame.shape[:2], np.uint8) # define mask for the item
							cv2.fillPoly(mask, camCP, 255); cv2.polylines(mask, camCP, 1, 0, thickness=2)
							avgBGR= cv2.mean(self.frame, mask)
							avgHSV= cv2.cvtColor(np.array([[avgBGR]],dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0][:3]
							d= self.nearlestColor(cm, avgHSV, face)[1] # distance to the color
							if maxd is None or d>maxd: maxd= d
						if maxd is not None and maxd<2000 and (bestw is None or bestw<X2-X1): # the candidate is better
							bestw,cf,x0,y0= X2-X1,_cf,X1,Y1

			# add to the stack and keep last 7 elements
			self.asstack.append((cf,x0,y0) if cf else None); self.asstack= self.asstack[-7:]; cf= None
			for i in range(0,len(self.asstack)): # if among last 7 elements we have 3 similar => thats it!
				e1= self.asstack[-i-1] # e1 goes from end of the stack
				if e1:
					_cf,_x0,_y0,cnt= e1[0],e1[1],e1[2],1 # average values counters
					for j in range(0,len(self.asstack)-1-i): # e2 is from 0 position till e1
						e2= self.asstack[j] # similar here means +-10% in each dimension, if so => add to average values counters
						if e2 and abs((e1[0]-e2[0])/e1[0])<0.1 and abs((e1[1]-e2[1])/e1[1])<0.1 and abs((e1[2]-e2[2])/e1[2])<0.1:
							_cf+= e2[0]; _x0+= e2[1]; _y0+= e2[2]; cnt+= 1
					if cnt>2: cf,x0,y0 = _cf/cnt, _x0/cnt, _y0/cnt; break # we have 3 similar

		else: # not auto-detect => place grid to 50% of height in the center
			cf= hh*0.5/(maxY-minY)
			x0,y0 = (ww-(maxX-minX)*cf)/2 , (hh-(maxY-minY)*cf)/2

		if cf:
			self.xy= {} # centers of face items
			self.avgBGR,self.avgHSV= {},{} # average colors in BGR and HSV
			self.camP= {} # face items polygons in cam coordinates item=>polygon
			for n,pp in filter( lambda x: x[0] in self.faceItems, cm.sch2d.poly.items() ): # items of the face
				pp= [ ((maxX-p[0])*cf+x0, (p[1]-minY)*cf+y0) for p in pp ] # points in cam coordinates
				x1,y1 = min([p[0] for p in pp]), min([p[1] for p in pp]) # item bounding square
				x2,y2 = max([p[0] for p in pp]), max([p[1] for p in pp])
				self.xy[n]= sum([p[0] for p in pp])/len(pp) , sum([p[1] for p in pp])/len(pp) # centers
				self.camP[n]= [ (x1+(x2-x1)*0.05+(p[0]-x1)*0.9, y1+(y2-y1)*0.05+(p[1]-y1)*0.9) for p in pp ]
				camCP= [ np.array( self.camP[n], dtype=np.int32 ) ] # array to draw it on the mask
				mask= np.zeros(self.frame.shape[:2], np.uint8) # define mask for the item
				cv2.fillPoly(mask, camCP, 255); cv2.polylines(mask, camCP, 1, 0, thickness=2)
				self.avgBGR[n]= cv2.mean(self.frame, mask) # get average BGR color of the poly & save to show on bar
				self.avgHSV[n]= cv2.cvtColor(np.array([[self.avgBGR[n]]],dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
				self.cols[n]= '' if calibrate else self.nearlestColor(cm, self.avgHSV[n], face)[0] # nearest or clear
		else:
			self.avgHSV={}; self.cols= {}

	def drawFrame(self, scr, x0, y0, w0, cm, calibrate): # draw the frame with positioned face 2d model
		if calibrate: self.getFrame() # for calibration mode - get new frame
		w,h= self.width, self.height; h0= int(w0*h/w)
		fx,fy= w0/self.scrFrame.shape[1], h0/self.scrFrame.shape[0] # ratios to fit screen position
		scr.img[y0:y0+h0,x0:x0+w0]= cv2.resize(self.scrFrame, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
		if cm.followFrame and cm.showDefMask and self.brcont:
			for r in self.brcont: # draw contours in screen coordinates
				pA= x0+int(w0*r[0]/w), y0+int(r[1]*h0/h)-1 # left top
				pB= x0+int(r[2]*w0/w), y0+int(h0*r[3]/h)-1 # right bottom
				cv2.rectangle(scr.img, pA, pB, (0,255,0), 1)
		if self.camP:
			for n in self.faceItems: # color boxes, convert from cam to screen coordinates
				camAP= [np.array( [ (x0+int(p[0]*w0/w),y0+int(p[1]*h0/h)) for p in self.camP[n] ], dtype=np.int32 )]
				cv2.polylines(scr.img, camAP, 1, (128,128,128) if calibrate else self.avgBGR[n], thickness=2)
				if calibrate: cv2.fillPoly(scr.img, camAP, self.avgBGR[n]) # if calibrating - fill the whole item
				x,y,d = x0+int(self.xy[n][0]*w0/w)-15, y0+int(self.xy[n][1]*h0/h)-15, 30 # box for color and name
				cv2.fillPoly(scr.img, [np.array([[x+1,y+1],[x+d,y+1],[x+d,y+d],[x+1,y+d]])], self.avgBGR[n])
				scr.putTextCenter(self.cols[n], (x+d//10,y+d//10), (255,255,255), None, 4*d//5, 4*d//5)
		cv2.imshow("window", scr.img)
		return y0+h0 # return height of cam frame



# PROGRAM STARTS HERE
#
file, filecmd = sys.argv[1] if len(sys.argv)>1 else None, sys.argv[2] if len(sys.argv)>2 else None
scr, cam = MainScreen(), ProcessCam() # screen and cam objects
print('model='+file if file else 'None', 'command='+filecmd if filecmd else 'None')
cm= CubeModel(file) if file else None # current model if defined

while True: # main loop - initialize detection of the cube and start reading colors

	if not cm: # choose a model if not defined in parameters
		cubModDict= {} # all models dictionary
		for j,f in enumerate(models): cubModDict[j] = CubeModel(f) # cube types - load all files
		scr.chooseModel(cubModDict)
		while not cm: # choose model loop if model is not defined
			c= cv2.waitKeyEx(33)
			if ord('1')<=c and c<=ord(str(len(models))): cm= cubModDict[c-ord('1')] # 1,2,3 - cube number
			elif c==27: exit(0) # exit from the program
		scr.img[:]= (32,32,32)
		scr.putTextCenter("compiling...",(scr.width/2,scr.height/2), fsz=2)
		scr.show(); # show waiting screen screen

	cm.algo.compile(cm, filecmd=='compile') # compile the model

	if filecmd=='rand':  cube= cm.algo.random(cm) # generate random cube
	elif filecmd is not None and len(filecmd)==len(cm.zero): cube= filecmd # the cube from command line
	else: cube= None

	c= None
	if not cube: # if cube is not defined
		face,cm.mode = 0,0, # current face; mode - find colors(=0) or calibration(=1); screen
		scr.warning,scr.preparing = False,True # inconsistent colors flag; model preparation flags
		scr.img[:]= (32,32,32); scr.show() # clear screen
		cm.sch2d.colors= cm.mod3d.colors= {n:'' for n in range(54,253)} if cm.file=='cube333gear' else {} # clear colors
		scr.drawColorMap(cm); cam.findColors(cm, face); scr.drawCamFrame(cm, cam); scr.drawModels(cm) # show everything
		cm.mod3d.draw(cm, scr, scr.modX1, scr.modY2+scr.modH-scr.modW, scr.modW, scr.modW, cm.defcol, start=True)
		while not cm.mod3d.draw(cm, scr): # show origami cartoon
			cam.getFrame(); cam.camP= {}; scr.drawCamFrame(cm, cam); cv2.imshow("window", scr.img); cv2.waitKey(1)
		scr.drawColorMap(cm); scr.drawModels(cm,face); scr.preparing= False # initialize screen objects

		time0,frameCnt= datetime.datetime.now(), 0
		while True: # find colors loop
			c= cv2.waitKeyEx(1) # keyboard processing & show screen updates

			if cm.mode==0: # find colors mode
				if face>=len(cm.faces): # all faces are filled
					if not scr.showColorsStat(cm,True): face-=1; scr.drawModels(cm,face)
					else: break
				cm.mod3d.draw(cm, scr); cam.findColors(cm, face); scr.drawCamFrame(cm, cam); scr.drawColorMap(cm, cam)
				if c==27: cm= None; break # escape
				elif c==13: face= len(cm.faces) # enter - try to checkout
				elif c==32 and cam.cols: # space - save & move to the next face
					cam.saveColors(cm); face+= 1; scr.drawModels(cm,face); scr.showColorsStat(cm)
				elif c in (65361,65364,65366,2424832,2228224,2621440,57) and face>0: face-= 1; scr.drawModels(cm,face) # left arrow
				elif c in (65363,65362,65365,2162688,2490368,2555904,48) and face<len(cm.faces)-1: face+= 1; scr.drawModels(cm,face) # right arrow
				elif c==50: cm.followFrame= not cm.followFrame # '2' - trigger auto-detect flag
				elif c==51: cm.showDefMask= not cm.showDefMask # '3' - trigger showDefMask flag
				elif c==52: cube= cm.algo.random(cm); break # '4' - random cube
				elif c==49: # '1' - go to calibration mode
					calcam= cam.findColors(cm, face, True)
					colorIndex= 0; cam.cols[cm.faces[face][colorIndex]]= '?'
					cm.mode= 1

			elif cm.mode==1: # color calibration mode
				scr.drawCamFrame(cm, cam)
				if c==27: cm.mode= 0; continue # go to normal mode
				elif c in (ord(x) for x in set(cm.zero.lower())): # color letter like r,g,b,o,w,y
					cam.cols[ cm.faces[face][colorIndex] ]= chr(c).upper() # set upper letter for current item
					colorIndex+= 1 # next item
					if colorIndex>=len(cm.faces[face]): # colors are set, go to next face
						cam.saveColors(cm,True); scr.drawColorMap(cm)
						face+= 1; scr.drawModels(cm,face); scr.showColorsStat(cm)
						cm.mode= 0
					else:
						cam.cols[ cm.faces[face][colorIndex] ]= '?' # highlight next item
				elif c==8 and colorIndex>0: # backspace - clear last color
					cam.cols[cm.faces[face][colorIndex]]= ''
					colorIndex-= 1; cam.cols[ cm.faces[face][colorIndex] ]= '?'
				elif c in (65535,3014656): cm.cmap.clear(); scr.drawColorMap(cm) # del - clear palette

			frameCnt+= 1; time1= datetime.datetime.now(); time= (time1-time0).total_seconds()
			if time>1:
				cv2.rectangle(scr.img, (0,0), (50,30), (32,32,32), -1)
				scr.putTextCenter( '%.1f'%(time*frameCnt), (25,15), (64,64,64), 0.5 )
				time0,frameCnt= time1,0

	else:
		cm.mod3d.origami(1)

	if c!=27: # colors are defined and not escape => run solver & show solution
		if not cube: # build cube string with defined colors from the model
			cube= ''.join([cm.sch2d.colors[c].lower() for c in sorted(cm.sch2d.colors.keys())])
		else:
			for i,c in enumerate(cube): cm.mod3d.colors[i]= c.upper() # predefined cube - write colors to the model
		print('cube=',cube,'\n')
		if cm.file=='cube333gear':
			for n in range(54,253): cm.mod3d.colors[n]= cm.sch2d.colors[n]= ''
		page= 0; pagePos= { 0:cm.mod3d.colors.copy() } # current page and first
		solPages= cm.algo.run(cm, scr, cube) # run solver and get paged solution

		scr.showSolutionPage(cm, solPages[page] if len(solPages)>0 else [], page, len(solPages)) # first page
		while True: # walk throw pages; wait for escape, then clear colors and start again
			c= cv2.waitKeyEx(33)
			if c==27:
				cube= file= filecmd= cm= None; break
			elif c in (65361,65364,65366,2424832,2228224,2621440) and page>0: # left arrow - previous page
				page-= 1; cm.mod3d.colors= pagePos[page].copy()
				scr.showSolutionPage(cm, solPages[page], page, len(solPages))
			elif c in (65363,65362,65365,2162688,2490368,2555904) and page<len(solPages)-1: # right arrow - next page
				page+= 1; pagePos[page]= cm.mod3d.colors.copy()
				scr.showSolutionPage(cm, solPages[page], page, len(solPages))
			elif ord('1')<=c and c<=ord('9'):
				p= c-ord('1')
				if p<len(solPages):
					if p<page:
						page= p; cm.mod3d.colors= pagePos[page].copy()
						scr.showSolutionPage(cm, solPages[page], page, len(solPages))
					else:
						for i in range(page,p):
							page+= 1; pagePos[page]= cm.mod3d.colors.copy()
							scr.showSolutionPage(cm, solPages[page], page, len(solPages))
			elif c!=-1: print('c=',c)
