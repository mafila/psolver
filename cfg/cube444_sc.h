{
	//char *z= ZERO_CUBE, z5= z[5], z21=z[21], z37=z[37], z53=z[53], z69=z[69], z85=z[85];
	static char z5= 'r', z21='b', z37='o', z53='g', z69='w', z85='y';
	return 
		  (a[5]==a[6])&&(a[5]==a[9])&&(a[5]==a[10])
		&&(a[21]==a[22])&&(a[21]==a[25])&&(a[21]==a[26])
		&&(a[69]==a[70])&&(a[69]==a[73])&&(a[69]==a[74])
		&&(a[37]==a[38])&&(a[37]==a[41])&&(a[37]==a[42])
		&&(a[53]==a[54])&&(a[53]==a[57])&&(a[53]==a[58])
		&&(
				  (z5==a[5]&&z21==a[21]&&z37==a[37]&&z53==a[53]&&z69==a[69]&&z85==a[85])
				||(z53==a[5]&&z5==a[21]&&z21==a[37]&&z37==a[53]&&z69==a[69]&&z85==a[85])
				||(z85==a[5]&&z21==a[21]&&z69==a[37]&&z53==a[53]&&z5==a[69]&&z37==a[85])
				||(z37==a[5]&&z53==a[21]&&z5==a[37]&&z21==a[53]&&z69==a[69]&&z85==a[85])
				||(z21==a[5]&&z37==a[21]&&z53==a[37]&&z5==a[53]&&z69==a[69]&&z85==a[85])
				||(z53==a[5]&&z85==a[21]&&z21==a[37]&&z69==a[53]&&z5==a[69]&&z37==a[85])
				||(z69==a[5]&&z53==a[21]&&z85==a[37]&&z21==a[53]&&z5==a[69]&&z37==a[85])
				||(z21==a[5]&&z69==a[21]&&z53==a[37]&&z85==a[53]&&z5==a[69]&&z37==a[85])
				||(z37==a[5]&&z21==a[21]&&z5==a[37]&&z53==a[53]&&z85==a[69]&&z69==a[85])
				||(z53==a[5]&&z37==a[21]&&z21==a[37]&&z5==a[53]&&z85==a[69]&&z69==a[85])
				||(z5==a[5]&&z53==a[21]&&z37==a[37]&&z21==a[53]&&z85==a[69]&&z69==a[85])
				||(z21==a[5]&&z5==a[21]&&z53==a[37]&&z37==a[53]&&z85==a[69]&&z69==a[85])
				||(z69==a[5]&&z21==a[21]&&z85==a[37]&&z53==a[53]&&z37==a[69]&&z5==a[85])
				||(z53==a[5]&&z69==a[21]&&z21==a[37]&&z85==a[53]&&z37==a[69]&&z5==a[85])
				||(z85==a[5]&&z53==a[21]&&z69==a[37]&&z21==a[53]&&z37==a[69]&&z5==a[85])
				||(z21==a[5]&&z85==a[21]&&z53==a[37]&&z69==a[53]&&z37==a[69]&&z5==a[85])
				||(z85==a[5]&&z5==a[21]&&z69==a[37]&&z37==a[53]&&z53==a[69]&&z21==a[85])
				||(z37==a[5]&&z85==a[21]&&z5==a[37]&&z69==a[53]&&z53==a[69]&&z21==a[85])
				||(z69==a[5]&&z37==a[21]&&z85==a[37]&&z5==a[53]&&z53==a[69]&&z21==a[85])
				||(z5==a[5]&&z69==a[21]&&z37==a[37]&&z85==a[53]&&z53==a[69]&&z21==a[85])
				||(z85==a[5]&&z37==a[21]&&z69==a[37]&&z5==a[53]&&z21==a[69]&&z53==a[85])
				||(z5==a[5]&&z85==a[21]&&z37==a[37]&&z69==a[53]&&z21==a[69]&&z53==a[85])
				||(z69==a[5]&&z5==a[21]&&z85==a[37]&&z37==a[53]&&z21==a[69]&&z53==a[85])
				||(z37==a[5]&&z69==a[21]&&z5==a[37]&&z85==a[53]&&z21==a[69]&&z53==a[85])
		);
}
