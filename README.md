# psolver
9 puzzle solver detects puzzle state from connected camera and solves the puzzle:
- Rubik's 2x2x3
- Rubik's 2x2x2
- Rubik's 3x3x3
- Rubik's 4x4x4
- Rubik's 5x5x5
- Pyraminx
- Skewb
- 3x3x3 gear
- Face-turning Octahedron

## installation
Requires Python 3.6+, OpenCV 4.1+, GCC 7.4+

###### Ubuntu 18.04
```
$ sudo apt update
$ sudo apt upgrade
$ python3 --version
Python 3.6.8
$ sudo apt install git python3-pip python3-opencv gcc make bc
$ sudo -H python3 -m pip install numpy opencv-python screeninfo
$ git clone https://github.com/mafila/psolver
$ cd psolver
$ python3 psolver.py
```

###### Windows 10
- Install latest Python from https://www.python.org/, add python to the Path environment variable
- Install latest Git from https://gitforwindows.org/
- Install 64-bit GCC from https://sourceware.org/cygwin/ (no additional packages required), add C:\cygwin64 to the Path environment variable
```
> git clone https://github.com/mafila/psolver
> python -m pip install numpy opencv-python screeninfo
> cd psolver
> python psolver.py
```

## usage
Connected web camera required for this application

Choose the puzzle by pressing key '1'-'9'.
On the first run you need to calibrate colors for your puzzle.
Accurately place the face of the puzzle in the grid and press '1', then enter face colors one by one by pressing corresponding key ('r' for red, 'w' for white, etc).
When the face is finished rotate puzzle as shown and enter colors for the next face.
Continue until all colors are defined

If you have different colors you should change 'zero' and 'colors' sections of corresponding cfg/\*.cr file

## console usage
Puzzles codes: cube223, cube222, cube333, cube444, cube555, pyraminx, ftoctahedron, skewb, cube333gear.
Colors order is defined in corresponding cfg/\*.cr file. For cubes the order is FRBLUD
```
./psolve.py cube333
./psolve.py cube333 rand
./psolve.py cube333 gbygbrwygogyyowrwgbobogbwgowoorrogybryrbwwwwbobygyryrr
./psolve.py cube333 compile
bin/cube333 `bin/cube333 rand`
bin/cube333 gbygbrwygogyyowrwgbobogbwgowoorrogybryrbwwwwbobygyryrr
tst/run cube333 00 100
tst/show cube333 00
```

## performance
Simple puzzles (cube223, cube222, pyraminx, skewb and cube333gear) are solved in an optimal way.
cube333 uses modified Kociemba's two-phase algorithm, it takes about 9 seconds.
cube444, cube555 and ftoctahedron uses several-phase algorithm, major know-how is in pairing edges.
Result depends on CPU frequency and number of cores, version of CPU AVX instructions and GCC version.
calculation takes about 1 min.


|CPU|Intel Core i7-4770 CPU @ 3.40GHz, 8 cores|Intel Xeon CPU E3-1270 V2 @ 3.50GHz, 8 cores|Intel Xeon Platinum 8124M CPU @ 3.00GHz, 36 cores|
|:---:|:---:|:---:|:---:|
|cube223| | |10.0|
|cube222| | |8.7|
|cube333| | | |
|cube333gear| | | |
|skewb| | |8.4|
|pyraminx| | |10.4|
|ftoctahedron| | | |
|cube444|47.7|47.4| |
|cube555|82.1|82.3| |
