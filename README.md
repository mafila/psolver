# psolver
9 puzzle solver detects puzzle state from connected camera and solves the puzzle:
- Rubik's 2x2x3 (Tower Cube)
- Rubik's 2x2x2 (Pocket Cube)
- Rubik's 3x3x3
- Rubik's 4x4x4 (Rubik's Revenge)
- Rubik's 5x5x5 (Professor's Cube)
- Pyraminx
- Skewb
- 3x3x3 Gear Cube
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
On the first run you need to choose and calibrate colors for your puzzle.
Accurately place the face of the puzzle in the grid and press '1', then enter unique colors of your puzzle ('r' for red, 'w' for white, etc).
After the unique colors are set enter face colors one by one by pressing corresponding key.
When the face is finished rotate puzzle as shown and enter colors for the next face.
Continue until all colors are defined

![sample](https://raw.githubusercontent.com/mafila/psolver/master/sample.png)

## console usage
Puzzles codes: cube223, cube222, cube333, cube444, cube555, pyraminx, ftoctahedron, skewb, cube333gear.
Colors order is defined in corresponding cfg/\*.cr file. For cubes the order is UFRBLD
```
./psolve.py cube333
./psolve.py cube333 rand
./psolve.py cube333 yyrwrwwyogbwowroyobobbgwgrgwrbgyywbyrboobgggbygyrowror
./psolve.py cube444 URFDLB RBFDDDLDUBFBRRFLBUBFRFDLURBDUUULFDDUFDRFRULFLFRRURDBLUUFBUFLDLRBDLRUBDBLBFBUBLFFLDUFULRLBRLRDBDR
./psolve.py cube555 compile
bin/cube555 `bin/cube555 rand`
./psolve.py cube333 compile
bin/cube333 yyrwrwwyogbwowroyobobbgwgrgwrbgyywbyrboobgggbygyrowror
tst/run cube333 00 100
tst/show cube333 00
```

## performance
Simple puzzles (cube223, cube222, pyraminx, skewb and cube333gear) are solved in an optimal way in several milliseconds

cube333 uses modified Kociemba's two-phase algorithm, it takes about 7 seconds

cube444, cube555 uses several-phase algorithm, major know-how is in pairing edges.
The search for the solution takes about 1 min

Face-turning Octahedron also works about 1 min

Result depends on CPU frequency and number of cores, version of CPU AVX instructions and GCC version.
Average move count based on 10000 runs for various CPU:

|CPU|cube333|ftoctahedron|cube444|cube555|
|:---|:---:|:---:|:---:|:---:|
|Intel Xeon Platinum 8124M @ 3.00GHz, 18 cores, 36 threads	|19.1|34.5|47.5|81.4|
|Intel Core i7-4770 @ 3.40GHz, 4 cores, 8 threads			|19.2|35.2|47.6|82.1|
|Intel Xeon CPU E3-1270 V2 @ 3.50GHz, 4 cores, 8 threads	|19.2|35.2|47.6|82.3|
|Intel Core i7-7500U @ 2.70GHz, 2 cores, 4 threads			|19.3|35.9|48.2|83.2|
|Intel Atom CPU  C2350  @ 1.74GHz, 2 cores, 2 threads		|19.6|36.5|51.2|85.3|
