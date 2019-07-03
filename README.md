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
- Face turning octahedron

## installation
Requires Python 3.6+, OpenCV 4.1+, GCC 7.4+

###### Ubuntu 18.04
```
$ sudo apt update
$ sudo apt upgrade
$ python3 --version
Python 3.6.8
$ sudo apt install git python3-pip gcc make bc
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

