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
Requires Git LFS, Python 3.6+, OpenCV 4.1+, GCC 7.4+

Connected web camera required for this application

Ubuntu 18.04 installation
```
$ python3 --version
Python 3.6.8
$ sudo apt update
$ sudo apt install git python3-pip gcc make curl software-properties-common
$ sudo -H python3 -m pip install opencv-python screeninfo
$ sudo add-apt-repository ppa:git-core/ppa
$ curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
$ sudo apt install git-lfs
$ git lfs install
$ git clone https://github.com/mafila/psolver
```


## usage
```
cd psolver
python3 psolver.py
```
Choose the puzzle by pressing key '1'-'9'.
At first run you need to calibrate colors for the puzzle, 
accurately place face of the puzzle in the grid and press '1', then enter face colors one by one.
When the face is finished rotate puzzle as shown and enter colors for the next face.
Continue until all colors are defined.

