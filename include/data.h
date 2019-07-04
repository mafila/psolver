#define ZERO_CUBE "rrrrrrrrrbbbbbbbbbooooooooogggggggggwwwwwwwwwyyyyyyyyy"
#define N_STEPS 6
#define N_BLOCKS 54
#define ALG_NAME "cube333"
#define N_MOVES 18
#define N_TURNS 23
#define N_MAX_SOL 30000
#define N_MAX_MOVESETS 1
#define N_MAX_PRUNES 2
#define N_MAX_PARAMS 16
#define REVCOL_FROM "rbw"
#define REVCOL_TO "ogy"
int stepHashDepth[N_STEPS]={6,6,6,7,7,7};
int stepStartDepth[N_STEPS]={0,0,0,0,0,0};
int stepMaxDepth[N_STEPS]={255,255,255,255,255,255};
int stepMaxWeights[N_STEPS]={0,0,0,0,0,0};
int stepSeq[N_STEPS]={1,0,0,0,0,-1};
int stepMaxSol[N_STEPS]={30000,30000,30000,1,1,1};
int solTime[N_STEPS]={1000,1000,1000,1000,1000,3000};
int solTimeHard[N_STEPS]={0,0,0,0,0,0};
int stepTime[N_STEPS]={1500,1500,30000,5000,5000,30000};
int stepLink[N_STEPS]={-1,-1,-1,0,1,2};
int stepLinkLocal[N_STEPS]={0,0,0,0,0,0};
float stepParams[N_STEPS][N_MAX_PARAMS]={{},{},{},{},{},{}};
int nStepPruneSymmetry[N_STEPS][N_MAX_PRUNES]={{0,0},{},{},{0,0},{},{}};
int nStepMasks[N_STEPS][N_MAX_PRUNES]={{0,0},{},{},{0,0},{},{}};
int stepMasks[N_STEPS][N_MAX_PRUNES][N_BLOCKS]={{{},{}},{{}},{{}},{{},{}},{{}},{{}}};
int nStepPrunes[N_STEPS]={2,2,2,2,2,2};
size_t stepPruneSize[N_STEPS][N_MAX_PRUNES]={{((size_t)1)<<24,((size_t)1)<<24},{((size_t)1)<<24,((size_t)1)<<24},{((size_t)1)<<24,((size_t)1)<<24},{((size_t)1)<<28,((size_t)1)<<28},{((size_t)1)<<28,((size_t)1)<<28},{((size_t)1)<<28,((size_t)1)<<28}};
char stepMd5[N_STEPS][N_MAX_PRUNES][33]={{"5b40fd8d0c66861497a5d294e54df216","194d501d82049834f9769714d2d013ce"},{"9fe67696d35e4d92e9936df890f3c048","ac36243f87bba2321729229aa81af0f8"},{"37d9bfa65d93c5d0fb081f84316c2312","9b6fb5ad02a007447b31b95c8571de94"},{"f8bbb898e1ffbb8e0f3ec725979fb09b","85c37fce3665fb2d4994f17507a51fb4"},{"a51f8ebff803fb8e3ebdc9bf6ccd95d4","f6d4bf6bbe488ce48423bf627c09d242"},{"767b8d0c443af28a53b013652b13494f","ce054d2ba6cb5066c5f171ab50558b1b"}};
int nStepMoveSets[N_STEPS]={1,1,1,1,1,1};
uint8_t stepMoveSetsLen[N_STEPS][N_MAX_MOVESETS]={{1},{1},{1},{1},{1},{1}};
uint8_t stepMoves[N_STEPS][N_MAX_MOVESETS][N_MAX_MS_MOVES][N_MOVES+1]={{{{18,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}}},{{{18,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}}},{{{18,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}}},{{{10,12,13,15,16,2,5,8,11,14,17}}},{{{10,0,1,3,4,2,5,8,11,14,17}}},{{{10,6,7,9,10,2,5,8,11,14,17}}}};
