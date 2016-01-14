TUNE_KBEST=$1
DEV_KBEST=$2
TEST_KBEST=$3

SOURCE_SENTS=jsalt/data/medium/dev2.cs
SOURCE_EMB=jsalt/emb/cs_en.cs.50.txt
TARGET_EMB=jsalt/emb/cs_en.en.50.txt
DEV_SENTS=jsalt/data/medium/dev1.cs
TEST_SENTS=jsalt/data/medium/test.cs

GAURAV_FLAG="--gaurav $SOURCE_SENTS $SOURCE_EMB $TARGET_EMB $DEV_SENTS"

#./bin/train $1 $2 -h 50 --adadelta $GAURAV_FLAG > model

echo 'Tuning scores:'
cat $TUNE_KBEST | python util/oracle.py --one_best
./bin/rerank model $TUNE_KBEST $SOURCE_SENTS 2>/dev/null | python util/oracle.py
cat $TUNE_KBEST | python util/oracle.py

echo 'Dev scores:'
cat $DEV_KBEST | python util/oracle.py --one_best
./bin/rerank model $DEV_KBEST $DEV_SENTS 2>/dev/null | python util/oracle.py
cat $DEV_KBEST | python util/oracle.py

echo 'Test scores:'
cat $TEST_KBEST | python util/oracle.py --one_best
./bin/rerank model $TEST_KBEST $TEST_SENTS 2>/dev/null | python util/oracle.py
cat $TEST_KBEST | python util/oracle.py

