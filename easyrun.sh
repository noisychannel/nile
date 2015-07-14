TUNE_KBEST=$1
DEV_KBEST=$2

./bin/pro_ebleu $1 $2 > model

echo 'Tuning scores:'
cat $TUNE_KBEST | python util/oracle.py --one_best
./bin/rerank model $TUNE_KBEST | python util/oracle.py
cat $TUNE_KBEST | python util/oracle.py

echo 'Dev scores:'
cat $DEV_KBEST | python util/oracle.py --one_best
./bin/rerank model $DEV_KBEST | python util/oracle.py
cat $DEV_KBEST | python util/oracle.py
