# Script to run hmmlearn and hmmdecode

if [ "$1" ]; then
  echo "Training on tagged file: $1.....";
  python3 hmmlearn.py $1;
  echo -e "DONE\n"
else
  echo "Training on default tagged file.....";
  python3 hmmlearn.py;
  echo -e "DONE\n"
fi

if [ "$2" ]; then
  echo "Tagging raw input file: $2.....";
  python3 hmmdecode.py $2;
else
  echo "Tagging default raw input file.....";
  python3 hmmdecode.py;
fi
