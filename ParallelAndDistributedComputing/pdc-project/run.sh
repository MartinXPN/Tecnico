make
./matFact$2 instances/$1.in > instances/$1-ours$2.out
printf "\033[0;31m\n"
cmp instances/$1.out instances/$1-ours$2.out || printf "✗ Error: files are different\n\n"
printf "\033[0m"