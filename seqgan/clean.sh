vars=("$@")
rm visdom.out
echo "moving experiment ${vars[0]}"
mv main.out "saved_output/main_${vars[0]}.out"