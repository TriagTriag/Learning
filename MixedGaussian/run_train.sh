mean_1=(0.1 0.3 2.0 6.0)
mean_2=(-0.1)
var_1=(0.1 2.0)
var_2=(0.1 0.5 4.0)
dim=1
num_samples=(10000 1000 10000)
num_tests=2000
#mean_1=(2.0)
#mean_2=(-0.1)
#var_1=(0.1)
#var_2=(0.1)
#dim=1
#num_samples=(10000)
#num_tests=2000
# code of a for loop to remove all folders between 1 to 100

# size_seeds=${#seeds[@]}
size_mean_1=${#mean_1[@]}
size_mean_2=${#mean_2[@]}
size_var_1=${#var_1[@]}
size_var_2=${#var_2[@]}
size_num_samples=${#num_samples[@]}
size_dim=${#dim[@]}
size_num_tests=${#num_tests[@]}


# find the remainder of $1 divided by size_mean_1, and call it $i, then find the divisor of $1 divided by size_mean_1, and find its remainder when divided by size_mean_2, and call it $j, then find the divisor of $1 divided by size_mean_1 and size_mean_2, and find its remainder when divided by size_var_1, and call it $k, then find the divisor of $1 divided by size_mean_1, size_mean_2, and size_var_1, and find its remainder when divided by size_var_2, and call it $l, then find the divisor of $1 divided by size_mean_1, size_mean_2, size_var_1, and size_var_2, and find its remainder when divided by size_num_samples, and call it $m, then find the divisor of $1 divided by size_mean_1, size_mean_2, size_var_1, size_var_2, and size_num_samples, and find its remainder when divided by size_dim, and call it $n, then find the divisor of $1 divided by size_mean_1, size_mean_2, size_var_1, size_var_2, size_num_samples, and size_dim, and find its remainder when divided by size_num_tests, and call it $o.

# i=$(( $1 %  ))
j=$(( ($1 ) % $size_mean_2 ))
k=$(( ($1  / $size_mean_2) % $size_var_1 ))
l=$(( ($1 / $size_mean_2 / $size_var_1) % $size_var_2 ))
m=$(( ($1 / $size_mean_2 / $size_var_1 / $size_var_2) % $size_num_samples ))
n=$(( ($1  / $size_mean_2 / $size_var_1 / $size_var_2 / $size_num_samples) % $size_dim ))
o=$(( ($1  / $size_mean_2 / $size_var_1 / $size_var_2 / $size_num_samples / $size_dim) % $size_num_tests ))
p=$(( ($1  / $size_mean_2 / $size_var_1 / $size_var_2 / $size_num_samples / $size_dim / $size_num_tests) % $size_mean_1 ))

# run over the values of $i, $j, $k, $l, $m, $n, and $o, and for each value, write a code that runs banded.py with the value of the corresponding variable of mean_1, mean_2, var_1, var_2, num_samples, dim, and num_tests, respectively.
# run over 20 seeds
for t in {0..19}
do
    echo "starting seed $t"
    echo $t
    python3 banded.py --hidden-dim 10 --mean-1 ${mean_1[$p]} --mean-2 ${mean_2[$j]} --var-1 ${var_1[$k]} --var-2 ${var_2[$l]} --N ${num_samples[$m]} --dim ${dim[$n]} --N-test ${num_tests[$o]} --seed $t
done
# python3 banded.py --hidden-dim 10 --mean-1 ${mean_1[$p]} --mean-2 ${mean_2[$j]} --var-1 ${var_1[$k]} --var-2 ${var_2[$l]} --N ${num_samples[$m]} --dim ${dim[$n]} --N-test ${num_tests[$o]} --seed ${seeds[$i]}

# # run tot.py similarly only if $1 is divisible by 100 and $1 is larger than 0
# if [ $1 -gt 0 ] && [ $(( $1 % 100 )) -eq 0 ]
# then
#     python3 avg.py --hidden-dim 10 --mean-1 ${mean_1[$i]} --mean-2 ${mean_2[$j]} --var-1 ${var_1[$k]} --var-2 ${var_2[$l]} --N ${num_samples[$m]} --dim ${dim[$n]} --N-test ${num_tests[$o]} --seed ${seeds[$p]}
# fi
