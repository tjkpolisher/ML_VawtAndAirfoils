#echo "Second arg: $1"

#points="$1/points"
points1="$1/polyMesh/points1"
points_computation="$1/polyMesh/points"
points_orig="$1/polyMesh/points_orig"

echo ${point1}


#cp ${points_computation} $1


sed -e '1,20d' ${points_computation} > a
sed -e '$d' a > b
sed -e '$d' b > a
sed -e '$d' a > b
sed -e '$d' b > ${points1}
rm a b
sed -e 's/.$//' ${points1} > a
cut -c2- a > ${points1}

