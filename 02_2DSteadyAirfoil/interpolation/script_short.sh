#echo "Second arg: $1"

faces="$1/faces"
points="$1/points"
boundary="$1/boundary"
owner="$1/owner"
faces1="$1/faces1"
points1="$1/points1"
startFace="$1/startFace"
owner1="$1/owner1"

sed -e '1,20d' ${faces} > a
sed -e '$d' a > b
sed -e '$d' b > a
sed -e '$d' a > b
sed -e '$d' b > ${faces1}
rm a b
sed -e 's/.$//' ${faces1} > a
cut -c3- a > ${faces1}

sed -e '1,20d' ${points} > a
sed -e '$d' a > b
sed -e '$d' b > a
sed -e '$d' a > b
sed -e '$d' b > ${points1}
rm a b
sed -e 's/.$//' ${points1} > a
cut -c2- a > ${points1}

grep startFace ${boundary} > a
sed -e 's/.$//' a > ${startFace}
rm a

sed -e '1,21d' ${owner} > a
sed -e '$d' a > b
sed -e '$d' b > a
sed -e '$d' a > b
sed -e '$d' b > ${owner1}
rm a b

 
