cd AD_o1/
for ((i=1;i<=10;i++))
do
rm -r $i
cp -r ../pack $i
cd $i
# conda env list
echo -e '\n'
echo -e '\n'
pwd
echo "num: $i"

python refrac_fw2_o1.py papers/paper_ad.md

cd ..
done
cd ..

# cd BC_o3/
# for ((i=1;i<=10;i++))
# do
# rm -r $i
# cp -r ../pack $i
# cd $i
# # conda env list
# echo -e '\n'
# echo -e '\n'
# pwd
# echo "num: $i"

# python refrac_fw2_o3.py papers/paper_BC_DN.md

# cd ..
# done
# cd ..

# cd ADR_o3/
# for ((i=1;i<=10;i++))
# do
# rm -r $i
# cp -r ../pack $i
# cd $i
# # conda env list
# echo -e '\n'
# echo -e '\n'
# pwd
# echo "num: $i"

# python refrac_fw2_o3.py papers/paper_ADR3.md

# cd ..
# done
# cd ..

# cd phys_test_claude
# for ((i=1;i<=10;i++))
# do
# rm -r $i
# cp -r ../pack $i
# cd $i
# # conda env list
# echo -e '\n'
# echo -e '\n'
# pwd
# echo "num: $i"

# python refracfw_v2_o3.py papers/paper_ad_physeval.md

# cd ..
# done
# cd ..

# cd NN_o3/
# for ((i=1;i<=10;i++))
# do
# rm -r $i
# cp -r ../pack $i
# cd $i
# # conda env list
# echo -e '\n'
# echo -e '\n'
# pwd
# echo "num: $i"

# python refrac_fw2_o3.py papers/paper_NN.md

# cd ..
# done