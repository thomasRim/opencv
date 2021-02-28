yourfilenames=`ls *.jpg`
for eachfile in $yourfilenames
do
   echo "/content/gdrive/MyDrive/darknet/ingress/images/$eachfile" >> train.txt
done