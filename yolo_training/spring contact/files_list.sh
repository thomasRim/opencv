
OUTPUTNAME="list.txt"
EXTENSION="*"
PATHBASE="/content/gdrive/MyDrive/darknet/yolov4_training/images"

if [ $# -eq 0 ]; then
    echo "Variable should be a path to project folder with pods!!! \n"
    echo "sh podrun.sh [path_to_project_with_podfile] \n"
    exit 1
fi

while [[ $# -gt 0 ]]
do
   key="$1"

   case $key in
      -o|--output)
      OUTPUTNAME="$2"
      shift # past argument
      shift # past value
      ;;
      -e|--extension)
      EXTENSION="$2"
      shift # past argument
      shift # past value
      ;;
      -p|--path)
      PATHBASE="$2"
      shift # past argument
      shift # past value
      ;;
   esac
done


yourfilenames=`ls *.$EXTENSION`
for eachfile in $yourfilenames
do
   echo "$PATHBASE/$eachfile" >> $OUTPUTNAME
done

