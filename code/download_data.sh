export DOWNLOAD_PATH="data/"
#wget https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/ -P  $download_path

# test download path
wget https://www.mpi-inf.mpg.de/fileadmin/inf/d2/xucong/MPIIGaze/model.png -P $DOWNLOAD_PATH
file=/data/model.png
if [-f $file];
then
    echo "successfully downloaded"
fi

wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip -P $DOWNLOAD_PATH
unzip code/MPIIFaceGaze.zip
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze_normalized.zip -P $DOWNLOAD_PATH
unzip code/MPIIFaceGaze_normalized.zip
