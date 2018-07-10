echo 'Building Dockerfile with image name pymarl/ngc`'
#docker build --no-cache -t deepmarl/pytorch .
docker build -t pymarl/ngc -f Dockerfile.ngc .
